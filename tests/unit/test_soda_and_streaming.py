"""Unit tests for incremental SODA client, failure-safe watermarks, and micro-batch streaming."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import requests
from pyspark.sql import SparkSession

from chicago_crimes.core.config import StreamingConfig
from chicago_crimes.ingestion.kafka_producer import ingest_soda_to_kafka
from chicago_crimes.ingestion.soda_client import (
    SodaWatermark,
    fetch_soda_incremental,
    load_watermark,
    save_watermark,
    soda_record_to_bronze_dict,
)
from chicago_crimes.storage.lakehouse import read_table
from chicago_crimes.streaming.crime_stream import (
    process_kafka_microbatch,
    reconcile_stream_targets,
)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        if not self._responses:
            return _FakeResponse(200, [])
        return self._responses.pop(0)

    def close(self) -> None:
        return None


def test_soda_pagination_retry_and_missing_geo_mapping(tmp_path: Path) -> None:
    """SODA fetch handles 429 retry, composite cursor order, pagination, and omitted geo keys."""
    wm_path = tmp_path / "wm.json"
    save_watermark(
        wm_path,
        SodaWatermark(
            last_updated_on="2026-11-12T10:00:00.000",
            last_id="9000",
            records_fetched=1,
        ),
    )

    page1 = [
        {
            "id": "1001",
            "case_number": "JK1001",
            "date": "2026-11-12T08:00:00.000",
            "block": "070XX S CREGIER AVE",
            "iucr": "0890",
            "primary_type": "THEFT",
            "description": "FROM BUILDING",
            "location_description": "APARTMENT",
            "arrest": False,
            "domestic": True,
            "beat": "0332",
            "district": "003",
            "ward": "5",
            "community_area": "43",
            "fbi_code": "06",
            "year": "2026",
            "updated_on": "2026-11-12T10:05:00.000",
            # Omitted x_coordinate, y_coordinate, latitude, longitude, location like live SODA
        },
        {
            "id": "1002",
            "case_number": "JK1002",
            "date": "2026-11-12T09:00:00.000",
            "block": "010XX N STATE ST",
            "iucr": "0486",
            "primary_type": "BATTERY",
            "description": "SIMPLE",
            "location_description": "STREET",
            "arrest": True,
            "domestic": False,
            "beat": "1832",
            "district": "018",
            "ward": "42",
            "community_area": "8",
            "fbi_code": "08B",
            "x_coordinate": "1176352",
            "y_coordinate": "1907549",
            "year": "2026",
            "updated_on": "2026-11-12T10:15:00.000",
            "latitude": "41.9012",
            "longitude": "-87.6285",
            "location": {"latitude": "41.9012", "longitude": "-87.6285"},
        },
    ]
    page2 = [
        {
            "id": "1003",
            "case_number": "JK1003",
            "date": "2026-11-12T09:30:00.000",
            "block": "020XX W MADISON ST",
            "iucr": "0560",
            "primary_type": "ASSAULT",
            "description": "SIMPLE",
            "location_description": "SIDEWALK",
            "arrest": False,
            "domestic": False,
            "beat": "1223",
            "district": "012",
            "ward": "27",
            "community_area": "28",
            "fbi_code": "08A",
            "year": "2026",
            "updated_on": "2026-11-12T10:20:00.000",
        }
    ]
    fake_session = _FakeSession(
        [
            _FakeResponse(429, {"error": "rate_limit"}, headers={"Retry-After": "0"}),
            _FakeResponse(200, page1),
            _FakeResponse(200, page2),
        ]
    )
    sleeps: list[float] = []

    res = fetch_soda_incremental(
        app_token="test-token",
        watermark_path=wm_path,
        page_size=2,
        overlap_seconds=60,
        session=fake_session,  # type: ignore[arg-type]
        sleep_fn=sleeps.append,
    )

    assert sleeps == [0.0]
    assert res.pages_fetched == 2
    assert len(res.records) == 3
    expected_where = (
        "updated_on >= '2026-11-12T09:59:00.000' "
        "AND (updated_on != '2026-11-12T10:00:00.000' OR id > '9000')"
    )
    assert expected_where == res.effective_where
    assert fake_session.calls[0]["headers"]["X-App-Token"] == "test-token"
    assert fake_session.calls[1]["params"]["$offset"] == 0
    assert fake_session.calls[2]["params"]["$offset"] == 2
    assert fake_session.calls[1]["params"]["$order"] == "updated_on ASC, id ASC"

    # Omitted coordinates remain None; present dict location formats cleanly
    assert res.records[0]["Latitude"] is None
    assert res.records[0]["Longitude"] is None
    assert res.records[0]["Arrest"] == "false"
    assert res.records[1]["Location"] == "(41.9012, -87.6285)"
    assert res.records[1]["Arrest"] == "true"

    # fetch_soda_incremental must NOT commit the watermark file on disk prematurely
    persisted = load_watermark(wm_path)
    assert persisted is not None and persisted.last_id == "9000"
    assert res.next_watermark.last_id == "1003"
    assert res.next_watermark.last_updated_on == "2026-11-12T10:20:00.000"


def test_watermark_not_advanced_on_kafka_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Kafka publish raises, the SODA watermark cursor must stay untouched."""
    wm_path = tmp_path / "wm.json"
    initial = SodaWatermark(
        last_updated_on="2026-11-12T10:00:00.000",
        last_id="500",
        records_fetched=5,
    )
    save_watermark(wm_path, initial)

    class _FakeFetch:
        records = [
            soda_record_to_bronze_dict(
                {"id": "501", "updated_on": "2026-11-12T10:05:00.000", "primary_type": "THEFT"},
                batch_id="b1",
            )
        ]
        next_watermark = SodaWatermark(
            last_updated_on="2026-11-12T10:05:00.000",
            last_id="501",
            records_fetched=1,
            last_batch_id="b1",
        )
        batch_id = "b1"
        effective_where = "updated_on >= '2026-11-12T09:59:00.000'"
        pages_fetched = 1
        has_more = False
        backlog_detected = False
        ingestion_lag_seconds = 12.0

    monkeypatch.setattr(
        "chicago_crimes.ingestion.kafka_producer.fetch_soda_incremental",
        lambda **kwargs: _FakeFetch(),
    )
    monkeypatch.setattr(
        "chicago_crimes.ingestion.kafka_producer.ensure_kafka_topics",
        lambda *args, **kwargs: {"chicago-crimes.raw": "exists"},
    )

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Kafka broker unavailable")

    monkeypatch.setattr(
        "chicago_crimes.ingestion.kafka_producer.publish_bronze_records_to_kafka",
        _boom,
    )

    cfg = StreamingConfig(watermark_path=str(wm_path))
    with pytest.raises(RuntimeError, match="Kafka broker unavailable"):
        ingest_soda_to_kafka(cfg, publish_kafka=True)

    after = load_watermark(wm_path)
    assert after is not None
    assert after.last_id == "500"
    assert after.last_updated_on == "2026-11-12T10:00:00.000"


def _kafka_row(
    payload: str,
    *,
    offset: int,
    topic: str = "chicago-crimes.raw",
    partition: int = 0,
) -> tuple[bytes, str, int, int, str]:
    return (payload.encode("utf-8"), topic, partition, offset, "2026-11-14T12:00:00")


def test_microbatch_dlq_quarantine_and_idempotent_out_of_order_merge(
    spark: SparkSession, tmp_path: Path
) -> None:
    """Verify DLQ, Silver quarantine, accounting, and monotonic Delta upsert."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"
    dlq_dir = tmp_path / "dlq"
    metrics_log = tmp_path / "metrics.jsonl"

    rec_101_v2 = soda_record_to_bronze_dict(
        {
            "id": "101",
            "case_number": "JK0101",
            "date": "2026-11-10T14:00:00.000",
            "block": "001XX N STATE ST",
            "iucr": "0890",
            "primary_type": "THEFT",
            "description": "RETAIL THEFT",
            "location_description": "DEPARTMENT STORE",
            "arrest": False,
            "domestic": False,
            "beat": "0111",
            "district": "001",
            "ward": "42",
            "community_area": "32",
            "fbi_code": "06",
            "year": "2026",
            "updated_on": "2026-11-10T16:00:00.000",
            # Omitted coords
        },
        batch_id="batch-1",
    )
    rec_101_v1_stale = {**rec_101_v2, "Arrest": "true", "Updated On": "2026-11-10T15:00:00.000"}
    rec_102 = {
        **rec_101_v2,
        "ID": "102",
        "Case Number": "JK0102",
        "Latitude": "41.881",
        "Longitude": "-87.627",
        "Updated On": "2026-11-10T16:05:00.000",
    }
    rec_missing_id = {**rec_101_v2, "ID": None, "Case Number": "JK0BAD"}

    batch1_rows = [
        _kafka_row(json.dumps(rec_101_v1_stale), offset=0),
        _kafka_row(json.dumps(rec_101_v2), offset=1),
        _kafka_row(json.dumps(rec_102), offset=2),
        _kafka_row(json.dumps(rec_missing_id), offset=3),
        _kafka_row("{broken-json", offset=4),
    ]
    schema = "value BINARY, topic STRING, partition INT, offset LONG, timestamp STRING"
    df1 = spark.createDataFrame(batch1_rows, schema=schema)

    stats1 = process_kafka_microbatch(
        df1,
        batch_id=1,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
        metrics_log_path=metrics_log,
    )
    assert stats1.kafka_input_rows == 5
    assert stats1.malformed_json_rows == 1
    assert stats1.bronze_written_rows == 4
    assert stats1.silver_accepted_deduped == 2
    assert stats1.silver_superseded == 1
    assert stats1.silver_quarantined == 1
    assert stats1.is_reconciled is True

    # Batch 2: stale event for ID 101 must NOT revert Arrest back to true; newer v3 for 102 wins
    rec_101_even_older = {**rec_101_v2, "Arrest": "true", "Updated On": "2026-11-10T12:00:00.000"}
    rec_102_v2 = {**rec_102, "Arrest": "true", "Updated On": "2026-11-11T09:00:00.000"}
    df2 = spark.createDataFrame(
        [
            _kafka_row(json.dumps(rec_101_even_older), offset=5),
            _kafka_row(json.dumps(rec_102), offset=6),  # duplicate replay
            _kafka_row(json.dumps(rec_102_v2), offset=7),
        ],
        schema=schema,
    )
    stats2 = process_kafka_microbatch(
        df2,
        batch_id=2,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
        metrics_log_path=metrics_log,
    )
    assert stats2.is_reconciled is True

    silver_rows = {
        r["ID"]: r
        for r in read_table(spark, silver_dir, table_format="delta").collect()
    }
    assert set(silver_rows.keys()) == {101, 102}
    assert silver_rows[101]["Arrest"] is False
    assert silver_rows[101]["has_valid_geo"] is False
    assert silver_rows[102]["Arrest"] is True
    assert silver_rows[102]["has_valid_geo"] is True

    recon = reconcile_stream_targets(
        spark,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
    )
    assert recon["bronze_rows"] == 7
    assert recon["silver_rows"] == 2
    assert recon["silver_id_unique"] is True
    assert recon["quarantine_rows"] == 1
    assert recon["dlq_rows"] == 1
    assert recon["all_sinks_unique"] is True


def test_microbatch_crash_replay_is_idempotent_across_all_sinks(
    spark: SparkSession, tmp_path: Path
) -> None:
    """Crash recovery replaying the same microbatch must not duplicate any sink or metric."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"
    dlq_dir = tmp_path / "dlq"
    metrics_log = tmp_path / "metrics.jsonl"

    base_rec = soda_record_to_bronze_dict(
        {
            "id": "201",
            "case_number": "JK0201",
            "date": "2026-11-12T10:00:00.000",
            "block": "001XX N STATE ST",
            "iucr": "0890",
            "primary_type": "THEFT",
            "description": "RETAIL THEFT",
            "location_description": "DEPARTMENT STORE",
            "arrest": False,
            "domestic": False,
            "beat": "0111",
            "district": "001",
            "ward": "42",
            "community_area": "32",
            "fbi_code": "06",
            "year": "2026",
            "latitude": "41.881",
            "longitude": "-87.627",
            "updated_on": "2026-11-12T12:00:00.000",
        },
        batch_id="crash-test",
    )
    bad_id_rec = {**base_rec, "ID": None, "Case Number": "JK_BAD"}
    rows = [
        _kafka_row(json.dumps(base_rec), offset=10),
        _kafka_row(json.dumps(bad_id_rec), offset=11),
        _kafka_row("{malformed", offset=12),
    ]
    df = spark.createDataFrame(
        rows, schema="value BINARY, topic STRING, partition INT, offset LONG, timestamp STRING"
    )

    def _fail_after_quarantine(stage: str, bid: int) -> None:
        if stage == "after_quarantine":
            raise RuntimeError(f"Injected crash after quarantine for batch {bid}")

    with pytest.raises(RuntimeError, match="Injected crash after quarantine"):
        process_kafka_microbatch(
            df,
            batch_id=0,
            bronze_target=bronze_dir,
            silver_target=silver_dir,
            quarantine_target=quarantine_dir,
            dlq_target=dlq_dir,
            table_format="delta",
            metrics_log_path=metrics_log,
            on_stage=_fail_after_quarantine,
        )

    assert not silver_dir.exists()
    assert not metrics_log.exists()

    # Replaying batch 0 cleanly must deduplicate DLQ, Bronze, Quarantine, Silver & Metrics
    stats = process_kafka_microbatch(
        df,
        batch_id=0,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
        metrics_log_path=metrics_log,
    )
    assert stats.is_reconciled is True
    # Also replay after full completion
    # (simulating crash between metrics write and checkpoint commit)
    stats_again = process_kafka_microbatch(
        df,
        batch_id=0,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
        metrics_log_path=metrics_log,
    )
    assert stats_again.is_reconciled is True

    recon = reconcile_stream_targets(
        spark,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
    )
    assert recon["bronze_rows"] == 2
    assert recon["quarantine_rows"] == 1
    assert recon["dlq_rows"] == 1
    assert recon["silver_rows"] == 1
    assert recon["all_sinks_unique"] is True

    lines = [ln for ln in metrics_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


def test_microbatch_crash_recovery_never_reverts_newer_silver_record(
    spark: SparkSession, tmp_path: Path
) -> None:
    """Crash recovery replaying a stale or null-timestamp batch must never revert Silver data."""
    bronze_dir = tmp_path / "bronze_crash"
    silver_dir = tmp_path / "silver_crash"
    quarantine_dir = tmp_path / "quarantine_crash"
    dlq_dir = tmp_path / "dlq_crash"
    metrics_log = tmp_path / "metrics_crash.jsonl"
    schema = "value BINARY, topic STRING, partition INT, offset LONG, timestamp STRING"

    rec_v2_new = soda_record_to_bronze_dict(
        {
            "id": "555",
            "case_number": "JK0555",
            "date": "2026-11-12T10:00:00.000",
            "block": "001XX N STATE ST",
            "iucr": "0890",
            "primary_type": "THEFT",
            "description": "CONFIRMED_NEWEST",
            "location_description": "STREET",
            "arrest": True,
            "domestic": False,
            "beat": "0111",
            "district": "001",
            "ward": "42",
            "community_area": "32",
            "fbi_code": "06",
            "year": "2026",
            "updated_on": "2026-11-12T18:00:00.000",
        },
        batch_id="b-new",
    )
    df_new = spark.createDataFrame(
        [_kafka_row(json.dumps(rec_v2_new), offset=100)], schema=schema
    )
    process_kafka_microbatch(
        df_new,
        batch_id=1,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
        metrics_log_path=metrics_log,
    )

    # Simulate a delayed/replayed batch containing both an older timestamp and a null timestamp
    # that crashes mid-flight after Bronze write and is then re-executed on recovery.
    rec_v1_old = {
        **rec_v2_new,
        "Description": "STALE_OLDER_VERSION",
        "Arrest": "false",
        "Updated On": "2026-11-12T09:00:00.000",
    }
    rec_null_ts = {
        **rec_v2_new,
        "Description": "NULL_TIMESTAMP_VERSION",
        "Arrest": "false",
        "Updated On": None,
    }
    df_stale_and_null = spark.createDataFrame(
        [
            _kafka_row(json.dumps(rec_v1_old), offset=101),
            _kafka_row(json.dumps(rec_null_ts), offset=102),
        ],
        schema=schema,
    )

    def _crash_after_bronze(stage: str, bid: int) -> None:
        if stage == "after_bronze":
            raise RuntimeError(f"Crash after bronze in batch {bid}")

    with pytest.raises(RuntimeError, match="Crash after bronze"):
        process_kafka_microbatch(
            df_stale_and_null,
            batch_id=2,
            bronze_target=bronze_dir,
            silver_target=silver_dir,
            quarantine_target=quarantine_dir,
            dlq_target=dlq_dir,
            table_format="delta",
            metrics_log_path=metrics_log,
            on_stage=_crash_after_bronze,
        )

    # Recovery replay of batch 2 completes cleanly without reverting ID=555 in Silver
    stats_recovered = process_kafka_microbatch(
        df_stale_and_null,
        batch_id=2,
        bronze_target=bronze_dir,
        silver_target=silver_dir,
        quarantine_target=quarantine_dir,
        dlq_target=dlq_dir,
        table_format="delta",
        metrics_log_path=metrics_log,
    )
    assert stats_recovered.is_reconciled is True
    row_555 = (
        read_table(spark, silver_dir, table_format="delta")
        .filter("ID = 555")
        .first()
    )
    assert row_555 is not None
    assert row_555["Description"] == "CONFIRMED_NEWEST"
    assert row_555["Arrest"] is True


def test_soda_consecutive_polls_catchup_and_identical_timestamps(tmp_path: Path) -> None:
    """Consecutive polls with >100 pending rows and identical `updated_on` never skip or stall."""
    wm_path = tmp_path / "wm_catchup.json"
    # 250 pending records: first 180 share the EXACT SAME updated_on timestamp,
    # next 70 share a second timestamp.
    all_rows = [
        {
            "id": f"{10000 + i}",
            "case_number": f"JK{10000 + i}",
            "date": "2026-11-10T12:00:00.000",
            "primary_type": "THEFT",
            "description": f"ITEM_{i}",
            "arrest": False,
            "domestic": False,
            "year": "2026",
            "updated_on": (
                "2026-11-10T15:41:56.000" if i < 180 else "2026-11-10T15:45:00.000"
            ),
        }
        for i in range(250)
    ]

    # Poll 1: limit=100, page_size=50 -> fetches 2 pages (rows 0..99),
    # detects has_more=True & backlog_detected=True.
    sess1 = _FakeSession(
        [
            _FakeResponse(200, all_rows[0:50]),
            _FakeResponse(200, all_rows[50:100]),
        ]
    )
    res1 = fetch_soda_incremental(
        watermark_path=wm_path,
        since="2026-11-10T15:00:00.000",
        page_size=50,
        max_records=100,
        overlap_seconds=60,
        session=sess1,  # type: ignore[arg-type]
        now_utc=datetime(2026, 11, 10, 22, 0, 0),
    )
    assert len(res1.records) == 100
    assert res1.pages_fetched == 2
    assert res1.has_more is True
    assert res1.backlog_detected is True
    assert res1.ingestion_lag_seconds is not None and res1.ingestion_lag_seconds > 0
    assert res1.next_watermark.last_updated_on == "2026-11-10T15:41:56.000"
    assert res1.next_watermark.last_id == "10099"
    save_watermark(wm_path, res1.next_watermark)

    # Poll 2: resumes from saved watermark (has_more=True) with limit=100 ->
    # must use strict composite cursor and fetch rows 100..199 without starvation!
    sess2 = _FakeSession(
        [
            _FakeResponse(200, all_rows[100:150]),
            _FakeResponse(200, all_rows[150:200]),
        ]
    )
    res2 = fetch_soda_incremental(
        watermark_path=wm_path,
        page_size=50,
        max_records=100,
        overlap_seconds=60,
        session=sess2,  # type: ignore[arg-type]
        now_utc=datetime(2026, 11, 10, 22, 5, 0),
    )
    expected_poll2_where = (
        "(updated_on > '2026-11-10T15:41:56.000' "
        "OR (updated_on = '2026-11-10T15:41:56.000' AND id > '10099'))"
    )
    assert res2.effective_where == expected_poll2_where
    assert len(res2.records) == 100
    assert res2.has_more is True
    assert res2.next_watermark.last_updated_on == "2026-11-10T15:45:00.000"
    assert res2.next_watermark.last_id == "10199"
    assert res2.next_watermark.cumulative_records_fetched == 200
    save_watermark(wm_path, res2.next_watermark)

    # Poll 3: fetches remaining 50 rows (200..249) and sees empty 2nd page ->
    # has_more=False, backlog drained!
    sess3 = _FakeSession(
        [
            _FakeResponse(200, all_rows[200:250]),
            _FakeResponse(200, []),
        ]
    )
    res3 = fetch_soda_incremental(
        watermark_path=wm_path,
        page_size=50,
        max_records=100,
        overlap_seconds=60,
        session=sess3,  # type: ignore[arg-type]
        now_utc=datetime(2026, 11, 10, 22, 10, 0),
    )
    assert len(res3.records) == 50
    assert res3.has_more is False
    assert res3.backlog_detected is False
    assert res3.next_watermark.last_id == "10249"
    assert res3.next_watermark.cumulative_records_fetched == 250
    save_watermark(wm_path, res3.next_watermark)

    fetched_ids = [r["ID"] for r in (*res1.records, *res2.records, *res3.records)]
    assert fetched_ids == [f"{10000 + i}" for i in range(250)]

    # Poll 4 (after full drain, has_more=False, overlap_seconds=60):
    # Applies 60-second overlap window while excluding `<= last_id` at `last_updated_on`!
    sess4 = _FakeSession([_FakeResponse(200, [])])
    res4 = fetch_soda_incremental(
        watermark_path=wm_path,
        page_size=50,
        max_records=100,
        overlap_seconds=60,
        session=sess4,  # type: ignore[arg-type]
        now_utc=datetime(2026, 11, 10, 22, 15, 0),
    )
    expected_poll4_where = (
        "updated_on >= '2026-11-10T15:44:00.000' "
        "AND (updated_on != '2026-11-10T15:45:00.000' OR id > '10249')"
    )
    assert res4.effective_where == expected_poll4_where
    assert len(res4.records) == 0
    assert res4.has_more is False


def test_merge_insert_only_delta_backfills_legacy_schema(
    spark: SparkSession, tmp_path: Path
) -> None:
    """Legacy Delta tables without `stream_event_id` must migrate cleanly on merge."""
    import hashlib

    from delta.tables import DeltaTable
    from pyspark.sql import Row

    from chicago_crimes.storage.delta import merge_insert_only_delta, write_delta

    expected_id_0 = hashlib.sha256(b"chicago_crimes_raw|0|10").hexdigest()
    expected_id_1 = hashlib.sha256(b"chicago_crimes_raw|0|11").hexdigest()

    # 1. Unmapped, unpartitioned legacy DLQ Delta table (no columnMapping.mode=name)
    legacy_dlq_dir = tmp_path / "legacy_dlq"
    legacy_dlq = spark.createDataFrame(
        [
            Row(
                raw_payload='{"bad": 1}',
                kafka_topic="chicago_crimes_raw",
                kafka_partition=0,
                kafka_offset=10,
                error_reason="malformed_json",
            )
        ]
    )
    legacy_dlq.write.format("delta").mode("overwrite").save(str(legacy_dlq_dir))

    incoming_dlq = spark.createDataFrame(
        [
            Row(
                raw_payload='{"bad": 1}',
                kafka_topic="chicago_crimes_raw",
                kafka_partition=0,
                kafka_offset=10,
                error_reason="malformed_json",
                stream_event_id=expected_id_0,
                microbatch_id=7,
            ),
            Row(
                raw_payload='{"bad": 2}',
                kafka_topic="chicago_crimes_raw",
                kafka_partition=0,
                kafka_offset=11,
                error_reason="malformed_json",
                stream_event_id=expected_id_1,
                microbatch_id=7,
            ),
        ]
    )
    res_dlq = merge_insert_only_delta(spark, incoming_dlq, legacy_dlq_dir)
    assert res_dlq["created"] is False

    migrated_dlq = spark.read.format("delta").load(str(legacy_dlq_dir))
    assert migrated_dlq.count() == 2
    assert "stream_event_id" in migrated_dlq.columns
    assert "microbatch_id" in migrated_dlq.columns
    by_offset_dlq = {r["kafka_offset"]: r for r in migrated_dlq.collect()}
    assert by_offset_dlq[10]["stream_event_id"] == expected_id_0
    assert by_offset_dlq[10]["microbatch_id"] is None
    assert by_offset_dlq[11]["stream_event_id"] == expected_id_1
    assert by_offset_dlq[11]["microbatch_id"] == 7

    # 2. Column-mapped, partitioned legacy Bronze Delta table with spaced columns
    legacy_bronze_dir = tmp_path / "legacy_bronze"
    legacy_bronze = spark.createDataFrame(
        [
            Row(
                **{
                    "Case Number": "HH100001",
                    "Primary Type": "THEFT",
                    "kafka_topic": "chicago_crimes_raw",
                    "kafka_partition": 0,
                    "kafka_offset": 10,
                    "year": 2026,
                }
            )
        ]
    )
    write_delta(legacy_bronze, legacy_bronze_dir, mode="overwrite", partition_by=("year",))

    incoming_bronze = spark.createDataFrame(
        [
            Row(
                **{
                    "Case Number": "HH100001",
                    "Primary Type": "THEFT",
                    "kafka_topic": "chicago_crimes_raw",
                    "kafka_partition": 0,
                    "kafka_offset": 10,
                    "year": 2026,
                    "stream_event_id": expected_id_0,
                    "microbatch_id": 9,
                }
            ),
            Row(
                **{
                    "Case Number": "HH100002",
                    "Primary Type": "BATTERY",
                    "kafka_topic": "chicago_crimes_raw",
                    "kafka_partition": 0,
                    "kafka_offset": 11,
                    "year": 2026,
                    "stream_event_id": expected_id_1,
                    "microbatch_id": 9,
                }
            ),
        ]
    )
    res_bronze = merge_insert_only_delta(
        spark, incoming_bronze, legacy_bronze_dir, partition_by=("year",)
    )
    assert res_bronze["created"] is False
    migrated_bronze = spark.read.format("delta").load(str(legacy_bronze_dir))
    assert migrated_bronze.count() == 2
    by_offset_bronze = {r["kafka_offset"]: r for r in migrated_bronze.collect()}
    assert by_offset_bronze[10]["stream_event_id"] == expected_id_0
    assert by_offset_bronze[11]["stream_event_id"] == expected_id_1
    assert by_offset_bronze[11]["microbatch_id"] == 9
    parts = (
        DeltaTable.forPath(spark, str(legacy_bronze_dir))
        .detail()
        .select("partitionColumns")
        .first()[0]
    )
    assert parts == ["year"]

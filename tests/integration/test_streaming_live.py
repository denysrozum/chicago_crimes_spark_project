"""Live integration tests for Kafka KRaft + MinIO S3A Delta crash/replay consistency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient

from chicago_crimes.core.config import AppConfig, load_config
from chicago_crimes.ingestion.kafka_producer import (
    ensure_kafka_topics,
    publish_bronze_records_to_kafka,
)
from chicago_crimes.ingestion.soda_client import soda_record_to_bronze_dict
from chicago_crimes.storage.minio import delete_prefix, ensure_bucket, list_objects, s3a_uri

REPO_ROOT = Path(__file__).resolve().parents[2]


def _kafka_bootstrap() -> str:
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS_EXTERNAL", "localhost:9094")


def _live_minio_url() -> str:
    return os.getenv("MINIO_ENDPOINT", "http://localhost:9000")


def _require_kafka_and_minio() -> tuple[str, AppConfig]:
    bootstrap = _kafka_bootstrap()
    try:
        admin = AdminClient({"bootstrap.servers": bootstrap})
        admin.list_topics(timeout=3.0)
    except Exception as exc:
        pytest.skip(f"Live Kafka broker not reachable at {bootstrap} ({exc})")

    cfg = load_config(profile="standalone")
    storage_cfg = cfg.storage.model_copy(update={"s3_endpoint": _live_minio_url()})
    try:
        ensure_bucket(storage_cfg)
    except Exception as exc:
        pytest.skip(f"Live MinIO S3 endpoint not reachable at {storage_cfg.s3_endpoint} ({exc})")

    return bootstrap, cfg.model_copy(update={"storage": storage_cfg})


_STREAM_RUNNER_CODE = """
import json, sys
from chicago_crimes.core.config import load_config
from chicago_crimes.core.spark import spark_session
from chicago_crimes.storage.lakehouse import read_table
from chicago_crimes.streaming.crime_stream import reconcile_stream_targets, run_crime_stream

params = json.loads(sys.argv[1])
base_cfg = load_config(profile="standalone")
cfg = base_cfg.model_copy(
    update={
        "spark": base_cfg.spark.model_copy(
            update={
                "profile": "local",
                "master": "local[2]",
                "driver_memory": "1g",
                "executor_memory": "1g",
                "enable_delta": True,
                "enable_kafka": True,
                "log_level": "ERROR",
            }
        ),
        "storage": base_cfg.storage.model_copy(
            update={"s3_endpoint": params["s3_endpoint"]}
        ),
        "streaming": base_cfg.streaming.model_copy(
            update={
                "kafka_bootstrap_servers": params["bootstrap"],
                "kafka_topic": params["topic"],
                "kafka_dlq_topic": params["dlq_topic"],
                "starting_offsets": "earliest",
                "checkpoint_location": params["checkpoint"],
                "bronze_stream_target": params["bronze"],
                "silver_stream_target": params["silver"],
                "quarantine_stream_target": params["quarantine"],
                "dlq_stream_target": params["dlq"],
                "metrics_log_path": params["metrics_log"],
            }
        ),
    }
)

if params["action"] == "stream":
    crash_stage = params.get("crash_stage")
    def _hook(stage: str, bid: int) -> None:
        if crash_stage and stage == crash_stage:
            raise RuntimeError(f"Injected crash at {stage} in micro-batch {bid}")
    res = run_crime_stream(
        cfg,
        available_now=True,
        ensure_topic=False,
        partition_by=("year",),
        on_stage=_hook if crash_stage else None,
    )
    print("__JSON__" + json.dumps(res, default=str))
elif params["action"] == "verify":
    with spark_session(cfg, app_name="VerifyLiveStreamState") as spark:
        rows = [
            r.asDict()
            for r in read_table(spark, params["silver"], table_format="delta").collect()
        ]
        recon = reconcile_stream_targets(
            spark,
            bronze_target=params["bronze"],
            silver_target=params["silver"],
            quarantine_target=params["quarantine"],
            dlq_target=params["dlq"],
            table_format="delta",
        )
        silver_map = {
            str(r["ID"]): {"Arrest": r["Arrest"], "has_valid_geo": r["has_valid_geo"]}
            for r in rows
        }
    print("__JSON__" + json.dumps({"silver_map": silver_map, "recon": recon}, default=str))
"""


def _run_isolated_stream_worker(payload: dict[str, object], *, check: bool = True) -> dict:
    """Run streaming/Spark work in a clean subprocess so JVM classpath always includes Kafka/S3A."""
    res = subprocess.run(
        [sys.executable, "-c", _STREAM_RUNNER_CODE, json.dumps(payload)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Stream worker failed (exit={res.returncode})\n"
            f"stdout={res.stdout}\nstderr={res.stderr}"
        )
    if not check:
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}

    for line in reversed(res.stdout.splitlines()):
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise AssertionError(f"Missing JSON output in stream worker:\n{res.stdout}\n{res.stderr}")


@pytest.mark.integration
def test_live_kafka_delta_crash_replay_quarantine_and_dlq(tmp_path: Path) -> None:
    """Publish unique Kafka events, inject inter-sink crash, recover from checkpoint & verify."""
    bootstrap, base_cfg = _require_kafka_and_minio()
    run_tag = uuid4().hex[:10]
    topic = f"chicago-crimes.it.{run_tag}"
    dlq_topic = f"chicago-crimes.dlq.it.{run_tag}"
    s3_prefix = f"test_runs/streaming_{run_tag}"

    bucket = base_cfg.storage.s3_bucket
    metrics_log = tmp_path / "microbatch_metrics.jsonl"
    common_params = {
        "bootstrap": bootstrap,
        "s3_endpoint": base_cfg.storage.s3_endpoint,
        "topic": topic,
        "dlq_topic": dlq_topic,
        "checkpoint": s3a_uri(bucket, f"{s3_prefix}/checkpoint"),
        "bronze": s3a_uri(bucket, f"{s3_prefix}/bronze"),
        "silver": s3a_uri(bucket, f"{s3_prefix}/silver"),
        "quarantine": s3a_uri(bucket, f"{s3_prefix}/quarantine"),
        "dlq": s3a_uri(bucket, f"{s3_prefix}/dlq"),
        "metrics_log": str(metrics_log),
    }

    ensure_kafka_topics(bootstrap, [topic, dlq_topic], num_partitions=1)
    try:
        # 1. Publish unique batch: newest wins for 900001, missing geo preserved for 900002,
        #    1 null-ID quarantined record, and 1 malformed JSON routed to DLQ.
        r1_old = soda_record_to_bronze_dict(
            {
                "id": "900001",
                "case_number": f"IT{run_tag[:6]}1",
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
                "updated_on": "2026-11-12T10:00:00.000",
            },
            batch_id=f"it-{run_tag}-1",
        )
        r1_new = {**r1_old, "Arrest": "true", "Updated On": "2026-11-12T12:00:00.000"}
        r2_missing_geo = {
            **r1_old,
            "ID": "900002",
            "Case Number": f"IT{run_tag[:6]}2",
            "Latitude": None,
            "Longitude": None,
            "Updated On": "2026-11-12T11:00:00.000",
        }
        r_quarantine = {
            **r1_old,
            "ID": None,
            "Case Number": f"ITQ{run_tag[:5]}",
        }
        pub = publish_bronze_records_to_kafka(
            [r1_old, r1_new, r2_missing_geo, r_quarantine],
            bootstrap_servers=bootstrap,
            topic=topic,
            batch_id=f"it-{run_tag}-1",
        )
        assert pub.delivered == 4

        raw_producer = Producer({"bootstrap.servers": bootstrap, "acks": "all"})
        raw_producer.produce(topic, key=b"bad-json", value=b"{corrupt-kafka-payload")
        assert raw_producer.flush(timeout=10.0) == 0

        # 2. Controlled crash after DLQ + Bronze writes, BEFORE Quarantine/Silver/checkpoint.
        crashed = _run_isolated_stream_worker(
            {**common_params, "action": "stream", "crash_stage": "after_bronze"},
            check=False,
        )
        assert crashed["returncode"] != 0
        assert "Injected crash at after_bronze" in (crashed["stdout"] + crashed["stderr"])
        assert list_objects(base_cfg.storage, prefix=f"{s3_prefix}/bronze/_delta_log/")
        assert list_objects(base_cfg.storage, prefix=f"{s3_prefix}/dlq/_delta_log/")
        assert not list_objects(base_cfg.storage, prefix=f"{s3_prefix}/silver/_delta_log/")
        assert not metrics_log.exists()

        # 3. Restart from exact same checkpoint; replay must stay duplicate-free.
        recovered = _run_isolated_stream_worker({**common_params, "action": "stream"})
        recon = recovered["reconciliation"]
        assert recon["bronze_rows"] == 4
        assert recon["bronze_distinct_events"] == 4
        assert recon["dlq_rows"] == 1
        assert recon["dlq_distinct_events"] == 1
        assert recon["quarantine_rows"] == 1
        assert recon["quarantine_distinct_events"] == 1
        assert recon["silver_rows"] == 2
        assert recon["silver_distinct_ids"] == 2
        assert recon["all_sinks_unique"] is True

        # 4. Publish follow-up batch: stale 900001 + newer 900002 -> monotonic Silver upsert.
        r1_stale = {**r1_old, "Arrest": "false", "Updated On": "2026-11-12T08:00:00.000"}
        r2_newer = {**r2_missing_geo, "Arrest": "true", "Updated On": "2026-11-12T15:00:00.000"}
        pub2 = publish_bronze_records_to_kafka(
            [r1_stale, r2_newer],
            bootstrap_servers=bootstrap,
            topic=topic,
            batch_id=f"it-{run_tag}-2",
        )
        assert pub2.delivered == 2

        second = _run_isolated_stream_worker({**common_params, "action": "stream"})
        recon2 = second["reconciliation"]
        assert recon2["bronze_rows"] == 6
        assert recon2["dlq_rows"] == 1
        assert recon2["quarantine_rows"] == 1
        assert recon2["silver_rows"] == 2
        assert recon2["all_sinks_unique"] is True

        audit = _run_isolated_stream_worker({**common_params, "action": "verify"})
        assert audit["silver_map"]["900001"]["Arrest"] is True
        assert audit["silver_map"]["900002"]["Arrest"] is True
        assert audit["silver_map"]["900002"]["has_valid_geo"] is False
        assert audit["recon"]["all_sinks_unique"] is True

        metric_entries = [
            json.loads(line)
            for line in metrics_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(metric_entries) == 2
        assert all(entry["is_reconciled"] is True for entry in metric_entries)
    finally:
        try:
            AdminClient({"bootstrap.servers": bootstrap}).delete_topics([topic, dlq_topic])
        except Exception:
            pass
        try:
            delete_prefix(base_cfg.storage, f"{s3_prefix}/")
        except Exception:
            pass

"""Idempotent Kafka topic management and crime event producer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from confluent_kafka import KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from chicago_crimes.core.config import StreamingConfig
from chicago_crimes.core.logging import get_logger
from chicago_crimes.ingestion.soda_client import (
    fetch_soda_incremental,
    save_watermark,
)

logger = get_logger("ingestion.kafka")


@dataclass(frozen=True)
class KafkaPublishResult:
    """Delivery summary returned once Kafka acknowledges all produced events."""

    topic: str
    batch_id: str
    attempted: int
    delivered: int
    failed: int
    partitions: list[int]
    watermark_saved_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_kafka_topics(
    bootstrap_servers: str,
    topics: Sequence[str],
    *,
    num_partitions: int = 2,
    replication_factor: int = 1,
    timeout_seconds: float = 15.0,
) -> dict[str, str]:
    """Idempotently create Kafka topics if absent and return per-topic status."""
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin.list_topics(timeout=timeout_seconds)
    statuses: dict[str, str] = {}
    missing: list[NewTopic] = []
    for topic in topics:
        if topic in metadata.topics and metadata.topics[topic].error is None:
            statuses[topic] = "exists"
        else:
            missing.append(
                NewTopic(
                    topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor,
                )
            )

    if not missing:
        return statuses

    futures = admin.create_topics(missing, request_timeout=timeout_seconds)
    for topic, future in futures.items():
        try:
            future.result()
            statuses[topic] = "created"
            logger.info("Created Kafka topic '%s' (partitions=%d)", topic, num_partitions)
        except KafkaException as exc:
            err = exc.args[0] if exc.args else None
            if err is not None and err.name() == "TOPIC_ALREADY_EXISTS":
                statuses[topic] = "exists"
            else:
                raise
    return statuses


def _record_key(record: dict[str, Any], index: int, batch_id: str) -> bytes:
    raw_id = record.get("ID")
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id).strip().encode("utf-8")
    case_num = record.get("Case Number")
    if case_num is not None and str(case_num).strip():
        return str(case_num).strip().encode("utf-8")
    return f"{batch_id}:{index}".encode()


def publish_bronze_records_to_kafka(
    records: Iterable[dict[str, Any]],
    *,
    bootstrap_servers: str,
    topic: str,
    batch_id: str | None = None,
    timeout_seconds: float = 30.0,
    producer: Any | None = None,
) -> KafkaPublishResult:
    """Serialize Bronze JSON records and flush synchronously to Kafka with idempotence."""
    effective_batch = batch_id or uuid4().hex
    record_list = list(records)
    if not record_list:
        return KafkaPublishResult(
            topic=topic,
            batch_id=effective_batch,
            attempted=0,
            delivered=0,
            failed=0,
            partitions=[],
        )

    client = producer or Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "acks": "all",
            "enable.idempotence": True,
            "retries": 5,
            "linger.ms": 10,
        }
    )

    delivered = 0
    failed = 0
    errors: list[str] = []
    partitions: set[int] = set()

    def _on_delivery(err: Any, msg: Any) -> None:
        nonlocal delivered, failed
        if err is not None:
            failed += 1
            errors.append(str(err))
            return
        delivered += 1
        if msg is not None and hasattr(msg, "partition"):
            partitions.add(int(msg.partition()))

    for idx, rec in enumerate(record_list):
        payload = json.dumps(rec, ensure_ascii=False).encode("utf-8")
        key = _record_key(rec, idx, effective_batch)
        while True:
            try:
                client.produce(
                    topic=topic,
                    key=key,
                    value=payload,
                    on_delivery=_on_delivery,
                )
                break
            except BufferError:
                client.poll(0.1)
        client.poll(0)

    remaining = client.flush(timeout_seconds)
    if remaining:
        raise RuntimeError(
            f"Kafka producer timed out with {remaining} undelivered message(s) for topic '{topic}'"
        )
    if failed > 0 or delivered != len(record_list):
        raise RuntimeError(
            f"Kafka publish failed for {failed}/{len(record_list)} record(s): {errors[:3]}"
        )

    logger.info(
        "Published %d record(s) to Kafka topic '%s' across partitions %s (batch=%s)",
        delivered,
        topic,
        sorted(partitions),
        effective_batch,
    )
    return KafkaPublishResult(
        topic=topic,
        batch_id=effective_batch,
        attempted=len(record_list),
        delivered=delivered,
        failed=0,
        partitions=sorted(partitions),
    )


def ingest_soda_to_kafka(
    streaming_cfg: StreamingConfig,
    *,
    watermark_path: str | Path | None = None,
    since: str | datetime | None = None,
    until: str | datetime | None = None,
    max_records: int | None = None,
    page_size: int | None = None,
    max_pages: int | None = None,
    default_lookback_hours: int = 72,
    publish_kafka: bool = True,
    commit_watermark: bool = True,
    output_jsonl: str | Path | None = None,
    bootstrap_servers: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """Fetch incremental SODA records, publish cleanly to Kafka, and persist watermark."""
    wm_path = Path(watermark_path or streaming_cfg.watermark_path)
    fetch_res = fetch_soda_incremental(
        streaming_cfg=streaming_cfg,
        watermark_path=wm_path,
        since=since,
        until=until,
        page_size=page_size,
        max_pages=max_pages,
        default_lookback_hours=default_lookback_hours,
        max_records=max_records,
    )

    if output_jsonl is not None:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in fetch_res.records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    kafka_summary: dict[str, Any] | None = None
    if publish_kafka:
        servers = bootstrap_servers or streaming_cfg.kafka_bootstrap_servers
        target_topic = topic or streaming_cfg.kafka_topic
        ensure_kafka_topics(servers, [target_topic, streaming_cfg.kafka_dlq_topic])
        pub_res = publish_bronze_records_to_kafka(
            fetch_res.records,
            bootstrap_servers=servers,
            topic=target_topic,
            batch_id=fetch_res.batch_id,
        )
        kafka_summary = pub_res.to_dict()

    saved_wm: str | None = None
    if commit_watermark and (fetch_res.records or fetch_res.next_watermark.last_updated_on):
        saved_wm = str(save_watermark(wm_path, fetch_res.next_watermark))

    return {
        "batch_id": fetch_res.batch_id,
        "effective_where": fetch_res.effective_where,
        "pages_fetched": fetch_res.pages_fetched,
        "fetched_count": len(fetch_res.records),
        "has_more": fetch_res.has_more,
        "backlog_detected": fetch_res.backlog_detected,
        "ingestion_lag_seconds": fetch_res.ingestion_lag_seconds,
        "watermark": fetch_res.next_watermark.to_dict(),
        "watermark_path": saved_wm,
        "kafka": kafka_summary,
        "output_jsonl": str(output_jsonl) if output_jsonl else None,
    }

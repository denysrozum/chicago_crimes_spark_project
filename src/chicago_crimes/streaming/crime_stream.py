"""Spark Structured Streaming consumer from Kafka to Bronze, Silver Delta, and DLQ."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from chicago_crimes.core.config import AppConfig
from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.schemas import RAW_CRIME_COLUMNS
from chicago_crimes.core.spark import spark_session
from chicago_crimes.ingestion.kafka_producer import ensure_kafka_topics
from chicago_crimes.processing.silver import run_silver_pipeline
from chicago_crimes.storage.lakehouse import (
    delta_history,
    is_delta_table,
    merge_insert_only_delta,
    merge_silver_delta,
    read_table,
    write_table,
)

logger = get_logger("streaming.crime_stream")

KAFKA_JSON_SCHEMA = StructType(
    [StructField(name, StringType(), True) for name in RAW_CRIME_COLUMNS]
    + [
        StructField("corrupt_record", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("source_id", StringType(), True),
        StructField("ingestion_batch_id", StringType(), True),
    ]
)

_KAFKA_PARSE_SCHEMA = StructType(
    list(KAFKA_JSON_SCHEMA.fields)
    + [StructField("_corrupt_json_record", StringType(), True)]
)


@dataclass(frozen=True)
class StreamingBatchStats:
    """Reconciliation accounting record emitted for every Spark micro-batch."""

    batch_id: int
    processed_at_utc: str
    kafka_input_rows: int
    malformed_json_rows: int
    bronze_written_rows: int
    silver_accepted_deduped: int
    silver_superseded: int
    silver_quarantined: int
    merge_version: int | None
    merge_metrics: dict[str, Any]
    is_reconciled: bool
    offsets_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stream_event_id_col() -> Any:
    return F.sha2(
        F.concat_ws(
            "|",
            F.coalesce(F.col("topic").cast("string"), F.lit("")),
            F.coalesce(F.col("partition").cast("string"), F.lit("-1")),
            F.coalesce(F.col("offset").cast("string"), F.lit("-1")),
        ),
        256,
    )


def _compute_offsets_digest(batch_df: DataFrame) -> str:
    rows = batch_df.select("topic", "partition", "offset").distinct().collect()
    tokens = sorted(f"{r['topic']}:{r['partition']}:{r['offset']}" for r in rows)
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()[:16]


def _write_replay_safe_event_sink(
    spark: SparkSession,
    df: DataFrame,
    target: str | Path,
    *,
    table_format: Literal["delta", "parquet"],
    partition_by: Sequence[str] | None = None,
) -> None:
    if table_format == "delta":
        merge_insert_only_delta(
            spark,
            df,
            target,
            key_col="stream_event_id",
            partition_by=partition_by,
        )
        return
    # Explicit Parquet fallback has no record-level ACID MERGE; Delta is required for replay safety.
    write_table(df, target, table_format="parquet", mode="append", partition_by=partition_by)


def parse_kafka_payloads(raw_df: DataFrame) -> DataFrame:
    """Decode Kafka binary/string values with `KAFKA_JSON_SCHEMA`, tagging malformed rows."""
    return (
        raw_df.withColumn("_raw_payload", F.col("value").cast("string"))
        .withColumn(
            "_parsed",
            F.from_json(
                F.col("_raw_payload"),
                _KAFKA_PARSE_SCHEMA,
                {
                    "mode": "PERMISSIVE",
                    "columnNameOfCorruptRecord": "_corrupt_json_record",
                },
            ),
        )
        .withColumn(
            "_is_malformed",
            F.col("_parsed").isNull()
            | F.col("_parsed._corrupt_json_record").isNotNull()
            | F.col("_raw_payload").isNull()
            | (F.length(F.trim(F.col("_raw_payload"))) == 0),
        )
    )


def _append_metrics_jsonl(metrics_path: str | Path | None, stats: StreamingBatchStats) -> None:
    if metrics_path is None:
        return
    target = Path(metrics_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if target.exists():
        for raw in target.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            entry = json.loads(raw)
            same_batch = int(entry.get("batch_id", -1)) == int(stats.batch_id)
            same_digest = entry.get("offsets_digest", "") == stats.offsets_digest
            if same_batch and same_digest:
                continue
            existing.append(entry)
    existing.append(stats.to_dict())
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        "\n".join(json.dumps(item) for item in existing) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, target)


def _extract_valid_bronze_df(parsed_df: DataFrame, *, batch_id: int, source_id: str) -> DataFrame:
    valid = parsed_df.filter(~F.col("_is_malformed"))
    parsed_cols = [F.col("_parsed")[name].alias(name) for name in RAW_CRIME_COLUMNS]
    return valid.select(
        *parsed_cols,
        F.col("_parsed.corrupt_record").alias("corrupt_record"),
        F.coalesce(
            F.expr(
                "try_to_timestamp(_parsed.ingestion_timestamp, \"yyyy-MM-dd'T'HH:mm:ss.SSS\")"
            ),
            F.expr("try_to_timestamp(_parsed.ingestion_timestamp, \"yyyy-MM-dd'T'HH:mm:ss\")"),
            F.expr("try_cast(_parsed.ingestion_timestamp AS TIMESTAMP)"),
            F.current_timestamp(),
        ).alias("ingestion_timestamp"),
        F.coalesce(F.col("_parsed.source_id"), F.lit(source_id)).cast("string").alias("source_id"),
        F.coalesce(
            F.col("_parsed.ingestion_batch_id"),
            F.lit(f"kafka-batch-{batch_id}"),
        )
        .cast("string")
        .alias("ingestion_batch_id"),
        F.col("topic").cast("string").alias("kafka_topic"),
        F.col("partition").cast("int").alias("kafka_partition"),
        F.col("offset").cast("long").alias("kafka_offset"),
        F.col("timestamp").cast("timestamp").alias("kafka_timestamp"),
        _stream_event_id_col().alias("stream_event_id"),
        F.lit(int(batch_id)).cast("long").alias("microbatch_id"),
    )


def _extract_malformed_dlq_df(parsed_df: DataFrame, *, batch_id: int) -> DataFrame:
    now_iso = datetime.now(UTC).isoformat()
    return parsed_df.filter(F.col("_is_malformed")).select(
        F.col("_raw_payload").alias("raw_payload"),
        F.col("topic").cast("string").alias("kafka_topic"),
        F.col("partition").cast("int").alias("kafka_partition"),
        F.col("offset").cast("long").alias("kafka_offset"),
        F.col("timestamp").cast("timestamp").alias("kafka_timestamp"),
        F.lit("malformed_json").alias("error_reason"),
        _stream_event_id_col().alias("stream_event_id"),
        F.lit(int(batch_id)).cast("long").alias("microbatch_id"),
        F.lit(now_iso).alias("quarantined_at_utc"),
    )


def process_kafka_microbatch(
    batch_df: DataFrame,
    batch_id: int,
    *,
    bronze_target: str | Path,
    silver_target: str | Path,
    quarantine_target: str | Path,
    dlq_target: str | Path,
    table_format: Literal["delta", "parquet"] = "delta",
    partition_by: Sequence[str] | None = ("year",),
    metrics_log_path: str | Path | None = None,
    source_id: str = "kafka_stream",
    on_stage: Callable[[str, int], None] | None = None,
) -> StreamingBatchStats:
    """Process one micro-batch into replay-safe Bronze, DLQ, Quarantine, and Silver Delta."""
    batch_df.persist()
    try:
        input_rows = int(batch_df.count())
        now_iso = datetime.now(UTC).isoformat()
        if input_rows == 0:
            stats = StreamingBatchStats(
                batch_id=int(batch_id),
                processed_at_utc=now_iso,
                kafka_input_rows=0,
                malformed_json_rows=0,
                bronze_written_rows=0,
                silver_accepted_deduped=0,
                silver_superseded=0,
                silver_quarantined=0,
                merge_version=None,
                merge_metrics={},
                is_reconciled=True,
                offsets_digest=hashlib.sha256(b"").hexdigest()[:16],
            )
            _append_metrics_jsonl(metrics_log_path, stats)
            return stats

        offsets_digest = _compute_offsets_digest(batch_df)
        parsed_df = parse_kafka_payloads(batch_df).persist()
        try:
            dlq_df = _extract_malformed_dlq_df(parsed_df, batch_id=int(batch_id))
            malformed_count = int(dlq_df.count())
            if malformed_count > 0:
                _write_replay_safe_event_sink(
                    batch_df.sparkSession,
                    dlq_df,
                    dlq_target,
                    table_format=table_format,
                )
                logger.warning(
                    "Micro-batch %d routed %d malformed JSON record(s) -> %s",
                    batch_id,
                    malformed_count,
                    dlq_target,
                )
            if on_stage is not None:
                on_stage("after_dlq", int(batch_id))

            bronze_df = _extract_valid_bronze_df(
                parsed_df, batch_id=int(batch_id), source_id=source_id
            ).persist()
            try:
                bronze_count = int(bronze_df.count())
                merge_version: int | None = None
                merge_metrics: dict[str, Any] = {}
                accepted = 0
                superseded = 0
                quarantined = 0

                if bronze_count > 0:
                    _write_replay_safe_event_sink(
                        batch_df.sparkSession,
                        bronze_df,
                        bronze_target,
                        table_format=table_format,
                    )
                    if on_stage is not None:
                        on_stage("after_bronze", int(batch_id))

                    silver_res = run_silver_pipeline(bronze_df, deduplicate=True)
                    silver_res.accounting.verify_invariant()
                    accepted = silver_res.accounting.accepted_records
                    superseded = silver_res.accounting.superseded_records
                    quarantined = silver_res.accounting.quarantined_records

                    if quarantined > 0:
                        _write_replay_safe_event_sink(
                            batch_df.sparkSession,
                            silver_res.quarantined_df,
                            quarantine_target,
                            table_format=table_format,
                        )
                    if on_stage is not None:
                        on_stage("after_quarantine", int(batch_id))

                    if accepted > 0:
                        if table_format == "delta":
                            merge_res = merge_silver_delta(
                                batch_df.sparkSession,
                                silver_res.silver_df,
                                silver_target,
                                partition_by=partition_by,
                            )
                            merge_version = int(merge_res["version"])
                            merge_metrics = dict(merge_res.get("metrics", {}))
                        else:
                            write_table(
                                silver_res.silver_df,
                                silver_target,
                                table_format="parquet",
                                mode="append",
                                partition_by=partition_by,
                            )
                    if on_stage is not None:
                        on_stage("after_silver", int(batch_id))

                reconciled = (input_rows == malformed_count + bronze_count) and (
                    bronze_count == accepted + superseded + quarantined
                )
                if not reconciled:
                    raise ValueError(
                        f"Streaming batch {batch_id} reconciliation failed: "
                        f"input={input_rows}, malformed={malformed_count}, "
                        f"bronze={bronze_count}, accepted={accepted}, "
                        f"superseded={superseded}, quarantined={quarantined}"
                    )

                stats = StreamingBatchStats(
                    batch_id=int(batch_id),
                    processed_at_utc=now_iso,
                    kafka_input_rows=input_rows,
                    malformed_json_rows=malformed_count,
                    bronze_written_rows=bronze_count,
                    silver_accepted_deduped=accepted,
                    silver_superseded=superseded,
                    silver_quarantined=quarantined,
                    merge_version=merge_version,
                    merge_metrics=merge_metrics,
                    is_reconciled=reconciled,
                    offsets_digest=offsets_digest,
                )
                _append_metrics_jsonl(metrics_log_path, stats)
                logger.info(
                    "Micro-batch %d committed: input=%d bronze=%d dlq=%d "
                    "silver_accepted=%d superseded=%d quarantined=%d version=%s",
                    batch_id,
                    input_rows,
                    bronze_count,
                    malformed_count,
                    accepted,
                    superseded,
                    quarantined,
                    merge_version,
                )
                return stats
            finally:
                bronze_df.unpersist()
        finally:
            parsed_df.unpersist()
    finally:
        batch_df.unpersist()


def reconcile_stream_targets(
    spark: SparkSession,
    *,
    bronze_target: str | Path,
    silver_target: str | Path,
    quarantine_target: str | Path | None = None,
    dlq_target: str | Path | None = None,
    table_format: Literal["delta", "parquet", "auto"] = "auto",
) -> dict[str, Any]:
    """Audit persisted streaming tables against unique ID & accounting expectations."""

    def _inspect_event_sink(target: str | Path | None) -> tuple[int, int, bool]:
        if target is None:
            return 0, 0, True
        target_str = str(target)
        try:
            sink_df = read_table(spark, target_str, table_format=table_format)
        except FileNotFoundError:
            return 0, 0, True
        total = int(sink_df.count())
        if "stream_event_id" in sink_df.columns:
            distinct_events = int(sink_df.select("stream_event_id").distinct().count())
            return total, distinct_events, total == distinct_events
        return total, total, True

    bronze_rows, bronze_distinct_events, bronze_event_unique = _inspect_event_sink(bronze_target)
    quarantine_rows, quarantine_distinct_events, quarantine_event_unique = _inspect_event_sink(
        quarantine_target
    )
    dlq_rows, dlq_distinct_events, dlq_event_unique = _inspect_event_sink(dlq_target)

    silver_rows = 0
    silver_distinct_ids = 0
    silver_version: int | None = None
    if is_delta_table(spark, silver_target) or (
        table_format != "delta" and Path(str(silver_target)).exists()
    ):
        silver_df = read_table(spark, silver_target, table_format=table_format)
        silver_rows = int(silver_df.count())
        silver_distinct_ids = int(silver_df.select("ID").distinct().count())
        if is_delta_table(spark, silver_target):
            hist = delta_history(spark, silver_target, limit=1)
            if hist:
                silver_version = int(hist[0]["version"])

    silver_id_unique = silver_rows == silver_distinct_ids
    return {
        "bronze_rows": bronze_rows,
        "bronze_distinct_events": bronze_distinct_events,
        "bronze_event_unique": bronze_event_unique,
        "silver_rows": silver_rows,
        "silver_distinct_ids": silver_distinct_ids,
        "silver_id_unique": silver_id_unique,
        "quarantine_rows": quarantine_rows,
        "quarantine_distinct_events": quarantine_distinct_events,
        "quarantine_event_unique": quarantine_event_unique,
        "dlq_rows": dlq_rows,
        "dlq_distinct_events": dlq_distinct_events,
        "dlq_event_unique": dlq_event_unique,
        "all_sinks_unique": (
            silver_id_unique
            and bronze_event_unique
            and quarantine_event_unique
            and dlq_event_unique
        ),
        "silver_delta_version": silver_version,
    }


def run_crime_stream(
    cfg: AppConfig,
    *,
    bootstrap_servers: str | None = None,
    topic: str | None = None,
    checkpoint_location: str | Path | None = None,
    bronze_target: str | Path | None = None,
    silver_target: str | Path | None = None,
    quarantine_target: str | Path | None = None,
    dlq_target: str | Path | None = None,
    metrics_log_path: str | Path | None = None,
    starting_offsets: str | None = None,
    available_now: bool = False,
    trigger_seconds: int | None = None,
    max_batches: int | None = None,
    timeout_seconds: float | None = None,
    fail_on_data_loss: bool = False,
    partition_by: Sequence[str] | None = ("year",),
    ensure_topic: bool = True,
    on_stage: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Run Spark Structured Streaming from Kafka into Bronze/Silver Delta with checkpoints."""
    servers = bootstrap_servers or cfg.streaming.kafka_bootstrap_servers
    source_topic = topic or cfg.streaming.kafka_topic
    ckpt = str(checkpoint_location or cfg.streaming.checkpoint_location)
    bronze_dest = str(bronze_target or cfg.streaming.bronze_stream_target)
    silver_dest = str(silver_target or cfg.streaming.silver_stream_target)
    quarantine_dest = str(quarantine_target or cfg.streaming.quarantine_stream_target)
    dlq_dest = str(dlq_target or cfg.streaming.dlq_stream_target)
    metrics_dest = metrics_log_path or cfg.streaming.metrics_log_path
    offsets = starting_offsets or cfg.streaming.starting_offsets
    interval = trigger_seconds or cfg.streaming.trigger_interval_seconds

    if ensure_topic:
        ensure_kafka_topics(servers, [source_topic, cfg.streaming.kafka_dlq_topic])

    stream_cfg = cfg.model_copy(
        update={
            "spark": cfg.spark.model_copy(update={"enable_kafka": True, "enable_delta": True})
        }
    )

    batch_stats: list[StreamingBatchStats] = []
    with spark_session(stream_cfg, app_name="ChicagoCrimes-KafkaStream") as spark:
        kafka_df = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", servers)
            .option("subscribe", source_topic)
            .option("startingOffsets", offsets)
            .option("failOnDataLoss", str(fail_on_data_loss).lower())
            .load()
        )

        def _handle_batch(df: DataFrame, bid: int) -> None:
            stats = process_kafka_microbatch(
                df,
                bid,
                bronze_target=bronze_dest,
                silver_target=silver_dest,
                quarantine_target=quarantine_dest,
                dlq_target=dlq_dest,
                table_format="delta",
                partition_by=partition_by,
                metrics_log_path=metrics_dest,
                on_stage=on_stage,
            )
            batch_stats.append(stats)

        writer = kafka_df.writeStream.outputMode("append").option("checkpointLocation", ckpt)
        if available_now:
            writer = writer.trigger(availableNow=True)
        else:
            writer = writer.trigger(processingTime=f"{interval} seconds")

        query = writer.foreachBatch(_handle_batch).start()
        try:
            if available_now:
                query.awaitTermination(timeout=timeout_seconds)
            else:
                deadline = (
                    time.monotonic() + timeout_seconds if timeout_seconds is not None else None
                )
                while query.isActive:
                    non_empty = sum(1 for b in batch_stats if b.kafka_input_rows > 0)
                    if max_batches is not None and non_empty >= max_batches:
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    time.sleep(0.5)
        finally:
            if query.isActive:
                query.stop()
            query.awaitTermination(timeout=15)

        reconciliation = reconcile_stream_targets(
            spark,
            bronze_target=bronze_dest,
            silver_target=silver_dest,
            quarantine_target=quarantine_dest,
            dlq_target=dlq_dest,
            table_format="delta",
        )

    return {
        "bootstrap_servers": servers,
        "topic": source_topic,
        "checkpoint_location": ckpt,
        "batches_processed": len(batch_stats),
        "batches": [b.to_dict() for b in batch_stats],
        "reconciliation": reconciliation,
    }

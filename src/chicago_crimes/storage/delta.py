"""Delta Lake storage adapter with ACID MERGE and transaction log inspection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from chicago_crimes.core.logging import get_logger
from chicago_crimes.storage.local import is_remote_uri

logger = get_logger("storage.delta")


def _normalize_target(path: str | Path) -> str:
    target_str = str(path)
    if is_remote_uri(target_str):
        return target_str
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


def is_delta_table(spark: SparkSession, path: str | Path) -> bool:
    """Return True if `path` contains an initialized Delta transaction log."""
    target_str = str(path)
    if not is_remote_uri(target_str):
        local_path = Path(target_str)
        if not local_path.exists() or not (local_path / "_delta_log").is_dir():
            return False
    try:
        return bool(DeltaTable.isDeltaTable(spark, target_str))
    except TypeError:
        return False


def write_delta(
    df: DataFrame,
    path: str | Path,
    *,
    mode: str = "overwrite",
    partition_by: Sequence[str] | None = None,
    overwrite_schema: bool = False,
) -> str:
    """Write a DataFrame in Delta Lake format to local disk or object storage."""
    target_str = _normalize_target(path)
    # Preserve spaced Chicago Open Data column names (`Case Number`, `Primary Type`, etc.).
    df.sparkSession.conf.set(
        "spark.databricks.delta.properties.defaults.columnMapping.mode", "name"
    )
    df.sparkSession.conf.set("spark.databricks.delta.properties.defaults.minReaderVersion", "2")
    df.sparkSession.conf.set("spark.databricks.delta.properties.defaults.minWriterVersion", "5")
    writer = df.write.format("delta").mode(mode)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    if overwrite_schema and mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    logger.info("Writing Delta table to %s (mode=%s)", target_str, mode)
    writer.save(target_str)
    return target_str


def read_delta(
    spark: SparkSession,
    path: str | Path,
    *,
    version_as_of: int | None = None,
    timestamp_as_of: str | None = None,
) -> DataFrame:
    """Read a Delta table with optional version/timestamp time travel."""
    if version_as_of is not None and timestamp_as_of is not None:
        raise ValueError("Specify at most one of version_as_of and timestamp_as_of.")
    target_str = str(path)
    if not is_delta_table(spark, target_str):
        raise FileNotFoundError(f"Delta table not found at: {target_str}")
    reader = spark.read.format("delta")
    if version_as_of is not None:
        reader = reader.option("versionAsOf", version_as_of)
    if timestamp_as_of is not None:
        reader = reader.option("timestampAsOf", timestamp_as_of)
    logger.info("Reading Delta table from %s (version=%s)", target_str, version_as_of)
    return reader.load(target_str)


def delta_history(
    spark: SparkSession, path: str | Path, *, limit: int = 10
) -> list[dict[str, Any]]:
    """Return recent commit entries from a Delta table's transaction log."""
    target_str = str(path)
    if not is_delta_table(spark, target_str):
        raise FileNotFoundError(f"Delta table not found at: {target_str}")
    hist_df = DeltaTable.forPath(spark, target_str).history(limit)
    rows = hist_df.collect()
    entries: list[dict[str, Any]] = []
    for row in rows:
        entries.append(
            {
                "version": int(row["version"]),
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "operation": row["operation"],
                "operationParameters": dict(row["operationParameters"] or {}),
                "operationMetrics": dict(row["operationMetrics"] or {}),
            }
        )
    return entries


def _deduplicate_source_batch(
    df: DataFrame,
    *,
    key_col: str,
    update_order_col: str,
) -> DataFrame:
    """Keep strictly one newest row per key inside a micro-batch before Delta MERGE."""
    order_cols = [F.col(update_order_col).desc_nulls_last()]
    if "ingestion_timestamp" in df.columns:
        order_cols.append(F.col("ingestion_timestamp").desc_nulls_last())
    if "kafka_partition" in df.columns:
        order_cols.append(F.col("kafka_partition").desc_nulls_last())
    if "kafka_offset" in df.columns:
        order_cols.append(F.col("kafka_offset").desc_nulls_last())
    win = Window.partitionBy(key_col).orderBy(*order_cols)
    return (
        df.withColumn("_merge_rn", F.row_number().over(win))
        .filter(F.col("_merge_rn") == 1)
        .drop("_merge_rn")
    )


@contextmanager
def _delta_schema_automerge(spark: SparkSession) -> Iterator[None]:
    """Temporarily enable Delta MERGE schema evolution for backwards-compatible sinks."""
    key = "spark.databricks.delta.schema.autoMerge.enabled"
    previous = spark.conf.get(key, "false")
    spark.conf.set(key, "true")
    try:
        yield
    finally:
        spark.conf.set(key, previous)


def _deterministic_legacy_key_expr(df: DataFrame) -> Any:
    """Compute a deterministic fallback key when backfilling a legacy Delta table."""
    if {"kafka_topic", "kafka_partition", "kafka_offset"}.issubset(df.columns):
        return F.sha2(
            F.concat_ws(
                "|",
                F.coalesce(F.col("kafka_topic").cast("string"), F.lit("")),
                F.coalesce(F.col("kafka_partition").cast("string"), F.lit("-1")),
                F.coalesce(F.col("kafka_offset").cast("string"), F.lit("-1")),
            ),
            256,
        )
    if df.columns:
        return F.sha2(
            F.concat_ws(
                "|",
                *[F.coalesce(F.col(f"`{col}`").cast("string"), F.lit("")) for col in df.columns],
            ),
            256,
        )
    return F.lit("")


def _ensure_insert_only_key_column(
    spark: SparkSession,
    target_str: str,
    *,
    key_col: str,
    source_cols: Sequence[str],
) -> DeltaTable:
    """Backfill `key_col` on legacy Delta sinks created before idempotent streaming keys."""
    delta_tbl = DeltaTable.forPath(spark, target_str)
    existing_df = delta_tbl.toDF()
    need_key_col = key_col not in existing_df.columns
    need_null_backfill = (
        not need_key_col
        and existing_df.filter(F.col(f"`{key_col}`").isNull()).limit(1).count() > 0
    )
    if not need_key_col and not need_null_backfill:
        return delta_tbl

    detail_row = delta_tbl.detail().select("partitionColumns", "properties").first()
    existing_partitions = (
        list(detail_row["partitionColumns"])
        if detail_row and detail_row["partitionColumns"]
        else []
    )
    existing_props = dict(detail_row["properties"] or {}) if detail_row else {}
    existing_col_mapping = existing_props.get("delta.columnMapping.mode")

    key_expr = _deterministic_legacy_key_expr(existing_df)
    migrated = (
        existing_df.withColumn(key_col, key_expr)
        if need_key_col
        else existing_df.withColumn(key_col, F.coalesce(F.col(f"`{key_col}`"), key_expr))
    )
    if "microbatch_id" in source_cols and "microbatch_id" not in migrated.columns:
        migrated = migrated.withColumn("microbatch_id", F.lit(None).cast("long"))

    # Eagerly materialize before overwriting so Spark breaks file-read lineage on `target_str`.
    materialized = migrated.localCheckpoint(eager=True)
    logger.info(
        "Backfilling legacy Delta table %s with key column `%s` (partitions=%s)",
        target_str,
        key_col,
        existing_partitions,
    )
    if existing_col_mapping == "name":
        write_delta(
            materialized,
            target_str,
            mode="overwrite",
            partition_by=existing_partitions or None,
            overwrite_schema=True,
        )
    else:
        writer = (
            materialized.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
        )
        if existing_partitions:
            writer = writer.partitionBy(*existing_partitions)
        writer.save(target_str)
    return DeltaTable.forPath(spark, target_str)


def merge_insert_only_delta(
    spark: SparkSession,
    source_df: DataFrame,
    target_path: str | Path,
    *,
    key_col: str = "stream_event_id",
    partition_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Idempotently insert unseen rows by `key_col` into an append-style Delta table."""
    target_str = _normalize_target(target_path)
    deduped = source_df.dropDuplicates([key_col])

    if not is_delta_table(spark, target_str):
        write_delta(deduped, target_str, mode="overwrite", partition_by=partition_by)
        latest = delta_history(spark, target_str, limit=1)[0]
        return {
            "target": target_str,
            "created": True,
            "version": latest["version"],
            "operation": latest["operation"],
            "metrics": latest["operationMetrics"],
        }

    delta_tbl = _ensure_insert_only_key_column(
        spark,
        target_str,
        key_col=key_col,
        source_cols=deduped.columns,
    )
    match_expr = f"target.`{key_col}` = src.`{key_col}`"
    with _delta_schema_automerge(spark):
        (
            delta_tbl.alias("target")
            .merge(deduped.alias("src"), match_expr)
            .whenNotMatchedInsertAll()
            .execute()
        )
    latest = delta_history(spark, target_str, limit=1)[0]
    logger.info(
        "Delta insert-only MERGE -> %s (version=%s metrics=%s)",
        target_str,
        latest["version"],
        latest["operationMetrics"],
    )
    return {
        "target": target_str,
        "created": False,
        "version": latest["version"],
        "operation": latest["operation"],
        "metrics": latest["operationMetrics"],
    }


def merge_silver_delta(
    spark: SparkSession,
    source_df: DataFrame,
    target_path: str | Path,
    *,
    key_col: str = "ID",
    update_order_col: str = "Updated On",
    partition_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Idempotently upsert `source_df` into a Silver Delta table ordered by update timestamp."""
    target_str = _normalize_target(target_path)
    deduped = _deduplicate_source_batch(
        source_df, key_col=key_col, update_order_col=update_order_col
    )

    if not is_delta_table(spark, target_str):
        write_delta(deduped, target_str, mode="overwrite", partition_by=partition_by)
        latest = delta_history(spark, target_str, limit=1)[0]
        return {
            "target": target_str,
            "created": True,
            "version": latest["version"],
            "operation": latest["operation"],
            "metrics": latest["operationMetrics"],
        }

    delta_tbl = DeltaTable.forPath(spark, target_str)
    match_expr = f"target.`{key_col}` = src.`{key_col}`"
    update_guard = (
        f"target.`{update_order_col}` IS NULL OR src.`{update_order_col}` IS NULL "
        f"OR src.`{update_order_col}` >= target.`{update_order_col}`"
    )
    with _delta_schema_automerge(spark):
        (
            delta_tbl.alias("target")
            .merge(deduped.alias("src"), match_expr)
            .whenMatchedUpdateAll(condition=update_guard)
            .whenNotMatchedInsertAll()
            .execute()
        )
    latest = delta_history(spark, target_str, limit=1)[0]
    logger.info(
        "Delta MERGE -> %s (version=%s metrics=%s)",
        target_str,
        latest["version"],
        latest["operationMetrics"],
    )
    return {
        "target": target_str,
        "created": False,
        "version": latest["version"],
        "operation": latest["operation"],
        "metrics": latest["operationMetrics"],
    }

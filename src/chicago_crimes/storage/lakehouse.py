"""Unified storage helpers abstracting local Parquet and Delta/MinIO URIs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pyspark.sql import DataFrame, SparkSession

from chicago_crimes.core.logging import get_logger
from chicago_crimes.storage.delta import (
    delta_history,
    is_delta_table,
    merge_insert_only_delta,
    merge_silver_delta,
    read_delta,
    write_delta,
)
from chicago_crimes.storage.local import read_parquet, write_parquet

logger = get_logger("storage.lakehouse")


def read_table(
    spark: SparkSession,
    path: str | Path,
    *,
    table_format: Literal["auto", "parquet", "delta"] = "auto",
    version_as_of: int | None = None,
) -> DataFrame:
    """Read a Spark table in Delta or Parquet format from local disk or S3A URI."""
    resolved = table_format
    if resolved == "auto":
        resolved = "delta" if is_delta_table(spark, path) else "parquet"

    if resolved == "delta":
        return read_delta(spark, path, version_as_of=version_as_of)
    if version_as_of is not None:
        raise ValueError("version_as_of time travel is only supported for Delta tables.")
    return read_parquet(spark, path)


def write_table(
    df: DataFrame,
    path: str | Path,
    *,
    table_format: Literal["parquet", "delta"] = "parquet",
    mode: str = "overwrite",
    partition_by: Sequence[str] | None = None,
) -> str:
    """Write a Spark DataFrame as Parquet or Delta to local disk or object storage."""
    if table_format == "delta":
        return write_delta(df, path, mode=mode, partition_by=partition_by)
    target_str = str(path)
    if partition_by:
        logger.info("Writing partitioned Parquet dataset to %s (mode=%s)", target_str, mode)
        df.write.mode(mode).partitionBy(*partition_by).parquet(target_str)
        return target_str
    return write_parquet(df, path, mode=mode)


def sync_to_delta(
    spark: SparkSession,
    source_path: str | Path,
    target_delta_path: str | Path,
    *,
    limit: int | None = None,
    mode: str = "overwrite",
    partition_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Copy a Parquet/Delta source dataset into a Delta target (e.g. MinIO S3A)."""
    df = read_table(spark, source_path, table_format="auto")
    if limit is not None:
        df = df.limit(limit)
    row_count = int(df.count())
    written_uri = write_delta(df, target_delta_path, mode=mode, partition_by=partition_by)
    latest = delta_history(spark, written_uri, limit=1)[0]
    return {
        "source": str(source_path),
        "target": written_uri,
        "row_count": row_count,
        "version": latest["version"],
        "operation": latest["operation"],
        "metrics": latest["operationMetrics"],
    }


__all__ = [
    "delta_history",
    "is_delta_table",
    "merge_insert_only_delta",
    "merge_silver_delta",
    "read_table",
    "sync_to_delta",
    "write_table",
]

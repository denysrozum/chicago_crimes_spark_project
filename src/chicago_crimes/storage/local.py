"""Local filesystem storage adapters for Parquet and CSV exports."""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from chicago_crimes.core.logging import get_logger

logger = get_logger("storage.local")


_REMOTE_PREFIXES: tuple[str, ...] = (
    "s3a://",
    "s3://",
    "s3n://",
    "hdfs://",
    "gs://",
    "abfs://",
    "abfss://",
)


def is_remote_uri(path: str | Path) -> bool:
    """Return True if `path` targets an object/distributed filesystem URI."""
    return str(path).startswith(_REMOTE_PREFIXES)


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read a Parquet dataset from local disk or a configured remote URI."""
    target_str = str(path)
    if not is_remote_uri(target_str):
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {target}")
        target_str = str(target)
    logger.info("Reading Parquet dataset from %s", target_str)
    return spark.read.parquet(target_str)


def write_parquet(df: DataFrame, path: str | Path, mode: str = "overwrite") -> str:
    """Write a DataFrame to Parquet at `path` (local path or object URI)."""
    target_str = str(path)
    if not is_remote_uri(target_str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target_str = str(target)
    logger.info("Writing Parquet dataset to %s (mode=%s)", target_str, mode)
    df.write.mode(mode).parquet(target_str)
    return target_str


def write_single_csv(
    df: DataFrame,
    target_dir: str | Path,
    filename: str = "result.csv",
) -> Path:
    """Write a Spark DataFrame to a single CSV file without collecting full data to driver."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / ".tmp_spark_csv"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_dir))
    part_files = glob.glob(os.path.join(str(temp_dir), "part-*.csv"))
    destination = out_dir / filename
    if destination.exists():
        destination.unlink()

    if not part_files:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if df.isEmpty():
            destination.write_text(",".join(df.columns) + "\n", encoding="utf-8")
            return destination
        raise RuntimeError(f"Spark produced no part-*.csv files in {temp_dir}")

    shutil.move(part_files[0], str(destination))
    shutil.rmtree(temp_dir, ignore_errors=True)
    return destination

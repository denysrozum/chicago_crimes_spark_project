"""Historical CSV ingestion for Chicago Crime data."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.schemas import (
    CORRUPT_RECORD_COL,
    RAW_CRIME_SCHEMA,
    RAW_CRIME_STRING_SCHEMA,
)

logger = get_logger("ingestion.csv")


def load_crime_csv(
    spark: SparkSession,
    path: str | Path,
    sep: str = ";",
    timestamp_format: str = "MM/dd/yyyy hh:mm:ss a",
    limit: int | None = None,
) -> DataFrame:
    """Load Chicago Crime CSV data using the canonical typed source schema."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Raw CSV path does not exist: {source}")

    logger.info("Loading CSV from %s (sep=%r, limit=%s)", source, sep, limit)
    df = (
        spark.read.option("header", True)
        .option("sep", sep)
        .option("quote", '"')
        .option("timestampFormat", timestamp_format)
        .schema(RAW_CRIME_SCHEMA)
        .csv(str(source))
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        df = df.limit(limit)
    return df


def load_raw_csv_for_bronze(
    spark: SparkSession,
    path: str | Path,
    sep: str = ";",
    limit: int | None = None,
) -> DataFrame:
    """Load Chicago Crime CSV losslessly as raw strings plus `_corrupt_record`."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Raw CSV path does not exist: {source}")

    logger.info("Loading raw Bronze CSV from %s (sep=%r, limit=%s)", source, sep, limit)
    df = (
        spark.read.option("header", True)
        .option("sep", sep)
        .option("quote", '"')
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_RECORD_COL)
        .schema(RAW_CRIME_STRING_SCHEMA)
        .csv(str(source))
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        df = df.limit(limit)
    return df

"""Non-destructive Bronze layer ingestion retaining full source fidelity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.schemas import CORRUPT_RECORD_COL, RAW_CRIME_COLUMNS
from chicago_crimes.ingestion.csv_loader import load_raw_csv_for_bronze
from chicago_crimes.storage.local import write_parquet

logger = get_logger("ingestion.bronze")


@dataclass(frozen=True)
class BronzeIngestionResult:
    output_path: Path
    source_id: str
    batch_id: str
    row_count: int
    corrupt_count: int


def build_bronze_dataframe(
    df: DataFrame,
    source_id: str,
    batch_id: str | None = None,
    ingested_at: datetime | None = None,
) -> DataFrame:
    """Annotate source DataFrame with Bronze lineage columns without destructive edits."""
    effective_batch = batch_id or uuid4().hex
    ts_value = ingested_at or datetime.now(UTC).replace(tzinfo=None)

    work = df
    for col_name in RAW_CRIME_COLUMNS:
        if col_name not in work.columns:
            work = work.withColumn(col_name, F.lit(None).cast(StringType()))

    if CORRUPT_RECORD_COL in work.columns:
        work = work.withColumn("corrupt_record", F.col(CORRUPT_RECORD_COL)).drop(CORRUPT_RECORD_COL)
    elif "corrupt_record" not in work.columns:
        work = work.withColumn("corrupt_record", F.lit(None).cast(StringType()))

    ordered_source = [c for c in RAW_CRIME_COLUMNS]
    extra_cols = [
        c
        for c in work.columns
        if c not in RAW_CRIME_COLUMNS
        and c not in {"ingestion_timestamp", "source_id", "ingestion_batch_id", "corrupt_record"}
    ]
    return work.select(*ordered_source, *extra_cols, "corrupt_record").withColumns(
        {
            "ingestion_timestamp": F.lit(ts_value).cast("timestamp"),
            "source_id": F.lit(source_id).cast("string"),
            "ingestion_batch_id": F.lit(effective_batch).cast("string"),
        }
    )


def ingest_csv_to_bronze(
    spark: SparkSession,
    csv_path: str | Path,
    bronze_dir: str | Path,
    sep: str = ";",
    limit: int | None = None,
    source_id: str | None = None,
    batch_id: str | None = None,
    mode: str = "errorifexists",
) -> BronzeIngestionResult:
    """Ingest raw CSV into Bronze Parquet preserving all source values and lineage."""
    src = Path(csv_path)
    out = Path(bronze_dir)
    if mode == "errorifexists" and out.exists():
        raise FileExistsError(
            f"Bronze output path already exists ({out}). Use --overwrite or --append explicitly."
        )

    effective_source = source_id or f"csv:{src.name}"
    effective_batch = batch_id or uuid4().hex

    raw_df = load_raw_csv_for_bronze(spark, src, sep=sep, limit=limit)
    bronze_df = build_bronze_dataframe(raw_df, source_id=effective_source, batch_id=effective_batch)

    write_parquet(bronze_df, out, mode=mode)
    persisted = spark.read.parquet(str(out))
    batch_view = (
        persisted.filter(F.col("ingestion_batch_id") == effective_batch)
        if mode == "append"
        else persisted
    )
    stats_row = batch_view.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(F.col("corrupt_record").isNotNull(), 1).otherwise(0)).alias("corrupt_count"),
    ).collect()[0]
    row_count = int(stats_row["row_count"] or 0)
    corrupt_count = int(stats_row["corrupt_count"] or 0)
    logger.info(
        "Bronze batch %s written to %s (%d rows, %d corrupt CSV rows)",
        effective_batch,
        out,
        row_count,
        corrupt_count,
    )
    return BronzeIngestionResult(
        output_path=out,
        source_id=effective_source,
        batch_id=effective_batch,
        row_count=row_count,
        corrupt_count=corrupt_count,
    )

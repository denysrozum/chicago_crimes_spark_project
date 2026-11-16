"""Reproducible Gold analytical aggregates built strictly from Silver records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from chicago_crimes.core.logging import get_logger
from chicago_crimes.processing.features import add_temporal_features
from chicago_crimes.storage.local import write_parquet

logger = get_logger("processing.gold")

METHODOLOGY_NOTES: dict[str, str] = {
    "incident_counts_definition": (
        "Raw reported crime incident counts per aggregation bucket; "
        "not normalized by resident or transient population."
    ),
    "arrest_rate_denominator": "arrest_count / incident_count * 100 within each group.",
    "domestic_rate_denominator": "domestic_count / incident_count * 100 within each group.",
    "spatial_filtering": (
        "spatial_valid_grid_summary includes only Silver records with has_valid_geo=true "
        "and non-null geo_grid; records lacking valid coordinates are excluded from spatial cells "
        "rather than assigned synthetic centroids."
    ),
}


@dataclass(frozen=True)
class GoldTableSummary:
    """Metadata describing one materialized Gold Parquet table."""

    name: str
    path: str
    row_count: int
    columns: list[str]


@dataclass(frozen=True)
class GoldBuildResult:
    """Complete result bundle returned by `build_gold_tables`."""

    gold_dir: Path
    manifest_path: Path
    source_silver_count: int
    valid_geo_count: int
    tables: dict[str, GoldTableSummary]


def _rate_expr(numerator_col: str, denominator_col: str = "incident_count") -> Any:
    return F.when(
        F.col(denominator_col) > 0,
        (F.col(numerator_col) / F.col(denominator_col)) * 100.0,
    ).otherwise(F.lit(0.0))


def _incident_aggs() -> tuple[Any, Any, Any]:
    return (
        F.count(F.lit(1)).alias("incident_count"),
        F.sum(F.when(F.col("Arrest") == True, 1).otherwise(0)).alias("arrest_count"),  # noqa: E712
        F.sum(F.when(F.col("Domestic") == True, 1).otherwise(0)).alias("domestic_count"),  # noqa: E712
    )


def compute_gold_dataframes(silver_df: DataFrame, total_incidents: int) -> dict[str, DataFrame]:
    """Compute all seven Gold aggregate DataFrames from a validated Silver DataFrame."""
    base = add_temporal_features(silver_df, date_col="Date").withColumn(
        "crime_date", F.to_date("ts")
    )

    daily_citywide = (
        base.groupBy("crime_date", "year", "month", "day_of_week_num", "day_of_week")
        .agg(*_incident_aggs())
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .withColumn("domestic_rate_percent", _rate_expr("domestic_count"))
        .orderBy(F.asc("crime_date"))
    )

    monthly_summary = (
        base.withColumn("year_month", F.date_format("ts", "yyyy-MM"))
        .groupBy("year_month", "year", "month")
        .agg(*_incident_aggs())
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .withColumn("domestic_rate_percent", _rate_expr("domestic_count"))
        .orderBy(F.asc("year_month"))
    )

    district_community = (
        base.filter(F.col("District").isNotNull() & F.col("Community Area").isNotNull())
        .groupBy("District", "Community Area")
        .agg(*_incident_aggs())
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .withColumn("domestic_rate_percent", _rate_expr("domestic_count"))
        .orderBy(F.desc("incident_count"), F.asc("District"), F.asc("Community Area"))
    )

    denom = max(1, total_incidents)
    category_distribution = (
        base.groupBy("Primary Type")
        .agg(*_incident_aggs())
        .withColumn("citywide_share_percent", (F.col("incident_count") / F.lit(denom)) * 100.0)
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .withColumn("domestic_rate_percent", _rate_expr("domestic_count"))
        .orderBy(F.desc("incident_count"), F.asc("Primary Type"))
    )

    temporal_hourly = (
        base.groupBy("hour", "day_of_week_num", "day_of_week", "month")
        .agg(*_incident_aggs())
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .orderBy(F.asc("month"), F.asc("day_of_week_num"), F.asc("hour"))
    )

    spatial_grid = (
        base.filter((F.col("has_valid_geo") == True) & F.col("geo_grid").isNotNull())  # noqa: E712
        .groupBy("geo_grid")
        .agg(
            *_incident_aggs(),
            F.avg("Latitude").alias("mean_latitude"),
            F.avg("Longitude").alias("mean_longitude"),
        )
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .orderBy(F.desc("incident_count"), F.asc("geo_grid"))
    )

    daily_district = (
        base.filter(F.col("District").isNotNull())
        .groupBy("crime_date", "District", "year", "month", "day_of_week_num")
        .agg(*_incident_aggs())
        .withColumn("arrest_rate_percent", _rate_expr("arrest_count"))
        .orderBy(F.asc("crime_date"), F.asc("District"))
    )

    return {
        "daily_citywide_counts": daily_citywide,
        "monthly_crime_summary": monthly_summary,
        "district_community_summary": district_community,
        "category_distribution": category_distribution,
        "temporal_hourly_profile": temporal_hourly,
        "spatial_valid_grid_summary": spatial_grid,
        "daily_district_counts": daily_district,
    }


def build_gold_tables(
    silver_df: DataFrame,
    gold_dir: str | Path,
    *,
    mode: str = "errorifexists",
) -> GoldBuildResult:
    """Materialize Gold Parquet aggregates and write a documented manifest JSON."""
    out_root = Path(gold_dir)
    if mode == "errorifexists" and out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(
            f"Gold directory already exists and is non-empty ({out_root}). "
            "Pass mode='overwrite' or --overwrite."
        )
    out_root.mkdir(parents=True, exist_ok=True)

    counts = silver_df.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(F.when(F.col("has_valid_geo") == True, 1).otherwise(0)).alias(  # noqa: E712
            "valid_geo"
        ),
    ).collect()[0]
    total_incidents = int(counts["total"] or 0)
    valid_geo_count = int(counts["valid_geo"] or 0)

    gold_frames = compute_gold_dataframes(silver_df, total_incidents=total_incidents)
    summaries: dict[str, GoldTableSummary] = {}

    for name, frame in gold_frames.items():
        table_dir = out_root / name
        if mode == "errorifexists" and table_dir.exists():
            raise FileExistsError(
                f"Gold table already exists ({table_dir}). Pass mode='overwrite'."
            )
        write_parquet(frame, table_dir, mode=mode)
        written = silver_df.sparkSession.read.parquet(str(table_dir))
        summaries[name] = GoldTableSummary(
            name=name,
            path=str(table_dir),
            row_count=int(written.count()),
            columns=list(written.columns),
        )

    manifest_path = out_root / "_gold_manifest.json"
    manifest_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_silver_count": total_incidents,
        "valid_geo_count": valid_geo_count,
        "excluded_invalid_or_missing_geo_count": total_incidents - valid_geo_count,
        "methodology": METHODOLOGY_NOTES,
        "tables": {
            name: {
                "path": item.path,
                "row_count": item.row_count,
                "columns": item.columns,
            }
            for name, item in summaries.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    logger.info("Gold tables built in %s (%s tables)", out_root, len(summaries))

    return GoldBuildResult(
        gold_dir=out_root,
        manifest_path=manifest_path,
        source_silver_count=total_incidents,
        valid_geo_count=valid_geo_count,
        tables=summaries,
    )

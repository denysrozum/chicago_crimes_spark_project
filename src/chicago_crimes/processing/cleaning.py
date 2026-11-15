"""Legacy preprocessing pipeline (preserved for Phase 1 regression parity)."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.schemas import (
    CHICAGO_LAT_MAX,
    CHICAGO_LAT_MIN,
    CHICAGO_LON_MAX,
    CHICAGO_LON_MIN,
    COMMUNITY_AREA_MAX,
    COMMUNITY_AREA_MIN,
    DISTRICT_MAX,
    DISTRICT_MIN,
    WARD_MAX,
    WARD_MIN,
)
from chicago_crimes.processing.features import extract_block_features
from chicago_crimes.processing.quality import (
    inspect_structural_anomalies,
    save_anomaly_report,
    validate_processed_dataframe,
)

logger = get_logger("processing.cleaning")

LEGACY_DROPPED_COLUMNS = ("ID", "Case Number", "X Coordinate", "Y Coordinate", "Location", "Year")


def drop_legacy_columns(df: DataFrame) -> DataFrame:
    """Legacy column drop (superseded in Phase 2 to retain record identity)."""
    return df.drop(*[c for c in LEGACY_DROPPED_COLUMNS if c in df.columns])


def impute_missing_legacy(df: DataFrame) -> DataFrame:
    """Legacy missing-value imputation preserved for Phase 1 parity before Phase 2 hardening."""
    location_mode = (
        df.filter(F.col("Location Description").isNotNull())
        .groupBy("Primary Type", "Description", "Location Description")
        .count()
    )
    location_window = Window.partitionBy("Primary Type", "Description").orderBy(F.desc("count"))
    location_mode_ranked = (
        location_mode.withColumn("rn", F.row_number().over(location_window))
        .withColumn(
            "group_total", F.sum("count").over(Window.partitionBy("Primary Type", "Description"))
        )
        .withColumn("dominance_ratio", F.col("count") / F.col("group_total"))
        .filter((F.col("rn") == 1) & (F.col("dominance_ratio") >= 0.8))
        .select(
            "Primary Type",
            "Description",
            F.col("Location Description").alias("Location_Description_imputed"),
        )
    )
    work = df.join(location_mode_ranked, on=["Primary Type", "Description"], how="left")
    work = work.withColumn(
        "Location Description",
        F.when(
            F.col("Location Description").isNull(),
            F.coalesce(F.col("Location_Description_imputed"), F.lit("Unknown")),
        ).otherwise(F.col("Location Description")),
    ).drop("Location_Description_imputed")

    district_mapping = (
        work.filter(F.col("District").isNotNull() & (F.col("District") != 0))
        .groupBy("Beat")
        .agg(F.first("District", ignorenulls=True).alias("District_from_beat"))
    )
    work = work.join(district_mapping, on="Beat", how="left")
    work = work.withColumn(
        "District",
        F.when(
            F.col("District").isNull() | (F.col("District") == 0),
            F.col("District_from_beat"),
        ).otherwise(F.col("District")),
    ).drop("District_from_beat")

    geo_valid = F.col("Latitude").isNotNull() & F.col("Longitude").isNotNull()
    community_means = (
        work.filter(
            F.col("Community Area").isNotNull() & (F.col("Community Area") != 0) & geo_valid
        )
        .groupBy("Community Area")
        .agg(F.avg("Latitude").alias("lat_ca"), F.avg("Longitude").alias("lon_ca"))
    )
    ward_means = (
        work.filter(F.col("Ward").isNotNull() & (F.col("Ward") != 0) & geo_valid)
        .groupBy("Ward")
        .agg(F.avg("Latitude").alias("lat_ward"), F.avg("Longitude").alias("lon_ward"))
    )
    district_means = (
        work.filter(F.col("District").isNotNull() & (F.col("District") != 0) & geo_valid)
        .groupBy("District")
        .agg(F.avg("Latitude").alias("lat_dist"), F.avg("Longitude").alias("lon_dist"))
    )
    global_means = (
        work.filter(geo_valid)
        .agg(F.avg("Latitude").alias("lat_global"), F.avg("Longitude").alias("lon_global"))
        .collect()[0]
    )
    lat_global = global_means["lat_global"]
    lon_global = global_means["lon_global"]

    work = (
        work.join(community_means, on="Community Area", how="left")
        .join(ward_means, on="Ward", how="left")
        .join(district_means, on="District", how="left")
    )
    work = (
        work.withColumn(
            "Latitude",
            F.when(
                F.col("Latitude").isNull(),
                F.coalesce(
                    F.col("lat_ca"), F.col("lat_ward"), F.col("lat_dist"), F.lit(lat_global)
                ),
            ).otherwise(F.col("Latitude")),
        )
        .withColumn(
            "Longitude",
            F.when(
                F.col("Longitude").isNull(),
                F.coalesce(
                    F.col("lon_ca"), F.col("lon_ward"), F.col("lon_dist"), F.lit(lon_global)
                ),
            ).otherwise(F.col("Longitude")),
        )
        .drop("lat_ca", "lon_ca", "lat_ward", "lon_ward", "lat_dist", "lon_dist")
    )

    for col_name in ("Ward", "Community Area", "District"):
        work = work.withColumn(
            col_name,
            F.when(F.col(col_name).isNull() | (F.col(col_name) == 0), F.lit(-1)).otherwise(
                F.col(col_name)
            ),
        )
    return work


def filter_structural_anomalies(
    df: DataFrame,
    report_path: str | Path | None = None,
) -> DataFrame:
    """Filter out rows failing Chicago coordinate and district/ward/community bounds."""
    summary = inspect_structural_anomalies(df)
    if report_path is not None:
        save_anomaly_report(summary, report_path)
    return df.filter(
        F.col("Latitude").between(CHICAGO_LAT_MIN, CHICAGO_LAT_MAX)
        & F.col("Longitude").between(CHICAGO_LON_MIN, CHICAGO_LON_MAX)
        & F.col("District").between(DISTRICT_MIN, DISTRICT_MAX)
        & F.col("Ward").between(WARD_MIN, WARD_MAX)
        & F.col("Community Area").between(COMMUNITY_AREA_MIN, COMMUNITY_AREA_MAX)
    )


def run_legacy_preprocessing(
    df: DataFrame,
    report_path: str | Path | None = None,
) -> DataFrame:
    """Execute the Phase 1 legacy preprocessing sequence and validate output."""
    logger.warning(
        "Running legacy preprocessing mode (drops ID and null-admin rows); "
        "non-destructive Bronze/Silver mode will replace this in Phase 2."
    )
    cleaned = drop_legacy_columns(df)
    cleaned = extract_block_features(cleaned)
    cleaned = impute_missing_legacy(cleaned)
    cleaned = filter_structural_anomalies(cleaned, report_path=report_path)
    validate_processed_dataframe(cleaned)
    return cleaned

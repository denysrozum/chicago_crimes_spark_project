"""Validated Silver pipeline with quarantine routing and deterministic deduplication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, IntegralType, TimestampType
from pyspark.storagelevel import StorageLevel

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
    MIN_VALID_YEAR,
    WARD_MAX,
    WARD_MIN,
)
from chicago_crimes.processing.features import (
    add_spatial_grid,
    add_temporal_features,
    extract_block_features,
)
from chicago_crimes.processing.quality import SilverAccountingReport, validate_silver_dataframe
from chicago_crimes.storage.local import write_parquet

logger = get_logger("processing.silver")

_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "MM/dd/yyyy hh:mm:ss a",
    "MM/dd/yyyy HH:mm:ss",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd",
)

SILVER_LINEAGE_COLUMNS: tuple[str, ...] = (
    "ingestion_timestamp",
    "source_id",
    "ingestion_batch_id",
)


@dataclass(frozen=True)
class SilverPipelineResult:
    """Outputs and row-level accounting from a Silver transformation run."""

    silver_df: DataFrame
    quarantined_df: DataFrame
    superseded_df: DataFrame
    accounting: SilverAccountingReport
    silver_path: Path | None = None
    quarantine_path: Path | None = None
    superseded_path: Path | None = None
    manifest_path: Path | None = None


def _clean_string_expr(df: DataFrame, col_name: str) -> Column:
    if col_name not in df.columns:
        return F.lit(None).cast("string")
    as_str = F.trim(F.col(col_name).cast("string"))
    return F.when(as_str.isNull() | (as_str == ""), F.lit(None).cast("string")).otherwise(as_str)


def _parse_int_col(df: DataFrame, col_name: str) -> tuple[Column, Column]:
    """Return `(parsed_int_col, raw_present_bool_col)` without throwing on malformed text."""
    if col_name not in df.columns:
        return F.lit(None).cast("int"), F.lit(False)
    dtype = df.schema[col_name].dataType
    if isinstance(dtype, IntegralType):
        col_ref = F.col(col_name)
        return col_ref.cast("int"), col_ref.isNotNull()

    cleaned = _clean_string_expr(df, col_name)
    parsed = F.expr(f"try_cast(nullif(trim(cast(`{col_name}` as string)), '') as int)")
    return parsed, cleaned.isNotNull()


def _parse_double_col(df: DataFrame, col_name: str) -> tuple[Column, Column]:
    """Return `(parsed_double_col, raw_present_bool_col)` without throwing on malformed text."""
    if col_name not in df.columns:
        return F.lit(None).cast("double"), F.lit(False)
    dtype = df.schema[col_name].dataType
    if isinstance(dtype, DoubleType | FloatType | IntegralType):
        col_ref = F.col(col_name)
        return col_ref.cast("double"), col_ref.isNotNull()

    cleaned = _clean_string_expr(df, col_name)
    parsed = F.expr(f"try_cast(nullif(trim(cast(`{col_name}` as string)), '') as double)")
    return parsed, cleaned.isNotNull()


def _parse_bool_col(df: DataFrame, col_name: str) -> Column:
    if col_name not in df.columns:
        return F.lit(None).cast("boolean")
    if isinstance(df.schema[col_name].dataType, BooleanType):
        return F.col(col_name)

    lowered = F.lower(_clean_string_expr(df, col_name))
    return (
        F.when(lowered.isin("true", "1", "t", "yes", "y"), F.lit(True))
        .when(lowered.isin("false", "0", "f", "no", "n"), F.lit(False))
        .otherwise(F.lit(None).cast("boolean"))
    )


def _parse_ts_col(df: DataFrame, col_name: str) -> tuple[Column, Column]:
    """Return `(parsed_timestamp_col, raw_present_bool_col)` across CSV and ISO formats."""
    if col_name not in df.columns:
        return F.lit(None).cast("timestamp"), F.lit(False)
    if isinstance(df.schema[col_name].dataType, TimestampType):
        col_ref = F.col(col_name)
        return col_ref, col_ref.isNotNull()

    cleaned = _clean_string_expr(df, col_name)
    parsed = F.coalesce(*[F.to_timestamp(cleaned, fmt) for fmt in _TIMESTAMP_FORMATS])
    return parsed, cleaned.isNotNull()


def _resolve_corrupt_col(df: DataFrame) -> Column:
    if "corrupt_record" in df.columns:
        return F.col("corrupt_record").cast("string")
    if "_corrupt_record" in df.columns:
        return F.col("_corrupt_record").cast("string")
    return F.lit(None).cast("string")


def _stage_silver_candidates(
    df: DataFrame,
    *,
    quarantine_out_of_bounds_geo: bool,
) -> DataFrame:
    """Normalize raw/Bronze records and annotate quarantine reason + quality flags."""
    for required in ("ID", "Date", "Primary Type"):
        if required not in df.columns:
            raise ValueError(f"Input DataFrame missing required column '{required}'")

    corrupt_raw = _resolve_corrupt_col(df)
    parsed_id, _ = _parse_int_col(df, "ID")
    parsed_date, _ = _parse_ts_col(df, "Date")
    parsed_updated, updated_present = _parse_ts_col(df, "Updated On")
    cleaned_primary_type = F.upper(_clean_string_expr(df, "Primary Type"))

    parsed_lat, lat_present = _parse_double_col(df, "Latitude")
    parsed_lon, lon_present = _parse_double_col(df, "Longitude")
    parsed_x, _ = _parse_int_col(df, "X Coordinate")
    parsed_y, _ = _parse_int_col(df, "Y Coordinate")
    parsed_beat, _ = _parse_int_col(df, "Beat")
    parsed_district, _ = _parse_int_col(df, "District")
    parsed_ward, _ = _parse_int_col(df, "Ward")
    parsed_ca, _ = _parse_int_col(df, "Community Area")
    parsed_source_year, _ = _parse_int_col(df, "Year")

    lat_valid_range = (parsed_lat >= CHICAGO_LAT_MIN) & (parsed_lat <= CHICAGO_LAT_MAX)
    lon_valid_range = (parsed_lon >= CHICAGO_LON_MIN) & (parsed_lon <= CHICAGO_LON_MAX)
    both_geo_present = parsed_lat.isNotNull() & parsed_lon.isNotNull()
    both_geo_missing = parsed_lat.isNull() & parsed_lon.isNull()
    partial_geo = parsed_lat.isNull() != parsed_lon.isNull()
    geo_in_bounds = both_geo_present & lat_valid_range & lon_valid_range
    geo_out_of_bounds = both_geo_present & (~geo_in_bounds)

    quarantine_reason = (
        F.when(corrupt_raw.isNotNull(), F.lit("corrupt_csv_row"))
        .when(parsed_id.isNull() | (parsed_id <= 0), F.lit("missing_or_invalid_id"))
        .when(parsed_date.isNull(), F.lit("missing_or_invalid_date"))
        .when(
            (F.year(parsed_date) < MIN_VALID_YEAR)
            | (parsed_date > F.expr("current_timestamp() + INTERVAL 7 DAYS")),
            F.lit("date_out_of_valid_range"),
        )
        .when(updated_present & parsed_updated.isNull(), F.lit("invalid_updated_on"))
        .when(cleaned_primary_type.isNull(), F.lit("missing_primary_type"))
        .when(
            (lat_present & parsed_lat.isNull()) | (lon_present & parsed_lon.isNull()),
            F.lit("invalid_coordinate_format"),
        )
    )
    if quarantine_out_of_bounds_geo:
        quarantine_reason = quarantine_reason.when(
            geo_out_of_bounds, F.lit("out_of_bounds_coordinates")
        )
    quarantine_reason = quarantine_reason.otherwise(F.lit(None).cast("string"))

    clean_lat = F.when(geo_in_bounds, parsed_lat).otherwise(F.lit(None).cast("double"))
    clean_lon = F.when(geo_in_bounds, parsed_lon).otherwise(F.lit(None).cast("double"))
    clean_x = F.when(geo_out_of_bounds, F.lit(None).cast("int")).otherwise(parsed_x)
    clean_y = F.when(geo_out_of_bounds, F.lit(None).cast("int")).otherwise(parsed_y)

    geo_quality_flag = (
        F.when(geo_in_bounds, F.lit("valid"))
        .when(geo_out_of_bounds, F.lit("out_of_bounds_nullified"))
        .when(partial_geo, F.lit("partial_missing_nullified"))
        .when(both_geo_missing, F.lit("missing"))
        .otherwise(F.lit("invalid"))
    )

    dist_oob = parsed_district.isNotNull() & (
        (parsed_district < DISTRICT_MIN) | (parsed_district > DISTRICT_MAX)
    )
    ward_oob = parsed_ward.isNotNull() & ((parsed_ward < WARD_MIN) | (parsed_ward > WARD_MAX))
    ca_oob = parsed_ca.isNotNull() & (
        (parsed_ca < COMMUNITY_AREA_MIN) | (parsed_ca > COMMUNITY_AREA_MAX)
    )
    clean_district = F.when(dist_oob, F.lit(None).cast("int")).otherwise(parsed_district)
    clean_ward = F.when(ward_oob, F.lit(None).cast("int")).otherwise(parsed_ward)
    clean_ca = F.when(ca_oob, F.lit(None).cast("int")).otherwise(parsed_ca)

    any_admin_oob = dist_oob | ward_oob | ca_oob
    all_admin_valid = clean_district.isNotNull() & clean_ward.isNotNull() & clean_ca.isNotNull()
    admin_quality_flag = (
        F.when(any_admin_oob, F.lit("out_of_range_nullified"))
        .when(all_admin_valid, F.lit("valid"))
        .otherwise(F.lit("partial_or_missing"))
    )

    staged = df.withColumns(
        {
            "_quarantine_reason": quarantine_reason,
            "_silver_id": parsed_id,
            "_silver_case_number": _clean_string_expr(df, "Case Number"),
            "_silver_date": parsed_date,
            "_silver_block": _clean_string_expr(df, "Block"),
            "_silver_iucr": _clean_string_expr(df, "IUCR"),
            "_silver_primary_type": cleaned_primary_type,
            "_silver_description": _clean_string_expr(df, "Description"),
            "_silver_location_desc": _clean_string_expr(df, "Location Description"),
            "_silver_arrest": _parse_bool_col(df, "Arrest"),
            "_silver_domestic": _parse_bool_col(df, "Domestic"),
            "_silver_beat": parsed_beat,
            "_silver_district": clean_district,
            "_silver_ward": clean_ward,
            "_silver_community_area": clean_ca,
            "_silver_fbi_code": _clean_string_expr(df, "FBI Code"),
            "_silver_x_coord": clean_x,
            "_silver_y_coord": clean_y,
            "_silver_source_year": parsed_source_year,
            "_silver_updated_on": parsed_updated,
            "_silver_latitude": clean_lat,
            "_silver_longitude": clean_lon,
            "_silver_location": _clean_string_expr(df, "Location"),
            "_has_valid_geo": geo_in_bounds,
            "_geo_quality_flag": geo_quality_flag,
            "_has_valid_admin": all_admin_valid,
            "_admin_quality_flag": admin_quality_flag,
        }
    )
    return staged


def _project_quarantine_df(staged_df: DataFrame, source_columns: list[str]) -> DataFrame:
    """Keep original columns alongside quarantine reason and timestamp."""
    raw_cols = [c for c in source_columns if c != "_corrupt_record"]
    now_lit = F.lit(datetime.now(UTC).replace(tzinfo=None)).cast("timestamp")
    selected = [*raw_cols]
    if "_corrupt_record" in source_columns and "corrupt_record" not in raw_cols:
        selected.append(F.col("_corrupt_record").alias("corrupt_record"))
    selected.extend(
        [
            F.col("_quarantine_reason").alias("quarantine_reason"),
            now_lit.alias("quarantined_at"),
        ]
    )
    return staged_df.filter(F.col("_quarantine_reason").isNotNull()).select(*selected)


def _project_silver_schema(candidate_df: DataFrame) -> DataFrame:
    """Project normalized Silver columns cleanly, dropping staging temporaries."""
    col_map: list[Column] = [
        F.col("_silver_id").alias("ID"),
        F.col("_silver_case_number").alias("Case Number"),
        F.col("_silver_date").alias("Date"),
        F.col("_silver_block").alias("Block"),
        F.col("_silver_iucr").alias("IUCR"),
        F.col("_silver_primary_type").alias("Primary Type"),
        F.col("_silver_description").alias("Description"),
        F.col("_silver_location_desc").alias("Location Description"),
        F.col("_silver_arrest").alias("Arrest"),
        F.col("_silver_domestic").alias("Domestic"),
        F.col("_silver_beat").alias("Beat"),
        F.col("_silver_district").alias("District"),
        F.col("_silver_ward").alias("Ward"),
        F.col("_silver_community_area").alias("Community Area"),
        F.col("_silver_fbi_code").alias("FBI Code"),
        F.col("_silver_x_coord").alias("X Coordinate"),
        F.col("_silver_y_coord").alias("Y Coordinate"),
        F.col("_silver_source_year").alias("source_year"),
        F.col("_silver_updated_on").alias("Updated On"),
        F.col("_silver_latitude").alias("Latitude"),
        F.col("_silver_longitude").alias("Longitude"),
        F.col("_silver_location").alias("Location"),
        F.col("_has_valid_geo").alias("has_valid_geo"),
        F.col("_geo_quality_flag").alias("geo_quality_flag"),
        F.col("_has_valid_admin").alias("has_valid_admin"),
        F.col("_admin_quality_flag").alias("admin_quality_flag"),
    ]
    for lineage_col in SILVER_LINEAGE_COLUMNS:
        if lineage_col in candidate_df.columns:
            col_map.append(F.col(lineage_col))
    return candidate_df.select(*col_map)


def _rank_by_id(projected_df: DataFrame) -> DataFrame:
    """Rank valid candidates per `ID` deterministically by newest `Updated On`."""
    order_cols: list[Column] = [F.col("Updated On").desc_nulls_last()]
    if "ingestion_timestamp" in projected_df.columns:
        order_cols.append(F.col("ingestion_timestamp").desc_nulls_last())
    if "source_id" in projected_df.columns:
        order_cols.append(F.col("source_id").asc_nulls_last())
    order_cols.extend([F.col("Date").desc_nulls_last(), F.col("Primary Type").asc_nulls_last()])

    win = Window.partitionBy("ID").orderBy(*order_cols)
    return projected_df.withColumn("_dedup_rank", F.row_number().over(win))


def _enrich_silver_features(df: DataFrame, *, grid_size: float = 0.01) -> DataFrame:
    """Add temporal, non-imputed address block, and valid-only spatial grid features."""
    enriched = add_temporal_features(df, date_col="Date")
    enriched = extract_block_features(enriched, impute_missing_block_number=False)
    enriched = add_spatial_grid(enriched, grid_size=grid_size)
    return enriched


def _check_output_targets(
    targets: tuple[Path | None, ...],
    mode: str,
) -> None:
    if mode != "errorifexists":
        return
    for target in targets:
        if target is not None and target.exists():
            raise FileExistsError(
                f"Output path already exists ({target}). Pass mode='overwrite' or --overwrite."
            )


def run_silver_pipeline(
    df: DataFrame,
    silver_dir: str | Path | None = None,
    quarantine_dir: str | Path | None = None,
    superseded_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    mode: str = "errorifexists",
    deduplicate: bool = True,
    quarantine_out_of_bounds_geo: bool = False,
    grid_size: float = 0.01,
) -> SilverPipelineResult:
    """Execute non-destructive Silver normalization, deduplication, and row accounting."""
    silver_path = Path(silver_dir) if silver_dir is not None else None
    quarantine_path = Path(quarantine_dir) if quarantine_dir is not None else None
    superseded_path = Path(superseded_dir) if superseded_dir is not None else None
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else (silver_path / "_silver_accounting.json" if silver_path is not None else None)
    )
    _check_output_targets((silver_path, quarantine_path, superseded_path), mode)

    source_columns = list(df.columns)
    input_count = int(df.count())

    in_memory_mode = silver_path is None
    storage_level = StorageLevel.MEMORY_AND_DISK if in_memory_mode else StorageLevel.DISK_ONLY

    staged_raw = _stage_silver_candidates(
        df, quarantine_out_of_bounds_geo=quarantine_out_of_bounds_geo
    )
    staged = staged_raw.persist(StorageLevel.MEMORY_AND_DISK) if in_memory_mode else staged_raw

    ranked_df: DataFrame | None = None
    try:
        quarantined_raw = _project_quarantine_df(staged, source_columns)
        if quarantine_path is not None:
            write_parquet(quarantined_raw, quarantine_path, mode=mode)
            quarantined_out = df.sparkSession.read.parquet(str(quarantine_path))
        else:
            quarantined_out = quarantined_raw.persist(StorageLevel.MEMORY_AND_DISK)

        reason_rows = quarantined_out.groupBy("quarantine_reason").count().collect()
        quarantine_reasons = {
            str(row["quarantine_reason"]): int(row["count"])
            for row in sorted(reason_rows, key=lambda r: str(r["quarantine_reason"]))
        }
        quarantined_count = sum(quarantine_reasons.values())

        candidates = _project_silver_schema(staged.filter(F.col("_quarantine_reason").isNull()))
        if deduplicate:
            ranked_df = _rank_by_id(candidates).persist(storage_level)
            superseded_raw = (
                ranked_df.filter(F.col("_dedup_rank") > 1)
                .drop("_dedup_rank")
                .withColumn("superseded_reason", F.lit("older_duplicate_id"))
            )
            accepted_raw = ranked_df.filter(F.col("_dedup_rank") == 1).drop("_dedup_rank")
        else:
            superseded_raw = candidates.limit(0).withColumn(
                "superseded_reason", F.lit("older_duplicate_id")
            )
            accepted_raw = candidates

        if superseded_path is not None:
            write_parquet(superseded_raw, superseded_path, mode=mode)
            superseded_out = df.sparkSession.read.parquet(str(superseded_path))
        else:
            superseded_out = superseded_raw.persist(StorageLevel.MEMORY_AND_DISK)

        superseded_count = int(superseded_out.count())

        enriched_silver = _enrich_silver_features(accepted_raw, grid_size=grid_size)
        if silver_path is not None:
            write_parquet(enriched_silver, silver_path, mode=mode)
            silver_out = df.sparkSession.read.parquet(str(silver_path))
        else:
            silver_out = enriched_silver.persist(StorageLevel.MEMORY_AND_DISK)

        metrics_row = silver_out.agg(
            F.count(F.lit(1)).alias("accepted_records"),
            F.sum(F.when(~F.col("has_valid_geo"), 1).otherwise(0)).alias("missing_geo_records"),
            F.sum(
                F.when(F.col("geo_quality_flag") == "out_of_bounds_nullified", 1).otherwise(0)
            ).alias("out_of_bounds_geo_nullified"),
            F.sum(F.when(~F.col("has_valid_admin"), 1).otherwise(0)).alias("missing_admin_records"),
            F.sum(
                F.when(F.col("admin_quality_flag") == "out_of_range_nullified", 1).otherwise(0)
            ).alias("out_of_range_admin_nullified"),
        ).collect()[0]

        accounting = SilverAccountingReport(
            input_records=input_count,
            accepted_records=int(metrics_row["accepted_records"] or 0),
            superseded_records=superseded_count,
            quarantined_records=quarantined_count,
            quarantine_reasons=quarantine_reasons,
            missing_geo_records=int(metrics_row["missing_geo_records"] or 0),
            out_of_bounds_geo_nullified=int(metrics_row["out_of_bounds_geo_nullified"] or 0),
            missing_admin_records=int(metrics_row["missing_admin_records"] or 0),
            out_of_range_admin_nullified=int(metrics_row["out_of_range_admin_nullified"] or 0),
        )

        validate_silver_dataframe(
            silver_out,
            accounting=accounting,
            require_unique_id=deduplicate,
        )

        if resolved_manifest is not None:
            resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
            resolved_manifest.write_text(
                json.dumps(accounting.to_dict(), indent=2),
                encoding="utf-8",
            )

        logger.info(
            "Silver pipeline finished: input=%s accepted=%s superseded=%s quarantined=%s",
            accounting.input_records,
            accounting.accepted_records,
            accounting.superseded_records,
            accounting.quarantined_records,
        )
        return SilverPipelineResult(
            silver_df=silver_out,
            quarantined_df=quarantined_out,
            superseded_df=superseded_out,
            accounting=accounting,
            silver_path=silver_path,
            quarantine_path=quarantine_path,
            superseded_path=superseded_path,
            manifest_path=resolved_manifest,
        )
    finally:
        if ranked_df is not None:
            ranked_df.unpersist()
        if in_memory_mode:
            staged.unpersist()

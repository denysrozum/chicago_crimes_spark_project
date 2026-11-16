"""Reusable spatial, temporal, and address block feature engineering."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

CHICAGO_TIMEZONE: str = "America/Chicago"

TIMESTAMP_DST_FLAGS: tuple[str, ...] = (
    "valid_cst",
    "valid_cdt",
    "dst_spring_gap_nonexistent",
    "dst_fall_fold_ambiguous",
    "missing",
)

_TEMPORAL_PARSE_FORMATS: tuple[str, ...] = (
    "MM/dd/yyyy hh:mm:ss a",
    "MM/dd/yyyy HH:mm:ss",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd",
)


def add_temporal_features(
    df: DataFrame,
    date_col: str = "Date",
    *,
    local_timezone: str = CHICAGO_TIMEZONE,
) -> DataFrame:
    """Add Chicago local wall-clock calendar features, UTC instants, and DST flags idempotently.

    Contract (`spark.sql.session.timeZone = "UTC"`):
    - `Date` and `ts` represent naive Chicago local wall-clock timestamps (`America/Chicago`).
    - `year`, `month`, `hour`, `day_of_week_num`, and `day_of_week` are derived directly from
      the Chicago local wall-clock timestamp `ts`, so historical daily/monthly aggregations and
      ML forecasting features follow Chicago local calendar boundaries.
    - `ts_utc` is the exact UTC instant corresponding to `ts` in `America/Chicago` when
      unambiguously derivable. When `ts` falls inside a nonexistent spring-forward DST gap
      (`02:00:00–02:59:59` on spring transition day) or an ambiguous fall-back DST fold
      (`01:00:00–01:59:59` on fall transition day without a source UTC offset), `ts_utc` is set
      to `NULL` rather than inventing precision, and `timestamp_dst_flag` records the status.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required timestamp column '{date_col}'")

    field = df.schema[date_col]
    if isinstance(field.dataType, TimestampType):
        ts_expr = F.col(date_col)
    else:
        cleaned = F.trim(F.col(date_col).cast("string"))
        ts_expr = F.coalesce(
            *[F.to_timestamp(cleaned, fmt) for fmt in _TEMPORAL_PARSE_FORMATS],
            F.to_timestamp(cleaned),
        )

    work = df.withColumn("ts", ts_expr)
    raw_utc = F.to_utc_timestamp(F.col("ts"), local_timezone)
    roundtrip_local = F.from_utc_timestamp(raw_utc, local_timezone)
    prev_hour_local = F.from_utc_timestamp(raw_utc - F.expr("INTERVAL 1 HOUR"), local_timezone)
    next_hour_local = F.from_utc_timestamp(raw_utc + F.expr("INTERVAL 1 HOUR"), local_timezone)

    is_gap = F.col("ts").isNotNull() & (roundtrip_local != F.col("ts"))
    is_fold = (
        F.col("ts").isNotNull()
        & (~is_gap)
        & ((prev_hour_local == F.col("ts")) | (next_hour_local == F.col("ts")))
    )
    utc_offset_seconds = F.unix_timestamp(raw_utc) - F.unix_timestamp(F.col("ts"))

    timestamp_dst_flag = (
        F.when(F.col("ts").isNull(), F.lit("missing"))
        .when(is_gap, F.lit("dst_spring_gap_nonexistent"))
        .when(is_fold, F.lit("dst_fall_fold_ambiguous"))
        .when(utc_offset_seconds == F.lit(18000), F.lit("valid_cdt"))
        .otherwise(F.lit("valid_cst"))
    )
    ts_utc = F.when(
        F.col("ts").isNotNull() & (~is_gap) & (~is_fold),
        raw_utc,
    ).otherwise(F.lit(None).cast("timestamp"))

    return (
        work.withColumn("ts_utc", ts_utc)
        .withColumn("timestamp_dst_flag", timestamp_dst_flag)
        .withColumn("hour", F.hour("ts"))
        .withColumn("year", F.year("ts"))
        .withColumn("month", F.month("ts"))
        .withColumn("day_of_week_num", F.dayofweek("ts"))
        .withColumn(
            "day_of_week",
            F.when(F.dayofweek("ts") == 1, F.lit("Sun"))
            .when(F.dayofweek("ts") == 2, F.lit("Mon"))
            .when(F.dayofweek("ts") == 3, F.lit("Tue"))
            .when(F.dayofweek("ts") == 4, F.lit("Wed"))
            .when(F.dayofweek("ts") == 5, F.lit("Thu"))
            .when(F.dayofweek("ts") == 6, F.lit("Fri"))
            .when(F.dayofweek("ts") == 7, F.lit("Sat")),
        )
    )


def add_spatial_grid(
    df: DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    grid_size: float = 0.01,
) -> DataFrame:
    """Discretize coordinates into fixed-resolution spatial grid cells."""
    factor = int(round(1.0 / grid_size))
    return df.withColumn(
        "geo_grid",
        F.when(
            F.col(lat_col).isNotNull() & F.col(lon_col).isNotNull(),
            F.concat_ws(
                "_",
                F.floor(F.col(lat_col) * F.lit(factor)).cast("int"),
                F.floor(F.col(lon_col) * F.lit(factor)).cast("int"),
            ),
        ),
    )


def _mode_block_number_table(
    df: DataFrame,
    group_cols: list[str],
    alias: str,
) -> DataFrame:
    """Compute modal block_number per group (tie-broken by smallest block_number)."""
    counts = (
        df.filter(F.col("block_number").isNotNull()).groupBy(*group_cols, "block_number").count()
    )
    win = Window.partitionBy(*group_cols).orderBy(F.desc("count"), F.asc("block_number"))
    return (
        counts.withColumn("rn", F.row_number().over(win))
        .filter(F.col("rn") == 1)
        .select(*group_cols, F.col("block_number").alias(alias))
    )


def extract_block_features(df: DataFrame, impute_missing_block_number: bool = True) -> DataFrame:
    """Parse block_number, direction, street_type, and street_name from `Block`."""
    work = df.withColumn("block_number_raw", F.regexp_extract(F.col("Block"), r"^(\d+)", 1))
    work = work.withColumn(
        "block_number",
        F.when(F.col("block_number_raw") != "", F.col("block_number_raw").cast("int")),
    ).drop("block_number_raw")

    work = work.withColumn("direction", F.regexp_extract(F.col("Block"), r"\b(N|S|E|W)\b", 1))
    work = work.withColumn(
        "street_type",
        F.upper(
            F.regexp_extract(
                F.col("Block"),
                r"\b(ST|AVE|AV|RD|BLVD|DR|LN|CT|PL|PKWY|HWY|TER|CIR|WAY)\b\.?$",
                1,
            )
        ),
    )
    work = work.withColumn(
        "street_name",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(F.col("Block"), r"^\d*XX\s*", ""),
                    r"\b(N|S|E|W)\b\s*",
                    "",
                ),
                r"\b(ST|AVE|AV|RD|BLVD|DR|LN|CT|PL|PKWY|HWY|TER|CIR|WAY)\b\.?$",
                "",
            )
        ),
    )

    work = (
        work.withColumn(
            "direction",
            F.when(F.col("direction") == "", F.lit("UNKNOWN")).otherwise(F.col("direction")),
        )
        .withColumn(
            "street_type",
            F.when(F.col("street_type") == "", F.lit("UNKNOWN")).otherwise(F.col("street_type")),
        )
        .withColumn(
            "street_name",
            F.when(
                F.col("street_name").isNull() | (F.col("street_name") == ""),
                F.lit("UNKNOWN"),
            ).otherwise(F.col("street_name")),
        )
    )

    if not impute_missing_block_number:
        return work

    exact_mode = _mode_block_number_table(
        work, ["street_name", "street_type", "direction", "District"], "block_number_exact_imputed"
    )
    street_mode = _mode_block_number_table(
        work, ["street_name", "street_type", "direction"], "block_number_street_imputed"
    )
    type_mode = _mode_block_number_table(
        work, ["street_name", "street_type"], "block_number_type_imputed"
    )

    work = (
        work.join(
            exact_mode, on=["street_name", "street_type", "direction", "District"], how="left"
        )
        .join(street_mode, on=["street_name", "street_type", "direction"], how="left")
        .join(type_mode, on=["street_name", "street_type"], how="left")
    )

    work = work.withColumn(
        "block_number",
        F.when(
            F.col("block_number").isNull(),
            F.coalesce(
                F.col("block_number_exact_imputed"),
                F.col("block_number_street_imputed"),
                F.col("block_number_type_imputed"),
            ),
        ).otherwise(F.col("block_number")),
    )
    return work.drop(
        "block_number_exact_imputed",
        "block_number_street_imputed",
        "block_number_type_imputed",
    )

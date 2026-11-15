"""Feature cardinality and class-imbalance profiling experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, ByteType, IntegerType, LongType, ShortType, StringType
from pyspark.storagelevel import StorageLevel

from chicago_crimes.processing.features import add_spatial_grid, add_temporal_features

DEFAULT_EXCLUDE_COLS: tuple[str, ...] = (
    "ID",
    "Case Number",
    "Location",
    "X Coordinate",
    "Y Coordinate",
    "Updated On",
    "source_year",
    "ingestion_timestamp",
    "source_id",
    "ingestion_batch_id",
    "corrupt_record",
    "has_valid_geo",
    "has_valid_admin",
)
DEFAULT_FORCE_INCLUDE_NUMERIC: tuple[str, ...] = (
    "Beat",
    "District",
    "Ward",
    "Community Area",
    "crime_year",
    "crime_month",
    "crime_hour",
    "crime_dayofweek",
    "block_number",
)


def _is_integer_like(field: Any) -> bool:
    return isinstance(field.dataType, IntegerType | LongType | ShortType | ByteType)


def _classify_imbalance(dominance_ratio: float) -> str:
    if dominance_ratio >= 0.9:
        return "almost_constant"
    if dominance_ratio >= 0.6:
        return "strong_imbalance"
    if dominance_ratio >= 0.3:
        return "moderate_imbalance"
    return "relatively_balanced"


def _add_crime_time_bins(df: DataFrame, date_col: str = "Date") -> DataFrame:
    temporal = add_temporal_features(df, date_col=date_col)
    return (
        temporal.withColumn("crime_year", F.col("year"))
        .withColumn("crime_month", F.col("month"))
        .withColumn("crime_hour", F.col("hour"))
        .withColumn("crime_dayofweek", F.col("day_of_week_num"))
        .drop("ts", "hour", "year", "month", "day_of_week_num", "day_of_week")
    )


def _should_analyze_column(
    df: DataFrame,
    field: Any,
    max_distinct: int,
    exclude_cols: set[str],
    force_include_numeric: set[str],
    auxiliary_used_cols: set[str],
) -> tuple[bool, int | None, str]:
    col_name = field.name
    if col_name in auxiliary_used_cols:
        return False, None, "used_for_derived_feature"
    if col_name in exclude_cols:
        return False, None, "excluded"

    if isinstance(field.dataType, StringType | BooleanType):
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            return False, distinct_count, "skipped_constant"
        return True, distinct_count, "categorical"

    if col_name in force_include_numeric and _is_integer_like(field):
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            return False, distinct_count, "skipped_constant"
        return True, distinct_count, "forced_numeric"

    if _is_integer_like(field):
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            return False, distinct_count, "skipped_constant"
        if distinct_count <= max_distinct:
            return True, distinct_count, "low_cardinality_numeric"
        return False, distinct_count, "skipped_high_cardinality_numeric"

    return False, None, "skipped_unsupported_type"


def _analyze_single_column(
    df: DataFrame,
    col_name: str,
    distinct_count: int,
    total_rows: int,
    top_n: int,
) -> dict[str, Any] | None:
    top_rows = (
        df.groupBy(col_name)
        .count()
        .orderBy(F.desc("count"), F.asc(col_name))
        .limit(top_n)
        .collect()
    )
    if not top_rows:
        return None

    top_value = top_rows[0][0]
    top_count = int(top_rows[0][1])
    dominance_ratio = top_count / total_rows if total_rows > 0 else 0.0
    label = _classify_imbalance(dominance_ratio)

    distribution_rows: list[dict[str, Any]] = []
    cumulative_top_n = 0
    for row in top_rows:
        cnt = int(row["count"])
        ratio = cnt / total_rows if total_rows > 0 else 0.0
        cumulative_top_n += cnt
        distribution_rows.append(
            {
                "feature": col_name,
                "value": str(row[col_name]),
                "count": cnt,
                "ratio": round(float(ratio), 6),
            }
        )

    return {
        "feature": col_name,
        "distinct_count": int(distinct_count),
        "top_value": str(top_value),
        "top_count": top_count,
        "dominance_ratio": round(float(dominance_ratio), 6),
        "top_n_ratio": round(float(cumulative_top_n / total_rows if total_rows > 0 else 0.0), 6),
        "imbalance_label": label,
        "distribution_rows": distribution_rows,
    }


def _format_section(title: str, section_df: pd.DataFrame, top_n: int) -> list[str]:
    lines = [title, "-" * len(title)]
    if section_df.empty:
        lines.extend(["None", ""])
        return lines

    for _, row in section_df.iterrows():
        lines.append(
            f"{row['feature']}: type={row['analysis_type']}, distinct={row['distinct_count']}, "
            f"top_value={row['top_value']}, top_count={row['top_count']}, "
            f"dominance_ratio={float(row['dominance_ratio']):.6f}, "
            f"top_{top_n}_ratio={float(row['top_n_ratio']):.6f}, "
            f"label={row['imbalance_label']}"
        )
    lines.append("")
    return lines


def run_imbalance_analysis(
    df: DataFrame,
    out_dir: str | Path = "artifacts/experiments/imbalance",
    max_distinct_numeric: int = 100,
    top_n: int = 10,
    exclude_cols: list[str] | tuple[str, ...] | None = None,
    force_include_numeric: list[str] | tuple[str, ...] | None = None,
    enable_spatial_grid: bool = True,
    grid_size: float = 0.01,
    enable_time_features: bool = True,
    date_col: str = "Date",
) -> dict[str, Any]:
    """Compute column dominance ratios and categorical/binned imbalance diagnostics."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    excluded = set(exclude_cols if exclude_cols is not None else DEFAULT_EXCLUDE_COLS)
    forced = set(
        force_include_numeric
        if force_include_numeric is not None
        else DEFAULT_FORCE_INCLUDE_NUMERIC
    )
    auxiliary_used: set[str] = set()
    work_df = df

    if enable_spatial_grid and {"Latitude", "Longitude"}.issubset(work_df.columns):
        work_df = add_spatial_grid(work_df, grid_size=grid_size)
        auxiliary_used.update({"Latitude", "Longitude"})

    if enable_time_features and date_col in work_df.columns:
        work_df = _add_crime_time_bins(work_df, date_col=date_col)
        auxiliary_used.add(date_col)

    work_df = work_df.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        total_rows = work_df.count()
        summary_rows: list[dict[str, Any]] = []
        distribution_all: list[dict[str, Any]] = []

        for field in work_df.schema.fields:
            should_analyze, distinct_count, reason = _should_analyze_column(
                work_df,
                field,
                max_distinct=max_distinct_numeric,
                exclude_cols=excluded,
                force_include_numeric=forced,
                auxiliary_used_cols=auxiliary_used,
            )
            if not should_analyze or distinct_count is None:
                summary_rows.append(
                    {
                        "feature": field.name,
                        "analysis_type": reason,
                        "distinct_count": int(distinct_count) if distinct_count is not None else "",
                        "top_value": "",
                        "top_count": "",
                        "dominance_ratio": "",
                        "top_n_ratio": "",
                        "imbalance_label": "skipped",
                    }
                )
                continue

            result = _analyze_single_column(
                work_df,
                col_name=field.name,
                distinct_count=distinct_count,
                total_rows=total_rows,
                top_n=top_n,
            )
            if result is None:
                continue

            summary_rows.append(
                {
                    "feature": result["feature"],
                    "analysis_type": reason,
                    "distinct_count": result["distinct_count"],
                    "top_value": result["top_value"],
                    "top_count": result["top_count"],
                    "dominance_ratio": result["dominance_ratio"],
                    "top_n_ratio": result["top_n_ratio"],
                    "imbalance_label": result["imbalance_label"],
                }
            )
            distribution_all.extend(result["distribution_rows"])
    finally:
        work_df.unpersist()

    summary_df = pd.DataFrame(summary_rows)
    distribution_df = (
        pd.DataFrame(distribution_all)
        if distribution_all
        else pd.DataFrame(columns=["feature", "value", "count", "ratio"])
    )

    label_order = {
        "almost_constant": 0,
        "strong_imbalance": 1,
        "moderate_imbalance": 2,
        "relatively_balanced": 3,
        "skipped": 4,
    }
    if not summary_df.empty:
        summary_df["label_rank"] = summary_df["imbalance_label"].map(label_order)
        summary_df["dominance_sort"] = pd.to_numeric(
            summary_df["dominance_ratio"], errors="coerce"
        ).fillna(-1)
        summary_df = summary_df.sort_values(
            by=["label_rank", "dominance_sort", "feature"],
            ascending=[True, False, True],
        ).drop(columns=["label_rank", "dominance_sort"])

    summary_csv_path = target / "imbalance_summary.csv"
    summary_txt_path = target / "imbalance_summary.txt"
    distribution_csv_path = target / "imbalance_top_values.csv"

    summary_df.to_csv(summary_csv_path, index=False)
    distribution_df.to_csv(distribution_csv_path, index=False)

    analyzed = summary_df[summary_df["imbalance_label"] != "skipped"]
    skipped = summary_df[
        (summary_df["imbalance_label"] == "skipped")
        & (summary_df["analysis_type"] != "used_for_derived_feature")
    ]
    auxiliary_df = summary_df[summary_df["analysis_type"] == "used_for_derived_feature"]

    almost_constant_df = analyzed[analyzed["imbalance_label"] == "almost_constant"]
    strong_df = analyzed[analyzed["imbalance_label"] == "strong_imbalance"]
    moderate_df = analyzed[analyzed["imbalance_label"] == "moderate_imbalance"]
    balanced_df = analyzed[analyzed["imbalance_label"] == "relatively_balanced"]

    lines = [
        "===== DATASET IMBALANCE ANALYSIS =====",
        "",
        f"Total rows: {total_rows}",
        f"Total features in report: {len(summary_df)}",
        f"Analyzed features: {len(analyzed)}",
        f"Skipped features: {len(skipped)}",
        f"Auxiliary source fields: {len(auxiliary_df)}",
        "",
    ]
    lines.extend(_format_section("Almost constant features", almost_constant_df, top_n))
    lines.extend(_format_section("Strongly imbalanced features", strong_df, top_n))
    lines.extend(_format_section("Moderately imbalanced features", moderate_df, top_n))
    lines.extend(_format_section("Relatively balanced features", balanced_df, top_n))

    lines.extend(["Skipped features", "----------------"])
    if skipped.empty:
        lines.append("None")
    else:
        for _, row in skipped.iterrows():
            lines.append(f"{row['feature']}: {row['analysis_type']}")

    lines.extend(["", "Auxiliary source fields", "-----------------------"])
    if auxiliary_df.empty:
        lines.append("None")
    else:
        for _, row in auxiliary_df.iterrows():
            lines.append(f"{row['feature']}: used_for_derived_feature")

    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_path_csv": str(summary_csv_path),
        "summary_path_txt": str(summary_txt_path),
        "distribution_path_csv": str(distribution_csv_path),
        "analyzed_count": int(len(analyzed)),
    }

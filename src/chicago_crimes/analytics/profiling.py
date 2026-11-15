"""Dataset profiling, correlation analysis, and non-destructive block statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    NumericType,
    StringType,
    TimestampType,
)
from pyspark.storagelevel import StorageLevel

from chicago_crimes.core.logging import get_logger
from chicago_crimes.processing.features import extract_block_features
from chicago_crimes.processing.quality import (
    StructuralAnomalySummary,
    inspect_structural_anomalies,
    save_anomaly_report,
)
from chicago_crimes.visualization.charts import plot_correlation_matrix

logger = get_logger("analytics.profiling")


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    numeric_column_count: int
    duplicate_count: int
    null_counts: dict[str, int]
    null_percentages: dict[str, float]
    anomalies: StructuralAnomalySummary | None = None
    report_path: Path | None = None


@dataclass(frozen=True)
class BlockProfile:
    total_rows: int
    null_before: dict[str, int]
    null_after: dict[str, int]
    imputed_count: int
    still_missing_count: int
    distinct_after: dict[str, int]
    out_dir: Path


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _format_size(source_path: str | Path | None) -> str:
    if source_path is None:
        return "Source path not provided."
    target = Path(source_path)
    if target.is_file():
        gb = round(target.stat().st_size / (1024**3), 3)
        return f"Source file size: {gb} GB ({target.stat().st_size:,} bytes)"
    if target.is_dir():
        total = _dir_size_bytes(target)
        gb = round(total / (1024**3), 3)
        return f"Source directory size: {gb} GB ({total:,} bytes)"
    return f"Source path not found: {target}"


def profile_dataset(
    df: DataFrame,
    out_dir: str | Path | None = None,
    dataset_name: str = "dataset",
    source_path: str | Path | None = None,
    include_anomalies: bool = True,
    cache_df: bool = False,
) -> DatasetProfile:
    """Compute schema, missingness, duplicate, informativeness, and anomaly statistics."""
    work = df.persist(StorageLevel.MEMORY_AND_DISK) if cache_df else df
    try:
        row_count = work.count()
        col_count = len(work.columns)
        numeric_cols = [f.name for f in work.schema.fields if isinstance(f.dataType, NumericType)]

        null_exprs = [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in work.columns]
        null_row = work.select(null_exprs).collect()[0].asDict() if work.columns else {}
        null_counts = {k: int(v) for k, v in null_row.items()}
        null_pct = {
            k: (float(v) / float(row_count) * 100.0 if row_count else 0.0)
            for k, v in null_counts.items()
        }

        duplicate_count = int(row_count - work.dropDuplicates().count()) if row_count else 0

        type_counts = Counter(f.dataType.simpleString() for f in work.schema.fields)
        string_cols = [f.name for f in work.schema.fields if isinstance(f.dataType, StringType)]
        distinct_map: dict[str, int] = {}
        if string_cols and row_count > 0:
            distinct_row = work.select(
                [F.countDistinct(F.col(c)).alias(c) for c in string_cols]
            ).collect()[0]
            distinct_map = {k: int(v) for k, v in distinct_row.asDict().items()}

        required_anomaly_cols = {
            "Date",
            "Latitude",
            "Longitude",
            "District",
            "Ward",
            "Community Area",
        }
        anomalies: StructuralAnomalySummary | None = None
        if include_anomalies and required_anomaly_cols <= set(work.columns):
            anomalies = inspect_structural_anomalies(work)

        report_path: Path | None = None
        if out_dir is not None:
            target_dir = Path(out_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            report_path = target_dir / f"dataset_stats_{dataset_name}.txt"

            lines: list[str] = [
                f"===== DATASET PROFILE: {dataset_name} =====",
                f"Rows: {row_count}",
                f"Columns: {col_count}",
                f"Numeric columns: {len(numeric_cols)}",
                f"Non-numeric columns: {col_count - len(numeric_cols)}",
                f"Exact duplicate rows: {duplicate_count}",
                _format_size(source_path),
                "",
                "Data types:",
            ]
            for dtype, count in sorted(type_counts.items()):
                lines.append(f"  {dtype}: {count}")

            lines.extend(["", "Null counts and percentages:"])
            for col_name in work.columns:
                lines.append(f"  {col_name}: {null_counts[col_name]} ({null_pct[col_name]:.5f}%)")

            if numeric_cols and row_count > 0:
                lines.extend(["", "Numeric summary:"])
                desc_pdf = work.select(numeric_cols).describe().toPandas()
                lines.append(desc_pdf.to_string(index=False))

            lines.extend(["", "Feature informativeness:"])
            for field in work.schema.fields:
                col_name = field.name
                dtype = field.dataType
                lines.append(f"--- {col_name} ({dtype.simpleString()}) ---")
                if isinstance(dtype, StringType) and row_count > 0:
                    top_freq = (
                        work.groupBy(col_name).count().orderBy(F.desc("count")).limit(1).collect()
                    )
                    if top_freq:
                        top_val, top_cnt = top_freq[0][0], int(top_freq[0][1])
                        ratio = round(top_cnt / row_count, 4)
                        lines.append(
                            f"  Distinct: {distinct_map.get(col_name)} | "
                            f"Top: {top_val} | Dominance: {ratio}"
                        )
                elif isinstance(dtype, IntegerType | DoubleType) and row_count > 0:
                    stats_row = work.select(
                        F.variance(col_name).alias("var"),
                        F.stddev(col_name).alias("std"),
                    ).collect()[0]
                    lines.append(f"  Variance: {stats_row['var']} | Stddev: {stats_row['std']}")
                elif isinstance(dtype, BooleanType | TimestampType):
                    lines.append("  Skipped (boolean/timestamp).")

            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if anomalies is not None:
                save_anomaly_report(
                    anomalies, target_dir / f"structural_anomalies_{dataset_name}.txt"
                )

        return DatasetProfile(
            row_count=row_count,
            column_count=col_count,
            numeric_column_count=len(numeric_cols),
            duplicate_count=duplicate_count,
            null_counts=null_counts,
            null_percentages=null_pct,
            anomalies=anomalies,
            report_path=report_path,
        )
    finally:
        if cache_df:
            work.unpersist()


def compute_correlation_matrix(df: DataFrame) -> pd.DataFrame | None:
    """Compute Pearson correlation matrix across numeric columns using Spark ML."""
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    if len(numeric_cols) < 2:
        logger.warning("Fewer than 2 numeric columns (%s); skipping correlation.", numeric_cols)
        return None

    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    vector_df = assembler.transform(df.select(numeric_cols)).select("features")
    if vector_df.isEmpty():
        logger.warning("No valid non-null numeric rows left for correlation.")
        return None

    corr_matrix = Correlation.corr(vector_df, "features", method="pearson").head()[0]
    return pd.DataFrame(corr_matrix.toArray(), index=numeric_cols, columns=numeric_cols)


def find_high_correlations(
    corr_df: pd.DataFrame, threshold: float = 0.9
) -> list[tuple[str, str, float]]:
    """Return pairs with |correlation| > threshold."""
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr_df.iloc[i, j])
            if not pd.isna(val) and abs(val) > threshold:
                pairs.append((cols[i], cols[j], val))
    return pairs


def run_correlation_analysis(
    df: DataFrame,
    out_dir: str | Path,
    threshold: float = 0.9,
) -> tuple[pd.DataFrame | None, list[tuple[str, str, float]]]:
    """Compute correlation matrix and write CSV, PNG heatmap, and high-correlation summary."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    corr_df = compute_correlation_matrix(df)
    if corr_df is None:
        return None, []

    corr_df.to_csv(target / "corr_matrix.csv")
    plot_correlation_matrix(corr_df, target / "corr_matrix.png")
    high_pairs = find_high_correlations(corr_df, threshold=threshold)

    lines = [f"High correlation pairs (|r| > {threshold}):"]
    if high_pairs:
        for left, right, val in high_pairs:
            lines.append(f"  {left} <-> {right}: {val:.4f}")
    else:
        lines.append("  None")
    (target / "high_correlations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corr_df, high_pairs


def _write_top_values(df: DataFrame, col_name: str, out_file: Path, top_n: int = 30) -> None:
    rows = (
        df.groupBy(col_name)
        .count()
        .orderBy(F.desc("count"), F.asc(col_name))
        .limit(top_n)
        .collect()
    )
    lines = [f"===== TOP {top_n} VALUES: {col_name} =====", ""]
    for row in rows:
        lines.append(f"{row[col_name]}: {row['count']}")
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_block_stats(
    df: DataFrame,
    out_dir: str | Path,
    max_sample_export: int = 10000,
) -> BlockProfile:
    """Analyze parsed address block features non-destructively (never mutates source data)."""
    if "Block" not in df.columns:
        raise ValueError("Missing required 'Block' column for block feature analysis.")

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    block_cols = ["street_name", "street_type", "direction", "block_number"]

    base_df = extract_block_features(df, impute_missing_block_number=False)
    total_rows = base_df.count()
    before_row = (
        base_df.select([F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in block_cols])
        .collect()[0]
        .asDict()
    )
    null_before = {k: int(v) for k, v in before_row.items()}

    base_with_orig = base_df.withColumn("block_number_original", F.col("block_number"))
    imputed_df = extract_block_features(base_with_orig, impute_missing_block_number=True)

    after_row = (
        imputed_df.select([F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in block_cols])
        .collect()[0]
        .asDict()
    )
    null_after = {k: int(v) for k, v in after_row.items()}
    distinct_after = {c: int(imputed_df.select(c).distinct().count()) for c in block_cols}

    imputed_rows_df = imputed_df.filter(
        F.col("block_number_original").isNull() & F.col("block_number").isNotNull()
    ).select(
        *[
            c
            for c in (
                "Block",
                "street_name",
                "street_type",
                "direction",
                "District",
                "block_number",
            )
            if c in imputed_df.columns
        ]
    )
    imputed_count = imputed_rows_df.count()
    still_missing_count = null_after["block_number"]

    if imputed_count > 0:
        imputed_rows_df.limit(max_sample_export).toPandas().to_csv(
            target / "imputed_block_number_sample.csv", index=False
        )

    lines = [
        "===== BLOCK-DERIVED FEATURES SUMMARY (NON-DESTRUCTIVE) =====",
        f"Total rows: {total_rows}",
        "",
        "Null counts BEFORE imputation:",
    ]
    for col_name in block_cols:
        cnt = null_before[col_name]
        pct = (cnt / total_rows * 100.0) if total_rows else 0.0
        lines.append(f"  {col_name}: {cnt} ({pct:.8f}%)")

    lines.extend(["", "Null counts AFTER imputation:"])
    for col_name in block_cols:
        cnt = null_after[col_name]
        pct = (cnt / total_rows * 100.0) if total_rows else 0.0
        lines.append(f"  {col_name}: {cnt} ({pct:.8f}%)")

    lines.extend(["", "Distinct counts AFTER imputation:"])
    for col_name in block_cols:
        lines.append(f"  {col_name}: {distinct_after[col_name]}")

    lines.extend(
        [
            "",
            "Imputation summary:",
            f"  Missing block_number before: {null_before['block_number']}",
            f"  Successfully imputed: {imputed_count}",
            f"  Still missing after: {still_missing_count}",
        ]
    )
    (target / "block_feature_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for col_name in block_cols:
        _write_top_values(imputed_df, col_name, target / f"top_{col_name}.txt", top_n=30)

    return BlockProfile(
        total_rows=total_rows,
        null_before=null_before,
        null_after=null_after,
        imputed_count=imputed_count,
        still_missing_count=still_missing_count,
        distinct_after=distinct_after,
        out_dir=target,
    )

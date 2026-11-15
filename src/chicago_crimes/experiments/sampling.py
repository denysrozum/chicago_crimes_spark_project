"""Class-imbalance resampling and 2D PCA comparison experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

from chicago_crimes.experiments._common import (
    select_numeric_and_bool_features,
    vector_to_list,
)
from chicago_crimes.storage.local import write_parquet


def _get_binary_distribution(df: DataFrame, target_col: str) -> dict[str, Any]:
    rows = df.groupBy(target_col).count().collect()
    if len(rows) != 2:
        raise ValueError(f"Target '{target_col}' must have exactly 2 classes, found {len(rows)}.")

    distribution = {row[target_col]: int(row["count"]) for row in rows}
    sorted_items = sorted(distribution.items(), key=lambda x: (x[1], str(x[0])))
    minority_value, minority_count = sorted_items[0]
    majority_value, majority_count = sorted_items[1]
    return {
        "distribution": distribution,
        "minority_value": minority_value,
        "minority_count": minority_count,
        "majority_value": majority_value,
        "majority_count": majority_count,
    }


def _prepare_modeling_df(df: DataFrame, target_col: str) -> tuple[DataFrame, list[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    work_df, feature_cols = select_numeric_and_bool_features(df, target_col=target_col)
    if len(feature_cols) < 2:
        raise ValueError(f"Not enough usable numeric/bool features for target '{target_col}'.")
    return work_df, feature_cols


_vector_to_list = vector_to_list


def _sample_exact_without_replacement(
    df: DataFrame, required_count: int, total_count: int, seed: int = 42
) -> DataFrame:
    if required_count >= total_count:
        return df

    fraction = min(1.0, (required_count / max(1, total_count)) * 1.15)
    sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    sampled_count = sampled.count()

    attempts = 0
    while sampled_count < required_count and attempts < 6:
        fraction = min(1.0, max(fraction * 1.35, (required_count + 1) / max(1, total_count)))
        sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed + attempts + 1)
        sampled_count = sampled.count()
        attempts += 1

    return sampled.limit(required_count)


def _sample_exact_with_replacement(
    df: DataFrame, required_count: int, base_count: int, seed: int = 42
) -> DataFrame:
    if required_count <= 0:
        return df.limit(0)

    fraction = max(1.0, (required_count / max(1, base_count)) * 1.25)
    sampled = df.sample(withReplacement=True, fraction=fraction, seed=seed)
    sampled_count = sampled.count()

    attempts = 0
    while sampled_count < required_count and attempts < 6:
        fraction *= 1.35
        sampled = df.sample(withReplacement=True, fraction=fraction, seed=seed + attempts + 1)
        sampled_count = sampled.count()
        attempts += 1

    return sampled.limit(required_count)


def _oversample_to_balance(
    df: DataFrame, target_col: str, seed: int = 42
) -> tuple[DataFrame, dict[str, Any]]:
    info = _get_binary_distribution(df, target_col)
    minority_df = df.filter(F.col(target_col) == info["minority_value"])
    majority_df = df.filter(F.col(target_col) == info["majority_value"])
    oversampled_minority = _sample_exact_with_replacement(
        minority_df,
        required_count=info["majority_count"],
        base_count=info["minority_count"],
        seed=seed,
    )
    return majority_df.unionByName(oversampled_minority), info


def _undersample_to_balance(
    df: DataFrame, target_col: str, seed: int = 42
) -> tuple[DataFrame, dict[str, Any]]:
    info = _get_binary_distribution(df, target_col)
    minority_df = df.filter(F.col(target_col) == info["minority_value"])
    majority_df = df.filter(F.col(target_col) == info["majority_value"])
    sampled_majority = _sample_exact_without_replacement(
        majority_df,
        required_count=info["minority_count"],
        total_count=info["majority_count"],
        seed=seed,
    )
    return minority_df.unionByName(sampled_majority), info


def _build_pca_projection(
    df: DataFrame, target_col: str, sample_max_rows: int = 50000, seed: int = 42
) -> tuple[pd.DataFrame, list[float], list[str]]:
    work_df, feature_names = _prepare_modeling_df(df, target_col)
    total_count = work_df.count()
    if total_count == 0:
        raise ValueError(f"Empty DataFrame for PCA projection on target '{target_col}'.")

    if total_count > sample_max_rows:
        fraction = min(1.0, (sample_max_rows / total_count) * 1.15)
        sampled_df = work_df.sample(withReplacement=False, fraction=fraction, seed=seed).limit(
            sample_max_rows
        )
    else:
        sampled_df = work_df
    sampled_df = sampled_df.dropna()

    assembler = VectorAssembler(inputCols=feature_names, outputCol="features", handleInvalid="skip")
    assembled_df = assembler.transform(sampled_df)
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withStd=True, withMean=True
    )
    scaled_df = scaler.fit(assembled_df).transform(assembled_df)

    pca_model = PCA(k=2, inputCol="scaled_features", outputCol="pca_features").fit(scaled_df)
    pca_df = pca_model.transform(scaled_df).select(target_col, "pca_features")

    rows = pca_df.limit(sample_max_rows).collect()
    data = [
        {target_col: row[target_col], "PC1": coords[0], "PC2": coords[1]}
        for row in rows
        for coords in [_vector_to_list(row["pca_features"])]
    ]
    return pd.DataFrame(data), [float(x) for x in pca_model.explainedVariance], feature_names


def _plot_pca_scatter(pdf: pd.DataFrame, target_col: str, out_path: Path, title: str) -> None:
    if pdf.empty:
        return
    plt.figure(figsize=(10, 7))
    for label, group in pdf.groupby(target_col):
        plt.scatter(group["PC1"], group["PC2"], s=8, alpha=0.5, label=str(label))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title=target_col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_distribution_report(
    before_info: dict[str, Any],
    after_df: DataFrame,
    target_col: str,
    method: str,
    out_path: Path,
    feature_cols: list[str],
) -> None:
    after_counts = {
        row[target_col]: int(row["count"]) for row in after_df.groupBy(target_col).count().collect()
    }
    before_dist = before_info["distribution"]
    b_min = before_info["minority_value"]
    b_min_cnt = before_info["minority_count"]
    b_maj = before_info["majority_value"]
    b_maj_cnt = before_info["majority_count"]

    a_min_cnt = after_counts.get(b_min, 0)
    a_maj_cnt = after_counts.get(b_maj, 0)
    min_delta = a_min_cnt - b_min_cnt
    maj_delta = a_maj_cnt - b_maj_cnt

    ratio_before = b_min_cnt / b_maj_cnt
    ratio_after = a_min_cnt / a_maj_cnt if a_maj_cnt != 0 else 0.0

    lines = [
        "===== SAMPLING REPORT =====",
        "",
        f"Target column: {target_col}",
        f"Method: {method}",
        f"Modeling features count: {len(feature_cols)}",
        f"Modeling features: {feature_cols}",
        "",
        "Before sampling",
        "----------------",
    ]
    for key, value in sorted(before_dist.items(), key=lambda x: str(x[0])):
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Detected roles",
            "--------------",
            f"Minority class: {b_min} ({b_min_cnt})",
            f"Majority class: {b_maj} ({b_maj_cnt})",
            f"Minority / Majority ratio before: {ratio_before:.6f}",
            "",
            "After sampling",
            "--------------",
        ]
    )
    for key, value in sorted(after_counts.items(), key=lambda x: str(x[0])):
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Change summary",
            "--------------",
            f"Minority class change: {min_delta:+d}",
            f"Majority class change: {maj_delta:+d}",
            f"Minority / Majority ratio after: {ratio_after:.6f}",
        ]
    )
    if method == "oversampling":
        growth_pct = (min_delta / b_min_cnt * 100.0) if b_min_cnt > 0 else 0.0
        lines.append(f"Minority class increase (%): {growth_pct:.4f}")
    else:
        shrink_pct = (-maj_delta / b_maj_cnt * 100.0) if b_maj_cnt > 0 else 0.0
        lines.append(f"Majority class reduction (%): {shrink_pct:.4f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_pca_report(
    explained_before: list[float],
    explained_after: list[float],
    out_path: Path,
    target_col: str,
    method: str,
) -> None:
    lines = [
        "===== PCA SAMPLING COMPARISON =====",
        "",
        f"Target column: {target_col}",
        f"Method: {method}",
        "",
        "Before sampling",
        "---------------",
        f"PC1 explained variance: {explained_before[0]:.6f}",
        f"PC2 explained variance: {explained_before[1]:.6f}",
        f"Cumulative: {sum(explained_before):.6f}",
        "",
        "After sampling",
        "--------------",
        f"PC1 explained variance: {explained_after[0]:.6f}",
        f"PC2 explained variance: {explained_after[1]:.6f}",
        f"Cumulative: {sum(explained_after):.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_sampling_case(
    df: DataFrame,
    target_col: str,
    method: str,
    out_dir: Path,
    seed: int = 42,
    pca_sample_max_rows: int = 50000,
    write_sampled_dataset: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    modeling_df, feature_cols = _prepare_modeling_df(df, target_col)
    modeling_df = modeling_df.dropna().persist(StorageLevel.MEMORY_AND_DISK)
    sampled_df: DataFrame | None = None
    try:
        if method == "oversampling":
            sampled_df, before_info = _oversample_to_balance(modeling_df, target_col, seed=seed)
        elif method == "undersampling":
            sampled_df, before_info = _undersample_to_balance(modeling_df, target_col, seed=seed)
        else:
            raise ValueError("Method must be 'oversampling' or 'undersampling'.")

        sampled_df = sampled_df.persist(StorageLevel.MEMORY_AND_DISK)
        sampled_path: Path | None = None
        if write_sampled_dataset:
            sampled_path = write_parquet(sampled_df, out_dir / "sampled_dataset", mode="overwrite")

        report_path = out_dir / "sampling_report.txt"
        _save_distribution_report(
            before_info=before_info,
            after_df=sampled_df,
            target_col=target_col,
            method=method,
            out_path=report_path,
            feature_cols=feature_cols,
        )

        before_pca_pdf, explained_before, before_feature_cols = _build_pca_projection(
            modeling_df, target_col=target_col, sample_max_rows=pca_sample_max_rows, seed=seed
        )
        after_pca_pdf, explained_after, after_feature_cols = _build_pca_projection(
            sampled_df, target_col=target_col, sample_max_rows=pca_sample_max_rows, seed=seed
        )

        modeling_rows = modeling_df.count()
        sampled_rows = sampled_df.count()
    finally:
        if sampled_df is not None:
            sampled_df.unpersist()
        modeling_df.unpersist()

    before_pca_pdf.to_csv(out_dir / "pca_before.csv", index=False)
    after_pca_pdf.to_csv(out_dir / "pca_after.csv", index=False)

    pca_before_path = out_dir / "pca_before.png"
    pca_after_path = out_dir / "pca_after.png"
    _plot_pca_scatter(
        before_pca_pdf,
        target_col=target_col,
        out_path=pca_before_path,
        title=f"PCA before sampling: {target_col}",
    )
    _plot_pca_scatter(
        after_pca_pdf,
        target_col=target_col,
        out_path=pca_after_path,
        title=f"PCA after {method}: {target_col}",
    )
    _save_pca_report(
        explained_before=explained_before,
        explained_after=explained_after,
        out_path=out_dir / "pca_report.txt",
        target_col=target_col,
        method=method,
    )

    run_info_path = out_dir / "run_info.txt"
    run_info_lines = [
        f"Target column: {target_col}",
        f"Method: {method}",
        f"Feature columns used: {feature_cols}",
        f"PCA sample max rows: {pca_sample_max_rows}",
        f"Rows in modeling_df: {modeling_rows}",
        f"Rows in sampled_df: {sampled_rows}",
        f"PCA features before: {before_feature_cols}",
        f"PCA features after: {after_feature_cols}",
    ]
    run_info_path.write_text("\n".join(run_info_lines) + "\n", encoding="utf-8")

    return {
        "sampled_path": str(sampled_path) if sampled_path is not None else None,
        "report_path": str(report_path),
        "pca_before_path": str(pca_before_path),
        "pca_after_path": str(pca_after_path),
        "run_info_path": str(run_info_path),
    }


def run_sampling_analysis(
    df: DataFrame,
    out_dir: str | Path = "artifacts/experiments/sampling",
    target_cols: tuple[str, ...] = ("Domestic", "Arrest"),
    methods: tuple[str, ...] = ("oversampling", "undersampling"),
    seed: int = 42,
    pca_sample_max_rows: int = 50000,
    write_sampled_dataset: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run oversampling and undersampling balance experiments across binary targets."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for target_col in target_cols:
        results[target_col] = {}
        for method in methods:
            case_out_dir = root / target_col.lower() / method
            results[target_col][method] = _run_single_sampling_case(
                df=df,
                target_col=target_col,
                method=method,
                out_dir=case_out_dir,
                seed=seed,
                pca_sample_max_rows=pca_sample_max_rows,
                write_sampled_dataset=write_sampled_dataset,
            )
    return results

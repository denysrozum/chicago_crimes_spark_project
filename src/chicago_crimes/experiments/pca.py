"""Spark MLlib PCA dimensionality-reduction experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.sql import DataFrame

from chicago_crimes.core.logging import get_logger
from chicago_crimes.experiments._common import (
    select_numeric_and_bool_features,
    vector_to_list,
)

logger = get_logger("experiments.pca")


def _prepare_numeric_df(df: DataFrame) -> tuple[DataFrame, list[str]]:
    return select_numeric_and_bool_features(df)


def _bounded_sample_df(
    df: DataFrame,
    fraction: float = 0.02,
    seed: int = 42,
    max_rows: int = 100000,
) -> DataFrame:
    sampled = (
        df if fraction >= 1.0 else df.sample(withReplacement=False, fraction=fraction, seed=seed)
    )
    count_sampled = sampled.count()
    if count_sampled > max_rows:
        ratio = max_rows / count_sampled
        sampled = sampled.sample(withReplacement=False, fraction=ratio, seed=seed)
    return sampled.limit(max_rows)


_vector_to_list = vector_to_list


def _save_summary(explained: list[float], out_path: Path) -> None:
    cumulative: list[float] = []
    running = 0.0
    for value in explained:
        running += float(value)
        cumulative.append(running)

    lines = [
        f"===== PCA with {len(explained)} components =====",
        "",
        "Explained variance by components:",
        f"{'Component':<12}{'Explained variance ratio':<28}{'Explained variance (%)':<24}",
    ]
    for i, ev in enumerate(explained, start=1):
        lines.append(f"{f'PC{i}':<12}{ev:<28.4f}{ev * 100:<24.4f}")

    lines.extend(
        ["", "Cumulative explained variance (%):", f"{'Component':<12}{'Cumulative (%)':<24}"]
    )
    for i, cum in enumerate(cumulative, start=1):
        lines.append(f"{f'PC{i}':<12}{cum * 100:<24.4f}")

    lines.extend(
        [
            "",
            f"Total explained variance by first {len(explained)} "
            f"components: {cumulative[-1] * 100:.2f}%",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_loadings(pc_matrix: Any, feature_names: list[str], out_path: Path) -> None:
    arr = pc_matrix.toArray()
    lines = ["PCA LOADINGS", "=" * 60, ""]
    for comp_idx in range(arr.shape[1]):
        lines.extend([f"PC{comp_idx + 1}", "-" * 40])
        pairs = [
            (feat_name, float(arr[feat_idx, comp_idx]))
            for feat_idx, feat_name in enumerate(feature_names)
        ]
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        for feat_name, weight in pairs:
            lines.append(f"{feat_name}: {weight:.6f}")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_cumulative_variance(explained: list[float], out_path: Path) -> None:
    cumulative: list[float] = []
    running = 0.0
    for value in explained:
        running += float(value)
        cumulative.append(running * 100.0)
    labels = [f"PC{i}" for i in range(1, len(cumulative) + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, cumulative, marker="o")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title(f"Cumulative Explained Variance ({len(labels)} Components)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_2d(scores_pdf: pd.DataFrame, out_path: Path) -> None:
    if not {"PC1", "PC2"}.issubset(scores_pdf.columns):
        return
    plt.figure(figsize=(10, 7))
    plt.scatter(scores_pdf["PC1"], scores_pdf["PC2"], s=8, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_3d(scores_pdf: pd.DataFrame, out_path: Path) -> None:
    if not {"PC1", "PC2", "PC3"}.issubset(scores_pdf.columns):
        return
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores_pdf["PC1"], scores_pdf["PC2"], scores_pdf["PC3"], s=8, alpha=0.45)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Projection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_pca_analysis(
    df: DataFrame,
    out_dir: str | Path = "artifacts/experiments/pca",
    n_components: int = 10,
    sample_fraction: float = 0.1,
    sample_max_rows: int = 50000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Spark StandardScaler + PCA on bounded numeric/boolean sample."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    numeric_df, feature_names = _prepare_numeric_df(df)
    if len(feature_names) < 2:
        raise ValueError(f"Not enough numeric/boolean features for PCA: {feature_names}")

    sampled_df = _bounded_sample_df(
        numeric_df, fraction=sample_fraction, seed=seed, max_rows=sample_max_rows
    ).dropna()

    row_count = sampled_df.count()
    if row_count < 2:
        raise ValueError(f"PCA requires at least 2 non-null sampled rows, got {row_count}.")

    effective_k = min(n_components, len(feature_names), row_count)
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features", handleInvalid="skip")
    assembled_df = assembler.transform(sampled_df).select("features")

    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withStd=True, withMean=True
    )
    scaled_df = scaler.fit(assembled_df).transform(assembled_df)

    pca_model = PCA(k=effective_k, inputCol="scaled_features", outputCol="pca_features").fit(
        scaled_df
    )
    pca_df = pca_model.transform(scaled_df)

    explained = [float(x) for x in pca_model.explainedVariance]
    _save_summary(explained, target / "pca_summary.txt")
    _save_loadings(pca_model.pc, feature_names, target / "pca_loadings.txt")
    _plot_cumulative_variance(explained, target / "pca_cumulative_variance.png")

    scores_rows = pca_df.select("pca_features").limit(sample_max_rows).collect()
    scores = [_vector_to_list(row["pca_features"]) for row in scores_rows]
    score_cols = [f"PC{i}" for i in range(1, effective_k + 1)]
    scores_pdf = pd.DataFrame(scores, columns=score_cols)
    scores_pdf.to_csv(target / "pca_scores_sample.csv", index=False)

    if effective_k >= 2:
        _plot_2d(scores_pdf, target / "pca_2d_scatter.png")
    if effective_k >= 3:
        _plot_3d(scores_pdf, target / "pca_3d_scatter.png")

    run_info_lines = [
        f"Input numeric features: {feature_names}",
        f"Number of numeric features: {len(feature_names)}",
        f"Requested components: {n_components}",
        f"Effective components: {effective_k}",
        f"Sample fraction: {sample_fraction}",
        f"Sample max rows: {sample_max_rows}",
        f"Used rows after sampling and dropna: {scores_pdf.shape[0]}",
    ]
    (target / "pca_run_info.txt").write_text("\n".join(run_info_lines) + "\n", encoding="utf-8")

    return {
        "feature_names": feature_names,
        "explained_variance": explained,
        "used_rows": int(scores_pdf.shape[0]),
        "out_dir": str(target),
    }

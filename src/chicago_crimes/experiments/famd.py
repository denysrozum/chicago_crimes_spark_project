"""Factor Analysis of Mixed Data (FAMD) experiment with bounded driver sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import prince
from pyspark.sql import DataFrame

from chicago_crimes.core.logging import get_logger

logger = get_logger("experiments.famd")

DEFAULT_NUMERIC_COLS: tuple[str, ...] = (
    "Beat",
    "District",
    "Ward",
    "Community Area",
    "Latitude",
    "Longitude",
)
DEFAULT_CATEGORICAL_COLS: tuple[str, ...] = (
    "Primary Type",
    "FBI Code",
    "Location Description",
    "Arrest",
    "Domestic",
)
DEFAULT_COLLAPSE_TOP_N: dict[str, int] = {"Location Description": 20}


def _sample_spark_df(
    df: DataFrame,
    fraction: float = 0.01,
    max_rows: int = 20000,
    seed: int = 42,
) -> DataFrame:
    if max_rows > 100000:
        raise ValueError(
            f"FAMD sample_max_rows={max_rows} exceeds safe driver collection limit of 100,000."
        )
    sampled = (
        df if fraction >= 1.0 else df.sample(withReplacement=False, fraction=fraction, seed=seed)
    )
    return sampled.limit(max_rows)


def _collapse_top_categories(
    series: pd.Series, top_n: int = 20, other_label: str = "OTHER"
) -> pd.Series:
    obj = series.astype("object").fillna("Unknown")
    top_values = obj.value_counts(dropna=False).head(top_n).index
    return obj.where(obj.isin(top_values), other_label)


def _prepare_famd_dataframe(
    pdf: pd.DataFrame,
    numeric_cols: list[str] | tuple[str, ...],
    categorical_cols: list[str] | tuple[str, ...],
    collapse_top_n: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    existing_numeric = [c for c in numeric_cols if c in pdf.columns]
    existing_categorical = [c for c in categorical_cols if c in pdf.columns]

    missing_numeric = [c for c in numeric_cols if c not in pdf.columns]
    missing_categorical = [c for c in categorical_cols if c not in pdf.columns]
    if missing_numeric:
        logger.warning("Missing numeric columns for FAMD: %s", missing_numeric)
    if missing_categorical:
        logger.warning("Missing categorical columns for FAMD: %s", missing_categorical)

    selected_cols = existing_numeric + existing_categorical
    if not selected_cols:
        raise ValueError("No valid columns found for FAMD.")

    work = pdf[selected_cols].copy()
    for col in existing_numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        median_val = work[col].median()
        work[col] = work[col].fillna(0.0 if pd.isna(median_val) else median_val)

    for col in existing_categorical:
        series = work[col].astype("object")
        work[col] = series.where(series.notna(), "Unknown")

    if collapse_top_n is not None:
        for col, top_n in collapse_top_n.items():
            if col in work.columns:
                work[col] = _collapse_top_categories(work[col], top_n=top_n)

    for col in existing_categorical:
        work[col] = work[col].astype("category")

    return work, existing_numeric, existing_categorical


def _extract_variance_info(model: prince.FAMD) -> tuple[list[float], list[float]]:
    explained_ratio: list[float] | None = None
    cumulative_ratio: list[float] | None = None

    if hasattr(model, "percentage_of_variance_"):
        explained_ratio = [float(x) / 100.0 for x in model.percentage_of_variance_]
    if hasattr(model, "cumulative_percentage_of_variance_"):
        cumulative_ratio = [float(x) / 100.0 for x in model.cumulative_percentage_of_variance_]

    if explained_ratio is None and hasattr(model, "eigenvalues_"):
        eigenvalues = [float(x) for x in model.eigenvalues_]
        total = sum(eigenvalues)
        if total > 0:
            explained_ratio = [x / total for x in eigenvalues]

    if explained_ratio is None:
        raise ValueError("Could not extract explained variance from FAMD model.")

    if cumulative_ratio is None:
        cumulative_ratio = []
        running = 0.0
        for value in explained_ratio:
            running += value
            cumulative_ratio.append(running)

    return explained_ratio, cumulative_ratio


def _save_summary(
    explained_ratio: list[float], cumulative_ratio: list[float], out_path: Path
) -> None:
    lines = [
        f"===== FAMD with {len(explained_ratio)} components =====",
        "",
        "Explained variance by components:",
        f"{'Component':<12}{'Explained variance ratio':<28}{'Explained variance (%)':<24}",
    ]
    for i, ev in enumerate(explained_ratio, start=1):
        lines.append(f"{f'FAMD{i}':<12}{ev:<28.6f}{ev * 100:<24.4f}")

    lines.extend(
        ["", "Cumulative explained variance (%):", f"{'Component':<12}{'Cumulative (%)':<24}"]
    )
    for i, cum in enumerate(cumulative_ratio, start=1):
        lines.append(f"{f'FAMD{i}':<12}{cum * 100:<24.4f}")

    lines.extend(
        [
            "",
            f"Total explained variance by first {len(explained_ratio)} "
            f"components: {cumulative_ratio[-1] * 100:.2f}%",
            "",
        ]
    )
    for threshold in (0.7, 0.8, 0.9, 0.95):
        needed = next(
            (i for i, cum in enumerate(cumulative_ratio, start=1) if cum >= threshold), None
        )
        if needed is None:
            lines.append(
                f"Threshold {threshold * 100:.0f}% was not reached. "
                f"Maximum cumulative variance = {cumulative_ratio[-1] * 100:.4f}%"
            )
        else:
            lines.append(f"Components needed for {threshold * 100:.0f}% variance: {needed}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_dataframe_txt(frame: pd.DataFrame, out_path: Path, title: str) -> None:
    content = f"{title}\n{'=' * len(title)}\n\n{frame.to_string()}\n"
    out_path.write_text(content, encoding="utf-8")


def _plot_cumulative_variance(
    explained_ratio: list[float], cumulative_ratio: list[float], out_path: Path
) -> None:
    labels = [f"FAMD{i}" for i in range(1, len(explained_ratio) + 1)]
    y = [x * 100.0 for x in cumulative_ratio]
    plt.figure(figsize=(10, 6))
    plt.plot(labels, y, marker="o")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("FAMD Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_2d(coords_df: pd.DataFrame, out_path: Path) -> None:
    if not {"Dim1", "Dim2"}.issubset(coords_df.columns):
        return
    plt.figure(figsize=(10, 7))
    plt.scatter(coords_df["Dim1"], coords_df["Dim2"], s=8, alpha=0.45)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.title("FAMD 2D Projection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_3d(coords_df: pd.DataFrame, out_path: Path) -> None:
    if not {"Dim1", "Dim2", "Dim3"}.issubset(coords_df.columns):
        return
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords_df["Dim1"], coords_df["Dim2"], coords_df["Dim3"], s=8, alpha=0.4)
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    ax.set_title("FAMD 3D Projection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_famd_analysis(
    spark_df: DataFrame,
    out_dir: str | Path = "artifacts/experiments/famd",
    sample_fraction: float = 0.1,
    sample_max_rows: int = 20000,
    n_components: int = 6,
    seed: int = 42,
    numeric_cols: list[str] | tuple[str, ...] | None = None,
    categorical_cols: list[str] | tuple[str, ...] | None = None,
    collapse_top_n: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run FAMD on a strictly bounded sample collected to the driver."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    num_cols = list(numeric_cols or DEFAULT_NUMERIC_COLS)
    cat_cols = list(categorical_cols or DEFAULT_CATEGORICAL_COLS)
    collapse = DEFAULT_COLLAPSE_TOP_N if collapse_top_n is None else collapse_top_n

    sampled_spark_df = _sample_spark_df(
        spark_df, fraction=sample_fraction, max_rows=sample_max_rows, seed=seed
    )
    select_cols = [c for c in (num_cols + cat_cols) if c in sampled_spark_df.columns]
    if not select_cols:
        raise ValueError("None of the requested FAMD columns exist in input DataFrame.")

    pdf = sampled_spark_df.select(*select_cols).toPandas()
    famd_df, used_numeric, used_categorical = _prepare_famd_dataframe(
        pdf, numeric_cols=num_cols, categorical_cols=cat_cols, collapse_top_n=collapse
    )

    if len(famd_df) < 3:
        raise ValueError(f"FAMD requires at least 3 sampled rows, got {len(famd_df)}.")

    # Drop completely constant columns so prince/sklearn SVD stays well-conditioned.
    variable_cols = [c for c in famd_df.columns if famd_df[c].nunique(dropna=False) > 1]
    if len(variable_cols) < 2:
        raise ValueError("FAMD requires at least 2 non-constant features after sampling.")
    famd_df = famd_df[variable_cols]
    used_numeric = [c for c in used_numeric if c in variable_cols]
    used_categorical = [c for c in used_categorical if c in variable_cols]

    effective_k = min(n_components, famd_df.shape[1], max(1, famd_df.shape[0] - 1))
    model = prince.FAMD(n_components=effective_k, random_state=seed, engine="sklearn").fit(famd_df)

    row_coords = model.row_coordinates(famd_df).copy()
    row_coords.columns = [f"Dim{i}" for i in range(1, row_coords.shape[1] + 1)]
    row_coords.to_csv(target / "famd_row_coordinates.csv", index=False)

    explained_ratio, cumulative_ratio = _extract_variance_info(model)
    _save_summary(explained_ratio, cumulative_ratio, target / "famd_summary.txt")

    if hasattr(model, "column_coordinates"):
        col_coords = model.column_coordinates(famd_df)
        if isinstance(col_coords, pd.DataFrame):
            col_coords.to_csv(target / "famd_column_coordinates.csv")
            _save_dataframe_txt(
                col_coords, target / "famd_column_coordinates.txt", "FAMD COLUMN COORDINATES"
            )

    if hasattr(model, "column_contributions_"):
        col_contrib = model.column_contributions_
        if isinstance(col_contrib, pd.DataFrame):
            col_contrib.to_csv(target / "famd_column_contributions.csv")
            _save_dataframe_txt(
                col_contrib, target / "famd_column_contributions.txt", "FAMD COLUMN CONTRIBUTIONS"
            )

    _plot_cumulative_variance(
        explained_ratio, cumulative_ratio, target / "famd_cumulative_variance.png"
    )
    if effective_k >= 2:
        _plot_2d(row_coords, target / "famd_2d_scatter.png")
    if effective_k >= 3:
        _plot_3d(row_coords, target / "famd_3d_scatter.png")

    run_info_lines = [
        f"Input rows after sampling: {len(pdf)}",
        f"Used numeric features: {used_numeric}",
        f"Used categorical features: {used_categorical}",
        f"Requested components: {n_components}",
        f"Effective components: {effective_k}",
        f"Sample fraction: {sample_fraction}",
        f"Sample max rows: {sample_max_rows}",
        f"Final dataframe shape for FAMD: {famd_df.shape}",
    ]
    (target / "famd_run_info.txt").write_text("\n".join(run_info_lines) + "\n", encoding="utf-8")

    return {
        "out_dir": str(target),
        "used_numeric": used_numeric,
        "used_categorical": used_categorical,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative_ratio,
    }

"""Unit tests for preprocessing, profiling, quality checks, and experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from chicago_crimes.analytics.profiling import (
    profile_dataset,
    run_block_stats,
    run_correlation_analysis,
)
from chicago_crimes.core.schemas import RAW_CRIME_SCHEMA
from chicago_crimes.experiments import (
    run_famd_analysis,
    run_imbalance_analysis,
    run_pca_analysis,
    run_sampling_analysis,
)
from chicago_crimes.ingestion.csv_loader import load_crime_csv
from chicago_crimes.processing.cleaning import LEGACY_DROPPED_COLUMNS, run_legacy_preprocessing
from chicago_crimes.processing.quality import inspect_structural_anomalies
from chicago_crimes.storage.local import write_single_csv
from tests.unit.conftest import _sample_rows, write_semicolon_csv


def test_csv_loader_and_legacy_preprocessing_parity(spark: SparkSession, tmp_path: Path) -> None:
    rows = list(_sample_rows())
    # Inject one bad row with out-of-range District=99 and null Location Description
    bad_upd = datetime(2025, 11, 1, 12, 0, 0)
    rows.append(
        (
            9999,
            "HX999",
            datetime(2025, 1, 2, 10, 0, 0),
            "001XX N STATE ST",
            "0820",
            "THEFT",
            "OVER $500",
            None,
            False,
            False,
            111,
            99,
            42,
            32,
            "06",
            1176000,
            1900000,
            2025,
            bad_upd,
            41.883,
            -87.628,
            "(41.883, -87.628)",
        )
    )
    csv_path = write_semicolon_csv(rows, tmp_path / "crimes.csv")
    raw_df = load_crime_csv(spark, csv_path)
    assert raw_df.count() == 25

    anomalies = inspect_structural_anomalies(raw_df)
    assert anomalies.invalid_district == 1
    assert anomalies.invalid_coordinates == 0

    report_path = tmp_path / "anomalies.txt"
    cleaned_df = run_legacy_preprocessing(raw_df, report_path=report_path)
    assert cleaned_df.count() == 24
    assert report_path.is_file()
    for col_name in LEGACY_DROPPED_COLUMNS:
        assert col_name not in cleaned_df.columns
    # Verify block_number on row 1024 ("XX N STATE ST") was imputed from modal "001XX N STATE ST"
    assert cleaned_df.filter("block_number IS NULL").count() == 0


def test_profiling_correlation_and_non_destructive_block_stats(
    sample_raw_df: DataFrame, tmp_path: Path
) -> None:
    stats_dir = tmp_path / "stats"
    profile = profile_dataset(
        sample_raw_df, out_dir=stats_dir, dataset_name="raw", include_anomalies=True
    )
    assert profile.row_count == 24
    assert profile.report_path is not None and profile.report_path.is_file()
    assert (stats_dir / "structural_anomalies_raw.txt").is_file()

    corr_df, _ = run_correlation_analysis(sample_raw_df, out_dir=stats_dir)
    assert corr_df is not None
    assert (stats_dir / "corr_matrix.csv").is_file()
    assert (stats_dir / "corr_matrix.png").is_file()
    assert (stats_dir / "high_correlations.txt").is_file()

    block_dir = tmp_path / "block_stats"
    block_profile = run_block_stats(sample_raw_df, out_dir=block_dir)
    assert block_profile.total_rows == 24
    assert block_profile.null_before["block_number"] == 1
    assert block_profile.imputed_count == 1
    assert block_profile.still_missing_count == 0
    assert (block_dir / "block_feature_summary.txt").is_file()
    assert (block_dir / "top_street_name.txt").is_file()


def test_pca_famd_imbalance_and_sampling_experiments(
    sample_processed_df: DataFrame, tmp_path: Path
) -> None:
    pca_res = run_pca_analysis(
        sample_processed_df,
        out_dir=tmp_path / "pca",
        n_components=4,
        sample_fraction=1.0,
        sample_max_rows=100,
    )
    assert len(pca_res["explained_variance"]) == 4
    assert (tmp_path / "pca" / "pca_summary.txt").is_file()
    assert (tmp_path / "pca" / "pca_2d_scatter.png").is_file()
    assert (tmp_path / "pca" / "pca_3d_scatter.png").is_file()

    famd_res = run_famd_analysis(
        sample_processed_df,
        out_dir=tmp_path / "famd",
        sample_fraction=1.0,
        sample_max_rows=100,
        n_components=4,
    )
    assert len(famd_res["explained_ratio"]) == 4
    assert (tmp_path / "famd" / "famd_summary.txt").is_file()
    assert (tmp_path / "famd" / "famd_row_coordinates.csv").is_file()

    with pytest.raises(ValueError, match="safe driver collection limit"):
        run_famd_analysis(
            sample_processed_df, out_dir=tmp_path / "famd_bad", sample_max_rows=200000
        )

    imb_res = run_imbalance_analysis(sample_processed_df, out_dir=tmp_path / "imbalance")
    assert imb_res["analyzed_count"] > 0
    imb_summary = pd.read_csv(imb_res["summary_path_csv"])
    assert "geo_grid" in set(imb_summary["feature"])

    samp_res = run_sampling_analysis(
        sample_processed_df,
        out_dir=tmp_path / "sampling",
        target_cols=("Domestic",),
        methods=("oversampling", "undersampling"),
        pca_sample_max_rows=100,
        write_sampled_dataset=False,
    )
    assert Path(samp_res["Domestic"]["oversampling"]["report_path"]).is_file()
    assert Path(samp_res["Domestic"]["undersampling"]["pca_after_path"]).is_file()
    oversample_text = Path(samp_res["Domestic"]["oversampling"]["report_path"]).read_text(
        encoding="utf-8"
    )
    assert "Minority / Majority ratio after: 1.000000" in oversample_text


def test_raw_schema_rejects_missing_file(spark: SparkSession, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_crime_csv(spark, tmp_path / "missing.csv")
    assert len(RAW_CRIME_SCHEMA.fields) == 22


def test_write_single_csv_non_empty_and_empty_df(sample_raw_df: DataFrame, tmp_path: Path) -> None:
    out_non_empty = write_single_csv(
        sample_raw_df.select("ID", "Primary Type").limit(3),
        tmp_path / "non_empty",
    )
    assert out_non_empty.is_file()
    lines = out_non_empty.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    out_empty = write_single_csv(
        sample_raw_df.select("ID", "Primary Type").limit(0),
        tmp_path / "empty",
    )
    assert out_empty.is_file()
    assert out_empty.read_text(encoding="utf-8").strip() == "ID,Primary Type"

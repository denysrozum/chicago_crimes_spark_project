"""Unit tests for the unified `chicago-crimes` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from chicago_crimes.cli import build_parser, main
from tests.unit.conftest import _sample_rows, write_semicolon_csv


def test_cli_help_and_invalid_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as bad_exc:
        parser.parse_args(["nonexistent-command"])
    assert bad_exc.value.code == 2


def test_cli_analyze_list(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["analyze", "--list"])
    assert code == 0
    out = capsys.readouterr().out
    for idx in range(1, 25):
        assert f"AQ-{idx:02d}" in out


def test_cli_returns_nonzero_on_missing_input(spark: SparkSession, tmp_path: Path) -> None:
    _ = spark
    code = main(
        [
            "--profile",
            "local",
            "ingest",
            "--input",
            str(tmp_path / "does_not_exist.csv"),
            "--output",
            str(tmp_path / "out_parquet"),
        ]
    )
    assert code == 1


def test_cli_end_to_end_smoke_and_overwrite_guard(spark: SparkSession, tmp_path: Path) -> None:
    _ = spark
    csv_path = write_semicolon_csv(_sample_rows(), tmp_path / "sample.csv")
    bronze_dir = tmp_path / "bronze"
    processed_dir = tmp_path / "processed"
    analytics_dir = tmp_path / "analytics"
    profile_dir = tmp_path / "profile"

    assert (
        main(
            [
                "--profile",
                "local",
                "ingest",
                "--input",
                str(csv_path),
                "--output",
                str(bronze_dir),
            ]
        )
        == 0
    )
    assert bronze_dir.exists()

    # Refusing silent overwrite without --overwrite flag
    assert (
        main(
            [
                "--profile",
                "local",
                "ingest",
                "--input",
                str(csv_path),
                "--output",
                str(bronze_dir),
            ]
        )
        == 1
    )

    assert (
        main(
            [
                "--profile",
                "local",
                "process",
                "--input",
                str(csv_path),
                "--output",
                str(processed_dir),
                "--report-path",
                str(tmp_path / "anomalies.txt"),
            ]
        )
        == 0
    )
    assert processed_dir.exists()

    assert (
        main(
            [
                "--profile",
                "local",
                "analyze",
                "--input",
                str(processed_dir),
                "--output",
                str(analytics_dir),
                "--tasks",
                "AQ-01",
                "AQ-02",
            ]
        )
        == 0
    )
    assert (analytics_dir / "AQ-01_night_crimes_by_type" / "result.csv").is_file()
    assert (analytics_dir / "AQ-02_weekend_vs_weekday_distribution" / "chart.png").is_file()
    assert (analytics_dir / "manifest.json").is_file()

    assert (
        main(
            [
                "--profile",
                "local",
                "profile",
                "--input",
                str(processed_dir),
                "--output",
                str(profile_dir),
                "--kind",
                "summary",
            ]
        )
        == 0
    )
    assert (profile_dir / "dataset_stats_dataset.txt").is_file()

    gold_dir = tmp_path / "gold"
    assert (
        main(
            [
                "--profile",
                "local",
                "gold",
                "--input",
                str(processed_dir),
                "--output",
                str(gold_dir),
            ]
        )
        == 0
    )
    assert (gold_dir / "_gold_manifest.json").is_file()
    assert (gold_dir / "daily_citywide_counts").is_dir()

"""End-to-end tests exercising the complete CLI workflow from Bronze ingestion to BigQuery DDL."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession

from chicago_crimes.cli import main
from tests.unit.conftest import _r, write_semicolon_csv


def _e2e_multi_day_rows() -> list[tuple]:
    start = datetime(2025, 1, 1, 14, 0, 0)
    rows: list[tuple] = []
    cid = 5000
    for day in range(45):
        cur = start + timedelta(days=day)
        for district, base_lat, base_lon in [(1, 41.8835, -87.6278), (8, 41.7780, -87.7080)]:
            count_today = 1 + ((day + district) % 3)
            for idx in range(count_today):
                cid += 1
                rows.append(
                    _r(
                        cid,
                        cur + timedelta(hours=idx),
                        "001XX N STATE ST" if district == 1 else "040XX W 63RD ST",
                        "0820" if idx % 2 == 0 else "0486",
                        "THEFT" if idx % 2 == 0 else "BATTERY",
                        "OVER $500" if idx % 2 == 0 else "DOMESTIC BATTERY SIMPLE",
                        "STREET" if district == 1 else "RESIDENCE",
                        idx % 3 == 0,
                        idx % 2 == 1,
                        district * 100 + 11,
                        district,
                        42 if district == 1 else 15,
                        32 if district == 1 else 66,
                        "06" if idx % 2 == 0 else "08B",
                        1176352,
                        1900927,
                        base_lat,
                        base_lon,
                    )
                )
    # Include one exact duplicate replay (same ID, same timestamp -> superseded)
    rows.append(rows[0])
    return rows


def test_full_cli_lifecycle_e2e(spark: SparkSession, tmp_path: Path) -> None:
    """Exercise Bronze -> Silver -> GE -> Gold -> Analytics -> ML -> BigQuery dry-run via CLI."""
    _ = spark
    csv_path = write_semicolon_csv(_e2e_multi_day_rows(), tmp_path / "crimes_e2e.csv")
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"
    superseded_dir = tmp_path / "superseded"
    silver_manifest = tmp_path / "silver_manifest.json"
    ge_report = tmp_path / "ge_report.json"
    gold_dir = tmp_path / "gold"
    analytics_dir = tmp_path / "analytics"
    ml_dir = tmp_path / "ml"
    bq_manifest = tmp_path / "bq_manifest.json"

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
    assert (
        main(
            [
                "--profile",
                "local",
                "process",
                "--pipeline",
                "silver",
                "--input",
                str(bronze_dir),
                "--input-format",
                "parquet",
                "--output",
                str(silver_dir),
                "--quarantine-dir",
                str(quarantine_dir),
                "--superseded-dir",
                str(superseded_dir),
                "--manifest-path",
                str(silver_manifest),
            ]
        )
        == 0
    )

    acc = json.loads(silver_manifest.read_text(encoding="utf-8"))
    assert acc["invariant_passed"] is True
    assert acc["superseded_records"] == 1
    assert acc["quarantined_records"] == 0

    assert (
        main(["--profile", "local", "gold", "--input", str(silver_dir), "--output", str(gold_dir)])
        == 0
    )
    assert (gold_dir / "_gold_manifest.json").is_file()

    assert (
        main(
            [
                "--profile",
                "local",
                "validate",
                "--layer",
                "both",
                "--silver-input",
                str(silver_dir),
                "--gold-input",
                str(gold_dir),
                "--output",
                str(ge_report),
            ]
        )
        == 0
    )
    assert json.loads(ge_report.read_text(encoding="utf-8"))["success"] is True

    assert (
        main(
            [
                "--profile",
                "local",
                "analyze",
                "--input",
                str(silver_dir),
                "--output",
                str(analytics_dir),
                "--tasks",
                "AQ-01",
                "AQ-08",
                "--no-plots",
            ]
        )
        == 0
    )
    manifest_data = json.loads((analytics_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["task_count"] == 2

    assert (
        main(
            [
                "--profile",
                "local",
                "ml",
                "--gold-input",
                str(gold_dir),
                "--output",
                str(ml_dir),
                "--experiment-name",
                "e2e_forecast",
                "--min-district-days",
                "20",
                "--no-spark-models",
            ]
        )
        == 0
    )
    ml_report = json.loads((ml_dir / "ml_evaluation_report.json").read_text(encoding="utf-8"))
    assert ml_report["champion_model"] in {
        "seasonal_naive_lag7",
        "rolling_mean_7_baseline",
        "xgboost_regressor",
    }
    assert (ml_dir / "mlflow.db").is_file()

    assert (
        main(
            [
                "--profile",
                "local",
                "publish-bigquery",
                "--project",
                "e2e-demo-project",
                "--dataset",
                "chicago_crimes_gold",
                "--gold-input",
                str(gold_dir),
                "--silver-input",
                str(silver_dir),
                "--output",
                str(bq_manifest),
            ]
        )
        == 0
    )
    bq_json = json.loads(bq_manifest.read_text(encoding="utf-8"))
    assert bq_json["dry_run"] is True
    assert bq_json["table_count"] == 6

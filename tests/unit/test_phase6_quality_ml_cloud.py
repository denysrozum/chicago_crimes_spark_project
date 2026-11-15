"""Unit tests for Phase 6: Great Expectations, ML forecasting + MLflow, BigQuery, and dbt."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from chicago_crimes.cli import main
from chicago_crimes.ingestion.bronze import ingest_csv_to_bronze
from chicago_crimes.ml.dataset import (
    assert_no_temporal_leakage,
    build_forecasting_features,
    split_chronologically,
)
from chicago_crimes.ml.models import run_forecasting_experiment
from chicago_crimes.processing.ge_validation import (
    validate_gold_with_gx,
    validate_silver_with_gx,
)
from chicago_crimes.processing.gold import build_gold_tables
from chicago_crimes.processing.silver import run_silver_pipeline
from chicago_crimes.storage.bigquery import (
    build_bq_ddl,
    build_bq_schema_spec,
    publish_gold_and_silver_to_bigquery,
    sanitize_bq_column_name,
)
from tests.unit.conftest import _sample_rows, write_semicolon_csv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_daily_district_df(days: int = 60) -> pd.DataFrame:
    start = date(2025, 1, 1)
    rows: list[dict[str, object]] = []
    for district in (1, 8):
        base = 20.0 if district == 1 else 35.0
        for d in range(days):
            cur = start + timedelta(days=d)
            dow = ((cur.weekday() + 1) % 7) + 1  # Spark dayofweek: 1=Sun .. 7=Sat
            weekly = 6.0 if dow in (6, 7) else -3.0
            trend = 0.08 * d
            incidents = int(round(base + weekly + trend + ((d * 3 + district) % 5)))
            arrests = max(1, int(incidents * 0.2))
            rows.append(
                {
                    "crime_date": cur,
                    "District": district,
                    "year": cur.year,
                    "month": cur.month,
                    "day_of_week_num": dow,
                    "incident_count": incidents,
                    "arrest_count": arrests,
                    "arrest_rate_percent": round(arrests / incidents * 100.0, 2),
                }
            )
    return pd.DataFrame(rows)


def test_great_expectations_silver_and_gold_pass_and_catch_bad_data(
    spark: SparkSession, tmp_path: Path
) -> None:
    csv_path = write_semicolon_csv(_sample_rows(), tmp_path / "sample.csv")
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"

    ingest_csv_to_bronze(spark, csv_path=csv_path, bronze_dir=bronze_dir, mode="overwrite")
    silver_res = run_silver_pipeline(
        spark.read.parquet(str(bronze_dir)),
        silver_dir=silver_dir,
        mode="overwrite",
    )
    build_gold_tables(silver_res.silver_df, gold_dir, mode="overwrite")

    silver_summary = validate_silver_with_gx(silver_res.silver_df)
    assert silver_summary["success"] is True
    assert silver_summary["statistics"]["unsuccessful_expectations"] == 0

    gold_summary = validate_gold_with_gx(gold_dir, spark=spark)
    assert gold_summary["success"] is True

    # Corrupted Silver data (null/dup ID, invalid district 99, invalid geo flag) must fail cleanly
    silver_pd = silver_res.silver_df.toPandas()
    bad_silver = silver_pd.copy()
    bad_silver.loc[bad_silver.index[0], "District"] = 99
    bad_silver.loc[bad_silver.index[0], "geo_quality_flag"] = "bogus_flag"
    bad_silver.loc[bad_silver.index[1], "ID"] = bad_silver.loc[bad_silver.index[0], "ID"]

    with pytest.raises(ValueError, match="Silver suite failed"):
        validate_silver_with_gx(bad_silver, raise_on_failure=True)

    failed_report = validate_silver_with_gx(bad_silver, raise_on_failure=False)
    assert failed_report["success"] is False
    assert len(failed_report["failed_expectations"]) >= 3

    # Corrupted Gold table (out-of-range arrest percentage > 100%) must fail cleanly
    bad_gold_city = spark.read.parquet(str(gold_dir / "daily_citywide_counts")).toPandas()
    bad_gold_city.loc[bad_gold_city.index[0], "arrest_rate_percent"] = 145.0
    with pytest.raises(ValueError, match="Gold suite failed"):
        validate_gold_with_gx(
            {"daily_citywide_counts": bad_gold_city},
            raise_on_failure=True,
        )


def test_ml_forecasting_temporal_split_leakage_guard_and_mlflow(
    spark: SparkSession, tmp_path: Path
) -> None:
    daily_pd = _synthetic_daily_district_df(days=60)
    featured = build_forecasting_features(daily_pd, min_district_days=30)

    # Confirm lag_1 matches prior day strictly within district boundary
    d1 = featured[featured["district_id"] == 1].sort_values("crime_date").reset_index(drop=True)
    assert np.isclose(d1.loc[1, "lag_1"], d1.loc[0, "incident_count"])

    split = split_chronologically(featured)
    assert_no_temporal_leakage(split)

    # Injecting overlapping dates between train and test must trigger temporal leakage assertion
    leaky = split.__class__(
        full_df=split.full_df,
        train_df=pd.concat([split.train_df, split.test_df.iloc[[0]]], ignore_index=True),
        val_df=split.val_df,
        test_df=split.test_df,
        feature_columns=split.feature_columns,
        target_column=split.target_column,
        districts=split.districts,
        train_date_range=split.train_date_range,
        val_date_range=split.val_date_range,
        test_date_range=split.test_date_range,
    )
    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_no_temporal_leakage(leaky)

    gold_dir = tmp_path / "gold"
    spark.createDataFrame(daily_pd).write.mode("overwrite").parquet(
        str(gold_dir / "daily_district_counts")
    )
    ml_out = tmp_path / "ml_out"
    bundle = run_forecasting_experiment(
        spark,
        gold_dir=gold_dir,
        output_dir=ml_out,
        experiment_name="unit_forecast_exp",
        min_district_days=30,
        include_spark_models=True,
    )
    assert bundle.report_path.is_file()
    assert (ml_out / "mlflow.db").is_file()
    assert len(bundle.results) == 5
    for res in bundle.results:
        assert res.mlflow_run_id
    for chart_path in bundle.charts.values():
        assert Path(chart_path).is_file()

    # Verify calendar reindexing fills gaps and excludes ultra-sparse catch-all districts (e.g. D31)
    d2 = daily_pd[daily_pd["District"] == 8].copy()
    gap_date = pd.Timestamp(date(2025, 1, 20))
    d2_gapped = d2[pd.to_datetime(d2["crime_date"]) != gap_date]
    d31_sparse = pd.DataFrame(
        [
            {
                "crime_date": date(2025, 1, 1) + timedelta(days=i * 25),
                "District": 31,
                "year": 2025,
                "month": 1,
                "day_of_week_num": 4,
                "incident_count": 1,
                "arrest_count": 0,
                "arrest_rate_percent": 0.0,
            }
            for i in range(35)
        ]
    )
    mixed = pd.concat(
        [daily_pd[daily_pd["District"] == 1], d2_gapped, d31_sparse],
        ignore_index=True,
    )
    clean_featured = build_forecasting_features(mixed, min_district_days=30)
    assert set(clean_featured["district_id"].unique()) == {1, 8}
    day_after_gap = clean_featured[
        (clean_featured["district_id"] == 8)
        & (clean_featured["crime_date"] == pd.Timestamp(date(2025, 1, 21)))
    ]
    assert float(day_after_gap["lag_1"].iloc[0]) == 0.0


def test_bigquery_dry_run_ddl_and_dbt_project_parse(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sanitize_bq_column_name("Primary Type") == "primary_type"
    assert sanitize_bq_column_name("Community Area") == "community_area"

    csv_path = write_semicolon_csv(_sample_rows(), tmp_path / "sample.csv")
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    ingest_csv_to_bronze(spark, csv_path=csv_path, bronze_dir=bronze_dir, mode="overwrite")
    silver_res = run_silver_pipeline(
        spark.read.parquet(str(bronze_dir)),
        silver_dir=silver_dir,
        mode="overwrite",
    )
    build_gold_tables(silver_res.silver_df, gold_dir, mode="overwrite")

    schema_spec = build_bq_schema_spec(
        spark.read.parquet(str(gold_dir / "daily_district_counts"))
    )
    ddl = build_bq_ddl(
        project_id="demo-project",
        dataset_id="chicago_crimes",
        table_name="daily_district_counts",
        schema_spec=schema_spec,
    )
    assert "PARTITION BY `crime_date`" in ddl
    assert "CLUSTER BY `district`" in ddl

    manifest_path = tmp_path / "bq_manifest.json"
    code = main(
        [
            "--profile",
            "local",
            "publish-bigquery",
            "--project",
            "demo-project",
            "--dataset",
            "chicago_crimes",
            "--gold-input",
            str(gold_dir),
            "--silver-input",
            str(silver_dir),
            "--output",
            str(manifest_path),
        ]
    )
    assert code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["table_count"] == 6
    assert all(t["status"] == "dry_run_planned" for t in manifest["tables"])

    # Live execution without ADC must raise descriptive error cleanly
    import google.auth

    def _fail_auth(*args, **kwargs):
        raise RuntimeError("No ADC configured")

    monkeypatch.setattr(google.auth, "default", _fail_auth)
    with pytest.raises(RuntimeError, match="Application Default Credentials"):
        publish_gold_and_silver_to_bigquery(
            spark,
            project_id="demo-project",
            dataset_id="chicago_crimes",
            gold_dir=gold_dir,
            dry_run=False,
        )

    # Unit-test mock verifying multi-part Parquet upload (no coalesce(1)) and staging cleanup
    import google.cloud.bigquery as bq_mod

    dispositions: list[str] = []

    class _FakeJob:
        def result(self) -> None:
            return None

    class _FakeBQClient:
        def __init__(self, project: str, location: str) -> None:
            self.project = project
            self.location = location

        def create_dataset(self, dataset_ref: Any, exists_ok: bool = True) -> None:
            return None

        def load_table_from_file(self, handle: Any, dest: str, job_config: Any) -> _FakeJob:
            dispositions.append(str(job_config.write_disposition))
            assert len(handle.read()) > 0
            return _FakeJob()

    monkeypatch.setattr(google.auth, "default", lambda: (object(), "demo-project"))
    monkeypatch.setattr(bq_mod, "Client", _FakeBQClient)

    stage_dir = tmp_path / "bq_stage"
    live_manifest = tmp_path / "bq_live_manifest.json"
    multi_part_df = spark.read.parquet(str(gold_dir / "daily_district_counts")).repartition(2)
    multi_part_df.write.mode("overwrite").parquet(str(gold_dir / "daily_district_counts"))
    orig_max_bytes = spark.conf.get("spark.sql.files.maxPartitionBytes")
    orig_open_cost = spark.conf.get("spark.sql.files.openCostInBytes")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "1")
    spark.conf.set("spark.sql.files.openCostInBytes", "1")
    try:
        publish_gold_and_silver_to_bigquery(
            spark,
            project_id="demo-project",
            dataset_id="chicago_crimes",
            gold_dir=gold_dir,
            tables=("daily_district_counts",),
            dry_run=False,
            staging_dir=stage_dir,
            output_manifest=live_manifest,
        )
    finally:
        spark.conf.set("spark.sql.files.maxPartitionBytes", orig_max_bytes)
        spark.conf.set("spark.sql.files.openCostInBytes", orig_open_cost)
    assert dispositions[0].endswith("WRITE_TRUNCATE")
    assert all(d.endswith("WRITE_APPEND") for d in dispositions[1:])
    assert len(dispositions) >= 2
    assert not (stage_dir / "daily_district_counts").exists()

    # Verify standalone dbt project parses cleanly with offline profile
    dbt_exe = shutil.which("dbt") or str(Path(sys.executable).parent / "dbt.exe")
    prof_dir = tmp_path / "dbt_prof"
    prof_dir.mkdir()
    shutil.copy(REPO_ROOT / "dbt" / "profiles.yml.example", prof_dir / "profiles.yml")
    res = subprocess.run(
        [
            dbt_exe,
            "parse",
            "--project-dir",
            str(REPO_ROOT / "dbt"),
            "--profiles-dir",
            str(prof_dir),
            "--no-send-anonymous-usage-stats",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"dbt parse failed:\n{res.stdout}\n{res.stderr}"

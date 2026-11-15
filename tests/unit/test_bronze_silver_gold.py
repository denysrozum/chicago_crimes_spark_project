"""Unit tests for Phase 2 Bronze, Silver, Quarantine, Accounting, and Gold pipelines."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from chicago_crimes.analytics.runner import run_analytical_tasks
from chicago_crimes.core.config import AnalyticsConfig
from chicago_crimes.core.schemas import BRONZE_METADATA_COLUMNS, RAW_CRIME_COLUMNS
from chicago_crimes.experiments import (
    run_imbalance_analysis,
    run_pca_analysis,
    run_sampling_analysis,
)
from chicago_crimes.ingestion.bronze import ingest_csv_to_bronze
from chicago_crimes.ingestion.csv_loader import load_raw_csv_for_bronze
from chicago_crimes.processing.gold import build_gold_tables
from chicago_crimes.processing.quality import SilverAccountingReport, validate_silver_dataframe
from chicago_crimes.processing.silver import run_silver_pipeline
from tests.unit.conftest import _sample_rows, write_semicolon_csv


def _write_raw_semicolon_lines(rows: list[list[str]], path: Path) -> Path:
    headers = list(RAW_CRIME_COLUMNS)
    lines = [";".join(headers), *[";".join(r) for r in rows]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_raw_row(
    *,
    cid: str = "1001",
    case_number: str = "HJ1001",
    date_str: str = "05/01/2024 02:30:00 PM",
    block: str = "001XX N STATE ST",
    iucr: str = "0820",
    primary_type: str = "THEFT",
    description: str = "OVER $500",
    loc_desc: str = "STREET",
    arrest: str = "false",
    domestic: str = "false",
    beat: str = "111",
    district: str = "1",
    ward: str = "42",
    ca: str = "32",
    fbi: str = "06",
    x: str = "1176352",
    y: str = "1900927",
    year: str = "2024",
    updated_on: str = "05/10/2024 12:00:00 PM",
    lat: str = "41.8835",
    lon: str = "-87.6278",
    location: str = "(41.8835, -87.6278)",
) -> list[str]:
    return [
        cid,
        case_number,
        date_str,
        block,
        iucr,
        primary_type,
        description,
        loc_desc,
        arrest,
        domestic,
        beat,
        district,
        ward,
        ca,
        fbi,
        x,
        y,
        year,
        updated_on,
        lat,
        lon,
        location,
    ]


def test_bronze_ingestion_preserves_strings_and_lineage(
    spark: SparkSession, tmp_path: Path
) -> None:
    csv_path = _write_raw_semicolon_lines(
        [
            _make_raw_row(cid="101", lat="41.88", lon="-87.62"),
            _make_raw_row(cid="NOT_AN_INT", lat="", lon="", ward="", ca=""),
            ["corrupt;unbalanced;only_three_cols"],
        ],
        tmp_path / "raw_mixed.csv",
    )
    bronze_dir = tmp_path / "bronze"
    result = ingest_csv_to_bronze(
        spark,
        csv_path=csv_path,
        bronze_dir=bronze_dir,
        source_id="unit-test-csv",
        batch_id="batch-001",
    )

    assert result.row_count == 3
    assert result.corrupt_count == 1
    assert result.source_id == "unit-test-csv"
    assert result.batch_id == "batch-001"

    bronze_df = spark.read.parquet(str(bronze_dir))
    for col_name in (*RAW_CRIME_COLUMNS, *BRONZE_METADATA_COLUMNS):
        assert col_name in bronze_df.columns
    for col_name in RAW_CRIME_COLUMNS:
        assert bronze_df.schema[col_name].dataType.typeName() == "string"

    assert bronze_df.filter(F.col("ID") == "NOT_AN_INT").count() == 1
    assert bronze_df.filter(F.col("corrupt_record").isNotNull()).count() == 1

    with pytest.raises(FileExistsError):
        ingest_csv_to_bronze(spark, csv_path=csv_path, bronze_dir=bronze_dir)


def test_silver_preserves_missing_geo_and_admin_without_centroids(
    spark: SparkSession, tmp_path: Path
) -> None:
    csv_path = _write_raw_semicolon_lines(
        [
            # 1: Completely valid record
            _make_raw_row(cid="2001", block="001XX N STATE ST"),
            # 2: Missing geo & admin & missing numeric block prefix ("UNKNOWN ST")
            _make_raw_row(
                cid="2002",
                block="UNKNOWN ST",
                district="",
                ward="",
                ca="",
                lat="",
                lon="",
                x="",
                y="",
                location="",
            ),
            # 3: Out-of-bounds coords + out-of-range sentinel admin codes (0 / 99)
            _make_raw_row(
                cid="2003",
                block="002XX S CLARK ST",
                district="99",
                ward="0",
                ca="0",
                lat="0.0",
                lon="0.0",
                x="0",
                y="0",
                location="(0.0, 0.0)",
            ),
            # 4: Partial geo (Latitude present, Longitude missing)
            _make_raw_row(
                cid="2004",
                lat="41.8835",
                lon="",
                location="",
            ),
        ],
        tmp_path / "null_geo_admin.csv",
    )
    bronze_dir = tmp_path / "bronze"
    ingest_csv_to_bronze(spark, csv_path=csv_path, bronze_dir=bronze_dir, source_id="csv-test")

    silver_dir = tmp_path / "silver"
    quarantine_dir = tmp_path / "quarantine"
    result = run_silver_pipeline(
        spark.read.parquet(str(bronze_dir)),
        silver_dir=silver_dir,
        quarantine_dir=quarantine_dir,
    )

    assert result.accounting.input_records == 4
    assert result.accounting.accepted_records == 4
    assert result.accounting.superseded_records == 0
    assert result.accounting.quarantined_records == 0
    assert result.accounting.missing_geo_records == 3
    assert result.accounting.out_of_bounds_geo_nullified == 1
    assert result.accounting.missing_admin_records == 2
    assert result.accounting.out_of_range_admin_nullified == 1
    assert result.accounting.is_balanced

    rows = {r["ID"]: r for r in result.silver_df.collect()}
    assert set(rows) == {2001, 2002, 2003, 2004}

    assert rows[2001]["has_valid_geo"] is True
    assert rows[2001]["geo_quality_flag"] == "valid"
    assert rows[2001]["geo_grid"] is not None
    assert rows[2001]["has_valid_admin"] is True
    assert rows[2001]["admin_quality_flag"] == "valid"

    assert rows[2002]["Latitude"] is None
    assert rows[2002]["Longitude"] is None
    assert rows[2002]["geo_grid"] is None
    assert rows[2002]["has_valid_geo"] is False
    assert rows[2002]["geo_quality_flag"] == "missing"
    assert rows[2002]["District"] is None
    assert rows[2002]["Ward"] is None
    assert rows[2002]["Community Area"] is None
    assert rows[2002]["has_valid_admin"] is False
    assert rows[2002]["admin_quality_flag"] == "partial_or_missing"
    assert rows[2002]["block_number"] is None  # No synthetic block imputation

    assert rows[2003]["Latitude"] is None
    assert rows[2003]["Longitude"] is None
    assert rows[2003]["X Coordinate"] is None
    assert rows[2003]["Y Coordinate"] is None
    assert rows[2003]["geo_quality_flag"] == "out_of_bounds_nullified"
    assert rows[2003]["District"] is None
    assert rows[2003]["Ward"] is None
    assert rows[2003]["Community Area"] is None
    assert rows[2003]["admin_quality_flag"] == "out_of_range_nullified"

    assert rows[2004]["Latitude"] is None
    assert rows[2004]["Longitude"] is None
    assert rows[2004]["geo_quality_flag"] == "partial_missing_nullified"

    assert (silver_dir / "_silver_accounting.json").is_file()


def test_silver_quarantine_reasons_and_accounting_invariant(
    spark: SparkSession, tmp_path: Path
) -> None:
    csv_path = _write_raw_semicolon_lines(
        [
            _make_raw_row(cid="3001"),  # valid
            _make_raw_row(cid="bad_id"),  # missing_or_invalid_id
            _make_raw_row(cid="-7"),  # missing_or_invalid_id
            _make_raw_row(cid="3002", date_str="not-a-date"),  # missing_or_invalid_date
            _make_raw_row(cid="3003", date_str="01/01/1998 10:00:00 AM"),  # out of range
            _make_raw_row(cid="3004", updated_on="not-an-update-ts"),  # invalid_updated_on
            _make_raw_row(cid="3005", primary_type=""),  # missing_primary_type
            _make_raw_row(cid="3006", lat="abc"),  # invalid_coordinate_format
            ["malformed_row_missing_delimiters"],  # corrupt_csv_row
        ],
        tmp_path / "quarantine_cases.csv",
    )
    bronze_df = load_raw_csv_for_bronze(spark, csv_path).withColumns(
        {
            "ingestion_timestamp": F.current_timestamp(),
            "source_id": F.lit("test-q"),
            "ingestion_batch_id": F.lit("b1"),
            "corrupt_record": F.col("_corrupt_record"),
        }
    )

    silver_dir = tmp_path / "silver_q"
    quarantine_dir = tmp_path / "quarantine_q"
    res = run_silver_pipeline(
        bronze_df,
        silver_dir=silver_dir,
        quarantine_dir=quarantine_dir,
    )

    acc = res.accounting
    assert acc.input_records == 9
    assert acc.accepted_records == 1
    assert acc.superseded_records == 0
    assert acc.quarantined_records == 8
    assert acc.accounted_records == acc.input_records
    assert acc.quarantine_reasons == {
        "corrupt_csv_row": 1,
        "date_out_of_valid_range": 1,
        "invalid_coordinate_format": 1,
        "invalid_updated_on": 1,
        "missing_or_invalid_date": 1,
        "missing_or_invalid_id": 2,
        "missing_primary_type": 1,
    }

    manifest_data = json.loads((silver_dir / "_silver_accounting.json").read_text(encoding="utf-8"))
    assert manifest_data["invariant_passed"] is True
    assert manifest_data["accounted_records"] == 9
    assert res.quarantined_df.count() == 8


def test_silver_accounting_and_validator_fail_fast() -> None:
    broken_acc = SilverAccountingReport(
        input_records=10,
        accepted_records=6,
        superseded_records=2,
        quarantined_records=1,
        quarantine_reasons={"corrupt_csv_row": 1},
        missing_geo_records=0,
        out_of_bounds_geo_nullified=0,
        missing_admin_records=0,
        out_of_range_admin_nullified=0,
    )
    assert not broken_acc.is_balanced
    with pytest.raises(ValueError, match="invariant violated"):
        broken_acc.verify_invariant()


def test_silver_deterministic_deduplication(spark: SparkSession, tmp_path: Path) -> None:
    csv_path = _write_raw_semicolon_lines(
        [
            # Middle version first
            _make_raw_row(
                cid="9001",
                description="MIDDLE_VER",
                arrest="false",
                updated_on="06/01/2024 12:00:00 PM",
            ),
            # Newest version (ISO timestamp!)
            _make_raw_row(
                cid="9001",
                description="NEWEST_VER",
                arrest="true",
                updated_on="2025-02-15T14:30:00.000",
            ),
            # Oldest version last
            _make_raw_row(
                cid="9001",
                description="OLDEST_VER",
                arrest="false",
                updated_on="01/01/2024 08:00:00 AM",
            ),
            _make_raw_row(cid="9002", description="UNIQUE_REC"),
        ],
        tmp_path / "dups.csv",
    )
    raw_df = load_raw_csv_for_bronze(spark, csv_path)
    superseded_dir = tmp_path / "superseded"
    res = run_silver_pipeline(
        raw_df,
        silver_dir=tmp_path / "silver",
        quarantine_dir=tmp_path / "quarantine",
        superseded_dir=superseded_dir,
    )

    assert res.accounting.input_records == 4
    assert res.accounting.accepted_records == 2
    assert res.accounting.superseded_records == 2
    assert res.accounting.quarantined_records == 0
    assert res.accounting.is_balanced

    accepted = {r["ID"]: r for r in res.silver_df.collect()}
    assert accepted[9001]["Description"] == "NEWEST_VER"
    assert accepted[9001]["Arrest"] is True
    assert (
        res.silver_df.filter(F.col("ID") == 9001)
        .select(F.date_format(F.col("Updated On"), "yyyy-MM-dd HH:mm:ss"))
        .first()[0]
        == "2025-02-15 14:30:00"
    )
    assert datetime.fromtimestamp(accepted[9001]["Updated On"].timestamp(), tz=UTC) == datetime(
        2025, 2, 15, 14, 30, 0, tzinfo=UTC
    )

    superseded = res.superseded_df.collect()
    assert len(superseded) == 2
    assert {r["Description"] for r in superseded} == {"OLDEST_VER", "MIDDLE_VER"}
    assert all(r["superseded_reason"] == "older_duplicate_id" for r in superseded)

    # Corrupting uniqueness manually triggers validate_silver_dataframe error
    dup_injected = res.silver_df.unionByName(res.silver_df.limit(1))
    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_silver_dataframe(dup_injected, require_unique_id=True)


def test_gold_aggregates_and_spatial_geo_filter(spark: SparkSession, tmp_path: Path) -> None:
    csv_path = _write_raw_semicolon_lines(
        [
            _make_raw_row(
                cid="4001",
                date_str="05/01/2024 01:00:00 AM",
                primary_type="THEFT",
                arrest="true",
                lat="41.8835",
                lon="-87.6278",
                district="1",
                ca="32",
            ),
            _make_raw_row(
                cid="4002",
                date_str="05/01/2024 02:00:00 PM",
                primary_type="BATTERY",
                arrest="false",
                lat="",
                lon="",
                location="",
                district="",
                ward="",
                ca="",
            ),
            _make_raw_row(
                cid="4003",
                date_str="06/15/2024 03:00:00 PM",
                primary_type="THEFT",
                arrest="false",
                lat="0.0",
                lon="0.0",
                district="1",
                ca="32",
            ),
        ],
        tmp_path / "gold_input.csv",
    )
    silver_res = run_silver_pipeline(load_raw_csv_for_bronze(spark, csv_path))
    gold_dir = tmp_path / "gold"
    gold_res = build_gold_tables(silver_res.silver_df, gold_dir=gold_dir)

    assert gold_res.source_silver_count == 3
    assert gold_res.valid_geo_count == 1
    assert set(gold_res.tables) == {
        "daily_citywide_counts",
        "monthly_crime_summary",
        "district_community_summary",
        "category_distribution",
        "temporal_hourly_profile",
        "spatial_valid_grid_summary",
        "daily_district_counts",
    }

    spatial_df = spark.read.parquet(gold_res.tables["spatial_valid_grid_summary"].path)
    assert spatial_df.agg(F.sum("incident_count")).collect()[0][0] == 1

    daily_df = spark.read.parquet(gold_res.tables["daily_citywide_counts"].path)
    assert daily_df.agg(F.sum("incident_count")).collect()[0][0] == 3
    # Verify year/month consistency with crime_date across UTC timezone sessions
    assert daily_df.filter(
        (F.year("crime_date") != F.col("year")) | (F.month("crime_date") != F.col("month"))
    ).count() == 0

    cat_df = spark.read.parquet(gold_res.tables["category_distribution"].path)
    assert cat_df.agg(F.sum("incident_count")).collect()[0][0] == 3

    manifest = json.loads(gold_res.manifest_path.read_text(encoding="utf-8"))
    assert manifest["valid_geo_count"] == 1
    assert manifest["excluded_invalid_or_missing_geo_count"] == 2
    assert "incident_counts_definition" in manifest["methodology"]

    with pytest.raises(FileExistsError):
        build_gold_tables(silver_res.silver_df, gold_dir=gold_dir)


def test_silver_compatible_with_analytics_and_experiments(
    spark: SparkSession, tmp_path: Path
) -> None:
    csv_path = write_semicolon_csv(_sample_rows(), tmp_path / "silver_sample.csv")
    raw_df = load_raw_csv_for_bronze(spark, csv_path)
    # Mix in one missing-geo & missing-admin incident to verify downstream null safety
    null_row = _write_raw_semicolon_lines(
        [
            _make_raw_row(
                cid="9999",
                date_str="05/01/2025 02:00:00 AM",
                primary_type="BATTERY",
                district="",
                ward="",
                ca="",
                lat="",
                lon="",
                location="",
            )
        ],
        tmp_path / "extra_null.csv",
    )
    combined = raw_df.unionByName(load_raw_csv_for_bronze(spark, null_row))
    silver_res = run_silver_pipeline(combined, silver_dir=tmp_path / "silver")

    cfg = AnalyticsConfig(
        min_district_crimes=2,
        community_area_id=40,
        arrest_share_year=2025,
        min_fbi_cases=2,
    )
    ana_res = run_analytical_tasks(
        silver_res.silver_df,
        output_dir=tmp_path / "analytics",
        tasks=["AQ-01", "AQ-07", "AQ-14", "AQ-20", "AQ-23"],
        config=cfg,
        render_plots=False,
    )
    assert len(ana_res) == 5

    pca_res = run_pca_analysis(
        silver_res.silver_df,
        out_dir=tmp_path / "exp_pca",
        n_components=4,
        sample_fraction=1.0,
        sample_max_rows=100,
    )
    assert "ID" not in pca_res["feature_names"]
    assert "has_valid_geo" not in pca_res["feature_names"]

    imb_res = run_imbalance_analysis(silver_res.silver_df, out_dir=tmp_path / "exp_imb")
    assert imb_res["analyzed_count"] > 0
    assert "Total rows: 25" in Path(imb_res["summary_path_txt"]).read_text(encoding="utf-8")

    from chicago_crimes.experiments.sampling import _prepare_modeling_df

    _, samp_features = _prepare_modeling_df(silver_res.silver_df, "Arrest")
    assert "ID" not in samp_features
    assert "has_valid_geo" not in samp_features

    samp_res = run_sampling_analysis(
        silver_res.silver_df,
        out_dir=tmp_path / "exp_samp",
        write_sampled_dataset=False,
    )
    assert Path(samp_res["Arrest"]["oversampling"]["report_path"]).is_file()
    assert Path(samp_res["Domestic"]["undersampling"]["pca_after_path"]).is_file()

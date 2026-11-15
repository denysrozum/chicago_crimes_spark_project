"""Great Expectations data quality suites for validated Silver and Gold tables."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.schemas import (
    CHICAGO_LAT_MAX,
    CHICAGO_LAT_MIN,
    CHICAGO_LON_MAX,
    CHICAGO_LON_MIN,
    COMMUNITY_AREA_MAX,
    COMMUNITY_AREA_MIN,
    DISTRICT_MAX,
    DISTRICT_MIN,
    WARD_MAX,
    WARD_MIN,
)
from chicago_crimes.storage.lakehouse import read_table

logger = get_logger("processing.ge_validation")

SILVER_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ID",
    "Date",
    "Primary Type",
    "Updated On",
    "Latitude",
    "Longitude",
    "District",
    "Ward",
    "Community Area",
    "has_valid_geo",
    "geo_quality_flag",
    "has_valid_admin",
    "admin_quality_flag",
)

GEO_QUALITY_FLAGS: tuple[str, ...] = (
    "valid",
    "missing",
    "partial_missing_nullified",
    "out_of_bounds_nullified",
    "invalid",
)

ADMIN_QUALITY_FLAGS: tuple[str, ...] = (
    "valid",
    "partial_or_missing",
    "out_of_range_nullified",
)


def _to_pandas(frame: DataFrame | pd.DataFrame, *, sample_limit: int | None = None) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.head(sample_limit) if sample_limit is not None else frame
    limited = frame.limit(sample_limit) if sample_limit is not None else frame
    return limited.toPandas()


def _build_silver_expectations() -> list[Any]:
    ex = gx.expectations
    expectations: list[Any] = [
        ex.ExpectColumnToExist(column=col) for col in SILVER_REQUIRED_COLUMNS
    ]
    expectations.extend(
        [
            ex.ExpectTableRowCountToBeBetween(min_value=1),
            ex.ExpectColumnValuesToNotBeNull(column="ID"),
            ex.ExpectColumnValuesToBeUnique(column="ID"),
            ex.ExpectColumnValuesToBeBetween(column="ID", min_value=1),
            ex.ExpectColumnValuesToNotBeNull(column="Date"),
            ex.ExpectColumnValuesToNotBeNull(column="Primary Type"),
            ex.ExpectColumnValuesToNotBeNull(column="has_valid_geo"),
            ex.ExpectColumnValuesToNotBeNull(column="has_valid_admin"),
            ex.ExpectColumnValuesToBeInSet(
                column="geo_quality_flag", value_set=list(GEO_QUALITY_FLAGS)
            ),
            ex.ExpectColumnValuesToBeInSet(
                column="admin_quality_flag", value_set=list(ADMIN_QUALITY_FLAGS)
            ),
            ex.ExpectColumnValuesToBeBetween(
                column="District", min_value=DISTRICT_MIN, max_value=DISTRICT_MAX
            ),
            ex.ExpectColumnValuesToBeBetween(column="Ward", min_value=WARD_MIN, max_value=WARD_MAX),
            ex.ExpectColumnValuesToBeBetween(
                column="Community Area",
                min_value=COMMUNITY_AREA_MIN,
                max_value=COMMUNITY_AREA_MAX,
            ),
            ex.ExpectColumnValuesToBeBetween(
                column="Latitude", min_value=CHICAGO_LAT_MIN, max_value=CHICAGO_LAT_MAX
            ),
            ex.ExpectColumnValuesToBeBetween(
                column="Longitude", min_value=CHICAGO_LON_MIN, max_value=CHICAGO_LON_MAX
            ),
        ]
    )
    return expectations


def _build_gold_expectations(table_name: str) -> list[Any]:
    ex = gx.expectations
    expectations: list[Any] = [
        ex.ExpectTableRowCountToBeBetween(min_value=1),
        ex.ExpectColumnValuesToNotBeNull(column="incident_count"),
        ex.ExpectColumnValuesToBeBetween(column="incident_count", min_value=1),
        ex.ExpectColumnValuesToBeBetween(column="arrest_count", min_value=0),
        ex.ExpectColumnValuesToBeBetween(
            column="arrest_rate_percent", min_value=0.0, max_value=100.0
        ),
    ]
    if table_name == "daily_citywide_counts":
        expectations.extend(
            [
                ex.ExpectColumnValuesToNotBeNull(column="crime_date"),
                ex.ExpectColumnValuesToBeUnique(column="crime_date"),
            ]
        )
    elif table_name == "daily_district_counts":
        expectations.extend(
            [
                ex.ExpectColumnValuesToNotBeNull(column="crime_date"),
                ex.ExpectColumnValuesToNotBeNull(column="District"),
                ex.ExpectColumnValuesToBeBetween(
                    column="District", min_value=DISTRICT_MIN, max_value=DISTRICT_MAX
                ),
            ]
        )
    elif table_name == "category_distribution":
        expectations.extend(
            [
                ex.ExpectColumnValuesToNotBeNull(column="Primary Type"),
                ex.ExpectColumnValuesToBeUnique(column="Primary Type"),
                ex.ExpectColumnValuesToBeBetween(
                    column="citywide_share_percent", min_value=0.0, max_value=100.0
                ),
            ]
        )
    return expectations


def evaluate_expectation_suite(
    pdf: pd.DataFrame,
    *,
    suite_name: str,
    expectations: list[Any],
) -> dict[str, Any]:
    """Run an ephemeral Great Expectations suite against a pandas DataFrame."""
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
    context = gx.get_context(mode="ephemeral")
    ds = context.data_sources.add_pandas(name=f"{suite_name}_source")
    asset = ds.add_dataframe_asset(name=f"{suite_name}_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("whole")

    suite = gx.ExpectationSuite(name=suite_name)
    for exp in expectations:
        suite.add_expectation(exp)
    context.suites.add(suite)

    validation_def = gx.ValidationDefinition(
        name=f"{suite_name}_validation",
        data=batch_def,
        suite=suite,
    )
    context.validation_definitions.add(validation_def)
    result = validation_def.run(batch_parameters={"dataframe": pdf})
    raw_dict = result.to_json_dict()

    details: list[dict[str, Any]] = []
    for item in raw_dict.get("results", []):
        exp_cfg = item.get("expectation_config", {})
        res = item.get("result", {})
        details.append(
            {
                "expectation_type": exp_cfg.get("type"),
                "column": exp_cfg.get("kwargs", {}).get("column"),
                "success": bool(item.get("success")),
                "unexpected_count": res.get("unexpected_count"),
                "unexpected_percent": res.get("unexpected_percent"),
            }
        )

    return {
        "suite_name": suite_name,
        "row_count": int(len(pdf)),
        "success": bool(result.success),
        "statistics": dict(raw_dict.get("statistics", {})),
        "failed_expectations": [d for d in details if not d["success"]],
        "expectations": details,
    }


def validate_silver_with_gx(
    silver_data: DataFrame | pd.DataFrame,
    *,
    sample_limit: int | None = 100_000,
    suite_name: str = "chicago_crimes_silver_suite",
    report_path: str | Path | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Execute Great Expectations validation suite on Silver crime records."""
    pdf = _to_pandas(silver_data, sample_limit=sample_limit)
    summary = evaluate_expectation_suite(
        pdf,
        suite_name=suite_name,
        expectations=_build_silver_expectations(),
    )
    if report_path is not None:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if raise_on_failure and not summary["success"]:
        raise ValueError(
            f"Great Expectations Silver suite failed: {summary['failed_expectations']}"
        )
    return summary


def validate_gold_with_gx(
    gold_source: str | Path | dict[str, DataFrame | pd.DataFrame],
    *,
    spark: SparkSession | None = None,
    table_names: tuple[str, ...] = (
        "daily_citywide_counts",
        "daily_district_counts",
        "category_distribution",
    ),
    report_path: str | Path | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Execute Great Expectations suites across materialized or in-memory Gold tables."""
    tables: dict[str, DataFrame | pd.DataFrame] = {}
    if isinstance(gold_source, dict):
        tables = gold_source
    else:
        if spark is None:
            raise ValueError("spark session is required when loading Gold tables from directory")
        base_dir = Path(gold_source)
        for name in table_names:
            table_dir = base_dir / name
            if table_dir.exists():
                tables[name] = spark.read.parquet(str(table_dir))

    if not tables:
        raise ValueError("No Gold tables found to validate.")

    table_results: dict[str, Any] = {}
    for name, frame in tables.items():
        if name not in table_names and not isinstance(gold_source, dict):
            continue
        pdf = _to_pandas(frame)
        table_results[name] = evaluate_expectation_suite(
            pdf,
            suite_name=f"gold_{name}_suite",
            expectations=_build_gold_expectations(name),
        )

    overall_success = all(item["success"] for item in table_results.values())
    payload = {
        "suite_name": "chicago_crimes_gold_suite",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "success": overall_success,
        "tables": table_results,
    }
    if report_path is not None:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if raise_on_failure and not overall_success:
        failed = {k: v["failed_expectations"] for k, v in table_results.items() if not v["success"]}
        raise ValueError(f"Great Expectations Gold suite failed: {failed}")
    return payload


def run_ge_validation_suite(
    spark: SparkSession,
    *,
    layer: str = "both",
    silver_input: str | None = None,
    gold_input: str | None = None,
    table_format: str = "auto",
    sample_limit: int = 100_000,
    output_path: str | Path = "artifacts/quality/ge_validation_report.json",
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Run Silver and/or Gold Great Expectations suites and save a combined JSON report."""
    report: dict[str, Any] = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "layer": layer,
        "sample_limit": sample_limit,
        "results": {},
    }
    overall_ok = True

    if layer in {"silver", "both"}:
        if not silver_input:
            raise ValueError("silver_input is required when layer includes 'silver'")
        silver_df = read_table(spark, silver_input, table_format=table_format)
        silver_res = validate_silver_with_gx(
            silver_df,
            sample_limit=sample_limit,
            raise_on_failure=False,
        )
        report["results"]["silver"] = silver_res
        overall_ok = overall_ok and silver_res["success"]

    if layer in {"gold", "both"}:
        if not gold_input:
            raise ValueError("gold_input is required when layer includes 'gold'")
        gold_res = validate_gold_with_gx(
            gold_input,
            spark=spark,
            raise_on_failure=False,
        )
        report["results"]["gold"] = gold_res
        overall_ok = overall_ok and gold_res["success"]

    report["success"] = overall_ok
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Great Expectations validation report saved to %s (success=%s)", out, overall_ok)

    if raise_on_failure and not overall_ok:
        raise ValueError(f"Great Expectations validation failed; inspect {out}")
    return report

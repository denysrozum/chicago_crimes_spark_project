"""Optional Google Cloud BigQuery batch publishing adapter and schema mapper."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegralType,
    TimestampType,
)

from chicago_crimes.core.logging import get_logger
from chicago_crimes.storage.lakehouse import read_table

logger = get_logger("storage.bigquery")

DEFAULT_GOLD_PUBLISH_TABLES: tuple[str, ...] = (
    "daily_citywide_counts",
    "daily_district_counts",
    "category_distribution",
    "monthly_crime_summary",
    "district_community_summary",
)

PARTITION_CLUSTER_HINTS: dict[str, dict[str, Any]] = {
    "daily_citywide_counts": {"partition_field": "crime_date", "clustering": ["year", "month"]},
    "daily_district_counts": {"partition_field": "crime_date", "clustering": ["district"]},
    "category_distribution": {"partition_field": None, "clustering": ["primary_type"]},
    "monthly_crime_summary": {"partition_field": None, "clustering": ["year", "month"]},
    "district_community_summary": {
        "partition_field": None,
        "clustering": ["district", "community_area"],
    },
    "silver_crimes": {"partition_field": "date", "clustering": ["district", "primary_type"]},
}


def sanitize_bq_column_name(name: str) -> str:
    """Normalize Spark/Gold column names to valid BigQuery identifiers."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned[:300]


def spark_type_to_bq_type(dtype: Any) -> str:
    """Map PySpark SQL DataType to BigQuery Standard SQL column type."""
    if isinstance(dtype, BooleanType):
        return "BOOL"
    if isinstance(dtype, IntegralType):
        return "INT64"
    if isinstance(dtype, DoubleType | FloatType):
        return "FLOAT64"
    if isinstance(dtype, DateType):
        return "DATE"
    if isinstance(dtype, TimestampType):
        return "TIMESTAMP"
    return "STRING"


def build_bq_schema_spec(df: DataFrame) -> list[dict[str, str]]:
    """Build deterministic BigQuery schema metadata from a Spark DataFrame."""
    return [
        {
            "name": sanitize_bq_column_name(field.name),
            "source_column": field.name,
            "type": spark_type_to_bq_type(field.dataType),
            "mode": "NULLABLE" if field.nullable else "REQUIRED",
        }
        for field in df.schema.fields
    ]


def build_bq_ddl(
    *,
    project_id: str,
    dataset_id: str,
    table_name: str,
    schema_spec: list[dict[str, str]],
) -> str:
    """Render reproducible BigQuery CREATE TABLE DDL for documentation/dry-run review."""
    col_lines = ",\n  ".join(f"`{col['name']}` {col['type']}" for col in schema_spec)
    hints = PARTITION_CLUSTER_HINTS.get(table_name, {})
    part_field = hints.get("partition_field")
    cluster_fields = [
        f
        for f in hints.get("clustering", [])
        if any(col["name"] == f for col in schema_spec)
    ]

    clauses = [
        f"CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{table_name}` (\n"
        f"  {col_lines}\n)"
    ]
    if part_field and any(col["name"] == part_field for col in schema_spec):
        p_type = next(col["type"] for col in schema_spec if col["name"] == part_field)
        if p_type == "DATE":
            clauses.append(f"PARTITION BY `{part_field}`")
        elif p_type == "TIMESTAMP":
            clauses.append(f"PARTITION BY DATE(`{part_field}`)")
    if cluster_fields:
        clauses.append(f"CLUSTER BY {', '.join(f'`{c}`' for c in cluster_fields)}")
    return "\n".join(clauses) + ";"


@dataclass(frozen=True)
class BigQueryPublishEntry:
    """Metadata and execution status for a single table planned or loaded to BigQuery."""

    table_name: str
    destination: str
    row_count: int
    schema: list[dict[str, str]]
    ddl: str
    status: str


def _prepare_sanitized_dataframe(df: DataFrame) -> DataFrame:
    renamed = df
    for col in df.columns:
        safe = sanitize_bq_column_name(col)
        if safe != col:
            renamed = renamed.withColumnRenamed(col, safe)
    return renamed


def publish_gold_and_silver_to_bigquery(
    spark: SparkSession,
    *,
    project_id: str,
    dataset_id: str,
    gold_dir: str | Path | None = "data/gold",
    silver_input: str | Path | None = None,
    table_format: str = "auto",
    tables: tuple[str, ...] = DEFAULT_GOLD_PUBLISH_TABLES,
    location: str = "US",
    dry_run: bool = True,
    output_manifest: str | Path = "artifacts/bigquery/publish_manifest.json",
    staging_dir: str | Path = "artifacts/bigquery/staging",
) -> dict[str, Any]:
    """Plan (`dry_run=True`) or execute Gold/Silver batch upload to Google BigQuery."""
    if not project_id.strip() or not dataset_id.strip():
        raise ValueError("Both project_id and dataset_id are required for BigQuery publishing.")

    sources: dict[str, DataFrame] = {}
    if gold_dir is not None:
        base = Path(gold_dir)
        for name in tables:
            tbl_path = base / name
            if tbl_path.exists():
                sources[name] = spark.read.parquet(str(tbl_path))

    if silver_input is not None:
        sources["silver_crimes"] = read_table(spark, silver_input, table_format=table_format)

    if not sources:
        raise FileNotFoundError("No Gold or Silver source tables found for BigQuery publishing.")

    entries: list[BigQueryPublishEntry] = []
    client = None
    bigquery_mod = None

    if not dry_run:
        try:
            import google.auth
            from google.cloud import bigquery as bq_module
        except ImportError as exc:
            raise RuntimeError(
                "BigQuery publishing requires optional `cloud` dependencies "
                "(`uv sync --extra cloud`)."
            ) from exc

        try:
            _, detected_project = google.auth.default()
        except Exception as exc:
            raise RuntimeError(
                "Google Cloud Application Default Credentials (ADC) are not configured "
                "or unavailable. Use --dry-run to inspect BigQuery schemas and DDL offline."
            ) from exc

        bigquery_mod = bq_module
        client = bq_module.Client(project=project_id or detected_project, location=location)
        dataset_ref = bq_module.Dataset(f"{project_id}.{dataset_id}")
        dataset_ref.location = location
        client.create_dataset(dataset_ref, exists_ok=True)

    staging_root = Path(staging_dir)

    for table_name, df in sources.items():
        schema_spec = build_bq_schema_spec(df)
        ddl = build_bq_ddl(
            project_id=project_id,
            dataset_id=dataset_id,
            table_name=table_name,
            schema_spec=schema_spec,
        )
        row_count = int(df.count())
        destination = f"{project_id}.{dataset_id}.{table_name}"

        if dry_run or client is None or bigquery_mod is None:
            entries.append(
                BigQueryPublishEntry(
                    table_name=table_name,
                    destination=destination,
                    row_count=row_count,
                    schema=schema_spec,
                    ddl=ddl,
                    status="dry_run_planned",
                )
            )
            continue

        # Materialize sanitized Parquet files locally and load cleanly via LoadJobConfig
        sanitized = _prepare_sanitized_dataframe(df)
        table_stage = staging_root / table_name
        try:
            sanitized.write.mode("overwrite").parquet(str(table_stage))
            parquet_files = sorted(table_stage.glob("*.parquet"))
            bq_schema = [
                bigquery_mod.SchemaField(col["name"], col["type"], mode=col["mode"])
                for col in schema_spec
            ]
            job_config = bigquery_mod.LoadJobConfig(
                source_format=bigquery_mod.SourceFormat.PARQUET,
                write_disposition=bigquery_mod.WriteDisposition.WRITE_TRUNCATE,
                schema=bq_schema,
            )
            hints = PARTITION_CLUSTER_HINTS.get(table_name, {})
            part_field = hints.get("partition_field")
            if part_field and any(col["name"] == part_field for col in schema_spec):
                job_config.time_partitioning = bigquery_mod.TimePartitioning(
                    type_=bigquery_mod.TimePartitioningType.DAY,
                    field=part_field,
                )
            cluster_fields = [
                f
                for f in hints.get("clustering", [])
                if any(col["name"] == f for col in schema_spec)
            ]
            if cluster_fields:
                job_config.clustering_fields = cluster_fields

            for idx, pq_file in enumerate(parquet_files):
                if idx > 0:
                    job_config.write_disposition = bigquery_mod.WriteDisposition.WRITE_APPEND
                with pq_file.open("rb") as handle:
                    load_job = client.load_table_from_file(
                        handle,
                        destination,
                        job_config=job_config,
                    )
                    load_job.result()
        finally:
            shutil.rmtree(table_stage, ignore_errors=True)

        entries.append(
            BigQueryPublishEntry(
                table_name=table_name,
                destination=destination,
                row_count=row_count,
                schema=schema_spec,
                ddl=ddl,
                status="loaded",
            )
        )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_id": project_id,
        "dataset_id": dataset_id,
        "location": location,
        "dry_run": dry_run,
        "table_count": len(entries),
        "tables": [
            {
                "table_name": e.table_name,
                "destination": e.destination,
                "row_count": e.row_count,
                "status": e.status,
                "schema": e.schema,
                "ddl": e.ddl,
            }
            for e in entries
        ],
    }
    manifest_path = Path(output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "BigQuery publish (%s) wrote manifest to %s (%s tables)",
        "dry_run" if dry_run else "live",
        manifest_path,
        len(entries),
    )
    return manifest

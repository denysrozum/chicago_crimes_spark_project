"""Unit tests for Phase 3 Delta Lake, MinIO config/URI helpers, and Spark benchmarks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from chicago_crimes.analytics.benchmark import (
    inspect_spark_master_cluster,
    run_distributed_benchmark,
)
from chicago_crimes.cli import main
from chicago_crimes.core.config import AppConfig, SparkConfig, StorageConfig, load_config
from chicago_crimes.core.schemas import RAW_CRIME_SCHEMA
from chicago_crimes.ingestion.csv_loader import load_raw_csv_for_bronze
from chicago_crimes.processing.silver import run_silver_pipeline
from chicago_crimes.storage.delta import (
    delta_history,
    is_delta_table,
    merge_silver_delta,
    read_delta,
    write_delta,
)
from chicago_crimes.storage.lakehouse import read_table, sync_to_delta, write_table
from chicago_crimes.storage.minio import build_s3_client, s3a_uri
from tests.unit.conftest import _sample_rows, write_semicolon_csv


def test_storage_config_and_minio_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert s3a_uri("chicago-crimes", "silver/crimes") == "s3a://chicago-crimes/silver/crimes"
    assert s3a_uri("/chicago-crimes/", "/gold/") == "s3a://chicago-crimes/gold/"

    monkeypatch.setenv("MINIO_ENDPOINT", "http://127.0.0.1:9999")
    monkeypatch.setenv("MINIO_BUCKET", "test-bucket")
    cfg = load_config(profile="standalone")
    assert cfg.spark.profile == "standalone"
    assert cfg.spark.enable_delta is True
    assert cfg.storage.backend == "s3a"
    assert cfg.storage.s3_endpoint == "http://127.0.0.1:9999"
    assert cfg.storage.s3_bucket == "test-bucket"

    client = build_s3_client(
        StorageConfig(
            backend="s3a",
            s3_endpoint="http://127.0.0.1:9999",
            s3_access_key="u",
            s3_secret_key="p",
        )
    )
    assert client.meta.endpoint_url == "http://127.0.0.1:9999"


def test_delta_write_time_travel_and_acid_merge(spark: SparkSession, tmp_path: Path) -> None:
    df_v0 = spark.createDataFrame(
        [
            (
                100,
                "HJ100",
                datetime(2024, 1, 1, 10, 0, 0),
                "001XX N STATE ST",
                "0820",
                "THEFT",
                "INITIAL",
                "STREET",
                False,
                False,
                111,
                1,
                42,
                32,
                "06",
                1176352,
                1900927,
                2024,
                datetime(2024, 1, 5, 12, 0, 0),
                41.8835,
                -87.6278,
                "(41.8835, -87.6278)",
            ),
            (
                101,
                "HJ101",
                datetime(2024, 2, 1, 10, 0, 0),
                "002XX S CLARK ST",
                "0460",
                "BATTERY",
                "SIMPLE",
                "SIDEWALK",
                True,
                False,
                112,
                1,
                42,
                32,
                "08B",
                1176100,
                1900500,
                2024,
                datetime(2024, 2, 5, 12, 0, 0),
                41.881,
                -87.63,
                "(41.881, -87.63)",
            ),
        ],
        schema=RAW_CRIME_SCHEMA,
    )

    delta_path = tmp_path / "silver_delta"
    assert not is_delta_table(spark, delta_path)

    first_res = merge_silver_delta(spark, df_v0, delta_path)
    assert first_res["created"] is True
    assert is_delta_table(spark, delta_path)
    assert read_delta(spark, delta_path).count() == 2

    # Second batch: updates ID=100 (newer), keeps ID=101 unchanged, inserts ID=102,
    # and carries an in-batch duplicate for ID=102 where newer must win cleanly.
    df_v1 = spark.createDataFrame(
        [
            (
                100,
                "HJ100",
                datetime(2024, 1, 1, 10, 0, 0),
                "001XX N STATE ST",
                "0820",
                "THEFT",
                "UPDATED_V1",
                "STREET",
                True,
                False,
                111,
                1,
                42,
                32,
                "06",
                1176352,
                1900927,
                2024,
                datetime(2025, 1, 10, 12, 0, 0),
                41.8835,
                -87.6278,
                "(41.8835, -87.6278)",
            ),
            (
                102,
                "HJ102",
                datetime(2025, 1, 2, 8, 0, 0),
                "003XX W MADISON ST",
                "0460",
                "BATTERY",
                "STALE_WITHIN_BATCH",
                "STREET",
                False,
                False,
                113,
                1,
                42,
                32,
                "08B",
                1175900,
                1900400,
                2025,
                datetime(2025, 1, 3, 12, 0, 0),
                41.882,
                -87.631,
                "(41.882, -87.631)",
            ),
            (
                102,
                "HJ102",
                datetime(2025, 1, 2, 8, 0, 0),
                "003XX W MADISON ST",
                "0460",
                "BATTERY",
                "NEWEST_WITHIN_BATCH",
                "STREET",
                True,
                False,
                113,
                1,
                42,
                32,
                "08B",
                1175900,
                1900400,
                2025,
                datetime(2025, 1, 4, 12, 0, 0),
                41.882,
                -87.631,
                "(41.882, -87.631)",
            ),
        ],
        schema=RAW_CRIME_SCHEMA,
    )
    second_res = merge_silver_delta(spark, df_v1, delta_path)
    assert second_res["created"] is False
    assert second_res["version"] == 1

    # Third batch: out-of-order stale update for ID=100 MUST NOT overwrite newer record
    df_stale = df_v0.filter(F.col("ID") == 100).withColumn("Description", F.lit("STALE_REPLAY"))
    merge_silver_delta(spark, df_stale, delta_path)

    latest_rows = {r["ID"]: r for r in read_table(spark, delta_path).collect()}
    assert set(latest_rows) == {100, 101, 102}
    assert latest_rows[100]["Description"] == "UPDATED_V1"
    assert latest_rows[102]["Description"] == "NEWEST_WITHIN_BATCH"

    # Time travel back to version 0 still shows INITIAL state
    v0_rows = {r["ID"]: r for r in read_table(spark, delta_path, version_as_of=0).collect()}
    assert set(v0_rows) == {100, 101}
    assert v0_rows[100]["Description"] == "INITIAL"

    hist = delta_history(spark, delta_path, limit=5)
    assert len(hist) == 3
    assert hist[0]["version"] == 2

    parquet_dir = tmp_path / "plain_parquet"
    write_table(df_v0, parquet_dir, table_format="parquet")
    with pytest.raises(ValueError, match="only supported for Delta"):
        read_table(spark, parquet_dir, version_as_of=0)


def test_sync_delta_and_benchmark_cli(spark: SparkSession, tmp_path: Path) -> None:
    csv_path = write_semicolon_csv(_sample_rows(), tmp_path / "sample.csv")
    raw_df = load_raw_csv_for_bronze(spark, csv_path)
    silver_dir = tmp_path / "silver"
    run_silver_pipeline(raw_df, silver_dir=silver_dir)

    delta_dir = tmp_path / "silver_delta"
    sync_res = sync_to_delta(spark, silver_dir, delta_dir, partition_by=("year",))
    assert sync_res["row_count"] == 24
    assert is_delta_table(spark, delta_dir)

    # CLI analyze directly on Delta table + benchmark execution
    ana_out = tmp_path / "ana_delta"
    assert (
        main(
            [
                "--profile",
                "local",
                "analyze",
                "--input",
                str(delta_dir),
                "--format",
                "delta",
                "--tasks",
                "AQ-01",
                "--no-plots",
                "--output",
                str(ana_out),
            ]
        )
        == 0
    )
    assert (ana_out / "AQ-01_night_crimes_by_type" / "result.csv").is_file()

    cfg = AppConfig(
        spark=SparkConfig(
            profile="local",
            master="local[1]",
            driver_memory="1g",
            shuffle_partitions=1,
            enable_adaptive_execution=False,
            enable_delta=True,
            ui_enabled=False,
        )
    )
    bench_json = tmp_path / "bench" / "report.json"
    report = run_distributed_benchmark(
        cfg,
        input_path=delta_dir,
        output_json=bench_json,
        partition_counts=(2, 4),
        write_delta_output=True,
    )
    assert report["input_rows"] == 24
    assert len(report["runs"]) == 2
    assert sum(report["runs"][0]["executor_distribution"].values()) == 24
    assert json.loads(bench_json.read_text(encoding="utf-8"))["input_rows"] == 24

    # Unreachable master UI degrades gracefully to None
    assert inspect_spark_master_cluster("http://127.0.0.1:19999") is None
    _ = write_delta

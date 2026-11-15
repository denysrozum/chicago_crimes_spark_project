"""Pytest fixtures for end-to-end tests."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from chicago_crimes.core.config import SparkConfig
from chicago_crimes.core.spark import build_spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory: pytest.TempPathFactory) -> SparkSession:
    """Session-scoped local SparkSession for E2E verification."""
    existing = SparkSession.getActiveSession()
    if existing is not None:
        yield existing
        return

    spark_tmp = tmp_path_factory.mktemp("spark_e2e")
    cfg = SparkConfig(
        app_name="ChicagoCrimes-E2ETests",
        profile="local",
        master="local[1]",
        driver_memory="1g",
        executor_memory="1g",
        shuffle_partitions=1,
        default_parallelism=1,
        enable_adaptive_execution=False,
        enable_delta=True,
        ui_enabled=False,
        local_dir=str(spark_tmp),
        session_timezone="UTC",
        log_level="ERROR",
    )
    session = build_spark(cfg)
    yield session
    if SparkSession.getActiveSession() is not None:
        session.stop()

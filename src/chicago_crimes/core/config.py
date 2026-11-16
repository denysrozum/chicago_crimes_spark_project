"""Centralized configuration loading via YAML and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SparkConfig(BaseModel):
    app_name: str = "ChicagoCrimesAnalytics"
    profile: Literal["local", "standalone"] = "local"
    master: str = "local[4]"
    driver_host: str | None = "127.0.0.1"
    driver_bind_address: str | None = "127.0.0.1"
    driver_memory: str = "4g"
    executor_memory: str = "2g"
    executor_cores: int = Field(default=2, ge=1)
    cores_max: int = Field(default=4, ge=1)
    shuffle_partitions: int = Field(default=64, ge=1)
    default_parallelism: int = Field(default=4, ge=1)
    local_dir: str | None = None
    log_level: str = "WARN"
    session_timezone: str = "UTC"
    enable_adaptive_execution: bool = True
    enable_delta: bool = False
    enable_kafka: bool = False
    extra_jars_dir: str | None = None
    ui_enabled: bool = True
    windows_bare_fs_jar: str | None = None


class StorageConfig(BaseModel):
    backend: Literal["local", "s3a"] = "local"
    table_format: Literal["parquet", "delta"] = "parquet"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "chicago-crimes"
    s3_ssl_enabled: bool = False


class PathsConfig(BaseModel):
    raw_csv: str = "data/chicago_crimes.csv"
    bronze_dir: str = "data/bronze/chicago_crimes"
    silver_dir: str = "data/silver/chicago_crimes"
    gold_dir: str = "data/gold"
    quarantine_dir: str = "data/quarantine/chicago_crimes"
    legacy_clean_dir: str = "data/processed/chicago_crimes_clean"
    analytics_output_dir: str = "artifacts/analytics"
    stats_output_dir: str = "artifacts/stats"
    experiments_output_dir: str = "artifacts/experiments"
    ml_output_dir: str = "artifacts/ml"


class AnalyticsConfig(BaseModel):
    min_district_crimes: int = Field(default=1000, ge=0)
    community_area_id: int = Field(default=40, ge=1, le=77)
    arrest_share_year: int = Field(default=2025, ge=2001)
    min_fbi_cases: int = Field(default=100, ge=0)


class ExperimentsConfig(BaseModel):
    seed: int = 42
    pca_components: int = Field(default=10, ge=2)
    pca_sample_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    pca_sample_max_rows: int = Field(default=50000, ge=100)
    famd_components: int = Field(default=6, ge=2)
    famd_sample_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    famd_sample_max_rows: int = Field(default=20000, ge=50, le=100000)
    sampling_pca_max_rows: int = Field(default=50000, ge=100)


class StreamingConfig(BaseModel):
    soda_endpoint: str = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    soda_app_token: str | None = None
    soda_page_size: int = Field(default=1000, ge=1, le=50000)
    soda_max_pages_per_run: int = Field(default=10, ge=1, le=500)
    soda_overlap_seconds: int = Field(default=60, ge=0)
    watermark_path: str = "artifacts/state/soda_watermark.json"
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_topic: str = "chicago-crimes.raw"
    kafka_dlq_topic: str = "chicago-crimes.dlq"
    starting_offsets: str = "earliest"
    trigger_interval_seconds: int = Field(default=10, ge=1)
    checkpoint_location: str = "artifacts/checkpoints/crime_stream"
    bronze_stream_target: str = "data/streaming/bronze_crimes"
    silver_stream_target: str = "data/streaming/silver_crimes"
    quarantine_stream_target: str = "data/streaming/quarantine_crimes"
    dlq_stream_target: str = "data/streaming/dlq_crimes"
    metrics_log_path: str = "artifacts/streaming/microbatch_metrics.jsonl"


class AppConfig(BaseModel):
    spark: SparkConfig = Field(default_factory=SparkConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path | None = None,
    profile: Literal["local", "standalone"] | None = None,
) -> AppConfig:
    """Load configuration from YAML, apply profile/env overrides, and validate."""
    data: dict[str, Any] = {}
    default_path = Path("configs/default.yaml")
    if default_path.is_file():
        with default_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    env_profile = os.getenv("CHICAGO_CRIMES_PROFILE")
    effective_profile = profile or (env_profile if env_profile in {"local", "standalone"} else None)
    if effective_profile == "standalone":
        standalone_path = Path("configs/standalone.yaml")
        if standalone_path.is_file():
            with standalone_path.open("r", encoding="utf-8") as fh:
                standalone_override = yaml.safe_load(fh) or {}
            data = _deep_merge(data, standalone_override)

    if config_path is not None:
        custom_path = Path(config_path)
        if not custom_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {custom_path}")
        with custom_path.open("r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        data = _deep_merge(data, override)

    spark_section = data.setdefault("spark", {})
    storage_section = data.setdefault("storage", {})
    streaming_section = data.setdefault("streaming", {})

    env_master = os.getenv("CHICAGO_CRIMES_SPARK_MASTER")
    env_driver_mem = os.getenv("CHICAGO_CRIMES_SPARK_DRIVER_MEMORY")
    env_exec_mem = os.getenv("CHICAGO_CRIMES_SPARK_EXECUTOR_MEMORY")
    env_driver_host = os.getenv("CHICAGO_CRIMES_SPARK_DRIVER_HOST")
    env_driver_bind = os.getenv("CHICAGO_CRIMES_SPARK_DRIVER_BIND_ADDRESS")
    env_cores_max = os.getenv("CHICAGO_CRIMES_SPARK_CORES_MAX")
    env_exec_cores = os.getenv("CHICAGO_CRIMES_SPARK_EXECUTOR_CORES")
    env_delta = os.getenv("CHICAGO_CRIMES_SPARK_ENABLE_DELTA")
    env_kafka = os.getenv("CHICAGO_CRIMES_SPARK_ENABLE_KAFKA")
    env_extra_jars = os.getenv("SPARK_EXTRA_JARS_DIR")

    if env_master:
        spark_section["master"] = env_master
    if env_driver_mem:
        spark_section["driver_memory"] = env_driver_mem
    if env_exec_mem:
        spark_section["executor_memory"] = env_exec_mem
    if env_driver_host:
        spark_section["driver_host"] = env_driver_host
    if env_driver_bind:
        spark_section["driver_bind_address"] = env_driver_bind
    if env_cores_max:
        spark_section["cores_max"] = int(env_cores_max)
    if env_exec_cores:
        spark_section["executor_cores"] = int(env_exec_cores)
    if env_delta is not None:
        spark_section["enable_delta"] = env_delta.strip().lower() in {"1", "true", "yes"}
    if env_kafka is not None:
        spark_section["enable_kafka"] = env_kafka.strip().lower() in {"1", "true", "yes"}
    if env_extra_jars:
        spark_section["extra_jars_dir"] = env_extra_jars

    if effective_profile is not None:
        spark_section["profile"] = effective_profile

    if spark_section.get("profile") == "standalone" and spark_section.get("master", "").startswith(
        "local"
    ):
        spark_section["master"] = os.getenv("SPARK_MASTER_URL", "spark://localhost:7077")
    elif spark_section.get("profile") == "local" and not env_master:
        spark_section["master"] = spark_section.get("master", "local[4]")

    if os.getenv("MINIO_ENDPOINT"):
        storage_section["s3_endpoint"] = os.getenv("MINIO_ENDPOINT")
    if os.getenv("MINIO_ROOT_USER"):
        storage_section["s3_access_key"] = os.getenv("MINIO_ROOT_USER")
    if os.getenv("MINIO_ROOT_PASSWORD"):
        storage_section["s3_secret_key"] = os.getenv("MINIO_ROOT_PASSWORD")
    if os.getenv("MINIO_BUCKET"):
        storage_section["s3_bucket"] = os.getenv("MINIO_BUCKET")
    if os.getenv("CHICAGO_CRIMES_STORAGE_BACKEND") in {"local", "s3a"}:
        storage_section["backend"] = os.getenv("CHICAGO_CRIMES_STORAGE_BACKEND")
    if os.getenv("CHICAGO_CRIMES_TABLE_FORMAT") in {"parquet", "delta"}:
        storage_section["table_format"] = os.getenv("CHICAGO_CRIMES_TABLE_FORMAT")

    if os.getenv("SOCRATA_APP_TOKEN"):
        streaming_section["soda_app_token"] = os.getenv("SOCRATA_APP_TOKEN")
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        streaming_section["kafka_bootstrap_servers"] = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if os.getenv("KAFKA_CRIME_TOPIC"):
        streaming_section["kafka_topic"] = os.getenv("KAFKA_CRIME_TOPIC")
    if os.getenv("KAFKA_DLQ_TOPIC"):
        streaming_section["kafka_dlq_topic"] = os.getenv("KAFKA_DLQ_TOPIC")
    if os.getenv("CHICAGO_CRIMES_STREAM_CHECKPOINT"):
        streaming_section["checkpoint_location"] = os.getenv("CHICAGO_CRIMES_STREAM_CHECKPOINT")

    return AppConfig.model_validate(data)

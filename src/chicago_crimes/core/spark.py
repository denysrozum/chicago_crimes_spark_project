"""Centralized SparkSession factory and lifecycle management."""

from __future__ import annotations

import os
import platform
import socket
import sys
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

from chicago_crimes.core.config import AppConfig, SparkConfig, StorageConfig
from chicago_crimes.core.logging import get_logger

logger = get_logger("spark")

_BARE_FS_ARTIFACT = "hadoop-bare-naked-local-fs-0.1.0.jar"
_BARE_FS_MAVEN_URL = (
    "https://repo1.maven.org/maven2/com/globalmentor/"
    "hadoop-bare-naked-local-fs/0.1.0/hadoop-bare-naked-local-fs-0.1.0.jar"
)

_DELTA_JARS: dict[str, str] = {
    "delta-spark_2.12-3.2.0.jar": (
        "https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.0/"
        "delta-spark_2.12-3.2.0.jar"
    ),
    "delta-storage-3.2.0.jar": (
        "https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar"
    ),
}

_S3A_JARS: dict[str, str] = {
    "hadoop-aws-3.3.4.jar": (
        "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar"
    ),
    "aws-java-sdk-bundle-1.12.262.jar": (
        "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/"
        "aws-java-sdk-bundle-1.12.262.jar"
    ),
}

_KAFKA_JARS: dict[str, str] = {
    "spark-sql-kafka-0-10_2.12-3.5.1.jar": (
        "https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/"
        "spark-sql-kafka-0-10_2.12-3.5.1.jar"
    ),
    "spark-token-provider-kafka-0-10_2.12-3.5.1.jar": (
        "https://repo1.maven.org/maven2/org/apache/spark/"
        "spark-token-provider-kafka-0-10_2.12/3.5.1/"
        "spark-token-provider-kafka-0-10_2.12-3.5.1.jar"
    ),
    "kafka-clients-3.4.1.jar": (
        "https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/"
        "kafka-clients-3.4.1.jar"
    ),
    "commons-pool2-2.11.1.jar": (
        "https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/"
        "commons-pool2-2.11.1.jar"
    ),
}


def _download_or_find_jar(
    filename: str,
    maven_url: str,
    search_dirs: list[Path],
    cache_dir: Path,
) -> Path:
    """Locate a required JAR locally or download once into `cache_dir`."""
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if not target.is_file():
        logger.info("Downloading %s from %s", filename, maven_url)
        urllib.request.urlretrieve(maven_url, target)
    return target.resolve()


def _pyspark_system_jars_dir() -> Path:
    return Path(pyspark.__file__).resolve().parent / "jars"


def _resolve_extra_jars(
    jar_specs: dict[str, str],
    extra_jars_dir: str | None,
) -> list[str]:
    """Return local JAR paths not already present in Spark's bundled `jars/` directory."""
    sys_jars = _pyspark_system_jars_dir()
    search_dirs: list[Path] = []
    if extra_jars_dir:
        search_dirs.append(Path(extra_jars_dir))
    if os.getenv("SPARK_EXTRA_JARS_DIR"):
        search_dirs.append(Path(os.environ["SPARK_EXTRA_JARS_DIR"]))
    search_dirs.extend([Path("/opt/spark-extra-jars"), Path(".cache") / "jars"])
    cache_dir = Path(".cache") / "jars"

    resolved: list[str] = []
    for filename, url in jar_specs.items():
        if (sys_jars / filename).is_file():
            continue
        jar_path = _download_or_find_jar(filename, url, search_dirs, cache_dir)
        resolved.append(str(jar_path))
    return resolved


def _resolve_windows_bare_fs_jar(configured_path: str | None) -> str | None:
    """Locate or download pure-Java BareLocalFileSystem jar when HADOOP_HOME is missing."""
    if platform.system() != "Windows":
        return None
    if os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir"):
        return None

    candidates: list[Path] = []
    if configured_path:
        candidates.append(Path(configured_path))
    candidates.extend(
        [
            Path.home() / ".ivy2" / "jars" / f"com.globalmentor_{_BARE_FS_ARTIFACT}",
            Path(".cache") / "jars" / _BARE_FS_ARTIFACT,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    target = Path(".cache") / "jars" / _BARE_FS_ARTIFACT
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s for Windows local Hadoop compatibility", _BARE_FS_ARTIFACT)
    urllib.request.urlretrieve(_BARE_FS_MAVEN_URL, target)
    return str(target.resolve())


def build_spark(
    config: AppConfig | SparkConfig | None = None,
    app_name: str | None = None,
) -> SparkSession:
    """Build a configured SparkSession supporting local, standalone, Delta, and S3A backends."""
    if isinstance(config, AppConfig):
        spark_cfg = config.spark
        storage_cfg = config.storage
    elif isinstance(config, SparkConfig):
        spark_cfg = config
        storage_cfg = StorageConfig()
    else:
        spark_cfg = SparkConfig()
        storage_cfg = StorageConfig()

    effective_name = app_name or spark_cfg.app_name
    local_dir = spark_cfg.local_dir or str(Path(tempfile.gettempdir()) / "chicago-crimes-spark")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Ensure worker Python matches the active virtualenv / container interpreter.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    if spark_cfg.profile == "local":
        # Avoid Spark URI failures when machine hostname contains underscores.
        os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    builder = (
        SparkSession.builder.master(spark_cfg.master)
        .appName(effective_name)
        .config("spark.driver.memory", spark_cfg.driver_memory)
        .config("spark.executor.memory", spark_cfg.executor_memory)
        .config("spark.sql.shuffle.partitions", str(spark_cfg.shuffle_partitions))
        .config("spark.default.parallelism", str(spark_cfg.default_parallelism))
        .config("spark.local.dir", local_dir)
        .config("spark.sql.adaptive.enabled", str(spark_cfg.enable_adaptive_execution).lower())
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            str(spark_cfg.enable_adaptive_execution).lower(),
        )
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.session.timeZone", spark_cfg.session_timezone)
        .config("spark.ui.enabled", str(spark_cfg.ui_enabled).lower())
    )

    if spark_cfg.profile == "local":
        if spark_cfg.driver_host:
            builder = builder.config("spark.driver.host", spark_cfg.driver_host)
        if spark_cfg.driver_bind_address:
            builder = builder.config("spark.driver.bindAddress", spark_cfg.driver_bind_address)
    else:
        resolved_driver_host = spark_cfg.driver_host or socket.gethostname()
        builder = (
            builder.config("spark.cores.max", str(spark_cfg.cores_max))
            .config("spark.executor.cores", str(spark_cfg.executor_cores))
            .config("spark.driver.host", resolved_driver_host)
            .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
        )
        if spark_cfg.driver_bind_address:
            builder = builder.config("spark.driver.bindAddress", spark_cfg.driver_bind_address)

    extra_jar_specs: dict[str, str] = {}
    delta_enabled = spark_cfg.enable_delta or storage_cfg.table_format == "delta"
    s3a_enabled = storage_cfg.backend == "s3a"
    kafka_enabled = spark_cfg.enable_kafka

    if kafka_enabled:
        extra_jar_specs.update(_KAFKA_JARS)

    if delta_enabled:
        extra_jar_specs.update(_DELTA_JARS)
        builder = (
            builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.databricks.delta.properties.defaults.columnMapping.mode", "name")
            .config("spark.databricks.delta.properties.defaults.minReaderVersion", "2")
            .config("spark.databricks.delta.properties.defaults.minWriterVersion", "5")
        )

    if s3a_enabled:
        extra_jar_specs.update(_S3A_JARS)
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", storage_cfg.s3_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", storage_cfg.s3_access_key)
            .config("spark.hadoop.fs.s3a.secret.key", storage_cfg.s3_secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config(
                "spark.hadoop.fs.s3a.connection.ssl.enabled",
                str(storage_cfg.s3_ssl_enabled).lower(),
            )
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
        )
        if delta_enabled:
            builder = builder.config(
                "spark.delta.logStore.s3a.impl",
                "org.apache.spark.sql.delta.storage.S3SingleDriverLogStore",
            )
        if platform.system() == "Windows":
            # Avoid S3A DiskBlockFactory calling NativeIO$Windows.access0 without winutils.
            builder = builder.config("spark.hadoop.fs.s3a.fast.upload", "true").config(
                "spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer"
            )

    resolved_jars = _resolve_extra_jars(extra_jar_specs, spark_cfg.extra_jars_dir)
    classpath_jars = list(resolved_jars)

    bare_jar = _resolve_windows_bare_fs_jar(spark_cfg.windows_bare_fs_jar)
    if bare_jar:
        classpath_jars.append(bare_jar)
        builder = builder.config(
            "spark.hadoop.fs.file.impl", "com.globalmentor.apache.hadoop.fs.BareLocalFileSystem"
        )
        if delta_enabled:
            builder = builder.config(
                "spark.delta.logStore.file.impl",
                "org.apache.spark.sql.delta.storage.LocalLogStore",
            )

    if resolved_jars:
        builder = builder.config("spark.jars", ",".join(resolved_jars))
    if classpath_jars:
        joined_cp = os.pathsep.join(classpath_jars)
        builder = builder.config("spark.driver.extraClassPath", joined_cp).config(
            "spark.executor.extraClassPath", joined_cp
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.log_level)
    logger.info(
        "Initialized SparkSession app=%s profile=%s master=%s delta=%s kafka=%s backend=%s",
        effective_name,
        spark_cfg.profile,
        spark_cfg.master,
        delta_enabled,
        kafka_enabled,
        storage_cfg.backend,
    )
    return spark


@contextmanager
def spark_session(
    config: AppConfig | SparkConfig | None = None,
    app_name: str | None = None,
) -> Iterator[SparkSession]:
    """Context manager managing SparkSession lifecycle without killing pre-existing sessions."""
    existing = SparkSession.getActiveSession()
    spark = build_spark(config=config, app_name=app_name)
    try:
        yield spark
    finally:
        if existing is None:
            spark.stop()
            logger.info("Stopped SparkSession")

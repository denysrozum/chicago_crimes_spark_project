"""Shared command builder for Airflow DAGs without duplicating Spark/pipeline logic."""

from __future__ import annotations

import os


def spark_cli_command(cli_args: str, *, local_direct: bool = False) -> str:
    """Build a shell command invoking `chicago_crimes` CLI inside the Spark container."""
    if local_direct or os.getenv("CHICAGO_CRIMES_AIRFLOW_DIRECT_CLI") == "1":
        return f"python -m chicago_crimes --profile standalone {cli_args}"

    network = os.getenv(
        "CHICAGO_CRIMES_DOCKER_NETWORK", "chicago-crimes-platform_chicago-crimes-net"
    )
    airflow_container = os.getenv("CHICAGO_CRIMES_AIRFLOW_CONTAINER", "chicago-crimes-airflow")
    spark_image = os.getenv("CHICAGO_CRIMES_SPARK_IMAGE", "chicago-crimes-spark:0.2.0")
    master_url = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    minio_user = os.getenv("MINIO_ROOT_USER", "minioadmin")
    minio_password = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    minio_bucket = os.getenv("MINIO_BUCKET", "chicago-crimes")
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

    return (
        f"docker run --rm "
        f"--network {network} "
        f"--volumes-from {airflow_container} "
        f"-w /app "
        f"-e CHICAGO_CRIMES_PROFILE=standalone "
        f"-e SPARK_MASTER_URL={master_url} "
        f"-e MINIO_ENDPOINT={minio_endpoint} "
        f"-e MINIO_ROOT_USER={minio_user} "
        f"-e MINIO_ROOT_PASSWORD={minio_password} "
        f"-e MINIO_BUCKET={minio_bucket} "
        f"-e CHICAGO_CRIMES_STORAGE_BACKEND=s3a "
        f"-e CHICAGO_CRIMES_TABLE_FORMAT=delta "
        f"-e KAFKA_BOOTSTRAP_SERVERS={kafka_servers} "
        f"{spark_image} "
        f"python -m chicago_crimes --profile standalone {cli_args}"
    )

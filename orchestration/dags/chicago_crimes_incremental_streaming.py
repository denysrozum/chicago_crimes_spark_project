"""Airflow DAG orchestrating SODA API incremental ingestion, Kafka drain, and analytics."""

from __future__ import annotations

from datetime import datetime, timedelta

from _common import spark_cli_command
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "chicago-crimes-data-eng",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="chicago_crimes_incremental_streaming",
    description="Orchestrate SODA incremental API fetch -> Kafka -> Structured Streaming Delta.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["chicago-crimes", "streaming", "soda", "kafka"],
    params={
        "since": Param("", type=["null", "string"]),
        "limit": Param(100, type="integer", minimum=1),
        "kafka_topic": Param("chicago-crimes.orchestrated", type="string"),
        "watermark_file": Param("artifacts/airflow/state/soda_watermark.json", type="string"),
        "checkpoint": Param("s3a://chicago-crimes/checkpoints/airflow_stream", type="string"),
        "bronze_target": Param("s3a://chicago-crimes/streaming/bronze_airflow", type="string"),
        "silver_target": Param("s3a://chicago-crimes/streaming/silver_airflow", type="string"),
        "quarantine_target": Param(
            "s3a://chicago-crimes/streaming/quarantine_airflow", type="string"
        ),
        "dlq_target": Param("s3a://chicago-crimes/streaming/dlq_airflow", type="string"),
        "metrics_log": Param(
            "artifacts/airflow/streaming/microbatch_metrics.jsonl", type="string"
        ),
        "analytics_output": Param("artifacts/airflow/streaming_analytics", type="string"),
    },
) as dag:
    ingest_soda_to_kafka = BashOperator(
        task_id="ingest_soda_to_kafka",
        bash_command=spark_cli_command(
            "ingest-soda "
            "{% if params.since %}--since {{ params.since }} {% endif %}"
            "--limit {{ params.limit }} "
            "--publish-kafka "
            "--bootstrap-servers kafka:9092 "
            "--topic {{ params.kafka_topic }} "
            "--watermark-file {{ params.watermark_file }}"
        ),
    )

    drain_kafka_stream = BashOperator(
        task_id="drain_kafka_stream",
        bash_command=spark_cli_command(
            "stream "
            "--bootstrap-servers kafka:9092 "
            "--topic {{ params.kafka_topic }} "
            "--checkpoint {{ params.checkpoint }} "
            "--bronze-target {{ params.bronze_target }} "
            "--silver-target {{ params.silver_target }} "
            "--quarantine-target {{ params.quarantine_target }} "
            "--dlq-target {{ params.dlq_target }} "
            "--metrics-log {{ params.metrics_log }} "
            "--available-now"
        ),
    )

    verify_streaming_analytics = BashOperator(
        task_id="verify_streaming_analytics",
        bash_command=spark_cli_command(
            "analyze --input {{ params.silver_target }} "
            "--format delta "
            "--tasks AQ-01 "
            "--output {{ params.analytics_output }} "
            "--no-plots"
        ),
    )

    ingest_soda_to_kafka >> drain_kafka_stream >> verify_streaming_analytics

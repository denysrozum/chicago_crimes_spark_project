"""Airflow DAG orchestrating batch Lakehouse Delta sync, Gold tables, and analytics."""

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
    dag_id="chicago_crimes_batch_pipeline",
    description="Orchestrate Spark Standalone Delta sync, Gold materialization, and analytics.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["chicago-crimes", "batch", "lakehouse", "delta"],
    params={
        "silver_input": Param("data/silver/chicago_crimes", type="string"),
        "silver_delta_target": Param(
            "s3a://chicago-crimes/silver/chicago_crimes_orchestrated", type="string"
        ),
        "limit": Param(50000, type="integer", minimum=1),
        "gold_output": Param("artifacts/airflow/batch_gold", type="string"),
        "analytics_output": Param("artifacts/airflow/batch_analytics", type="string"),
        "analyze_tasks": Param("AQ-01 AQ-08", type="string"),
    },
) as dag:
    ensure_minio_bucket = BashOperator(
        task_id="ensure_minio_bucket",
        bash_command=spark_cli_command("lakehouse ensure-bucket"),
    )

    sync_silver_delta = BashOperator(
        task_id="sync_silver_delta",
        bash_command=spark_cli_command(
            "lakehouse sync-delta "
            "--input {{ params.silver_input }} "
            "--target {{ params.silver_delta_target }} "
            "--limit {{ params.limit }} "
            "--partition-by year --overwrite"
        ),
    )

    build_gold_tables = BashOperator(
        task_id="build_gold_tables",
        bash_command=spark_cli_command(
            "gold --input {{ params.silver_input }} "
            "--output {{ params.gold_output }} --overwrite"
        ),
    )

    run_silver_analytics = BashOperator(
        task_id="run_silver_analytics",
        bash_command=spark_cli_command(
            "analyze --input {{ params.silver_delta_target }} "
            "--format delta "
            "--tasks {{ params.analyze_tasks }} "
            "--output {{ params.analytics_output }} "
            "--no-plots"
        ),
    )

    ensure_minio_bucket >> sync_silver_delta >> build_gold_tables >> run_silver_analytics

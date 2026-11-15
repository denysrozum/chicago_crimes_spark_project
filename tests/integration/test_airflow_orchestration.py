"""Orchestration structural verification and isolated live Airflow DAG tests."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from confluent_kafka.admin import AdminClient

from chicago_crimes.core.config import load_config
from chicago_crimes.storage.minio import delete_prefix, ensure_bucket, list_objects, s3a_uri

REPO_ROOT = Path(__file__).resolve().parents[2]
DAGS_DIR = REPO_ROOT / "orchestration" / "dags"
AIRFLOW_CONTAINER = os.getenv("CHICAGO_CRIMES_AIRFLOW_CONTAINER", "chicago-crimes-airflow")
AIRFLOW_CLI_TIMEOUT = int(os.getenv("CHICAGO_CRIMES_AIRFLOW_CLI_TIMEOUT", "60"))
AIRFLOW_DAG_TIMEOUT = int(os.getenv("CHICAGO_CRIMES_AIRFLOW_DAG_WAIT_SECONDS", "720"))


def _require_running_airflow_container() -> str:
    try:
        res = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}}|"
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                AIRFLOW_CONTAINER,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Docker CLI unavailable for Airflow tests ({exc})")

    if res.returncode != 0:
        pytest.skip(f"Container '{AIRFLOW_CONTAINER}' does not exist ({res.stderr.strip()})")
    running, health = (res.stdout.strip().split("|") + ["unknown"])[:2]
    if running.lower() != "true":
        pytest.skip(f"Container '{AIRFLOW_CONTAINER}' is not running (status={res.stdout.strip()})")
    return health


def _run_airflow_cli(args: list[str], *, expect_json: bool = True) -> object:
    cmd = ["docker", "exec", AIRFLOW_CONTAINER, "airflow", *args]
    if expect_json and "-o" not in args:
        cmd.extend(["-o", "json"])
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AIRFLOW_CLI_TIMEOUT,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        container_health = _require_running_airflow_container()
        pytest.fail(
            f"[TRANSIENT_INFRA_TIMEOUT] Airflow CLI timed out after {AIRFLOW_CLI_TIMEOUT}s "
            f"(container_health={container_health}): {' '.join(cmd)}\n"
            f"stdout={exc.stdout}\nstderr={exc.stderr}"
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"[FUNCTIONAL_CLI_FAILURE] Airflow CLI exit={exc.returncode}: {' '.join(cmd)}\n"
            f"stdout={exc.stdout}\nstderr={exc.stderr}"
        )
    return json.loads(res.stdout) if expect_json else res.stdout


def _wait_for_dag_run(
    dag_id: str, run_id: str, *, timeout_seconds: int = AIRFLOW_DAG_TIMEOUT
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_state = "unknown"
    while time.monotonic() < deadline:
        runs = _run_airflow_cli(["dags", "list-runs", "-d", dag_id])
        assert isinstance(runs, list)
        match = next((r for r in runs if r.get("run_id") == run_id), None)
        if match is not None:
            last_state = str(match.get("state", "unknown"))
            if last_state in {"success", "failed"}:
                return last_state
        time.sleep(3.0)

    tasks = _run_airflow_cli(["tasks", "states-for-dag-run", dag_id, run_id])
    pytest.fail(
        f"[TRANSIENT_INFRA_TIMEOUT] DAG '{dag_id}' run '{run_id}' did not finish within "
        f"{timeout_seconds}s (last_state={last_state}, task_states={tasks})"
    )


def _write_isolated_silver_fixture(silver_dir: Path, *, rows: int = 60) -> None:
    csv_path = silver_dir.parent / "raw_airflow.csv"
    headers = (
        "ID;Case Number;Date;Block;IUCR;Primary Type;Description;"
        "Location Description;Arrest;Domestic;Beat;District;Ward;"
        "Community Area;FBI Code;X Coordinate;Y Coordinate;Year;Updated On;"
        "Latitude;Longitude;Location"
    )
    lines = [headers]
    for i in range(1, rows + 1):
        year = 2024 if i % 2 == 0 else 2025
        month = ((i - 1) % 12) + 1
        day = ((i - 1) % 28) + 1
        district = ((i - 1) % 2) + 1
        cid = 700000 + i
        arrest_val = "true" if i % 3 == 0 else "false"
        lines.append(
            f"{cid};HA{cid:06d};{month:02d}/{day:02d}/{year} 01:30:00 PM;"
            f"001XX N STATE ST;0820;THEFT;OVER $500;STREET;{arrest_val};false;"
            f"0111;{district:03d};42;32;06;1176000;1900000;{year};"
            f"01/15/2026 09:00:00 AM;41.881000;-87.627000;(41.881000, -87.627000)"
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Build the Parquet fixture in an isolated subprocess so pytest's JVM classpath
    # is not initialized without Delta/Kafka/S3A packages needed by other test suites.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "chicago_crimes",
            "--profile",
            "local",
            "--log-level",
            "ERROR",
            "process",
            "--pipeline",
            "silver",
            "--input",
            str(csv_path),
            "--input-format",
            "csv",
            "--output",
            str(silver_dir),
            "--overwrite",
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=AIRFLOW_CLI_TIMEOUT * 2,
    )


def test_airflow_dags_delegate_without_duplicating_business_logic() -> None:
    """DAG modules must parse cleanly and delegate all pipeline work to the CLI."""
    airflowignore = (DAGS_DIR / ".airflowignore").read_text(encoding="utf-8")
    assert r"_common\.py" in airflowignore

    spec = importlib.util.spec_from_file_location("airflow_dag_common", DAGS_DIR / "_common.py")
    assert spec is not None and spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    docker_cmd = common.spark_cli_command("gold --input a --output b")
    assert "docker run --rm" in docker_cmd
    assert "python -m chicago_crimes --profile standalone gold --input a --output b" in docker_cmd

    direct_cmd = common.spark_cli_command("gold --input a --output b", local_direct=True)
    assert direct_cmd == "python -m chicago_crimes --profile standalone gold --input a --output b"

    expected_dags = {
        "chicago_crimes_batch_pipeline.py": "chicago_crimes_batch_pipeline",
        "chicago_crimes_incremental_streaming.py": "chicago_crimes_incremental_streaming",
    }
    for filename, dag_id in expected_dags.items():
        source = (DAGS_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        assert dag_id in source
        assert "spark_cli_command" in source
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("chicago_crimes"), (
                    f"{filename} must not import pipeline internals ({node.module})"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "pyspark"


@pytest.mark.integration
def test_live_airflow_isolated_batch_success_failure_and_streaming_dags() -> None:
    """Trigger fresh isolated Airflow DAG runs, assert task graph outcomes, and verify outputs."""
    _require_running_airflow_container()
    dags = _run_airflow_cli(["dags", "list"])
    assert isinstance(dags, list)
    dag_map = {item["dag_id"]: item for item in dags}
    assert "chicago_crimes_batch_pipeline" in dag_map
    assert "chicago_crimes_incremental_streaming" in dag_map

    _run_airflow_cli(["dags", "unpause", "chicago_crimes_batch_pipeline"], expect_json=False)
    _run_airflow_cli(["dags", "unpause", "chicago_crimes_incremental_streaming"], expect_json=False)

    cfg = load_config(profile="standalone")
    storage_cfg = cfg.storage.model_copy(
        update={"s3_endpoint": os.getenv("MINIO_ENDPOINT", "http://localhost:9000")}
    )
    ensure_bucket(storage_cfg)

    run_tag = uuid4().hex[:8]
    host_run_dir = REPO_ROOT / "artifacts" / "test_runs" / f"af_{run_tag}"
    rel_run_dir = f"artifacts/test_runs/af_{run_tag}"
    s3_prefix = f"test_runs/af_{run_tag}"
    kafka_topic = f"chicago-crimes.it.af.{run_tag}"

    _write_isolated_silver_fixture(host_run_dir / "input_silver", rows=60)
    try:
        # 1. Isolated batch success run
        ok_run_id = f"it_batch_ok_{run_tag}"
        ok_conf = {
            "silver_input": f"{rel_run_dir}/input_silver",
            "silver_delta_target": s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/silver_delta"),
            "limit": 60,
            "gold_output": f"{rel_run_dir}/gold_out",
            "analytics_output": f"{rel_run_dir}/analytics_out",
            "analyze_tasks": "AQ-01 AQ-08",
        }
        _run_airflow_cli(
            [
                "dags",
                "trigger",
                "-r",
                ok_run_id,
                "-c",
                json.dumps(ok_conf),
                "chicago_crimes_batch_pipeline",
            ],
            expect_json=False,
        )
        ok_state = _wait_for_dag_run("chicago_crimes_batch_pipeline", ok_run_id)
        assert ok_state == "success"
        ok_tasks = _run_airflow_cli(
            ["tasks", "states-for-dag-run", "chicago_crimes_batch_pipeline", ok_run_id]
        )
        assert {t["task_id"]: t["state"] for t in ok_tasks} == {
            "ensure_minio_bucket": "success",
            "sync_silver_delta": "success",
            "build_gold_tables": "success",
            "run_silver_analytics": "success",
        }

        assert list_objects(storage_cfg, prefix=f"{s3_prefix}/silver_delta/_delta_log/")
        gold_manifest = json.loads(
            (host_run_dir / "gold_out" / "_gold_manifest.json").read_text(encoding="utf-8")
        )
        assert gold_manifest["source_silver_count"] == 60
        analytics_manifest = json.loads(
            (host_run_dir / "analytics_out" / "manifest.json").read_text(encoding="utf-8")
        )
        assert analytics_manifest["task_count"] == 2

        # 2. Isolated failure-propagation run on nonexistent silver_input
        fail_run_id = f"it_batch_fail_{run_tag}"
        fail_conf = {
            **ok_conf,
            "silver_input": f"{rel_run_dir}/nonexistent_silver_input",
        }
        _run_airflow_cli(
            [
                "dags",
                "trigger",
                "-r",
                fail_run_id,
                "-c",
                json.dumps(fail_conf),
                "chicago_crimes_batch_pipeline",
            ],
            expect_json=False,
        )
        fail_state = _wait_for_dag_run("chicago_crimes_batch_pipeline", fail_run_id)
        assert fail_state == "failed"
        fail_tasks = _run_airflow_cli(
            ["tasks", "states-for-dag-run", "chicago_crimes_batch_pipeline", fail_run_id]
        )
        assert {t["task_id"]: t["state"] for t in fail_tasks} == {
            "ensure_minio_bucket": "success",
            "sync_silver_delta": "failed",
            "build_gold_tables": "upstream_failed",
            "run_silver_analytics": "upstream_failed",
        }

        # 3. Isolated incremental streaming DAG run
        stream_run_id = f"it_stream_ok_{run_tag}"
        stream_conf = {
            "since": "2026-11-01T00:00:00",
            "limit": 5,
            "kafka_topic": kafka_topic,
            "watermark_file": f"{rel_run_dir}/soda_watermark.json",
            "checkpoint": s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/stream_ckpt"),
            "bronze_target": s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/stream_bronze"),
            "silver_target": s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/stream_silver"),
            "quarantine_target": s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/stream_quarantine"),
            "dlq_target": s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/stream_dlq"),
            "metrics_log": f"{rel_run_dir}/stream_metrics.jsonl",
            "analytics_output": f"{rel_run_dir}/stream_analytics",
        }
        _run_airflow_cli(
            [
                "dags",
                "trigger",
                "-r",
                stream_run_id,
                "-c",
                json.dumps(stream_conf),
                "chicago_crimes_incremental_streaming",
            ],
            expect_json=False,
        )
        stream_state = _wait_for_dag_run("chicago_crimes_incremental_streaming", stream_run_id)
        assert stream_state == "success"
        stream_tasks = _run_airflow_cli(
            ["tasks", "states-for-dag-run", "chicago_crimes_incremental_streaming", stream_run_id]
        )
        assert {t["task_id"]: t["state"] for t in stream_tasks} == {
            "ingest_soda_to_kafka": "success",
            "drain_kafka_stream": "success",
            "verify_streaming_analytics": "success",
        }

        wm = json.loads((host_run_dir / "soda_watermark.json").read_text(encoding="utf-8"))
        assert wm["records_fetched"] == 5
        metrics_entries = [
            json.loads(line)
            for line in (host_run_dir / "stream_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len(metrics_entries) >= 1
        assert all(entry["is_reconciled"] is True for entry in metrics_entries)
        assert list_objects(storage_cfg, prefix=f"{s3_prefix}/stream_silver/_delta_log/")
    finally:
        shutil.rmtree(host_run_dir, ignore_errors=True)
        try:
            delete_prefix(storage_cfg, f"{s3_prefix}/")
        except Exception:
            pass
        try:
            bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS_EXTERNAL", "localhost:9094")
            AdminClient({"bootstrap.servers": bootstrap}).delete_topics([kafka_topic])
        except Exception:
            pass

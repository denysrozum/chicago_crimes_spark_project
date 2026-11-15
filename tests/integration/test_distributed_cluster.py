"""Live integration tests against the Spark Standalone + MinIO S3A + Delta cluster."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from chicago_crimes.analytics.benchmark import inspect_spark_master_cluster
from chicago_crimes.core.config import load_config
from chicago_crimes.storage.minio import delete_prefix, ensure_bucket, list_objects, s3a_uri

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_UI_URL = os.getenv("SPARK_MASTER_UI_URL", "http://localhost:8080")
SPARK_MASTER_CONTAINER = os.getenv("CHICAGO_CRIMES_SPARK_CONTAINER", "chicago-crimes-spark-master")
DOCKER_CMD_TIMEOUT = int(os.getenv("CHICAGO_CRIMES_DOCKER_CLI_TIMEOUT", "300"))


def _require_live_cluster() -> dict:
    info = inspect_spark_master_cluster(MASTER_UI_URL)
    if info is None or info.get("aliveworkers", 0) < 2:
        pytest.skip(
            f"Live 2-worker Spark Standalone cluster not reachable on {MASTER_UI_URL}."
        )
    return info


def _write_synthetic_raw_csv(path: Path, *, rows: int, arrest_flag: str = "false") -> None:
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
        district = ((i - 1) % 4) + 1
        cid = 500000 + i
        lat = 41.80 + (i % 10) * 0.01
        lon = -87.70 + (i % 10) * 0.01
        lines.append(
            f"{cid};HZ{cid:06d};{month:02d}/{day:02d}/{year} 02:15:00 PM;"
            f"001XX N STATE ST;0820;THEFT;OVER $500;STREET;{arrest_flag};false;"
            f"0111;{district:03d};42;32;06;1176000;1900000;{year};"
            f"01/15/2026 10:00:00 AM;{lat:.6f};{lon:.6f};({lat:.6f}, {lon:.6f})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _exec_in_spark_master(cli_args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [
        "docker",
        "exec",
        SPARK_MASTER_CONTAINER,
        "python",
        "-m",
        "chicago_crimes",
        "--profile",
        "standalone",
        *cli_args,
    ]
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOCKER_CMD_TIMEOUT,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"[TRANSIENT_INFRA_TIMEOUT] Spark container CLI exceeded {DOCKER_CMD_TIMEOUT}s: "
            f"{' '.join(cmd)}\nstdout={exc.stdout}\nstderr={exc.stderr}"
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"[FUNCTIONAL_CLI_FAILURE] Spark CLI exit={exc.returncode}: {' '.join(cmd)}\n"
            f"stdout={exc.stdout}\nstderr={exc.stderr}"
        )


@pytest.mark.integration
def test_live_spark_standalone_two_workers_alive() -> None:
    """Spark Standalone Master must report ALIVE with 2 workers and 4 cores."""
    info = _require_live_cluster()
    assert info["status"] == "ALIVE"
    assert info["aliveworkers"] == 2
    assert info["cores"] == 4
    alive_hosts = [w["host"] for w in info["workers"] if w["state"] == "ALIVE"]
    assert len(alive_hosts) == 2


@pytest.mark.integration
def test_live_standalone_delta_sync_merge_and_executor_distribution() -> None:
    """Execute isolated sync-delta, merge-silver, and 2-worker benchmark on MinIO S3A Delta."""
    _require_live_cluster()
    cfg = load_config(profile="standalone")
    storage_cfg = cfg.storage.model_copy(
        update={"s3_endpoint": os.getenv("MINIO_ENDPOINT", "http://localhost:9000")}
    )
    ensure_bucket(storage_cfg)

    run_tag = uuid4().hex[:10]
    host_run_dir = REPO_ROOT / "artifacts" / "test_runs" / f"cluster_{run_tag}"
    container_run_rel = f"artifacts/test_runs/cluster_{run_tag}"
    s3_prefix = f"test_runs/cluster_{run_tag}"
    delta_uri = s3a_uri(storage_cfg.s3_bucket, f"{s3_prefix}/silver_delta")

    total_rows = 240
    _write_synthetic_raw_csv(host_run_dir / "raw_base.csv", rows=total_rows, arrest_flag="false")
    _write_synthetic_raw_csv(host_run_dir / "raw_upd.csv", rows=40, arrest_flag="true")
    try:
        for csv_name, out_name in (
            ("raw_base.csv", "silver_base"),
            ("raw_upd.csv", "silver_upd"),
        ):
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
                    str(host_run_dir / csv_name),
                    "--input-format",
                    "csv",
                    "--output",
                    str(host_run_dir / out_name),
                    "--overwrite",
                ],
                cwd=str(REPO_ROOT),
                check=True,
                capture_output=True,
                text=True,
                timeout=DOCKER_CMD_TIMEOUT,
            )

        # 1. Sync isolated Silver dataset into MinIO Delta via Spark Standalone executors
        _exec_in_spark_master(
            [
                "lakehouse",
                "sync-delta",
                "--input",
                f"{container_run_rel}/silver_base",
                "--target",
                delta_uri,
                "--partition-by",
                "year",
                "--overwrite",
            ]
        )
        # 2. Idempotently merge updates into the same Delta table (version 1)
        merge_res = _exec_in_spark_master(
            [
                "lakehouse",
                "merge-silver",
                "--input",
                f"{container_run_rel}/silver_upd",
                "--target",
                delta_uri,
                "--partition-by",
                "year",
            ]
        )
        merge_json = json.loads(merge_res.stdout)
        assert merge_json["version"] == 1
        assert int(merge_json["metrics"]["numTargetRowsUpdated"]) == 40
        assert int(merge_json["metrics"]["numTargetRowsInserted"]) == 0

        delta_logs = list_objects(storage_cfg, prefix=f"{s3_prefix}/silver_delta/_delta_log/")
        assert any(key.endswith("00000000000000000000.json") for key in delta_logs)
        assert any(key.endswith("00000000000000000001.json") for key in delta_logs)

        # 3. Execute benchmark on spark://spark-master:7077 and verify both workers run tasks
        _exec_in_spark_master(
            [
                "benchmark",
                "--input",
                delta_uri,
                "--output",
                f"{container_run_rel}/bench.json",
                "--partitions",
                "16",
                "--no-delta",
            ]
        )
        report = json.loads((host_run_dir / "bench.json").read_text(encoding="utf-8"))
        assert report["master"] == "spark://spark-master:7077"
        assert report["storage_backend"] == "s3a"
        assert report["input_rows"] == total_rows
        dist = report["runs"][0]["executor_distribution"]
        assert set(dist.keys()) == {"spark-worker-1", "spark-worker-2"}
        assert sum(dist.values()) == total_rows
    finally:
        shutil.rmtree(host_run_dir, ignore_errors=True)
        try:
            delete_prefix(storage_cfg, f"{s3_prefix}/")
        except Exception:
            pass

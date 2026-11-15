"""Distributed Spark execution telemetry and performance benchmarking."""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from chicago_crimes.analytics.runner import prepare_analytics_dataframe
from chicago_crimes.core.config import AnalyticsConfig, AppConfig
from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.spark import spark_session
from chicago_crimes.storage.delta import delta_history, write_delta
from chicago_crimes.storage.lakehouse import read_table
from chicago_crimes.storage.minio import ensure_bucket, s3a_uri

logger = get_logger("analytics.benchmark")


@dataclass(frozen=True)
class BenchmarkStepResult:
    """Measured latency and row throughput for a single benchmark scenario."""

    name: str
    shuffle_partitions: int
    cores_max: int | None
    duration_seconds: float
    input_rows: int
    output_rows: int
    rows_per_second: float
    executor_distribution: dict[str, int]


def inspect_spark_master_cluster(master_ui_url: str) -> dict[str, Any] | None:
    """Query Spark Standalone Master `/json/` API for worker counts and state."""
    url = f"{master_ui_url.rstrip('/')}/json/"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        workers = [
            {
                "id": w.get("id"),
                "host": w.get("host"),
                "port": w.get("port"),
                "cores": w.get("cores"),
                "coresused": w.get("coresused"),
                "memory": w.get("memory"),
                "memoryused": w.get("memoryused"),
                "state": w.get("state"),
            }
            for w in payload.get("workers", [])
        ]
        return {
            "url": payload.get("url"),
            "status": payload.get("status"),
            "aliveworkers": int(payload.get("aliveworkers", 0)),
            "cores": int(payload.get("cores", 0)),
            "coresused": int(payload.get("coresused", 0)),
            "memory": int(payload.get("memory", 0)),
            "workers": workers,
        }
    except Exception as exc:
        logger.warning("Could not query Spark Master UI at %s: %s", url, exc)
        return None


def _default_master_ui_url(master: str) -> str | None:
    if os.getenv("SPARK_MASTER_UI_URL"):
        return os.environ["SPARK_MASTER_UI_URL"]
    if not master.startswith("spark://"):
        return None
    parsed = urlparse(master)
    host = parsed.hostname or "localhost"
    return f"http://{host}:8080"


def measure_executor_distribution(df: DataFrame) -> dict[str, int]:
    """Count rows processed per Spark executor host/container cleanly without custom closures."""

    def _partition_hosts(_idx: int, iterator: Iterator[Any]) -> Iterator[tuple[str, int]]:
        count = 0
        for _ in iterator:
            count += 1
        yield (socket.gethostname(), count)

    counts_per_part = df.rdd.mapPartitionsWithIndex(_partition_hosts).collect()
    totals: dict[str, int] = {}
    for host, cnt in counts_per_part:
        totals[host] = totals.get(host, 0) + int(cnt)
    return dict(sorted(totals.items()))


def _run_benchmark_workload(
    spark: SparkSession,
    df: DataFrame,
    *,
    analytics_cfg: AnalyticsConfig,
    delta_target: str | None = None,
) -> tuple[int, dict[str, int]]:
    """Run representative analytics + spatial aggregation + optional Delta write."""
    prepared = prepare_analytics_dataframe(df)
    _ = analytics_cfg

    # Trigger shuffle-heavy aggregations across executors & verify worker distribution.
    monthly = prepared.groupBy("year", "month").agg(
        F.count(F.lit(1)).alias("incident_count"),
        F.sum(F.when(F.col("Arrest") == True, 1).otherwise(0)).alias("arrest_count"),  # noqa: E712
    )
    district_type = (
        prepared.filter(F.col("District").isNotNull()).groupBy("District", "Primary Type").count()
    )

    m_count = int(monthly.count())
    d_count = int(district_type.count())
    distribution = measure_executor_distribution(prepared)

    if delta_target is not None:
        write_delta(monthly, delta_target, mode="overwrite")
        delta_history(spark, delta_target, limit=1)

    return m_count + d_count, distribution


def run_distributed_benchmark(
    cfg: AppConfig,
    *,
    input_path: str | Path,
    output_json: str | Path,
    partition_counts: Sequence[int] = (8, 16, 32),
    sample_limit: int | None = None,
    write_delta_output: bool = True,
) -> dict[str, Any]:
    """Run benchmark sweeps and record cluster topology, worker distribution, and timings."""
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if write_delta_output and cfg.storage.backend == "s3a":
        ensure_bucket(cfg.storage)

    master_ui = _default_master_ui_url(cfg.spark.master)
    master_info_before = inspect_spark_master_cluster(master_ui) if master_ui else None

    results: list[BenchmarkStepResult] = []
    with spark_session(cfg, app_name="ChicagoCrimes-Benchmark") as spark:
        source_df = read_table(spark, input_path, table_format="auto")
        if sample_limit is not None:
            source_df = source_df.limit(sample_limit)
        input_rows = int(source_df.count())

        # Warm JVM/Catalyst once on small count before timing sweeps.
        source_df.select("ID").limit(10).collect()

        for num_parts in partition_counts:
            spark.conf.set("spark.sql.shuffle.partitions", str(num_parts))
            repart_df = source_df.repartition(num_parts)

            delta_uri: str | None = None
            if write_delta_output:
                if cfg.storage.backend == "s3a":
                    delta_uri = s3a_uri(
                        cfg.storage.s3_bucket,
                        f"benchmarks/monthly_summary_p{num_parts}",
                    )
                else:
                    delta_uri = str(out_path.parent / f"benchmark_delta_p{num_parts}")

            t0 = time.perf_counter()
            out_rows, dist = _run_benchmark_workload(
                spark,
                repart_df,
                analytics_cfg=cfg.analytics,
                delta_target=delta_uri,
            )
            elapsed = max(1e-6, time.perf_counter() - t0)
            results.append(
                BenchmarkStepResult(
                    name=f"partitions_{num_parts}",
                    shuffle_partitions=int(num_parts),
                    cores_max=cfg.spark.cores_max if cfg.spark.profile == "standalone" else None,
                    duration_seconds=round(elapsed, 3),
                    input_rows=input_rows,
                    output_rows=out_rows,
                    rows_per_second=round(input_rows / elapsed, 1),
                    executor_distribution=dist,
                )
            )

    master_info_after = inspect_spark_master_cluster(master_ui) if master_ui else None
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "profile": cfg.spark.profile,
        "master": cfg.spark.master,
        "storage_backend": cfg.storage.backend,
        "table_format": cfg.storage.table_format,
        "input_path": str(input_path),
        "input_rows": results[0].input_rows if results else 0,
        "cluster_before": master_info_before,
        "cluster_after": master_info_after,
        "runs": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Benchmark report saved -> %s (%d runs)", out_path, len(results))
    return report

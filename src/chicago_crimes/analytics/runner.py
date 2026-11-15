"""Analytical task execution engine with strict error handling and manifests."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

from chicago_crimes.analytics.registry import resolve_tasks

if TYPE_CHECKING:
    import pandas as pd
from chicago_crimes.core.config import AnalyticsConfig
from chicago_crimes.core.logging import get_logger
from chicago_crimes.processing.features import add_temporal_features, extract_block_features
from chicago_crimes.storage.local import write_single_csv

logger = get_logger("analytics.runner")

MAX_PLOT_ROWS = 5000


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    legacy_name: str
    domain_name: str
    category: str
    row_count: int
    csv_path: str | None
    chart_path: str | None
    duration_seconds: float


def prepare_analytics_dataframe(df: DataFrame) -> DataFrame:
    """Derive missing temporal or street features idempotently for analytical queries."""
    work = df
    needed_temporal = {"hour", "year", "day_of_week", "day_of_week_num"}
    if not needed_temporal.issubset(set(work.columns)) and "Date" in work.columns:
        temporal_names = ("ts", "hour", "year", "month", "day_of_week_num", "day_of_week")
        preserved = {c: f"__orig_{c}" for c in temporal_names if c in work.columns}
        for orig, alias in preserved.items():
            work = work.withColumnRenamed(orig, alias)
        work = add_temporal_features(work, date_col="Date")
        for orig, alias in preserved.items():
            work = work.withColumn(orig, F.col(alias)).drop(alias)

    if "street_name" not in work.columns and "Block" in work.columns:
        work = extract_block_features(work, impute_missing_block_number=False)

    return work


def run_analytical_tasks(
    df: DataFrame,
    output_dir: str | Path,
    tasks: list[str] | None = None,
    category: str | None = None,
    config: AnalyticsConfig | None = None,
    render_plots: bool = True,
    write_csv: bool = True,
) -> list[TaskExecutionResult]:
    """Execute selected analytical tasks fail-fast and persist CSV/PNG/manifest outputs."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = config or AnalyticsConfig()

    selected = resolve_tasks(identifiers=tasks, category=category)
    prepared_df = prepare_analytics_dataframe(df)
    should_cache_prepared = len(selected) > 1
    if should_cache_prepared:
        prepared_df = prepared_df.persist(StorageLevel.MEMORY_AND_DISK)

    results: list[TaskExecutionResult] = []
    try:
        for task in selected:
            start = time.perf_counter()
            logger.info("Executing %s (%s / %s)", task.task_id, task.domain_name, task.legacy_name)

            result_df = task.execute(prepared_df, cfg).persist(StorageLevel.MEMORY_AND_DISK)
            task_dir = out_root / f"{task.task_id}_{task.domain_name}"
            task_dir.mkdir(parents=True, exist_ok=True)

            csv_path_str: str | None = None
            chart_path_str: str | None = None
            try:
                row_count = result_df.count()
                pdf: pd.DataFrame | None = None
                if row_count <= MAX_PLOT_ROWS and (write_csv or render_plots):
                    pdf = result_df.toPandas()
                    if write_csv:
                        csv_file = task_dir / "result.csv"
                        pdf.to_csv(csv_file, index=False)
                        csv_path_str = str(csv_file)
                elif write_csv:
                    csv_file = write_single_csv(result_df, task_dir, filename="result.csv")
                    csv_path_str = str(csv_file)

                if render_plots and row_count > 0:
                    if pdf is None:
                        logger.warning(
                            "Task %s returned %d rows (> %d); plotting head(%d).",
                            task.task_id,
                            row_count,
                            MAX_PLOT_ROWS,
                            MAX_PLOT_ROWS,
                        )
                        pdf = result_df.limit(MAX_PLOT_ROWS).toPandas()
                    chart_file = task.draw_plot(pdf, task_dir / "chart.png")
                    chart_path_str = str(chart_file) if chart_file is not None else None
            finally:
                result_df.unpersist()

            elapsed = round(time.perf_counter() - start, 3)
            results.append(
                TaskExecutionResult(
                    task_id=task.task_id,
                    legacy_name=task.legacy_name,
                    domain_name=task.domain_name,
                    category=task.category,
                    row_count=row_count,
                    csv_path=csv_path_str,
                    chart_path=chart_path_str,
                    duration_seconds=elapsed,
                )
            )
    finally:
        if should_cache_prepared:
            prepared_df.unpersist()

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "task_count": len(results),
        "analytics_config": cfg.model_dump(),
        "tasks": [asdict(item) for item in results],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return results

"""Analytical task registry, runner, and dataset profiling utilities."""

from chicago_crimes.analytics.base import AnalyticalTask
from chicago_crimes.analytics.profiling import (
    BlockProfile,
    DatasetProfile,
    compute_correlation_matrix,
    find_high_correlations,
    profile_dataset,
    run_block_stats,
    run_correlation_analysis,
)
from chicago_crimes.analytics.registry import ALL_TASKS, ANALYTICAL_TASKS, get_task, resolve_tasks
from chicago_crimes.analytics.runner import (
    TaskExecutionResult,
    prepare_analytics_dataframe,
    run_analytical_tasks,
)

__all__ = [
    "ALL_TASKS",
    "ANALYTICAL_TASKS",
    "AnalyticalTask",
    "BlockProfile",
    "DatasetProfile",
    "TaskExecutionResult",
    "compute_correlation_matrix",
    "find_high_correlations",
    "get_task",
    "prepare_analytics_dataframe",
    "profile_dataset",
    "resolve_tasks",
    "run_analytical_tasks",
    "run_block_stats",
    "run_correlation_analysis",
]

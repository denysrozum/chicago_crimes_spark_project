"""Base contract for registered analytical tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame

from chicago_crimes.core.config import AnalyticsConfig

ExecuteFn = Callable[[DataFrame, AnalyticsConfig], DataFrame]
PlotFn = Callable[[pd.DataFrame, str, Path], Path]


@dataclass(frozen=True)
class AnalyticalTask:
    task_id: str
    legacy_name: str
    domain_name: str
    category: str
    title: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    execute_fn: ExecuteFn
    plot_fn: PlotFn

    def execute(self, df: DataFrame, config: AnalyticsConfig | None = None) -> DataFrame:
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Task {self.task_id} ({self.domain_name}) missing required columns: {missing}"
            )
        cfg = config or AnalyticsConfig()
        return self.execute_fn(df, cfg)

    def draw_plot(self, pdf: pd.DataFrame, out_path: str | Path) -> Path | None:
        if pdf.empty:
            return None
        return self.plot_fn(pdf, self.title, Path(out_path))

"""Crime forecasting dataset preparation, baselines, models, and evaluation."""

from chicago_crimes.ml.dataset import (
    FORECAST_FEATURE_COLUMNS,
    FORECAST_TARGET_COLUMN,
    ForecastingDataset,
    assert_no_temporal_leakage,
    build_forecasting_features,
    prepare_forecasting_dataset,
    split_chronologically,
)
from chicago_crimes.ml.evaluate import ModelEvaluationResult, compute_regression_metrics
from chicago_crimes.ml.models import ForecastingRunBundle, run_forecasting_experiment

__all__ = [
    "FORECAST_FEATURE_COLUMNS",
    "FORECAST_TARGET_COLUMN",
    "ForecastingDataset",
    "ForecastingRunBundle",
    "ModelEvaluationResult",
    "assert_no_temporal_leakage",
    "build_forecasting_features",
    "compute_regression_metrics",
    "prepare_forecasting_dataset",
    "run_forecasting_experiment",
    "split_chronologically",
]

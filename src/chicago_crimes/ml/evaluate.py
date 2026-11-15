"""Regression evaluation metrics and Matplotlib forecast diagnostic charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Standardized evaluation result for one forecasting model or baseline."""

    model_name: str
    framework: str
    params: dict[str, object]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importances: dict[str, float]
    mlflow_run_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize model evaluation metrics cleanly for reports."""
        return {
            "model_name": self.model_name,
            "framework": self.framework,
            "params": self.params,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "feature_importances": self.feature_importances,
            "mlflow_run_id": self.mlflow_run_id,
        }


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute RMSE, MAE, R2, and WMAPE for crime count regression."""
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    errors = actual - pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    denom = max(1e-9, float(np.sum(np.abs(actual))))
    wmape = float(np.sum(np.abs(errors)) / denom * 100.0)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "wmape_percent": round(wmape, 4),
    }


def plot_model_comparison(results: list[ModelEvaluationResult], out_path: str | Path) -> Path:
    """Plot held-out test RMSE and MAE across baselines and ML models."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = [r.model_name for r in results]
    rmses = [r.test_metrics["rmse"] for r in results]
    maes = [r.test_metrics["mae"] for r in results]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width / 2, rmses, width, label="Test RMSE", color="steelblue")
    b2 = ax.bar(x + width / 2, maes, width, label="Test MAE", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Daily Incident Error (Incidents / District)")
    ax.set_title("Forecasting Model Comparison on Chronological Held-Out Test Split")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_actual_vs_predicted(
    test_df: pd.DataFrame,
    baseline_preds: np.ndarray,
    champion_preds: np.ndarray,
    *,
    baseline_name: str,
    champion_name: str,
    out_path: str | Path,
) -> Path:
    """Plot daily summed test actual incidents against naive baseline and champion model."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "crime_date": pd.to_datetime(test_df["crime_date"]),
            "actual": test_df["incident_count"].to_numpy(dtype=float),
            "baseline": np.asarray(baseline_preds, dtype=float),
            "champion": np.asarray(champion_preds, dtype=float),
        }
    )
    daily = (
        frame.groupby("crime_date", as_index=False)
        .sum(numeric_only=True)
        .sort_values("crime_date")
    )
    if len(daily) > 120:
        daily = daily.tail(120)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        daily["crime_date"],
        daily["actual"],
        label="Actual Daily Incidents",
        color="black",
        linewidth=1.8,
    )
    ax.plot(
        daily["crime_date"],
        daily["baseline"],
        label=f"Baseline ({baseline_name})",
        color="gray",
        linestyle="--",
        linewidth=1.3,
    )
    ax.plot(
        daily["crime_date"],
        daily["champion"],
        label=f"Champion ({champion_name})",
        color="teal",
        linewidth=1.6,
    )
    ax.set_title("Held-Out Test Window: Daily Citywide Summed Forecast vs Actuals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Incidents (Sum Across Districts)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_feature_importances(
    importances: dict[str, float],
    *,
    model_name: str,
    out_path: str | Path,
) -> Path:
    """Render horizontal bar chart of model feature importances."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="navy")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Forecasting Feature Importances ({model_name})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target

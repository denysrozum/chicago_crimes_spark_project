"""Forecasting baselines, Spark MLlib & XGBoost trainers, and MLflow experiment tracking."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

from chicago_crimes.core.logging import get_logger
from chicago_crimes.ml.dataset import ForecastingDataset, prepare_forecasting_dataset
from chicago_crimes.ml.evaluate import (
    ModelEvaluationResult,
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importances,
    plot_model_comparison,
)

logger = get_logger("ml.models")


@dataclass(frozen=True)
class ForecastingRunBundle:
    """Complete result bundle returned by `run_forecasting_experiment`."""

    output_dir: Path
    report_path: Path
    tracking_uri: str
    experiment_name: str
    champion_model: str
    results: list[ModelEvaluationResult]
    charts: dict[str, str]
    report: dict[str, Any]


def _ensure_mlflow_experiment(tracking_dir: Path, experiment_name: str) -> str:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = (tracking_dir / "mlflow.db").resolve()
    artifact_dir = (tracking_dir / "mlartifacts").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment_name)
    if existing is None:
        client.create_experiment(experiment_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def _log_model_run(
    *,
    model_name: str,
    framework: str,
    params: dict[str, object],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    feature_importances: dict[str, float],
    dataset_summary: dict[str, object],
    xgb_model: xgb.XGBRegressor | None = None,
    chart_paths: list[Path] | None = None,
) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("framework", framework)
        mlflow.set_tag("task", "daily_district_crime_forecasting")
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_param("train_rows", int(dataset_summary["train_rows"]))  # type: ignore[arg-type]
        mlflow.log_param("val_rows", int(dataset_summary["val_rows"]))  # type: ignore[arg-type]
        mlflow.log_param("test_rows", int(dataset_summary["test_rows"]))  # type: ignore[arg-type]
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        if feature_importances:
            mlflow.log_dict(feature_importances, "feature_importances.json")
        if xgb_model is not None:
            mlflow.xgboost.log_model(xgb_model, name="model", pip_requirements=["xgboost"])
        for chart in chart_paths or []:
            if chart.exists():
                mlflow.log_artifact(str(chart), artifact_path="charts")
        return str(run.info.run_id)


def _train_spark_regressor(
    spark: SparkSession,
    dataset: ForecastingDataset,
    *,
    estimator_kind: str,
    max_depth: int,
    num_iters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    feature_cols = list(dataset.feature_columns)
    target_col = dataset.target_column

    def _to_spark_df(pdf: pd.DataFrame) -> Any:
        indexed = pdf[[*feature_cols, target_col]].copy()
        for col in [*feature_cols, target_col]:
            indexed[col] = indexed[col].astype(float)
        indexed["_row_idx"] = np.arange(len(indexed), dtype=np.int64)
        return spark.createDataFrame(indexed)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    train_sdf = (
        assembler.transform(_to_spark_df(dataset.train_df))
        .select("_row_idx", "features", target_col)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    train_sdf.count()
    val_sdf = assembler.transform(_to_spark_df(dataset.val_df)).select("_row_idx", "features")
    test_sdf = assembler.transform(_to_spark_df(dataset.test_df)).select("_row_idx", "features")

    try:
        if estimator_kind == "rf":
            estimator = RandomForestRegressor(
                featuresCol="features",
                labelCol=target_col,
                numTrees=num_iters,
                maxDepth=max_depth,
                seed=42,
            )
        elif estimator_kind == "gbt":
            estimator = GBTRegressor(
                featuresCol="features",
                labelCol=target_col,
                maxIter=num_iters,
                maxDepth=max_depth,
                seed=42,
            )
        else:
            raise ValueError(f"Unsupported Spark estimator_kind '{estimator_kind}'")

        model = estimator.fit(train_sdf)
        val_pdf = (
            model.transform(val_sdf)
            .select("_row_idx", "prediction")
            .orderBy("_row_idx")
            .toPandas()
        )
        test_pdf = (
            model.transform(test_sdf)
            .select("_row_idx", "prediction")
            .orderBy("_row_idx")
            .toPandas()
        )
        importances = {
            col: round(float(val), 6)
            for col, val in zip(feature_cols, model.featureImportances.toArray(), strict=True)
        }
    finally:
        train_sdf.unpersist()
    return (
        val_pdf["prediction"].to_numpy(dtype=float),
        test_pdf["prediction"].to_numpy(dtype=float),
        importances,
    )


def run_forecasting_experiment(
    spark: SparkSession,
    *,
    gold_dir: str | Path | None = "data/gold",
    silver_input: str | Path | None = None,
    table_format: str = "auto",
    output_dir: str | Path = "artifacts/ml",
    experiment_name: str = "chicago_crimes_daily_forecast",
    min_district_days: int = 30,
    max_history_days: int | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    include_spark_models: bool = True,
) -> ForecastingRunBundle:
    """Train baselines, Spark MLlib regressors, and XGBoost with MLflow tracking."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = _ensure_mlflow_experiment(out_root, experiment_name)

    dataset = prepare_forecasting_dataset(
        spark,
        gold_dir=gold_dir,
        silver_input=silver_input,
        table_format=table_format,
        min_district_days=min_district_days,
        max_history_days=max_history_days,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    ds_summary = dataset.summary_dict()
    y_val = dataset.val_df[dataset.target_column].to_numpy(dtype=float)
    y_test = dataset.test_df[dataset.target_column].to_numpy(dtype=float)

    results: list[ModelEvaluationResult] = []
    test_predictions: dict[str, np.ndarray] = {}

    def _evaluate_and_log(
        *,
        model_name: str,
        framework: str,
        params: dict[str, Any],
        val_pred: np.ndarray,
        test_pred: np.ndarray,
        feature_importances: dict[str, float],
    ) -> None:
        val_m = compute_regression_metrics(y_val, val_pred)
        test_m = compute_regression_metrics(y_test, test_pred)
        results.append(
            ModelEvaluationResult(
                model_name=model_name,
                framework=framework,
                params=params,
                val_metrics=val_m,
                test_metrics=test_m,
                feature_importances=feature_importances,
                mlflow_run_id=_log_model_run(
                    model_name=model_name,
                    framework=framework,
                    params=params,
                    val_metrics=val_m,
                    test_metrics=test_m,
                    feature_importances=feature_importances,
                    dataset_summary=ds_summary,
                ),
            )
        )
        test_predictions[model_name] = test_pred

    # 1. Seasonal Naive Baseline (t - 7 days same district)
    _evaluate_and_log(
        model_name="seasonal_naive_lag7",
        framework="baseline",
        params={"rule": "incident_count(t - 7)"},
        val_pred=dataset.val_df["lag_7"].to_numpy(dtype=float),
        test_pred=dataset.test_df["lag_7"].to_numpy(dtype=float),
        feature_importances={"lag_7": 1.0},
    )

    # 2. 7-Day Moving Average Baseline
    _evaluate_and_log(
        model_name="rolling_mean_7_baseline",
        framework="baseline",
        params={"rule": "mean(incident_count[t-7..t-1])"},
        val_pred=dataset.val_df["rolling_mean_7"].to_numpy(dtype=float),
        test_pred=dataset.test_df["rolling_mean_7"].to_numpy(dtype=float),
        feature_importances={"rolling_mean_7": 1.0},
    )

    # 3. Spark MLlib Baselines (Random Forest & GBT)
    if include_spark_models:
        rf_params = {"numTrees": 25, "maxDepth": 6, "seed": 42}
        val_rf, test_rf, rf_imp = _train_spark_regressor(
            spark, dataset, estimator_kind="rf", max_depth=6, num_iters=25
        )
        _evaluate_and_log(
            model_name="spark_mllib_rf",
            framework="spark_mllib",
            params=rf_params,
            val_pred=val_rf,
            test_pred=test_rf,
            feature_importances=rf_imp,
        )

        gbt_params = {"maxIter": 30, "maxDepth": 5, "seed": 42}
        val_gbt, test_gbt, gbt_imp = _train_spark_regressor(
            spark, dataset, estimator_kind="gbt", max_depth=5, num_iters=30
        )
        _evaluate_and_log(
            model_name="spark_mllib_gbt",
            framework="spark_mllib",
            params=gbt_params,
            val_pred=val_gbt,
            test_pred=test_gbt,
            feature_importances=gbt_imp,
        )

    # 4. XGBoost Regressor selected using validation split
    feature_cols = list(dataset.feature_columns)
    X_train = dataset.train_df[feature_cols]
    y_train = dataset.train_df[dataset.target_column].to_numpy(dtype=float)
    X_val = dataset.val_df[feature_cols]
    X_test = dataset.test_df[feature_cols]

    xgb_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
    }
    xgb_model = xgb.XGBRegressor(**xgb_params, n_jobs=2, early_stopping_rounds=20)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_xgb = xgb_model.predict(X_val)
    test_xgb = xgb_model.predict(X_test)
    xgb_imp = {
        col: round(float(val), 6)
        for col, val in zip(feature_cols, xgb_model.feature_importances_, strict=True)
    }
    xgb_val_m = compute_regression_metrics(y_val, val_xgb)
    xgb_test_m = compute_regression_metrics(y_test, test_xgb)

    # Select champion strictly on validation RMSE (never test set)
    preliminary_by_val = sorted(
        [
            *results,
            ModelEvaluationResult(
                model_name="xgboost_regressor",
                framework="xgboost",
                params=xgb_params,
                val_metrics=xgb_val_m,
                test_metrics=xgb_test_m,
                feature_importances=xgb_imp,
            ),
        ],
        key=lambda r: r.val_metrics["rmse"],
    )
    champion_stub = preliminary_by_val[0]
    test_predictions["xgboost_regressor"] = test_xgb

    charts_dir = out_root / "charts"
    comp_chart = plot_model_comparison(
        [
            *results,
            ModelEvaluationResult(
                model_name="xgboost_regressor",
                framework="xgboost",
                params=xgb_params,
                val_metrics=xgb_val_m,
                test_metrics=xgb_test_m,
                feature_importances=xgb_imp,
            ),
        ],
        charts_dir / "model_comparison.png",
    )
    pred_chart = plot_actual_vs_predicted(
        dataset.test_df,
        test_predictions["seasonal_naive_lag7"],
        test_predictions[champion_stub.model_name],
        baseline_name="seasonal_naive_lag7",
        champion_name=champion_stub.model_name,
        out_path=charts_dir / "actual_vs_predicted_test.png",
    )
    imp_chart = plot_feature_importances(
        champion_stub.feature_importances or xgb_imp,
        model_name=champion_stub.model_name,
        out_path=charts_dir / "feature_importance.png",
    )

    xgb_run_id = _log_model_run(
        model_name="xgboost_regressor",
        framework="xgboost",
        params=xgb_params,
        val_metrics=xgb_val_m,
        test_metrics=xgb_test_m,
        feature_importances=xgb_imp,
        dataset_summary=ds_summary,
        xgb_model=xgb_model,
        chart_paths=[comp_chart, pred_chart, imp_chart],
    )
    results.append(
        ModelEvaluationResult(
            model_name="xgboost_regressor",
            framework="xgboost",
            params=xgb_params,
            val_metrics=xgb_val_m,
            test_metrics=xgb_test_m,
            feature_importances=xgb_imp,
            mlflow_run_id=xgb_run_id,
        )
    )

    champion = min(results, key=lambda r: r.val_metrics["rmse"])
    naive_res = next(r for r in results if r.model_name == "seasonal_naive_lag7")
    roll_res = next(r for r in results if r.model_name == "rolling_mean_7_baseline")

    charts = {
        "model_comparison": str(comp_chart),
        "actual_vs_predicted_test": str(pred_chart),
        "feature_importance": str(imp_chart),
    }
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mlflow_tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "dataset": ds_summary,
        "champion_selection_criterion": "lowest validation RMSE",
        "champion_model": champion.model_name,
        "champion_test_metrics": champion.test_metrics,
        "champion_beats_seasonal_naive": bool(
            champion.test_metrics["rmse"] < naive_res.test_metrics["rmse"]
        ),
        "champion_beats_rolling_mean": bool(
            champion.test_metrics["rmse"] < roll_res.test_metrics["rmse"]
        ),
        "models": [r.to_dict() for r in results],
        "charts": charts,
    }
    report_path = out_root / "ml_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "ML forecasting complete (champion=%s, test_rmse=%.4f, mlflow_uri=%s)",
        champion.model_name,
        champion.test_metrics["rmse"],
        tracking_uri,
    )

    return ForecastingRunBundle(
        output_dir=out_root,
        report_path=report_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        champion_model=champion.model_name,
        results=results,
        charts=charts,
        report=report,
    )

"""Leakage-free daily district crime forecasting dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from chicago_crimes.processing.gold import compute_gold_dataframes
from chicago_crimes.storage.lakehouse import read_table

FORECAST_FEATURE_COLUMNS: tuple[str, ...] = (
    "district_id",
    "year",
    "month",
    "day_of_month",
    "day_of_week_num",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "arrest_rate_lag_1",
    "arrest_rate_roll_7",
)

FORECAST_TARGET_COLUMN: str = "incident_count"


@dataclass(frozen=True)
class ForecastingDataset:
    """Chronologically partitioned daily district crime forecasting splits."""

    full_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    districts: list[int]
    train_date_range: tuple[str, str]
    val_date_range: tuple[str, str]
    test_date_range: tuple[str, str]

    def summary_dict(self) -> dict[str, object]:
        """Return dataset split metadata for manifests and MLflow params."""
        return {
            "total_rows": int(len(self.full_df)),
            "train_rows": int(len(self.train_df)),
            "val_rows": int(len(self.val_df)),
            "test_rows": int(len(self.test_df)),
            "district_count": len(self.districts),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "train_date_range": list(self.train_date_range),
            "val_date_range": list(self.val_date_range),
            "test_date_range": list(self.test_date_range),
        }


def load_daily_district_source(
    spark: SparkSession,
    *,
    gold_dir: str | Path | None = None,
    silver_input: str | Path | None = None,
    table_format: str = "auto",
) -> DataFrame:
    """Load `daily_district_counts` from Gold Parquet or derive it from Silver."""
    if gold_dir is not None:
        candidate = Path(gold_dir) / "daily_district_counts"
        if candidate.exists():
            return spark.read.parquet(str(candidate))

    if silver_input is None:
        raise FileNotFoundError(
            "Neither Gold `daily_district_counts` nor fallback `silver_input` was provided/found."
        )
    silver_df = read_table(spark, silver_input, table_format=table_format)
    total = int(silver_df.count())
    return compute_gold_dataframes(silver_df, total_incidents=total)["daily_district_counts"]


def build_forecasting_features(
    daily_district_pd: pd.DataFrame,
    *,
    min_district_days: int = 30,
    max_history_days: int | None = None,
    min_calendar_coverage: float = 0.5,
) -> pd.DataFrame:
    """Construct causal calendar lag/rolling features per district without future leakage."""
    required = {"crime_date", "District", "incident_count", "arrest_rate_percent"}
    missing = required - set(daily_district_pd.columns)
    if missing:
        raise ValueError(f"Forecasting source missing required columns: {sorted(missing)}")

    work = daily_district_pd.copy()
    work["crime_date"] = pd.to_datetime(work["crime_date"]).dt.normalize()
    work["District"] = pd.to_numeric(work["District"], errors="coerce")
    work = work.dropna(subset=["crime_date", "District"]).copy()
    work["district_id"] = work["District"].astype(int)

    stats = (
        work.groupby("district_id")["crime_date"]
        .agg(
            unique_days="nunique",
            min_date="min",
            max_date="max",
        )
    )
    stats["span_days"] = (stats["max_date"] - stats["min_date"]).dt.days + 1
    stats["coverage"] = stats["unique_days"] / stats["span_days"].clip(lower=1)

    base_eligible = stats[stats["unique_days"] >= min_district_days]
    if base_eligible.empty:
        raise ValueError(
            f"No districts have at least {min_district_days} distinct days in forecasting source."
        )
    dense_eligible = base_eligible[base_eligible["coverage"] >= min_calendar_coverage]
    eligible_districts = sorted(
        (dense_eligible if not dense_eligible.empty else base_eligible).index.tolist()
    )
    work = work[work["district_id"].isin(eligible_districts)].copy()

    if max_history_days is not None and max_history_days > 0:
        max_dt = work["crime_date"].max()
        cutoff = max_dt - pd.Timedelta(days=int(max_history_days))
        work = work[work["crime_date"] >= cutoff].copy()

    reindexed_parts: list[pd.DataFrame] = []
    for district_id, group in work.groupby("district_id", sort=True):
        full_dates = pd.date_range(group["crime_date"].min(), group["crime_date"].max(), freq="D")
        agg = (
            group.groupby("crime_date", as_index=True)
            .agg({"incident_count": "sum", "arrest_rate_percent": "mean"})
            .reindex(full_dates, fill_value=0.0)
        )
        agg.index.name = "crime_date"
        part = agg.reset_index()
        part["district_id"] = int(district_id)
        reindexed_parts.append(part)
    work = pd.concat(reindexed_parts, ignore_index=True).sort_values(
        ["district_id", "crime_date"]
    ).reset_index(drop=True)

    # Strictly causal shift(>=1) within each district boundary so row t uses only <= t-1
    grp = work.groupby("district_id", sort=False)
    work["lag_1"] = grp["incident_count"].shift(1)
    work["lag_7"] = grp["incident_count"].shift(7)
    work["rolling_mean_7"] = grp["incident_count"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )
    work["rolling_std_7"] = (
        grp["incident_count"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=3).std())
        .fillna(0.0)
    )
    work["rolling_mean_14"] = grp["incident_count"].transform(
        lambda s: s.shift(1).rolling(window=14, min_periods=5).mean()
    )
    work["arrest_rate_lag_1"] = grp["arrest_rate_percent"].shift(1)
    work["arrest_rate_roll_7"] = grp["arrest_rate_percent"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=3).mean()
    )

    work["year"] = work["crime_date"].dt.year.astype(int)
    work["month"] = work["crime_date"].dt.month.astype(int)
    work["day_of_month"] = work["crime_date"].dt.day.astype(int)
    # Match Spark dayofweek convention (1=Sunday .. 7=Saturday) across all calendar dates
    work["day_of_week_num"] = (((work["crime_date"].dt.dayofweek + 1) % 7) + 1).astype(int)
    work["is_weekend"] = work["day_of_week_num"].isin([1, 7]).astype(int)

    model_cols = ["crime_date", *FORECAST_FEATURE_COLUMNS, FORECAST_TARGET_COLUMN]
    cleaned = (
        work[model_cols]
        .dropna()
        .sort_values(["crime_date", "district_id"])
        .reset_index(drop=True)
    )
    return cleaned


def split_chronologically(
    featured_df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> ForecastingDataset:
    """Split forecasting features strictly by unique sorted crime_date cutoffs."""
    if not (0.3 <= train_ratio < 0.9) or not (0.05 <= val_ratio < 0.5):
        raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}")
    if train_ratio + val_ratio >= 0.98:
        raise ValueError("train_ratio + val_ratio must leave room for a held-out test split.")

    unique_dates = pd.Series(sorted(featured_df["crime_date"].unique()))
    n_dates = len(unique_dates)
    if n_dates < 6:
        raise ValueError(f"Need at least 6 distinct dates for train/val/test split, got {n_dates}")

    train_end_idx = max(2, int(n_dates * train_ratio))
    val_end_idx = max(train_end_idx + 1, int(n_dates * (train_ratio + val_ratio)))
    if val_end_idx >= n_dates:
        val_end_idx = n_dates - 1
        train_end_idx = min(train_end_idx, val_end_idx - 1)

    train_cutoff = unique_dates.iloc[train_end_idx - 1]
    val_cutoff = unique_dates.iloc[val_end_idx - 1]

    train_df = featured_df[featured_df["crime_date"] <= train_cutoff].reset_index(drop=True)
    val_df = featured_df[
        (featured_df["crime_date"] > train_cutoff) & (featured_df["crime_date"] <= val_cutoff)
    ].reset_index(drop=True)
    test_df = featured_df[featured_df["crime_date"] > val_cutoff].reset_index(drop=True)

    dataset = ForecastingDataset(
        full_df=featured_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=FORECAST_FEATURE_COLUMNS,
        target_column=FORECAST_TARGET_COLUMN,
        districts=sorted(featured_df["district_id"].unique().astype(int).tolist()),
        train_date_range=(
            train_df["crime_date"].min().strftime("%Y-%m-%d"),
            train_df["crime_date"].max().strftime("%Y-%m-%d"),
        ),
        val_date_range=(
            val_df["crime_date"].min().strftime("%Y-%m-%d"),
            val_df["crime_date"].max().strftime("%Y-%m-%d"),
        ),
        test_date_range=(
            test_df["crime_date"].min().strftime("%Y-%m-%d"),
            test_df["crime_date"].max().strftime("%Y-%m-%d"),
        ),
    )
    assert_no_temporal_leakage(dataset)
    return dataset


def assert_no_temporal_leakage(dataset: ForecastingDataset) -> None:
    """Verify strict chronological separation across train, validation, and test splits."""
    splits = [("train", dataset.train_df), ("val", dataset.val_df), ("test", dataset.test_df)]
    for name, part in splits:
        if part.empty:
            raise ValueError(f"Forecasting split '{name}' is empty.")

    train_max = pd.to_datetime(dataset.train_df["crime_date"]).max()
    val_min = pd.to_datetime(dataset.val_df["crime_date"]).min()
    val_max = pd.to_datetime(dataset.val_df["crime_date"]).max()
    test_min = pd.to_datetime(dataset.test_df["crime_date"]).min()

    if not (train_max < val_min <= val_max < test_min):
        raise ValueError(
            "Temporal leakage detected in forecasting split: "
            f"train_max={train_max}, val_min={val_min}, val_max={val_max}, test_min={test_min}"
        )


def prepare_forecasting_dataset(
    spark: SparkSession,
    *,
    gold_dir: str | Path | None = None,
    silver_input: str | Path | None = None,
    table_format: str = "auto",
    min_district_days: int = 30,
    max_history_days: int | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> ForecastingDataset:
    """Load daily district Gold counts, build causal features, and split chronologically."""
    spark_df = load_daily_district_source(
        spark,
        gold_dir=gold_dir,
        silver_input=silver_input,
        table_format=table_format,
    )
    selected = spark_df.select(
        F.col("crime_date"),
        F.col("District"),
        F.col("day_of_week_num"),
        F.col("incident_count"),
        F.col("arrest_rate_percent"),
    ).toPandas()
    featured = build_forecasting_features(
        selected,
        min_district_days=min_district_days,
        max_history_days=max_history_days,
    )
    return split_chronologically(featured, train_ratio=train_ratio, val_ratio=val_ratio)

"""Generate publication-grade analytical and ML evaluation figures for README.md."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb

from chicago_crimes.ml.dataset import build_forecasting_features, split_chronologically

# Consistent publication color palette (colorblind-safe Okabe-Ito / Slate inspired)
COLOR_NAVY = "#1b365d"
COLOR_TEAL = "#008080"
COLOR_AMBER = "#d95f02"
COLOR_SLATE = "#637381"
COLOR_LIGHT_BG = "#f8fafc"
COLOR_GRID = "#cbd5e1"
COLOR_RED_ACCENT = "#b91c1c"


def _apply_base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
            "axes.edgecolor": "#94a3b8",
            "axes.linewidth": 0.9,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
        }
    )


def _read_parquet_dir(path: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(path / "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No Parquet part files found in {path}")
    return ds.dataset(files, format="parquet").to_table().to_pandas()


def render_historical_trends(gold_dir: Path, out_path: Path) -> dict[str, object]:
    """Render Figure 1: Historical Chicago Crime Volume & Arrest Rate (2001-2025)."""
    monthly = _read_parquet_dir(gold_dir / "monthly_crime_summary").sort_values("year_month")
    monthly["dt"] = pd.to_datetime(monthly["year_month"] + "-01")

    # Separate complete calendar years (2001-2025) for annual summary bars,
    # while keeping full monthly series through 2025-12 for continuous monthly trend.
    monthly_complete = monthly[(monthly["year"] >= 2001) & (monthly["year"] <= 2025)].copy()
    monthly_complete["rolling_12m"] = (
        monthly_complete["incident_count"].rolling(window=12, min_periods=6).mean()
    )

    annual = (
        monthly_complete.groupby("year", as_index=False)
        .agg(
            incidents=("incident_count", "sum"),
            arrests=("arrest_count", "sum"),
            domestic=("domestic_count", "sum"),
        )
        .sort_values("year")
    )
    annual["arrest_rate_pct"] = annual["arrests"] / annual["incidents"] * 100.0
    annual["domestic_rate_pct"] = annual["domestic"] / annual["incidents"] * 100.0

    total_all_rows = int(monthly["incident_count"].sum())
    max_ym = str(monthly["year_month"].max())

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.4),
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.24},
    )

    # Panel A: Monthly Reported Incidents + 12-Month Rolling Mean
    ax1.fill_between(
        monthly_complete["dt"],
        monthly_complete["incident_count"] / 1_000.0,
        color="#93c5fd",
        alpha=0.30,
    )
    ax1.plot(
        monthly_complete["dt"],
        monthly_complete["incident_count"] / 1_000.0,
        color="#3b82f6",
        linewidth=1.1,
        alpha=0.75,
        label="Monthly Reported Incidents",
    )
    ax1.plot(
        monthly_complete["dt"],
        monthly_complete["rolling_12m"] / 1_000.0,
        color=COLOR_NAVY,
        linewidth=2.4,
        label="12-Month Rolling Average",
    )

    peak_row = monthly_complete.loc[monthly_complete["incident_count"].idxmax()]
    min_row = monthly_complete.loc[monthly_complete["incident_count"].idxmin()]
    ax1.annotate(
        f"Peak: {peak_row['year_month']} ({int(peak_row['incident_count']):,})",
        xy=(peak_row["dt"], peak_row["incident_count"] / 1_000.0),
        xytext=(pd.Timestamp("2004-06-01"), 45.2),
        arrowprops={"arrowstyle": "->", "color": COLOR_NAVY, "lw": 1.1},
        fontsize=9,
        fontweight="bold",
        color=COLOR_NAVY,
    )
    ax1.annotate(
        f"COVID-19 Low: {min_row['year_month']} ({int(min_row['incident_count']):,})",
        xy=(min_row["dt"], min_row["incident_count"] / 1_000.0),
        xytext=(pd.Timestamp("2009-06-01"), 9.2),
        arrowprops={
            "arrowstyle": "->",
            "color": COLOR_RED_ACCENT,
            "lw": 1.1,
            "connectionstyle": "arc3,rad=0.15",
        },
        fontsize=9,
        fontweight="bold",
        color=COLOR_RED_ACCENT,
    )

    ax1.set_title("A. Monthly Reported Crime Volume & Seasonal Cycle (2001–2025)")
    ax1.set_xlabel("Month (America/Chicago Local Calendar)")
    ax1.set_ylabel("Reported Incidents (Thousands / Month)")
    ax1.set_ylim(0, 50)
    ax1.xaxis.set_major_locator(mdates.YearLocator(4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(True, linestyle="--", color=COLOR_GRID, alpha=0.65)
    ax1.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    # Panel B: Annual Incident Totals (Bars) vs Arrest Rate % & Domestic Share % (Lines)
    ax2_twin = ax2.twinx()
    ax2.bar(
        annual["year"],
        annual["incidents"] / 1_000.0,
        color="#cbd5e1",
        edgecolor="#64748b",
        linewidth=0.6,
        width=0.72,
        label="Annual Incidents (k)",
        zorder=2,
    )
    line1 = ax2_twin.plot(
        annual["year"],
        annual["arrest_rate_pct"],
        color=COLOR_TEAL,
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        label="Arrest Rate (%)",
        zorder=4,
    )
    line2 = ax2_twin.plot(
        annual["year"],
        annual["domestic_rate_pct"],
        color=COLOR_AMBER,
        marker="s",
        markersize=4.0,
        linewidth=1.9,
        linestyle="--",
        label="Domestic Incident Share (%)",
        zorder=4,
    )

    ax2.set_title("B. Annual Volume vs. Arrest & Domestic Rates (2001–2025)")
    ax2.set_xlabel("Calendar Year")
    ax2.set_ylabel("Annual Reported Incidents (Thousands)", color="#334155")
    ax2_twin.set_ylabel("Rate (% of Reported Incidents)", color=COLOR_NAVY)
    ax2.set_ylim(0, 540)
    ax2_twin.set_ylim(0, 36)
    ax2.set_xticks(range(2001, 2026, 4))
    ax2.grid(True, axis="y", linestyle="--", color=COLOR_GRID, alpha=0.65)

    bars_handle, bars_label = ax2.get_legend_handles_labels()
    lines_handles = line1 + line2
    lines_labels = [str(h.get_label()) for h in lines_handles]
    ax2_twin.legend(
        bars_handle + lines_handles,
        bars_label + lines_labels,
        loc="upper right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
    )

    fig.suptitle(
        "Chicago Historical Crime Dynamics "
        f"(Gold Monthly Summary — {total_all_rows:,} Total Verified Incidents Through {max_ym})",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.008,
        "Source: Gold `monthly_crime_summary` derived from validated Silver "
        "(`data/silver/chicago_crimes`). "
        "Partial year 2026 excluded from annual bar totals for calendar comparability.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "total_incidents": total_all_rows,
        "peak_month": str(peak_row["year_month"]),
        "peak_month_incidents": int(peak_row["incident_count"]),
        "2001_incidents": int(annual.loc[annual["year"] == 2001, "incidents"].iloc[0]),
        "2025_incidents": int(annual.loc[annual["year"] == 2025, "incidents"].iloc[0]),
        "2001_arrest_rate_pct": round(
            float(annual.loc[annual["year"] == 2001, "arrest_rate_pct"].iloc[0]), 2
        ),
        "2025_arrest_rate_pct": round(
            float(annual.loc[annual["year"] == 2025, "arrest_rate_pct"].iloc[0]), 2
        ),
    }


def render_categories_and_districts(gold_dir: Path, out_path: Path) -> dict[str, object]:
    """Render Figure 2: Top Offense Categories & Police District Distribution."""
    cats = _read_parquet_dir(gold_dir / "category_distribution").sort_values(
        "incident_count", ascending=False
    )
    top_cats = cats.head(10).iloc[::-1].copy()

    dist_comm = _read_parquet_dir(gold_dir / "district_community_summary")
    dist = (
        dist_comm.groupby("District", as_index=False)
        .agg(
            incident_count=("incident_count", "sum"),
            arrest_count=("arrest_count", "sum"),
            domestic_count=("domestic_count", "sum"),
        )
        .sort_values("incident_count", ascending=False)
    )
    # Show top 15 active police districts
    top_dist = dist.head(15).iloc[::-1].copy()
    top_dist["arrest_rate_pct"] = top_dist["arrest_count"] / top_dist["incident_count"] * 100.0
    top_dist["district_label"] = top_dist["District"].apply(lambda d: f"District {int(d):02d}")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.8),
        gridspec_kw={"width_ratios": [1.14, 1.0], "wspace": 0.34},
    )

    # Panel A: Top 10 Offense Categories (Horizontal Bars + Arrest Rate Annotation)
    y_pos = np.arange(len(top_cats))
    bars1 = ax1.barh(
        y_pos,
        top_cats["incident_count"] / 1_000_000.0,
        color=COLOR_NAVY,
        alpha=0.88,
        height=0.68,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_cats["Primary Type"], fontsize=9.2, fontweight="bold")
    ax1.set_xlabel("Reported Incidents (Millions, 2001–2026)")
    ax1.set_title("A. Top 10 Offense Categories (Share & Arrest Rate)")
    ax1.set_xlim(0, 2.55)
    ax1.grid(True, axis="x", linestyle="--", color=COLOR_GRID, alpha=0.65)

    for bar, (_, row) in zip(bars1, top_cats.iterrows(), strict=True):
        width = bar.get_width()
        share_pct = row["citywide_share_percent"]
        arrest_pct = row["arrest_rate_percent"]
        ax1.text(
            width + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}M ({share_pct:.1f}% | Arrest: {arrest_pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#1e293b",
        )

    # Panel B: Top 15 Police Districts by Incident Volume (colored by Arrest Rate %)
    y_dist = np.arange(len(top_dist))
    norm = plt.Normalize(
        vmin=float(top_dist["arrest_rate_pct"].min() - 2.0),
        vmax=float(top_dist["arrest_rate_pct"].max() + 2.0),
    )
    cmap = plt.cm.viridis
    colors = cmap(norm(top_dist["arrest_rate_pct"].to_numpy()))

    bars2 = ax2.barh(
        y_dist,
        top_dist["incident_count"] / 1_000.0,
        color=colors,
        height=0.68,
        edgecolor="#334155",
        linewidth=0.4,
    )
    ax2.set_yticks(y_dist)
    ax2.set_yticklabels(top_dist["district_label"], fontsize=9.2)
    ax2.set_xlabel("Reported Incidents (Thousands, Valid District + Community Area)")
    ax2.set_title("B. Top 15 Police Districts by Volume & Arrest Rate")
    ax2.set_xlim(0, 680)
    ax2.grid(True, axis="x", linestyle="--", color=COLOR_GRID, alpha=0.65)

    for bar, (_, row) in zip(bars2, top_dist.iterrows(), strict=True):
        width = bar.get_width()
        ax2.text(
            width + 8.0,
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}k (Arrest: {row['arrest_rate_pct']:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#1e293b",
        )

    fig.suptitle(
        "Offense Category Concentration & Police District Profile (Gold Aggregations)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Source: Gold `category_distribution` (34 offense categories) and "
        "`district_community_summary` "
        "(raw incident counts; not population-normalized).",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.15, right=0.96)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    top_cat_row = cats.iloc[0]
    top_dist_row = dist.iloc[0]
    return {
        "top_category": str(top_cat_row["Primary Type"]),
        "top_category_count": int(top_cat_row["incident_count"]),
        "top_category_share_pct": round(float(top_cat_row["citywide_share_percent"]), 2),
        "top_district": int(top_dist_row["District"]),
        "top_district_count": int(top_dist_row["incident_count"]),
    }


def render_ml_evaluation(gold_dir: Path, ml_report_path: Path, out_path: Path) -> dict[str, object]:
    """Render Figure 3: ML Forecasting Model Comparison & Held-Out Test Trajectory."""
    report = json.loads(ml_report_path.read_text(encoding="utf-8"))
    models_list = report["models"]

    # Reconstruct exact test split and champion XGBoost predictions deterministically
    daily_dist = _read_parquet_dir(gold_dir / "daily_district_counts")
    featured = build_forecasting_features(daily_dist, min_district_days=30)
    dataset = split_chronologically(featured, train_ratio=0.70, val_ratio=0.15)

    feature_cols = list(dataset.feature_columns)
    X_train = dataset.train_df[feature_cols]
    y_train = dataset.train_df[dataset.target_column].to_numpy(dtype=float)
    X_val = dataset.val_df[feature_cols]
    y_val = dataset.val_df[dataset.target_column].to_numpy(dtype=float)
    X_test = dataset.test_df[feature_cols]

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=2,
        early_stopping_rounds=20,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_test_pred = xgb_model.predict(X_test)

    test_frame = pd.DataFrame(
        {
            "crime_date": pd.to_datetime(dataset.test_df["crime_date"]),
            "actual": dataset.test_df["incident_count"].to_numpy(dtype=float),
            "naive_lag7": dataset.test_df["lag_7"].to_numpy(dtype=float),
            "xgb_pred": xgb_test_pred,
        }
    )
    daily_sum = (
        test_frame.groupby("crime_date", as_index=False)
        .sum(numeric_only=True)
        .sort_values("crime_date")
    )
    # Show the last 120 complete calendar days through 2025-12-31
    window_df = daily_sum[daily_sum["crime_date"] <= pd.Timestamp("2025-12-31")].tail(120).copy()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.4),
        gridspec_kw={"width_ratios": [1.0, 1.22], "wspace": 0.26},
    )

    # Panel A: 5-Model Comparison (Val RMSE, Test RMSE, Test MAE)
    display_labels = {
        "seasonal_naive_lag7": "Seasonal Naive\n(lag_7)",
        "rolling_mean_7_baseline": "7-Day Moving\nAverage",
        "spark_mllib_rf": "Spark MLlib\nRandom Forest",
        "spark_mllib_gbt": "Spark MLlib\nGBT",
        "xgboost_regressor": "XGBoost\n(Champion)",
    }
    labels = [display_labels.get(m["model_name"], m["model_name"]) for m in models_list]
    val_rmses = [float(m["val_metrics"]["rmse"]) for m in models_list]
    test_rmses = [float(m["test_metrics"]["rmse"]) for m in models_list]
    test_maes = [float(m["test_metrics"]["mae"]) for m in models_list]

    x = np.arange(len(labels))
    w = 0.26
    b_val = ax1.bar(x - w, val_rmses, width=w, color="#94a3b8", label="Validation RMSE")
    b_tst = ax1.bar(x, test_rmses, width=w, color=COLOR_NAVY, label="Held-Out Test RMSE")
    b_mae = ax1.bar(x + w, test_maes, width=w, color=COLOR_TEAL, label="Held-Out Test MAE")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8)
    ax1.set_ylabel("Error (Daily Incidents / District)")
    ax1.set_title("A. Validation & Held-Out Test Error Across 5 Models")
    ax1.set_ylim(0, 11.2)
    ax1.grid(True, axis="y", linestyle="--", color=COLOR_GRID, alpha=0.65)
    ax1.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    for bars in (b_val, b_tst, b_mae):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.12,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold" if bars is b_tst else "normal",
            )

    # Panel B: Last 120 Days of Held-Out Test Split (Actual vs Champion XGBoost vs Naive Lag-7)
    ax2.plot(
        window_df["crime_date"],
        window_df["actual"],
        color="#0f172a",
        linewidth=1.8,
        label="Observed Daily Incidents (Summed Across 22 Districts)",
    )
    naive_rmse = models_list[0]["test_metrics"]["rmse"]
    ax2.plot(
        window_df["crime_date"],
        window_df["naive_lag7"],
        color="#94a3b8",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=f"Seasonal Naive (lag_7) — Test RMSE {naive_rmse:.2f}",
    )
    ax2.plot(
        window_df["crime_date"],
        window_df["xgb_pred"],
        color=COLOR_TEAL,
        linewidth=2.0,
        label=(
            f"XGBoost Champion — Test RMSE {report['champion_test_metrics']['rmse']:.2f} "
            f"(R²={report['champion_test_metrics']['r2']:.3f})"
        ),
    )

    ax2.set_title("B. Held-Out Test Forecast vs. Observed Daily Crime (Sep–Dec 2025)")
    ax2.set_xlabel("Test Date (America/Chicago Calendar)")
    ax2.set_ylabel("Daily Incidents Across 22 Active Districts")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.grid(True, linestyle="--", color=COLOR_GRID, alpha=0.65)
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    ds_meta = report["dataset"]
    train_r = f"{ds_meta['train_date_range'][0]}..{ds_meta['train_date_range'][1]}"
    val_r = f"{ds_meta['val_date_range'][0]}..{ds_meta['val_date_range'][1]}"
    test_r = f"{ds_meta['test_date_range'][0]}..{ds_meta['test_date_range'][1]}"
    fig.suptitle(
        "Chronological Daily District Crime Forecasting Evaluation "
        f"({ds_meta['total_rows']:,} District-Days, Test Split {test_r})",
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        f"Source: `artifacts/ml/ml_evaluation_report.json` | "
        f"Train: {train_r} ({ds_meta['train_rows']:,} rows) | "
        f"Val: {val_r} ({ds_meta['val_rows']:,} rows) | "
        f"Test: {test_r} ({ds_meta['test_rows']:,} rows).",
        ha="center",
        fontsize=8.4,
        color="#475569",
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "champion_model": report["champion_model"],
        "champion_test_metrics": report["champion_test_metrics"],
        "dataset": ds_meta,
    }


def main() -> None:
    _apply_base_style()
    root = Path(__file__).resolve().parents[1]
    gold_dir = root / "data" / "gold"
    ml_report = root / "artifacts" / "ml" / "ml_evaluation_report.json"
    assets_dir = root / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "historical_trends": render_historical_trends(
            gold_dir, assets_dir / "historical_crime_trends.png"
        ),
        "categories_and_districts": render_categories_and_districts(
            gold_dir, assets_dir / "crime_categories_and_districts.png"
        ),
        "ml_evaluation": render_ml_evaluation(
            gold_dir, ml_report, assets_dir / "ml_forecasting_evaluation.png"
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

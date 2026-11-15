"""Matplotlib visualization helpers and analytical task chart renderers."""

from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_correlation_matrix(
    corr_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (12, 10),
    decimals: int = 1,
    font_size: int = 9,
) -> Path:
    """Render and save a numeric correlation heatmap."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = corr_df.values
    n = data.shape[0]

    plt.figure(figsize=figsize)
    im = plt.imshow(data, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title(title)

    ticks = np.arange(n)
    plt.xticks(ticks, corr_df.columns, rotation=90)
    plt.yticks(ticks, corr_df.index)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            txt = "NaN" if np.isnan(val) else f"{val:.{decimals}f}"
            text_obj = plt.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color="white",
            )
            text_obj.set_path_effects(
                [path_effects.Stroke(linewidth=1.6, foreground="black"), path_effects.Normal()]
            )

    plt.tight_layout()
    plt.savefig(target, dpi=220)
    plt.close()
    return target


def plot_horizontal_bar(
    pdf: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    top_n: int | None = None,
    color: str = "midnightblue",
    edgecolor: str = "black",
) -> Path:
    """Generic horizontal bar chart with value annotations."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = pdf.head(top_n) if top_n is not None else pdf

    plt.figure(figsize=(12, 8))
    bars = plt.barh(data[label_col].astype(str), data[value_col], color=color, edgecolor=edgecolor)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + max(width * 0.01, 0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            fontsize=10,
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_weekend_vs_weekday(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = float(pdf["count"].sum()) or 1.0
    colors = ["#5dade2", "#f39c12"][: len(pdf)]
    explode = tuple([0.05] + [0.0] * max(0, len(pdf) - 1))

    plt.figure(figsize=(8, 8))
    plt.pie(
        pdf["count"],
        labels=pdf["is_weekend"],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100.0)):,})",
        startangle=140,
        colors=colors,
        explode=explode if len(explode) == len(pdf) else None,
        shadow=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_narcotics_vs_assault(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sizes = np.clip(pdf["ratio"].astype(float) * 100.0, 30.0, 800.0)
    plt.scatter(pdf["assaults"], pdf["narcotics"], s=sizes, alpha=0.6, color="blue")
    for i, txt in enumerate(pdf["District"]):
        plt.annotate(f"D:{txt}", (pdf["assaults"].iloc[i], pdf["narcotics"].iloc[i]))
    plt.title(title)
    plt.xlabel("Assaults Count")
    plt.ylabel("Narcotics Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_vertical_bar(
    pdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    color: str = "rebeccapurple",
    is_percent: bool = False,
) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 6))
    bars = plt.bar(
        pdf[x_col].astype(str),
        pdf[y_col],
        color=color,
        edgecolor="black",
        alpha=0.85,
    )
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.1f}%" if is_percent else f"{int(height):,}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(height * 0.01, 0.2),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_moving_average(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(
        pdf["YearMonth"],
        pdf["count"],
        label="Monthly Count",
        color="lightgray",
        marker=".",
        alpha=0.7,
    )
    plt.plot(
        pdf["YearMonth"],
        pdf["moving_avg_3m"],
        label="3-Month Moving Avg",
        color="red",
        linewidth=2.5,
    )
    plt.title(title)
    plt.ylabel("Crime Count")
    step = max(1, len(pdf) // 20)
    plt.xticks(
        ticks=range(0, len(pdf), step),
        labels=pdf["YearMonth"].iloc[::step],
        rotation=45,
    )
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_domestic_monthly(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = pdf.copy()
    work["MonthName"] = work["MonthNum"].apply(lambda x: calendar.month_name[int(x)])
    plt.figure(figsize=(10, 6))
    plt.plot(
        work["MonthName"],
        work["count"],
        marker="o",
        linestyle="-",
        color="crimson",
        linewidth=2,
        markersize=8,
    )
    plt.fill_between(work["MonthName"], work["count"], color="crimson", alpha=0.1)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Incidents", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_top_crime_per_district(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sub = pdf.head(15)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sub["District"].astype(str), sub["count"], color="teal")
    plt.title(title)
    plt.xlabel("District")
    plt.ylabel("Count")
    for bar, crime in zip(bars, sub["Primary Type"], strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(crime),
            ha="center",
            va="bottom",
            rotation=45,
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_block_share(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sorted_pdf = pdf.sort_values(by="share_percent", ascending=True)
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        sorted_pdf["Block"],
        sorted_pdf["share_percent"],
        color="steelblue",
        edgecolor="black",
    )
    plt.title(title, fontsize=14)
    plt.xlabel("District Share (%)")
    plt.ylabel("Block")
    max_val = float(sorted_pdf["share_percent"].max()) if not sorted_pdf.empty else 1.0
    plt.xlim(0, max(max_val * 1.4, 1.0))
    for bar, count, dist_total, dist in zip(
        bars,
        sorted_pdf["count"],
        sorted_pdf["dist_total"],
        sorted_pdf["District"],
        strict=False,
    ):
        width = bar.get_width()
        label_text = f"{width:.1f}% ({int(count):,} of {int(dist_total):,} in Dist {dist})"
        plt.text(
            width + max(max_val * 0.02, 0.1),
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            fontsize=9,
        )
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_lollipop_share(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    top_pdf = pdf.head(20).reset_index(drop=True)
    plt.figure(figsize=(12, 9))
    plt.hlines(y=top_pdf["Primary Type"], xmin=0, xmax=top_pdf["share"], color="grey", alpha=0.5)
    plt.scatter(
        top_pdf["share"], top_pdf["Primary Type"], color="darkcyan", s=100, edgecolors="black"
    )
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Arrest Share")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for i, row in top_pdf.iterrows():
        plt.text(row["share"] + 0.01, i, f"{row['share'] * 100:.1f}%", va="center")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_grouped_weekday_weekend(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(pdf["District"]))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, pdf["weekday_count"], width, label="Weekday", color="skyblue")
    plt.bar(x + width / 2, pdf["weekend_count"], width, label="Weekend", color="salmon")
    plt.title(title)
    plt.xlabel("District")
    plt.ylabel("Crime Count")
    plt.xticks(x, pdf["District"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_stacked_domestic_streets(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.barh(pdf["street_name"], pdf["non_domestic"], color="lightblue", label="Non-Domestic")
    plt.barh(
        pdf["street_name"],
        pdf["domestic"],
        left=pdf["non_domestic"],
        color="orange",
        label="Domestic",
    )
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_hourly_polar(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 10))
    angles = [n / 24.0 * 2 * np.pi for n in pdf["hour"].astype(float)]
    angles += angles[:1]
    values = pdf["count"].tolist()
    values += values[:1]
    ax = plt.subplot(111, projection="polar")
    ax.plot(angles, values, color="darkorange", linewidth=2, label="Incidents")
    ax.fill(angles, values, color="orange", alpha=0.3)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels([f"{h}h" for h in range(24)])
    plt.title(title, fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_cumulative_arrests(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(pdf["Year"], pdf["cumulative_arrests"], color="green", marker="o", linewidth=2)
    plt.fill_between(pdf["Year"], pdf["cumulative_arrests"], color="green", alpha=0.2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Cumulative Arrests")
    plt.xticks(pdf["Year"], rotation=45)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def plot_above_avg_beats(pdf: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.bar(pdf["Beat"].astype(str), pdf["count"], color="darkred")
    if not pdf.empty:
        plt.axhline(
            y=float(pdf["avg_count"].iloc[0]),
            color="black",
            linestyle="--",
            label="Citywide Beat Mean",
        )
    plt.title(title)
    plt.xlabel("Beat")
    plt.ylabel("Crime Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target

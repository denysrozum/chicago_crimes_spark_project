"""Temporal analytical tasks (AQ-01 through AQ-06)."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from chicago_crimes.analytics.base import AnalyticalTask
from chicago_crimes.core.config import AnalyticsConfig
from chicago_crimes.visualization.charts import (
    plot_domestic_monthly,
    plot_horizontal_bar,
    plot_hourly_polar,
    plot_moving_average,
    plot_weekend_vs_weekday,
)


def _night_crimes_by_type(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter((F.col("hour") >= 0) & (F.col("hour") < 6))
        .groupBy("Primary Type")
        .count()
        .orderBy(F.desc("count"), F.asc("Primary Type"))
    )


def _weekend_vs_weekday_distribution(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("day_of_week").isNotNull())
        .withColumn(
            "is_weekend",
            F.when(F.col("day_of_week").isin("Sat", "Sun"), "weekend").otherwise("weekday"),
        )
        .groupBy("is_weekend")
        .count()
        .orderBy(F.asc("is_weekend"))
    )


def _monthly_moving_average_crimes(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    monthly = (
        df.filter(F.col("Date").isNotNull())
        .withColumn("YearMonth", F.date_format(F.col("Date"), "yyyy-MM"))
        .groupBy("YearMonth")
        .count()
    )
    window_3m = Window.orderBy("YearMonth").rowsBetween(-2, 0)
    return monthly.withColumn("moving_avg_3m", F.avg("count").over(window_3m)).orderBy(
        F.asc("YearMonth")
    )


def _domestic_crimes_by_month(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter((F.col("Domestic") == True) & F.col("Date").isNotNull())  # noqa: E712
        .withColumn("MonthNum", F.month(F.col("Date")))
        .groupBy("MonthNum")
        .count()
        .orderBy(F.asc("MonthNum"))
    )


def _hourly_crime_distribution_citywide(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return df.filter(F.col("hour").isNotNull()).groupBy("hour").count().orderBy(F.asc("hour"))


def _daytime_crimes_by_type(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter((F.col("hour") >= 12) & (F.col("hour") < 18))
        .groupBy("Primary Type")
        .count()
        .orderBy(F.desc("count"), F.asc("Primary Type"))
    )


TEMPORAL_TASKS: tuple[AnalyticalTask, ...] = (
    AnalyticalTask(
        task_id="AQ-01",
        legacy_name="Q1_D_Night_Crimes",
        domain_name="night_crimes_by_type",
        category="temporal",
        title="Crimes Committed at Night (00:00 - 06:00)",
        input_columns=("Primary Type", "hour"),
        output_columns=("Primary Type", "count"),
        execute_fn=_night_crimes_by_type,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Primary Type",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Crime Type",
            out_path=out,
            top_n=15,
            color="midnightblue",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-02",
        legacy_name="Q2_D_Weekend_vs_Weekday",
        domain_name="weekend_vs_weekday_distribution",
        category="temporal",
        title="Crime Share: Weekdays vs Weekends",
        input_columns=("day_of_week",),
        output_columns=("is_weekend", "count"),
        execute_fn=_weekend_vs_weekday_distribution,
        plot_fn=plot_weekend_vs_weekday,
    ),
    AnalyticalTask(
        task_id="AQ-03",
        legacy_name="Q6_D_Moving_Average_Crimes",
        domain_name="monthly_moving_average_crimes",
        category="temporal",
        title="Crime Dynamics with 3-Month Moving Average",
        input_columns=("Date",),
        output_columns=("YearMonth", "count", "moving_avg_3m"),
        execute_fn=_monthly_moving_average_crimes,
        plot_fn=plot_moving_average,
    ),
    AnalyticalTask(
        task_id="AQ-04",
        legacy_name="Q2_J_Domestic_Monthly",
        domain_name="domestic_crimes_by_month",
        category="temporal",
        title="Monthly Seasonality of Domestic Crimes",
        input_columns=("Domestic", "Date"),
        output_columns=("MonthNum", "count"),
        execute_fn=_domestic_crimes_by_month,
        plot_fn=plot_domestic_monthly,
    ),
    AnalyticalTask(
        task_id="AQ-05",
        legacy_name="Q6_M_Hourly_All_City",
        domain_name="hourly_crime_distribution_citywide",
        category="temporal",
        title="24-Hour Crime Clock (Citywide)",
        input_columns=("hour",),
        output_columns=("hour", "count"),
        execute_fn=_hourly_crime_distribution_citywide,
        plot_fn=plot_hourly_polar,
    ),
    AnalyticalTask(
        task_id="AQ-06",
        legacy_name="Q3_V_Day_Crimes",
        domain_name="daytime_crimes_by_type",
        category="temporal",
        title="Crimes Committed During Daytime (12:00 - 18:00)",
        input_columns=("Primary Type", "hour"),
        output_columns=("Primary Type", "count"),
        execute_fn=_daytime_crimes_by_type,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Primary Type",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Crime Type",
            out_path=out,
            top_n=15,
            color="gold",
            edgecolor="darkgoldenrod",
        ),
    ),
)

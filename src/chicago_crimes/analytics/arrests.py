"""Arrest outcome analytical tasks (AQ-20 through AQ-24)."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from chicago_crimes.analytics.base import AnalyticalTask
from chicago_crimes.core.config import AnalyticsConfig
from chicago_crimes.visualization.charts import (
    plot_cumulative_arrests,
    plot_horizontal_bar,
    plot_lollipop_share,
    plot_vertical_bar,
)


def _arrest_rate_by_ward(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("Ward").isNotNull())
        .groupBy("Ward")
        .agg(
            F.count("*").alias("total_crimes"),
            F.sum(F.when(F.col("Arrest") == True, 1).otherwise(0)).alias("arrests"),  # noqa: E712
        )
        .filter(F.col("arrests") > 0)
        .withColumn("arrest_rate", (F.col("arrests") / F.col("total_crimes")) * 100.0)
        .orderBy(F.desc("arrest_rate"), F.asc("Ward"))
        .limit(15)
    )


def _arrests_by_crime_type(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("Arrest") == True)  # noqa: E712
        .groupBy("Primary Type")
        .count()
        .orderBy(F.desc("count"), F.asc("Primary Type"))
    )


def _arrest_share_by_crime_type_year(df: DataFrame, config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("year") == config.arrest_share_year)
        .groupBy("Primary Type")
        .agg(
            F.count("*").alias("total_count"),
            F.sum(F.when(F.col("Arrest") == True, 1).otherwise(0)).alias(  # noqa: E712
                "arrest_count"
            ),
        )
        .filter((F.col("total_count") > 0) & (F.col("arrest_count") > 0))
        .withColumn("share", F.col("arrest_count") / F.col("total_count"))
        .select("Primary Type", "share")
        .orderBy(F.desc("share"), F.asc("Primary Type"))
    )


def _cumulative_arrests_by_year(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    yearly = (
        df.filter((F.col("Arrest") == True) & F.col("Date").isNotNull())  # noqa: E712
        .withColumn("Year", F.date_format(F.col("Date"), "yyyy"))
        .groupBy("Year")
        .count()
    )
    cum_win = Window.orderBy("Year").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return yearly.withColumn("cumulative_arrests", F.sum("count").over(cum_win)).orderBy(
        F.asc("Year")
    )


def _arrest_success_by_fbi_code(df: DataFrame, config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("FBI Code").isNotNull())
        .groupBy("FBI Code")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("Arrest") == True, 1).otherwise(0)).alias("arrests"),  # noqa: E712
        )
        .filter((F.col("total") > config.min_fbi_cases) & (F.col("arrests") > 0))
        .withColumn("success_rate", (F.col("arrests") / F.col("total")) * 100.0)
        .orderBy(F.desc("success_rate"), F.asc("FBI Code"))
        .limit(12)
    )


ARREST_TASKS: tuple[AnalyticalTask, ...] = (
    AnalyticalTask(
        task_id="AQ-20",
        legacy_name="Q5_D_Arrest_Ratio_By_Ward",
        domain_name="arrest_rate_by_ward",
        category="arrests",
        title="Top 15 Wards by Arrest Rate (%)",
        input_columns=("Ward", "Arrest"),
        output_columns=("Ward", "total_crimes", "arrests", "arrest_rate"),
        execute_fn=_arrest_rate_by_ward,
        plot_fn=lambda pdf, title, out: plot_vertical_bar(
            pdf,
            x_col="Ward",
            y_col="arrest_rate",
            title=title,
            xlabel="City Ward",
            ylabel="Arrest Rate (%)",
            out_path=out,
            color="coral",
            is_percent=True,
        ),
    ),
    AnalyticalTask(
        task_id="AQ-21",
        legacy_name="Q1_M_Arrest_By_Type",
        domain_name="arrests_by_crime_type",
        category="arrests",
        title="Arrest Counts by Crime Type (Top 20)",
        input_columns=("Arrest", "Primary Type"),
        output_columns=("Primary Type", "count"),
        execute_fn=_arrests_by_crime_type,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Primary Type",
            value_col="count",
            title=title,
            xlabel="Arrest Count",
            ylabel="Crime Type",
            out_path=out,
            top_n=20,
            color="forestgreen",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-22",
        legacy_name="Q2_M_Arrest_Share_2025",
        domain_name="arrest_share_by_crime_type_year",
        category="arrests",
        title="Arrest Share by Crime Type (Configured Year)",
        input_columns=("year", "Primary Type", "Arrest"),
        output_columns=("Primary Type", "share"),
        execute_fn=_arrest_share_by_crime_type_year,
        plot_fn=plot_lollipop_share,
    ),
    AnalyticalTask(
        task_id="AQ-23",
        legacy_name="Q4_V_Cumulative_Arrests_By_Year",
        domain_name="cumulative_arrests_by_year",
        category="arrests",
        title="Cumulative Arrests Over Years",
        input_columns=("Arrest", "Date"),
        output_columns=("Year", "count", "cumulative_arrests"),
        execute_fn=_cumulative_arrests_by_year,
        plot_fn=plot_cumulative_arrests,
    ),
    AnalyticalTask(
        task_id="AQ-24",
        legacy_name="Q6_V_FBI_Code_Arrest_Success",
        domain_name="arrest_success_by_fbi_code",
        category="arrests",
        title="Top 12 FBI Codes by Arrest Success Rate (%)",
        input_columns=("FBI Code", "Arrest"),
        output_columns=("FBI Code", "total", "arrests", "success_rate"),
        execute_fn=_arrest_success_by_fbi_code,
        plot_fn=lambda pdf, title, out: plot_vertical_bar(
            pdf,
            x_col="FBI Code",
            y_col="success_rate",
            title=title,
            xlabel="FBI Code",
            ylabel="Arrest Success Rate (%)",
            out_path=out,
            color="gold",
            is_percent=True,
        ),
    ),
)

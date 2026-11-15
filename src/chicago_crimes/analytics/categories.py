"""Crime category and domestic-incident analytical tasks (AQ-14 through AQ-19)."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from chicago_crimes.analytics.base import AnalyticalTask
from chicago_crimes.core.config import AnalyticsConfig
from chicago_crimes.visualization.charts import (
    plot_grouped_weekday_weekend,
    plot_horizontal_bar,
    plot_narcotics_vs_assault,
    plot_stacked_domestic_streets,
    plot_vertical_bar,
)


def _narcotics_vs_assault_by_district(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(
            F.col("District").isNotNull() & F.col("Primary Type").isin("NARCOTICS", "ASSAULT")
        )
        .groupBy("District")
        .agg(
            F.sum(F.when(F.col("Primary Type") == "NARCOTICS", 1).otherwise(0)).alias("narcotics"),
            F.sum(F.when(F.col("Primary Type") == "ASSAULT", 1).otherwise(0)).alias("assaults"),
        )
        .filter((F.col("narcotics") > 0) & (F.col("assaults") > 0))
        .withColumn("ratio", F.col("narcotics") / F.col("assaults"))
        .orderBy(F.desc("narcotics"), F.asc("District"))
        .limit(15)
    )


def _domestic_crimes_by_district(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter((F.col("Domestic") == True) & F.col("District").isNotNull())  # noqa: E712
        .groupBy("District")
        .count()
        .orderBy(F.desc("count"), F.asc("District"))
    )


def _weekday_vs_weekend_by_district(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("District").isNotNull() & F.col("day_of_week_num").isNotNull())
        .groupBy("District")
        .agg(
            F.sum(F.when(~F.col("day_of_week_num").isin(1, 7), 1).otherwise(0)).alias(
                "weekday_count"
            ),
            F.sum(F.when(F.col("day_of_week_num").isin(1, 7), 1).otherwise(0)).alias(
                "weekend_count"
            ),
        )
        .orderBy(F.desc("weekend_count"), F.asc("District"))
        .limit(10)
    )


def _domestic_vs_nondomestic_top_streets(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    agg = (
        df.filter(
            F.col("street_name").isNotNull()
            & (F.col("street_name") != "")
            & F.col("Domestic").isNotNull()
        )
        .groupBy("street_name")
        .agg(
            F.sum(F.when(F.col("Domestic") == True, 1).otherwise(0)).alias(  # noqa: E712
                "domestic"
            ),
            F.sum(F.when(F.col("Domestic") == False, 1).otherwise(0)).alias(  # noqa: E712
                "non_domestic"
            ),
        )
        .withColumn("total", F.col("domestic") + F.col("non_domestic"))
    )
    rank_win = Window.orderBy(F.desc("total"), F.asc("street_name"))
    return (
        agg.withColumn("rank", F.rank().over(rank_win))
        .filter(F.col("rank") <= 10)
        .drop("rank")
        .orderBy(F.desc("total"), F.asc("street_name"))
    )


def _top_crimes_in_community_area(df: DataFrame, config: AnalyticsConfig) -> DataFrame:
    rank_win = Window.orderBy(F.desc("count"), F.asc("Primary Type"))
    return (
        df.filter(F.col("Community Area") == config.community_area_id)
        .groupBy("Primary Type")
        .count()
        .withColumn("rank", F.dense_rank().over(rank_win))
        .filter(F.col("rank") <= 10)
        .drop("rank")
        .orderBy(F.desc("count"), F.asc("Primary Type"))
    )


def _street_crimes_by_type(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("Location Description") == "STREET")
        .groupBy("Primary Type")
        .count()
        .orderBy(F.desc("count"), F.asc("Primary Type"))
    )


CATEGORY_TASKS: tuple[AnalyticalTask, ...] = (
    AnalyticalTask(
        task_id="AQ-14",
        legacy_name="Q3_D_Narcotics_vs_Assault_By_District",
        domain_name="narcotics_vs_assault_by_district",
        category="categories",
        title="Narcotics vs Assault Incidents by District (Top 15)",
        input_columns=("District", "Primary Type"),
        output_columns=("District", "narcotics", "assaults", "ratio"),
        execute_fn=_narcotics_vs_assault_by_district,
        plot_fn=plot_narcotics_vs_assault,
    ),
    AnalyticalTask(
        task_id="AQ-15",
        legacy_name="Q4_D_Domestic_By_District",
        domain_name="domestic_crimes_by_district",
        category="categories",
        title="Domestic Crimes by Police District",
        input_columns=("Domestic", "District"),
        output_columns=("District", "count"),
        execute_fn=_domestic_crimes_by_district,
        plot_fn=lambda pdf, title, out: plot_vertical_bar(
            pdf,
            x_col="District",
            y_col="count",
            title=title,
            xlabel="District",
            ylabel="Domestic Incidents",
            out_path=out,
            color="rebeccapurple",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-16",
        legacy_name="Q3_M_Weekday_Vs_Weekend_By_District",
        domain_name="weekday_vs_weekend_by_district",
        category="categories",
        title="Weekday vs Weekend Crimes by District (Top 10)",
        input_columns=("District", "day_of_week_num"),
        output_columns=("District", "weekday_count", "weekend_count"),
        execute_fn=_weekday_vs_weekend_by_district,
        plot_fn=plot_grouped_weekday_weekend,
    ),
    AnalyticalTask(
        task_id="AQ-17",
        legacy_name="Q4_M_Domestic_Vs_NonDomestic_Streets",
        domain_name="domestic_vs_nondomestic_top_streets",
        category="categories",
        title="Top 10 Streets: Domestic vs Non-Domestic Crimes",
        input_columns=("street_name", "Domestic"),
        output_columns=("street_name", "domestic", "non_domestic", "total"),
        execute_fn=_domestic_vs_nondomestic_top_streets,
        plot_fn=plot_stacked_domestic_streets,
    ),
    AnalyticalTask(
        task_id="AQ-18",
        legacy_name="Q5_M_Crime_Type_Rank_In_Community",
        domain_name="top_crimes_in_community_area",
        category="categories",
        title="Top 10 Crime Types in Configured Community Area",
        input_columns=("Community Area", "Primary Type"),
        output_columns=("Primary Type", "count"),
        execute_fn=_top_crimes_in_community_area,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Primary Type",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Crime Type",
            out_path=out,
            color="magenta",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-19",
        legacy_name="Q2_V_Street_Crimes",
        domain_name="street_crimes_by_type",
        category="categories",
        title="Most Common Crimes on Streets",
        input_columns=("Location Description", "Primary Type"),
        output_columns=("Primary Type", "count"),
        execute_fn=_street_crimes_by_type,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Primary Type",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Crime Type",
            out_path=out,
            top_n=15,
            color="slategray",
        ),
    ),
)

"""Geographic and location analytical tasks (AQ-07 through AQ-13)."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from chicago_crimes.analytics.base import AnalyticalTask
from chicago_crimes.core.config import AnalyticsConfig
from chicago_crimes.visualization.charts import (
    plot_above_avg_beats,
    plot_block_share,
    plot_horizontal_bar,
    plot_top_crime_per_district,
    plot_vertical_bar,
)


def _top_theft_districts(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter((F.col("Primary Type") == "THEFT") & F.col("District").isNotNull())
        .groupBy("District")
        .count()
        .orderBy(F.desc("count"), F.asc("District"))
        .limit(10)
    )


def _top5_crime_types_locations(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    top_types = (
        df.groupBy("Primary Type")
        .count()
        .orderBy(F.desc("count"), F.asc("Primary Type"))
        .limit(5)
        .select("Primary Type")
    )
    return (
        df.join(F.broadcast(top_types), on="Primary Type", how="inner")
        .filter(F.col("Location Description").isNotNull())
        .groupBy("Location Description")
        .count()
        .orderBy(F.desc("count"), F.asc("Location Description"))
        .limit(10)
    )


def _top_crime_streets(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(F.col("street_name").isNotNull() & (F.col("street_name") != ""))
        .groupBy("street_name")
        .count()
        .orderBy(F.desc("count"), F.asc("street_name"))
        .limit(10)
    )


def _top_crime_type_per_district(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    window_spec = Window.partitionBy("District").orderBy(F.desc("count"), F.asc("Primary Type"))
    return (
        df.filter(F.col("District").isNotNull())
        .groupBy("District", "Primary Type")
        .count()
        .withColumn("rank", F.rank().over(window_spec))
        .filter(F.col("rank") == 1)
        .drop("rank")
        .orderBy(F.asc("District"))
    )


def _most_concentrated_blocks_by_district_share(
    df: DataFrame, config: AnalyticsConfig
) -> DataFrame:
    dist_totals = (
        df.filter(F.col("District").isNotNull())
        .groupBy("District")
        .agg(F.count("*").alias("dist_total"))
        .filter(F.col("dist_total") > config.min_district_crimes)
    )
    block_window = Window.partitionBy("District").orderBy(F.desc("count"), F.asc("Block"))
    top_block_per_dist = (
        df.filter(F.col("District").isNotNull() & F.col("Block").isNotNull())
        .groupBy("District", "Block")
        .count()
        .withColumn("rank", F.row_number().over(block_window))
        .filter(F.col("rank") == 1)
        .drop("rank")
    )
    return (
        top_block_per_dist.join(dist_totals, on="District", how="inner")
        .withColumn("share_percent", (F.col("count") / F.col("dist_total")) * 100.0)
        .orderBy(F.desc("share_percent"), F.asc("District"))
        .limit(10)
    )


def _top_criminal_damage_locations(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    return (
        df.filter(
            (F.col("Primary Type") == "CRIMINAL DAMAGE") & F.col("Location Description").isNotNull()
        )
        .groupBy("Location Description")
        .count()
        .orderBy(F.desc("count"), F.asc("Location Description"))
        .limit(10)
    )


def _above_average_crime_beats(df: DataFrame, _config: AnalyticsConfig) -> DataFrame:
    beat_counts = df.filter(F.col("Beat").isNotNull()).groupBy("Beat").count()
    avg_df = beat_counts.select(F.avg("count").alias("avg_count"))
    return (
        beat_counts.crossJoin(avg_df)
        .filter(F.col("count") > F.col("avg_count"))
        .orderBy(F.desc("count"), F.asc("Beat"))
        .limit(15)
    )


GEOGRAPHIC_TASKS: tuple[AnalyticalTask, ...] = (
    AnalyticalTask(
        task_id="AQ-07",
        legacy_name="Q1_J_Top_Theft_Districts",
        domain_name="top_theft_districts",
        category="geographic",
        title="Top 10 Districts by Theft Count",
        input_columns=("Primary Type", "District"),
        output_columns=("District", "count"),
        execute_fn=_top_theft_districts,
        plot_fn=lambda pdf, title, out: plot_vertical_bar(
            pdf,
            x_col="District",
            y_col="count",
            title=title,
            xlabel="Police District",
            ylabel="Theft Incidents",
            out_path=out,
            color="skyblue",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-08",
        legacy_name="Q3_J_Top5_Crime_Locations",
        domain_name="top5_crime_types_locations",
        category="geographic",
        title="Top 10 Locations Among Top 5 Crime Types",
        input_columns=("Primary Type", "Location Description"),
        output_columns=("Location Description", "count"),
        execute_fn=_top5_crime_types_locations,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Location Description",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Location Description",
            out_path=out,
            color="purple",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-09",
        legacy_name="Q4_J_Top_Streets",
        domain_name="top_crime_streets",
        category="geographic",
        title="Top 10 Streets by Crime Count",
        input_columns=("street_name",),
        output_columns=("street_name", "count"),
        execute_fn=_top_crime_streets,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="street_name",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Street Name",
            out_path=out,
            color="indianred",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-10",
        legacy_name="Q5_J_Top_Crime_Per_District",
        domain_name="top_crime_type_per_district",
        category="geographic",
        title="Most Frequent Crime Type per District",
        input_columns=("District", "Primary Type"),
        output_columns=("District", "Primary Type", "count"),
        execute_fn=_top_crime_type_per_district,
        plot_fn=plot_top_crime_per_district,
    ),
    AnalyticalTask(
        task_id="AQ-11",
        legacy_name="Q6_J_Most_Dangerous_Blocks_Share",
        domain_name="most_concentrated_blocks_by_district_share",
        category="geographic",
        title="Top 10 Blocks with Highest Share of District Crime",
        input_columns=("District", "Block"),
        output_columns=("District", "Block", "count", "dist_total", "share_percent"),
        execute_fn=_most_concentrated_blocks_by_district_share,
        plot_fn=plot_block_share,
    ),
    AnalyticalTask(
        task_id="AQ-12",
        legacy_name="Q1_V_Criminal_Damage_Locations",
        domain_name="top_criminal_damage_locations",
        category="geographic",
        title="Top 10 Locations for Criminal Damage",
        input_columns=("Primary Type", "Location Description"),
        output_columns=("Location Description", "count"),
        execute_fn=_top_criminal_damage_locations,
        plot_fn=lambda pdf, title, out: plot_horizontal_bar(
            pdf,
            label_col="Location Description",
            value_col="count",
            title=title,
            xlabel="Number of Incidents",
            ylabel="Location Description",
            out_path=out,
            color="firebrick",
            edgecolor="maroon",
        ),
    ),
    AnalyticalTask(
        task_id="AQ-13",
        legacy_name="Q5_V_High_Crime_Beats_Vs_Avg",
        domain_name="above_average_crime_beats",
        category="geographic",
        title="Top 15 Police Beats Above Citywide Average",
        input_columns=("Beat",),
        output_columns=("Beat", "count", "avg_count"),
        execute_fn=_above_average_crime_beats,
        plot_fn=plot_above_avg_beats,
    ),
)

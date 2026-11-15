"""Regression and schema contract tests for all 24 analytical tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyspark.sql import DataFrame

from chicago_crimes.analytics.registry import ALL_TASKS, ANALYTICAL_TASKS, get_task, resolve_tasks
from chicago_crimes.analytics.runner import prepare_analytics_dataframe, run_analytical_tasks
from chicago_crimes.core.config import AnalyticsConfig


def test_registry_completeness_and_aliases() -> None:
    assert len(ALL_TASKS) == 24
    assert len(ANALYTICAL_TASKS) == 24
    assert len({t.task_id for t in ALL_TASKS}) == 24
    assert len({t.legacy_name for t in ALL_TASKS}) == 24
    assert len({t.domain_name for t in ALL_TASKS}) == 24

    for idx in range(1, 25):
        tid = f"AQ-{idx:02d}"
        task = get_task(tid)
        assert get_task(task.legacy_name.lower()) is task
        assert get_task(task.domain_name.upper()) is task

    with pytest.raises(KeyError):
        get_task("AQ-99")


def test_category_filtering() -> None:
    assert len(resolve_tasks(category="temporal")) == 6
    assert len(resolve_tasks(category="geographic")) == 7
    assert len(resolve_tasks(category="categories")) == 6
    assert len(resolve_tasks(category="arrests")) == 5
    with pytest.raises(ValueError):
        resolve_tasks(category="nonexistent")


def test_all_24_tasks_execute_and_render_charts(
    sample_processed_df: DataFrame,
    analytics_config: AnalyticsConfig,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "all_24"
    results = run_analytical_tasks(
        sample_processed_df,
        output_dir=out_dir,
        config=analytics_config,
        render_plots=True,
        write_csv=True,
    )
    assert len(results) == 24

    for item in results:
        assert item.row_count > 0, f"{item.task_id} returned empty DataFrame on fixture"
        assert item.csv_path is not None and Path(item.csv_path).stat().st_size > 0
        assert item.chart_path is not None and Path(item.chart_path).stat().st_size > 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_count"] == 24
    assert len(manifest["tasks"]) == 24


def test_specific_task_semantics(
    sample_processed_df: DataFrame, analytics_config: AnalyticsConfig
) -> None:
    prepared = prepare_analytics_dataframe(sample_processed_df)

    def _run(tid: str) -> tuple[list[str], list]:
        res_df = get_task(tid).execute(prepared, analytics_config)
        return list(res_df.columns), res_df.collect()

    # Temporal (AQ-01..AQ-06)
    cols, rows = _run("AQ-01")
    assert cols == ["Primary Type", "count"]
    assert {r["Primary Type"]: r["count"] for r in rows} == {
        "THEFT": 3,
        "NARCOTICS": 2,
        "ASSAULT": 1,
        "CRIMINAL DAMAGE": 1,
        "ROBBERY": 1,
    }

    cols, rows = _run("AQ-02")
    assert cols == ["is_weekend", "count"]
    assert {r["is_weekend"]: r["count"] for r in rows} == {"weekday": 10, "weekend": 14}

    cols, rows = _run("AQ-03")
    assert cols == ["YearMonth", "count", "moving_avg_3m"]
    assert len(rows) == 23
    assert (rows[0]["YearMonth"], rows[0]["count"], rows[0]["moving_avg_3m"]) == ("2024-01", 2, 2.0)
    assert rows[1]["moving_avg_3m"] == pytest.approx(1.5)
    assert rows[2]["moving_avg_3m"] == pytest.approx(4.0 / 3.0)

    cols, rows = _run("AQ-04")
    assert cols == ["MonthNum", "count"]
    assert sum(r["count"] for r in rows) == 8
    assert {r["MonthNum"]: r["count"] for r in rows}[8] == 2

    cols, rows = _run("AQ-05")
    assert cols == ["hour", "count"]
    assert len(rows) == 18
    assert sum(r["count"] for r in rows) == 24
    hour_map = {r["hour"]: r["count"] for r in rows}
    assert hour_map[0] == 1 and hour_map[1] == 2

    cols, rows = _run("AQ-06")
    assert cols == ["Primary Type", "count"]
    assert {r["Primary Type"]: r["count"] for r in rows} == {
        "NARCOTICS": 3,
        "ASSAULT": 2,
        "CRIMINAL DAMAGE": 2,
        "THEFT": 2,
        "BATTERY": 1,
    }

    # Geographic (AQ-07..AQ-13)
    cols, rows = _run("AQ-07")
    assert cols == ["District", "count"]
    assert {r["District"]: r["count"] for r in rows} == {1: 4, 2: 1, 11: 1}

    cols, rows = _run("AQ-08")
    assert cols == ["Location Description", "count"]
    assert {r["Location Description"]: r["count"] for r in rows} == {
        "STREET": 10,
        "RESIDENCE": 5,
        "SIDEWALK": 5,
        "APARTMENT": 3,
    }

    cols, rows = _run("AQ-09")
    assert cols == ["street_name", "count"]
    assert len(rows) == 8
    assert rows[0]["street_name"] == "STATE" and rows[0]["count"] == 5

    cols, rows = _run("AQ-10")
    assert cols == ["District", "Primary Type", "count"]
    assert {r["District"]: (r["Primary Type"], r["count"]) for r in rows} == {
        1: ("THEFT", 4),
        2: ("BATTERY", 2),
        11: ("CRIMINAL DAMAGE", 2),
    }

    cols, rows = _run("AQ-11")
    assert cols == ["District", "Block", "count", "dist_total", "share_percent"]
    aq11_map = {r["District"]: r for r in rows}
    assert aq11_map[2]["Block"] == "005XX N CLARK ST"
    assert aq11_map[2]["share_percent"] == pytest.approx(60.0)
    assert aq11_map[11]["share_percent"] == pytest.approx(37.5)
    assert aq11_map[1]["share_percent"] == pytest.approx(400.0 / 11.0)

    cols, rows = _run("AQ-12")
    assert cols == ["Location Description", "count"]
    assert {r["Location Description"]: r["count"] for r in rows} == {"RESIDENCE": 3, "STREET": 1}

    cols, rows = _run("AQ-13")
    assert cols == ["Beat", "count", "avg_count"]
    assert {r["Beat"] for r in rows} == {111, 113, 1111, 1112}
    assert all(r["count"] == 4 for r in rows)
    assert rows[0]["avg_count"] == pytest.approx(24.0 / 7.0)

    # Categories (AQ-14..AQ-19)
    cols, rows = _run("AQ-14")
    assert cols == ["District", "narcotics", "assaults", "ratio"]
    d11 = [r for r in rows if r["District"] == 11][0]
    assert (d11["narcotics"], d11["assaults"]) == (2, 1)
    assert d11["ratio"] == pytest.approx(2.0)

    cols, rows = _run("AQ-15")
    assert cols == ["District", "count"]
    assert {r["District"]: r["count"] for r in rows} == {1: 4, 2: 2, 11: 2}

    cols, rows = _run("AQ-16")
    assert cols == ["District", "weekday_count", "weekend_count"]
    assert {r["District"]: (r["weekday_count"], r["weekend_count"]) for r in rows} == {
        1: (4, 7),
        11: (3, 5),
        2: (3, 2),
    }

    cols, rows = _run("AQ-17")
    assert cols == ["street_name", "domestic", "non_domestic", "total"]
    aq17_map = {r["street_name"]: r for r in rows}
    assert (aq17_map["STATE"]["domestic"], aq17_map["STATE"]["non_domestic"]) == (1, 4)
    assert (aq17_map["MADISON"]["domestic"], aq17_map["MADISON"]["non_domestic"]) == (3, 0)

    cols, rows = _run("AQ-18")
    assert cols == ["Primary Type", "count"]
    aq18_map = {r["Primary Type"]: r["count"] for r in rows}
    assert aq18_map["BATTERY"] == 3 and aq18_map["ASSAULT"] == 2 and sum(aq18_map.values()) == 10

    cols, rows = _run("AQ-19")
    assert cols == ["Primary Type", "count"]
    assert {r["Primary Type"]: r["count"] for r in rows} == {
        "THEFT": 5,
        "NARCOTICS": 3,
        "ASSAULT": 1,
        "CRIMINAL DAMAGE": 1,
        "ROBBERY": 1,
    }

    # Arrests (AQ-20..AQ-24)
    cols, rows = _run("AQ-20")
    assert cols == ["Ward", "total_crimes", "arrests", "arrest_rate"]
    aq20_map = {r["Ward"]: r for r in rows}
    assert aq20_map[27]["total_crimes"] == 3 and aq20_map[27]["arrests"] == 2
    assert aq20_map[27]["arrest_rate"] == pytest.approx(200.0 / 3.0)

    cols, rows = _run("AQ-21")
    assert cols == ["Primary Type", "count"]
    assert {r["Primary Type"]: r["count"] for r in rows} == {
        "NARCOTICS": 5,
        "ASSAULT": 3,
        "THEFT": 2,
        "BATTERY": 1,
        "CRIMINAL DAMAGE": 1,
    }

    cols, rows = _run("AQ-22")
    assert cols == ["Primary Type", "share"]
    aq22_map = {r["Primary Type"]: r["share"] for r in rows}
    assert aq22_map["NARCOTICS"] == pytest.approx(1.0)
    assert aq22_map["ASSAULT"] == pytest.approx(1.0)
    assert aq22_map["THEFT"] == pytest.approx(0.5)
    assert aq22_map["CRIMINAL DAMAGE"] == pytest.approx(1.0 / 3.0)

    cols, rows = _run("AQ-23")
    assert cols == ["Year", "count", "cumulative_arrests"]
    assert [(r["Year"], r["count"], r["cumulative_arrests"]) for r in rows] == [
        ("2024", 6, 6),
        ("2025", 6, 12),
    ]

    cols, rows = _run("AQ-24")
    assert cols == ["FBI Code", "total", "arrests", "success_rate"]
    aq24_map = {r["FBI Code"]: r for r in rows}
    assert aq24_map["18"]["success_rate"] == pytest.approx(100.0)
    assert aq24_map["08A"]["success_rate"] == pytest.approx(75.0)
    assert aq24_map["06"]["success_rate"] == pytest.approx(100.0 / 3.0)


def test_task_raises_on_missing_required_column(
    sample_processed_df: DataFrame, analytics_config: AnalyticsConfig
) -> None:
    broken_df = sample_processed_df.drop("District")
    task = get_task("AQ-07")
    with pytest.raises(ValueError, match="District"):
        task.execute(broken_df, analytics_config)

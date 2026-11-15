"""Shared Spark feature-preparation helpers for multivariate experiments."""

from __future__ import annotations

from typing import Any

from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, NumericType

EXCLUDED_EXPERIMENT_COLS: frozenset[str] = frozenset(
    {
        "ID",
        "X Coordinate",
        "Y Coordinate",
        "source_year",
        "has_valid_geo",
        "has_valid_admin",
    }
)


def vector_to_list(vec: Any) -> list[float]:
    """Convert a Spark ML DenseVector or SparseVector to a Python float list."""
    if isinstance(vec, DenseVector):
        return [float(x) for x in vec]
    if isinstance(vec, SparseVector):
        return [float(x) for x in vec.toArray().tolist()]
    return [float(x) for x in vec]


def select_numeric_and_bool_features(
    df: DataFrame,
    *,
    target_col: str | None = None,
) -> tuple[DataFrame, list[str]]:
    """Cast boolean columns to double and select usable numeric/boolean features."""
    excluded = EXCLUDED_EXPERIMENT_COLS | ({target_col} if target_col else set())
    numeric_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, NumericType) and f.name not in excluded
    ]
    bool_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, BooleanType) and f.name not in excluded
    ]
    feature_cols = numeric_cols + bool_cols

    work_df = df
    for col_name in bool_cols:
        work_df = work_df.withColumn(col_name, F.col(col_name).cast("double"))

    selected = [*feature_cols, target_col] if target_col else feature_cols
    return work_df.select(*selected), feature_cols

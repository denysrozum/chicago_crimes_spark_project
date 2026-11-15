"""Data cleaning, validation, quality checks, feature engineering, Silver, and Gold."""

from chicago_crimes.processing.cleaning import run_legacy_preprocessing
from chicago_crimes.processing.features import (
    add_spatial_grid,
    add_temporal_features,
    extract_block_features,
)
from chicago_crimes.processing.ge_validation import (
    run_ge_validation_suite,
    validate_gold_with_gx,
    validate_silver_with_gx,
)
from chicago_crimes.processing.gold import GoldBuildResult, GoldTableSummary, build_gold_tables
from chicago_crimes.processing.quality import (
    SilverAccountingReport,
    StructuralAnomalySummary,
    inspect_structural_anomalies,
    save_anomaly_report,
    validate_processed_dataframe,
    validate_silver_dataframe,
)
from chicago_crimes.processing.silver import SilverPipelineResult, run_silver_pipeline

__all__ = [
    "GoldBuildResult",
    "GoldTableSummary",
    "SilverAccountingReport",
    "SilverPipelineResult",
    "StructuralAnomalySummary",
    "add_spatial_grid",
    "add_temporal_features",
    "build_gold_tables",
    "extract_block_features",
    "inspect_structural_anomalies",
    "run_ge_validation_suite",
    "run_legacy_preprocessing",
    "run_silver_pipeline",
    "save_anomaly_report",
    "validate_gold_with_gx",
    "validate_processed_dataframe",
    "validate_silver_dataframe",
    "validate_silver_with_gx",
]

"""Data validation and structural anomaly inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

from chicago_crimes.core.schemas import (
    CHICAGO_LAT_MAX,
    CHICAGO_LAT_MIN,
    CHICAGO_LON_MAX,
    CHICAGO_LON_MIN,
    COMMUNITY_AREA_MAX,
    COMMUNITY_AREA_MIN,
    DISTRICT_MAX,
    DISTRICT_MIN,
    WARD_MAX,
    WARD_MIN,
)


@dataclass(frozen=True)
class StructuralAnomalySummary:
    total_rows: int
    invalid_coordinates: int
    invalid_district: int
    invalid_ward: int
    invalid_community_area: int
    suspicious_dates: int

    @property
    def raw_anomaly_sum(self) -> int:
        return (
            self.invalid_coordinates
            + self.invalid_district
            + self.invalid_ward
            + self.invalid_community_area
        )

    def format_report(self) -> str:
        lines = [
            "===== STRUCTURAL ANOMALY REPORT =====",
            f"Total rows: {self.total_rows}",
            f"Invalid coordinates: {self.invalid_coordinates}",
            f"Invalid District values: {self.invalid_district}",
            f"Invalid Ward values: {self.invalid_ward}",
            f"Invalid Community Area values: {self.invalid_community_area}",
            f"Suspicious dates (< 2000): {self.suspicious_dates}",
            f"Total detected anomalies (raw sum): {self.raw_anomaly_sum}",
        ]
        if self.raw_anomaly_sum == 0:
            lines.append("VERDICT: No structural anomalies detected.")
        elif self.total_rows > 0:
            pct = (self.raw_anomaly_sum / self.total_rows) * 100.0
            lines.append(f"VERDICT: Structural anomalies detected ({pct:.4f}% of data).")
        else:
            lines.append("VERDICT: Empty dataset.")
        return "\n".join(lines) + "\n"


def inspect_structural_anomalies(df: DataFrame) -> StructuralAnomalySummary:
    """Compute counts of out-of-range coordinates, administrative codes, and dates."""
    agg_row = df.agg(
        F.count(F.lit(1)).alias("total_rows"),
        F.sum(
            F.when(
                F.col("Latitude").isNotNull()
                & F.col("Longitude").isNotNull()
                & (
                    (F.col("Latitude") < CHICAGO_LAT_MIN)
                    | (F.col("Latitude") > CHICAGO_LAT_MAX)
                    | (F.col("Longitude") < CHICAGO_LON_MIN)
                    | (F.col("Longitude") > CHICAGO_LON_MAX)
                ),
                1,
            ).otherwise(0)
        ).alias("invalid_coordinates"),
        F.sum(
            F.when(
                F.col("District").isNotNull()
                & ((F.col("District") < DISTRICT_MIN) | (F.col("District") > DISTRICT_MAX)),
                1,
            ).otherwise(0)
        ).alias("invalid_district"),
        F.sum(
            F.when(
                F.col("Ward").isNotNull()
                & ((F.col("Ward") < WARD_MIN) | (F.col("Ward") > WARD_MAX)),
                1,
            ).otherwise(0)
        ).alias("invalid_ward"),
        F.sum(
            F.when(
                F.col("Community Area").isNotNull()
                & (
                    (F.col("Community Area") < COMMUNITY_AREA_MIN)
                    | (F.col("Community Area") > COMMUNITY_AREA_MAX)
                ),
                1,
            ).otherwise(0)
        ).alias("invalid_community_area"),
        F.sum(F.when(F.year("Date") < 2000, 1).otherwise(0)).alias("suspicious_dates"),
    ).collect()[0]

    return StructuralAnomalySummary(
        total_rows=int(agg_row["total_rows"] or 0),
        invalid_coordinates=int(agg_row["invalid_coordinates"] or 0),
        invalid_district=int(agg_row["invalid_district"] or 0),
        invalid_ward=int(agg_row["invalid_ward"] or 0),
        invalid_community_area=int(agg_row["invalid_community_area"] or 0),
        suspicious_dates=int(agg_row["suspicious_dates"] or 0),
    )


def save_anomaly_report(summary: StructuralAnomalySummary, report_path: str | Path) -> Path:
    """Persist a structural anomaly report to disk."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.format_report(), encoding="utf-8")
    return path


def validate_processed_dataframe(df: DataFrame) -> None:
    """Validate basic schema and non-emptiness constraints on a processed DataFrame."""
    if df.limit(1).count() == 0:
        raise ValueError("DataFrame is empty")

    if "Date" not in df.columns:
        raise ValueError("Date column missing")
    if not isinstance(df.schema["Date"].dataType, TimestampType):
        raise TypeError(f"Date column must be timestamp, got {df.schema['Date'].dataType}")

    required_cols = ["Date", "Latitude", "Longitude", "District"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_rows = (
        df.filter(
            F.col("Date").isNull()
            | F.col("Latitude").isNull()
            | F.col("Longitude").isNull()
            | F.col("District").isNull()
            | (F.col("District") < DISTRICT_MIN)
            | (F.col("District") > DISTRICT_MAX)
        )
        .limit(1)
        .count()
    )
    if invalid_rows > 0:
        raise ValueError("DataFrame contains invalid critical values")


@dataclass(frozen=True)
class SilverAccountingReport:
    """Strict row-level accounting record for Silver transformation."""

    input_records: int
    accepted_records: int
    superseded_records: int
    quarantined_records: int
    quarantine_reasons: dict[str, int]
    missing_geo_records: int
    out_of_bounds_geo_nullified: int
    missing_admin_records: int
    out_of_range_admin_nullified: int

    @property
    def accounted_records(self) -> int:
        """Total records accounted for across accepted, superseded, and quarantined outputs."""
        return self.accepted_records + self.superseded_records + self.quarantined_records

    @property
    def is_balanced(self) -> bool:
        """Return True iff every input record is accounted for cleanly."""
        return (
            self.input_records >= 0
            and self.accepted_records >= 0
            and self.superseded_records >= 0
            and self.quarantined_records >= 0
            and self.accounted_records == self.input_records
        )

    def verify_invariant(self) -> None:
        """Raise ValueError if row accounting fails the zero-silent-loss invariant."""
        if not self.is_balanced:
            raise ValueError(
                "Silver row accounting invariant violated: "
                f"input={self.input_records}, accepted={self.accepted_records}, "
                f"superseded={self.superseded_records}, quarantined={self.quarantined_records}, "
                f"accounted={self.accounted_records}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize report to a JSON-compatible dictionary."""
        return {
            "input_records": self.input_records,
            "accepted_records": self.accepted_records,
            "superseded_records": self.superseded_records,
            "quarantined_records": self.quarantined_records,
            "accounted_records": self.accounted_records,
            "invariant_passed": self.is_balanced,
            "quarantine_reasons": dict(self.quarantine_reasons),
            "missing_geo_records": self.missing_geo_records,
            "out_of_bounds_geo_nullified": self.out_of_bounds_geo_nullified,
            "missing_admin_records": self.missing_admin_records,
            "out_of_range_admin_nullified": self.out_of_range_admin_nullified,
        }


def validate_silver_dataframe(
    df: DataFrame,
    accounting: SilverAccountingReport | None = None,
    *,
    require_unique_id: bool = True,
) -> None:
    """Validate Silver invariants: required keys, quality flags, bounds on non-nulls, accounting."""
    from chicago_crimes.core import schemas as _schemas

    if accounting is not None:
        accounting.verify_invariant()

    required_cols = [
        "ID",
        "Date",
        "Primary Type",
        "Updated On",
        "Latitude",
        "Longitude",
        "District",
        "Ward",
        "Community Area",
        "has_valid_geo",
        "geo_quality_flag",
        "has_valid_admin",
        "admin_quality_flag",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Silver DataFrame missing required columns: {missing_cols}")

    if df.schema["Date"].dataType.typeName() != "timestamp":
        raise ValueError(
            f"Silver column 'Date' must be timestamp, got {df.schema['Date'].dataType}"
        )

    bad_rows = (
        df.filter(
            F.col("ID").isNull()
            | (F.col("ID") <= 0)
            | F.col("Date").isNull()
            | (F.year(F.col("Date")) < _schemas.MIN_VALID_YEAR)
            | F.col("Primary Type").isNull()
            | (F.trim(F.col("Primary Type")) == "")
            | F.col("has_valid_geo").isNull()
            | F.col("geo_quality_flag").isNull()
            | F.col("has_valid_admin").isNull()
            | F.col("admin_quality_flag").isNull()
            | (F.col("Latitude").isNotNull() != F.col("Longitude").isNotNull())
            | (
                F.col("has_valid_geo")
                != (F.col("Latitude").isNotNull() & F.col("Longitude").isNotNull())
            )
            | (
                F.col("Latitude").isNotNull()
                & (
                    (F.col("Latitude") < _schemas.CHICAGO_LAT_MIN)
                    | (F.col("Latitude") > _schemas.CHICAGO_LAT_MAX)
                )
            )
            | (
                F.col("Longitude").isNotNull()
                & (
                    (F.col("Longitude") < _schemas.CHICAGO_LON_MIN)
                    | (F.col("Longitude") > _schemas.CHICAGO_LON_MAX)
                )
            )
            | (
                F.col("District").isNotNull()
                & ((F.col("District") < DISTRICT_MIN) | (F.col("District") > DISTRICT_MAX))
            )
            | (
                F.col("Ward").isNotNull()
                & ((F.col("Ward") < _schemas.WARD_MIN) | (F.col("Ward") > _schemas.WARD_MAX))
            )
            | (
                F.col("Community Area").isNotNull()
                & (
                    (F.col("Community Area") < _schemas.COMMUNITY_AREA_MIN)
                    | (F.col("Community Area") > _schemas.COMMUNITY_AREA_MAX)
                )
            )
        )
        .limit(1)
        .count()
    )
    if bad_rows > 0:
        raise ValueError("Silver DataFrame failed critical schema/bounds/quality flag validation")

    if require_unique_id:
        dup_rows = df.groupBy("ID").count().filter(F.col("count") > 1).limit(1).count()
        if dup_rows > 0:
            raise ValueError("Silver DataFrame contains duplicate IDs after deduplication")

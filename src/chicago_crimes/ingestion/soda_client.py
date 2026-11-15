"""Incremental Chicago Open Data (SODA API v2) ingestion client with watermarks."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from chicago_crimes.core.config import StreamingConfig
from chicago_crimes.core.logging import get_logger
from chicago_crimes.core.schemas import RAW_CRIME_COLUMNS

logger = get_logger("ingestion.soda")

DEFAULT_SODA_ENDPOINT = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

SODA_TO_BRONZE_FIELD_MAP: dict[str, str] = {
    "ID": "id",
    "Case Number": "case_number",
    "Date": "date",
    "Block": "block",
    "IUCR": "iucr",
    "Primary Type": "primary_type",
    "Description": "description",
    "Location Description": "location_description",
    "Arrest": "arrest",
    "Domestic": "domestic",
    "Beat": "beat",
    "District": "district",
    "Ward": "ward",
    "Community Area": "community_area",
    "FBI Code": "fbi_code",
    "X Coordinate": "x_coordinate",
    "Y Coordinate": "y_coordinate",
    "Year": "year",
    "Updated On": "updated_on",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Location": "location",
}

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class SodaWatermark:
    """Durable watermark tracking the last committed SODA incremental cursor."""

    last_updated_on: str | None = None
    last_id: str | None = None
    records_fetched: int = 0
    last_batch_id: str | None = None
    updated_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SodaFetchResult:
    """Incremental fetch result prior to committing watermark downstream."""

    records: list[dict[str, Any]]
    next_watermark: SodaWatermark
    batch_id: str
    source_id: str
    effective_where: str
    pages_fetched: int
    raw_count: int


def parse_soda_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO-8601 SODA timestamp into a naive UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is not None:
        return parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def format_soda_timestamp(value: str | datetime) -> str:
    """Format a datetime as a SODA SoQL floating timestamp (`YYYY-MM-DDTHH:MM:SS.fff`)."""
    dt = parse_soda_timestamp(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def load_watermark(path: str | Path) -> SodaWatermark | None:
    """Load persisted SODA watermark from JSON if it exists."""
    target = Path(path)
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    return SodaWatermark(
        last_updated_on=payload.get("last_updated_on"),
        last_id=payload.get("last_id"),
        records_fetched=int(payload.get("records_fetched", 0)),
        last_batch_id=payload.get("last_batch_id"),
        updated_at_utc=payload.get("updated_at_utc"),
    )


def save_watermark(path: str | Path, watermark: SodaWatermark) -> Path:
    """Atomically persist SODA watermark JSON so interrupted writes cannot corrupt state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{uuid4().hex[:8]}.tmp")
    tmp_path.write_text(json.dumps(watermark.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp_path, target)
    return target


def build_incremental_where(
    *,
    watermark: SodaWatermark | None = None,
    since: str | datetime | None = None,
    until: str | datetime | None = None,
    overlap_seconds: int = 60,
    default_lookback_hours: int = 72,
    now_utc: datetime | None = None,
) -> str:
    """Build deterministic SoQL `$where` filter using `since` or overlapped watermark."""
    if since is not None:
        start_dt = parse_soda_timestamp(since)
    elif watermark is not None and watermark.last_updated_on:
        start_dt = parse_soda_timestamp(watermark.last_updated_on) - timedelta(
            seconds=max(0, overlap_seconds)
        )
    else:
        anchor = (
            parse_soda_timestamp(now_utc)
            if now_utc
            else datetime.now(UTC).replace(tzinfo=None)
        )
        start_dt = anchor - timedelta(hours=max(1, default_lookback_hours))

    clauses = [f"updated_on >= '{format_soda_timestamp(start_dt)}'"]
    if until is not None:
        clauses.append(f"updated_on <= '{format_soda_timestamp(until)}'")
    return " AND ".join(clauses)


def _format_location_value(raw_location: Any) -> str | None:
    if raw_location is None:
        return None
    if isinstance(raw_location, dict):
        lat = raw_location.get("latitude")
        lon = raw_location.get("longitude")
        if lat is not None and lon is not None:
            return f"({lat}, {lon})"
        return None
    text = str(raw_location).strip()
    return text if text else None


def _stringify_raw_field(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text if text else None


def soda_record_to_bronze_dict(
    record: dict[str, Any],
    *,
    source_id: str = "chicago_soda_api",
    batch_id: str,
    ingested_at: datetime | None = None,
) -> dict[str, Any]:
    """Translate one SODA API JSON object to canonical raw Bronze column names + lineage."""
    ts = ingested_at or datetime.now(UTC).replace(tzinfo=None)
    out: dict[str, Any] = {}
    for canonical_col in RAW_CRIME_COLUMNS:
        soda_key = SODA_TO_BRONZE_FIELD_MAP[canonical_col]
        if canonical_col == "Location":
            out[canonical_col] = _format_location_value(record.get(soda_key))
        else:
            out[canonical_col] = _stringify_raw_field(record.get(soda_key))

    out["corrupt_record"] = None
    out["ingestion_timestamp"] = format_soda_timestamp(ts)
    out["source_id"] = source_id
    out["ingestion_batch_id"] = batch_id
    return out


def _compute_retry_delay(
    response: requests.Response | None,
    attempt: int,
    base_backoff_seconds: float,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return float(base_backoff_seconds * (2 ** attempt))


def _request_page_with_retries(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout_seconds: float,
    max_retries: int,
    base_backoff_seconds: float,
    sleep_fn: Callable[[float], None],
) -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        response: requests.Response | None = None
        try:
            response = session.get(
                endpoint, params=params, headers=headers, timeout=timeout_seconds
            )
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = _compute_retry_delay(response, attempt, base_backoff_seconds)
                logger.warning(
                    "SODA API transient HTTP %s (attempt %d/%d); retrying in %.2fs",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                sleep_fn(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Expected SODA JSON array, got {type(payload).__name__}")
            return payload
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            delay = _compute_retry_delay(response, attempt, base_backoff_seconds)
            logger.warning(
                "SODA API request error (%s, attempt %d/%d); retrying in %.2fs",
                exc,
                attempt + 1,
                max_retries,
                delay,
            )
            sleep_fn(delay)

    if last_exc is not None:
        raise last_exc
    return []


def fetch_soda_incremental(
    streaming_cfg: StreamingConfig | None = None,
    *,
    endpoint: str | None = None,
    app_token: str | None = None,
    watermark: SodaWatermark | None = None,
    watermark_path: str | Path | None = None,
    since: str | datetime | None = None,
    until: str | datetime | None = None,
    page_size: int | None = None,
    overlap_seconds: int | None = None,
    default_lookback_hours: int = 72,
    max_records: int | None = None,
    source_id: str = "chicago_soda_api",
    batch_id: str | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 4,
    base_backoff_seconds: float = 1.0,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> SodaFetchResult:
    """Fetch ordered incremental crime records from the official SODA endpoint."""
    cfg = streaming_cfg or StreamingConfig()
    effective_endpoint = endpoint or cfg.soda_endpoint
    effective_token = app_token if app_token is not None else cfg.soda_app_token
    effective_page_size = page_size or cfg.soda_page_size
    effective_overlap = cfg.soda_overlap_seconds if overlap_seconds is None else overlap_seconds

    prior_watermark = watermark
    if prior_watermark is None and watermark_path is not None:
        prior_watermark = load_watermark(watermark_path)

    effective_where = build_incremental_where(
        watermark=prior_watermark,
        since=since,
        until=until,
        overlap_seconds=effective_overlap,
        default_lookback_hours=default_lookback_hours,
    )

    headers: dict[str, str] = {"Accept": "application/json"}
    if effective_token:
        headers["X-App-Token"] = effective_token

    run_batch_id = batch_id or uuid4().hex
    ingested_at = datetime.now(UTC).replace(tzinfo=None)
    own_session = session is None
    client = session or requests.Session()

    raw_records: list[dict[str, Any]] = []
    pages_fetched = 0
    offset = 0

    try:
        while True:
            remaining = (
                effective_page_size
                if max_records is None
                else max(0, min(effective_page_size, max_records - len(raw_records)))
            )
            if remaining == 0:
                break

            params = {
                "$where": effective_where,
                "$order": "updated_on ASC, id ASC",
                "$limit": remaining,
                "$offset": offset,
            }
            page = _request_page_with_retries(
                client,
                effective_endpoint,
                params,
                headers,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                base_backoff_seconds=base_backoff_seconds,
                sleep_fn=sleep_fn,
            )
            pages_fetched += 1
            if not page:
                break
            raw_records.extend(page)
            if len(page) < remaining:
                break
            offset += len(page)
    finally:
        if own_session:
            client.close()

    bronze_records = [
        soda_record_to_bronze_dict(
            row,
            source_id=source_id,
            batch_id=run_batch_id,
            ingested_at=ingested_at,
        )
        for row in raw_records
    ]

    latest_updated_on = prior_watermark.last_updated_on if prior_watermark else None
    latest_id = prior_watermark.last_id if prior_watermark else None
    for row in raw_records:
        row_updated = row.get("updated_on")
        row_id = str(row.get("id")) if row.get("id") is not None else None
        if not row_updated:
            continue
        formatted_updated = format_soda_timestamp(str(row_updated))
        if latest_updated_on is None or (formatted_updated, row_id or "") >= (
            latest_updated_on,
            latest_id or "",
        ):
            latest_updated_on = formatted_updated
            latest_id = row_id

    next_watermark = SodaWatermark(
        last_updated_on=latest_updated_on,
        last_id=latest_id,
        records_fetched=len(bronze_records),
        last_batch_id=run_batch_id,
        updated_at_utc=ingested_at.replace(tzinfo=UTC).isoformat(),
    )
    logger.info(
        "Fetched %d incremental SODA rows across %d page(s) (where=%s next_updated_on=%s)",
        len(bronze_records),
        pages_fetched,
        effective_where,
        next_watermark.last_updated_on,
    )
    return SodaFetchResult(
        records=bronze_records,
        next_watermark=next_watermark,
        batch_id=run_batch_id,
        source_id=source_id,
        effective_where=effective_where,
        pages_fetched=pages_fetched,
        raw_count=len(raw_records),
    )

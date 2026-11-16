# Known Limitations & Release Constraints

## 1. Scheduled Incremental Ingestion vs. True Push Event Streaming
- The City of Chicago SODA API (`ijzp-q8t2.json`) is a pull-based REST endpoint rather than a real-time event broker. Chicago Open Data does **not** push events directly into Kafka.
- Instead, the `chicago_crimes_incremental_streaming` Airflow DAG runs on a 30-minute schedule (`*/30 * * * *`) and invokes `chicago_crimes.ingestion.soda_client` (`ingest-soda`) to poll records incrementally using a composite cursor `(updated_on, id)` Ordered by `updated_on ASC, id ASC`.
- **Bounded Catch-Up & Backlog Telemetry**: Each scheduled poll executes bounded multi-page catch-up (`page_size=500`, `max_pages=5`, `limit=2000` by default in the DAG; configurable via `--page-size`, `--max-pages`, and `--limit`) and probes one extra record (`fetch_limit + 1`) to set `has_more` / `backlog_detected` and compute `ingestion_lag_seconds` relative to the latest upstream `updated_on`. When `has_more=true` (or `overlap_seconds=0`), the next poll uses the strict cursor `(updated_on > T OR (updated_on = T AND id > LAST_ID))` so records sharing an identical `updated_on` timestamp are never skipped or re-fetched in an infinite loop.
- **Downstream Scope**: Incremental streaming updates the Bronze, DLQ, Quarantine, and Silver Delta tables (`merge_silver_delta`). Gold aggregations (`chicago-crimes gold`) and ML forecasting models (`chicago-crimes ml`) are rebuilt by the scheduled batch pipeline (`chicago_crimes_batch_pipeline`) rather than recalculated on every streaming micro-batch.

## 2. Streaming Consistency & Cross-Sink Atomicity
- Each Structured Streaming sink is individually replay-safe under Delta Lake:
  - DLQ, Bronze, and Quarantine use deterministic `stream_event_id = sha2(concat_ws('|', topic, partition, offset), 256)` insert-only Delta `MERGE` (`merge_insert_only_delta`).
  - Silver uses monotonic, deterministic Delta `MERGE` keyed by `ID` (`merge_silver_delta`): newer non-null `Updated On` replaces older target rows, valid `Updated On` upgrades null-timestamp targets, null-timestamp sources never overwrite valid targets, and equal timestamps resolve deterministically via SHA-256 business-payload digest comparison.
  - Micro-batch reconciliation metrics use atomic temp-file replacement (`os.replace`) deduplicating `(batch_id, offsets_digest)`.
- Spark Structured Streaming does not provide distributed two-phase commit across multiple independent sinks (multiple Delta tables, local JSONL metrics logs, Kafka DLQ topic, and SODA file watermarks). Recovery guarantees idempotency across replay rather than global cross-sink transactional atomicity.

## 3. Small/Medium-Scale Distributed Overhead
- At `100,000` Silver rows under equal CPU budget (`4` cores), `local[4]` completes the analytical benchmark faster (`16.633s` at `p=16`) than single-host Docker Spark Standalone (`41.447s` at `p=16`) because JVM task serialization, inter-container shuffle networking, and MinIO S3A HTTP overhead exceed multi-worker parallelism gains at this scale on one workstation.

## 4. Google Cloud BigQuery & Live dbt Execution
- `publish-bigquery` has been exercised end-to-end in `--dry-run` mode (exporting Parquet staging files plus JSON schemas and BigQuery load/DDL manifests), and `dbt/` models have been validated offline via `dbt parse`.
- Live table creation inside a Google Cloud BigQuery project and `dbt run` / `dbt test` against live GCP infrastructure remain unexecuted in this environment because no GCP project credentials or cloud billing account are configured.

## 5. Timestamp & Timezone Semantics (`America/Chicago` vs. `UTC`)
- Historical CSV (`MM/dd/yyyy hh:mm:ss a`) and SODA (`yyyy-MM-dd'T'HH:mm:ss.SSS`) timestamps represent local wall-clock time in `America/Chicago` (`CST = UTC-6` in winter, `CDT = UTC-5` in summer). For example, `2001-01-01 00:00:00` Chicago local time (`CST`) corresponds to `2001-01-01T06:00:00Z` in UTC.
- Because the upstream dataset does not include explicit UTC offsets on each row, `1,662` historical records falling inside the autumn DST fall-back hour (`01:00:00..01:59:59` on transition Sundays) are ambiguous between CDT (`UTC-5`) and CST (`UTC-6`), and `253` records falling inside the spring DST forward gap (`02:00:00..02:59:59`) represent nonexistent local wall-clock times. Rather than fabricating offsets, Silver preserves the reported local wall-clock timestamp in `Date`/`ts`, flags these rows via `timestamp_dst_flag` (`dst_fall_fold_ambiguous` or `dst_spring_gap_nonexistent`), and sets `ts_utc = NULL` on those `1,915` records (`0.0225%` of the dataset).

## 6. Upstream Licensing & Redistribution Status
- **Public redistribution status: BLOCKED pending contributor licensing approval.**
- The repository currently contains no `LICENSE` file. Git history includes commits from four contributors (`denysrozum` formerly `denusbtw`, `Eclipsexc`, `Evilcheshire`, `ThLivSt451`), and the raw/derived datasets originate from the City of Chicago Open Data portal (`https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data`).
- Do not redistribute dataset derivatives or publish under an open-source license without explicit written sign-off from repository collaborators and proper Chicago Open Data terms attribution.

## 7. Host Environment Notes (Windows / Docker Desktop)
- Running Docker Compose stacks concurrently (Airflow, PostgreSQL, Kafka, MinIO, Spark Master + 2 Workers) requires significant memory (~16 GB host RAM recommended) and causes Docker Desktop `ext4.vhdx` storage growth over time.
- Running PySpark locally on Windows uses a bundled bare-filesystem helper JAR in `core/spark.py` (`_resolve_windows_bare_fs_jar`); Linux/Docker/CI environments bypass that helper automatically.

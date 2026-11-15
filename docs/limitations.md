# Known Limitations & Release Constraints

## 1. Incremental Ingestion Architecture (Polling, Not Push)
- The `chicago_crimes_incremental_streaming` Airflow DAG polls the City of Chicago SODA API (`ijzp-q8t2.json`) on a scheduled interval using `chicago_crimes.ingestion.soda_client` and publishes newly fetched JSON payloads into Kafka (`chicago-crimes.raw`).
- Chicago Open Data does not push events directly to Kafka; end-to-end incremental latency is therefore bounded by Airflow polling frequency plus Spark Structured Streaming micro-batch triggering.

## 2. Streaming Consistency & Cross-Sink Atomicity
- Each Structured Streaming sink is individually replay-safe under Delta Lake:
  - DLQ, Bronze, and Quarantine use deterministic `stream_event_id = sha2(concat_ws('|', topic, partition, offset), 256)` insert-only Delta `MERGE` (`merge_insert_only_delta`).
  - Silver uses latest-`Updated On` Delta `MERGE` keyed by `ID` (`merge_silver_delta`).
  - Micro-batch reconciliation metrics use atomic temp-file replacement (`os.replace`) deduplicating `(batch_id, offsets_digest)`.
- Spark Structured Streaming does not provide distributed two-phase commit across multiple independent sinks (multiple Delta tables, local JSONL metrics logs, Kafka DLQ topic, and SODA file watermarks). Recovery guarantees idempotency across replay rather than global cross-sink transactional atomicity.

## 3. Small/Medium-Scale Distributed Overhead
- At `100,000` Silver rows under equal CPU budget (`4` cores), `local[4]` completes the analytical benchmark faster (`16.633s` at `p=16`) than single-host Docker Spark Standalone (`41.447s` at `p=16`) because JVM task serialization, inter-container shuffle networking, and MinIO S3A HTTP overhead exceed multi-worker parallelism gains at this scale on one workstation.

## 4. Google Cloud BigQuery & Live dbt Execution
- `publish-bigquery` has been exercised end-to-end in `--dry-run` mode (exporting Parquet staging files plus JSON schemas and BigQuery load/DDL manifests), and `dbt/` models have been validated offline via `dbt parse`.
- Live table creation inside a Google Cloud BigQuery project and `dbt run` / `dbt test` against live GCP infrastructure remain unexecuted in this environment because no GCP project credentials or cloud billing account are configured.

## 5. Timestamp & Timezone Semantics
- Historical CSV timestamps originate as Chicago local wall-clock strings (`MM/dd/yyyy hh:mm:ss a`), while Spark sessions run in UTC (`spark.sql.session.timeZone=UTC`). Boundary midnight records (such as `2001-01-01 00:00:00` local vs UTC conversions around `2000-12-31T22:00:00Z`) require explicit timezone awareness if cross-checking against UTC-shifted extracts.

## 6. Upstream Licensing & Redistribution Status
- **Public redistribution status: BLOCKED pending contributor licensing approval.**
- The repository currently contains no `LICENSE` file. Git history includes commits from four contributors (`denysrozum` formerly `denusbtw`, `Eclipsexc`, `Evilcheshire`, `ThLivSt451`), and the raw/derived datasets originate from the City of Chicago Open Data portal (`https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data`).
- Do not redistribute dataset derivatives or publish under an open-source license without explicit written sign-off from repository collaborators and proper Chicago Open Data terms attribution.

## 7. Host Environment Notes (Windows / Docker Desktop)
- Running Docker Compose stacks concurrently (Airflow, PostgreSQL, Kafka, MinIO, Spark Master + 2 Workers) requires significant memory (~16 GB host RAM recommended) and causes Docker Desktop `ext4.vhdx` storage growth over time.
- Running PySpark locally on Windows uses a bundled bare-filesystem helper JAR in `core/spark.py` (`_resolve_windows_bare_fs_jar`); Linux/Docker/CI environments bypass that helper automatically.

# Architecture Overview

The **Chicago Crime Analytics Platform** is a modular PySpark lakehouse, streaming, orchestration, data-quality, analytics, and forecasting engineering platform built around the City of Chicago *Crimes - 2001 to Present* dataset (`ijzp-q8t2`).

## 1. Package & Component Boundaries

All core runtime logic lives under `src/chicago_crimes/` (`chicago-crimes` CLI entry point, `pyproject.toml`):

| Package | Responsibility | Key Modules |
|---|---|---|
| `core` | Profile/YAML/env configuration, structured logging, canonical 22-column raw CSV schema & domain bounds, SparkSession factory | `config.py`, `logging.py`, `schemas.py`, `spark.py` |
| `ingestion` | Non-destructive Bronze CSV ingestion, City of Chicago SODA API v2 polling client with durable watermark, Kafka producer | `bronze.py`, `csv_loader.py`, `soda_client.py`, `kafka_producer.py` |
| `processing` | Silver normalization/quarantine/deduplication, temporal/block/spatial feature engineering, accounting invariant, Great Expectations suites, Gold aggregations | `cleaning.py`, `silver.py`, `features.py`, `quality.py`, `ge_validation.py`, `gold.py` |
| `analytics` | Registry of 24 analytical tasks (`AQ-01`–`AQ-24`), task runner, dataset profiling, Spark/Delta shuffle benchmark | `base.py`, `registry.py`, `temporal.py`, `geographic.py`, `categories.py`, `arrests.py`, `runner.py`, `profiling.py`, `benchmark.py` |
| `streaming` | Spark Structured Streaming Kafka consumer, DLQ/Bronze/Quarantine routing, Silver Delta `MERGE`, micro-batch reconciliation | `crime_stream.py` |
| `storage` | Local Parquet I/O, Delta Lake ACID & `MERGE`, MinIO (`s3a://`) bucket bootstrap, unified lakehouse dispatcher, BigQuery export & DDL manifest generator | `local.py`, `delta.py`, `minio.py`, `lakehouse.py`, `bigquery.py` |
| `ml` | Leakage-free daily district time-series feature builder, chronological split, multi-model trainer, MLflow SQLite logger, regression evaluation | `dataset.py`, `models.py`, `evaluate.py` |
| `experiments` & `visualization` | PCA, FAMD, class imbalance, and sampling studies plus `matplotlib` chart rendering | `pca.py`, `famd.py`, `imbalance.py`, `sampling.py`, `charts.py` |

Supporting assets outside `src/`:

- `configs/`: `default.yaml` and `standalone.yaml` runtime profiles.
- `orchestration/dags/`: Airflow DAG definitions (`chicago_crimes_batch_pipeline` and `chicago_crimes_incremental_streaming`).
- `dbt/`: Offline-validated BigQuery SQL models (`stg_chicago_crimes_silver` view and 3 analytical marts).
- `infrastructure/`: Container build definitions (`infrastructure/spark/Dockerfile`, `infrastructure/spark/entrypoint.sh`, `infrastructure/airflow/Dockerfile`).

## 2. Spark Execution Modes & Lakehouse Storage

The platform runs identically through two configuration profiles (`--profile local` vs `--profile standalone`):

1. **Local Mode (`local`)**:
   - Runs embedded PySpark (`local[4]` by default) reading and writing local filesystem Parquet or Delta tables.
   - Uses `configs/default.yaml` (overrideable via `--config` or `CHICAGO_CRIMES_*` environment variables).
   - On Windows hosts, `core/spark.py` loads a helper JAR via `_resolve_windows_bare_fs_jar` to avoid native Hadoop `winutils.exe` failures; on Linux (including CI and Docker containers), this helper is bypassed.
2. **Standalone Cluster Mode (`standalone`)**:
   - Targets Docker Compose Spark Standalone (`spark://spark-master:7077`, Web UI `8080`) with two workers (`spark-worker-1` on `8081` and `spark-worker-2` on `8082`, allocated 2 CPU cores and 1800 MB RAM each).
   - Uses Delta Lake 3.2.0 (`io.delta:delta-spark_2.12:3.2.0`) and Hadoop AWS 3.3.4 (`org.apache.hadoop:hadoop-aws:3.3.4`) writing via `s3a://` to MinIO (`http://minio:9000` in Compose / `http://localhost:9000` from host, bucket `chicago-crimes`).
   - Built into `chicago-crimes-spark:0.2.0` (`infrastructure/spark/Dockerfile`).

## 3. Batch vs Incremental Streaming Flows

### Batch Pipeline
1. **Bronze Ingestion (`ingest`)**: Reads semicolon-delimited CSV non-destructively as strings (`RAW_CRIME_STRING_SCHEMA` with permissive `_corrupt_record` capture), attaching `ingestion_timestamp`, `source_id`, `ingestion_batch_id`, and `corrupt_record`.
2. **Silver Cleaning, Feature Engineering & Deduplication (`process`)**: Parses types safely (`try_cast` / multi-format timestamps), routes invalid rows to `quarantine`, preserves nullified out-of-bounds geographic/administrative fields with explicit quality flags (`has_valid_geo`, `geo_quality_flag`, `has_valid_admin`, `admin_quality_flag`), enriches temporal/address-block/grid columns, and deduplicates by `ID` ranked descending by `Updated On` (writing older duplicates to `superseded`). Every run asserts the accounting invariant: `input == accepted + superseded + quarantined`.
3. **Quality Gate & Gold Aggregation (`validate`, `gold`)**: Validates Silver (`28` Great Expectations rules) and Gold (`23` Great Expectations rules), then materializes 7 aggregate tables (`daily_citywide_counts`, `monthly_crime_summary`, `district_community_summary`, `category_distribution`, `temporal_hourly_profile`, `spatial_valid_grid_summary`, `daily_district_counts`).
4. **Analytics, ML & Serving (`analyze`, `ml`, `publish-bigquery`, `dbt/`)**: Executes registered tasks (`AQ-01`–`AQ-24`), trains 5 daily district forecasting models logged to local SQLite MLflow (`seasonal_naive_lag7`, `rolling_mean_7_baseline`, `spark_mllib_rf`, `spark_mllib_gbt`, `xgboost_regressor` champion selected strictly on validation RMSE), and optionally exports BigQuery schemas/load manifests consumed by `dbt/` models.

### Incremental Scheduled SODA-to-Kafka Streaming Pipeline
Because the City of Chicago SODA API (`ijzp-q8t2.json`) is a REST endpoint rather than a push broker:
1. Airflow (`chicago_crimes_incremental_streaming`, PostgreSQL metadata DB, `LocalExecutor`) periodically polls `chicago_crimes.ingestion.soda_client` via `chicago-crimes ingest-soda`.
2. `soda_client.py` queries records incrementally beyond the durable `(last_updated_on, last_id)` JSON watermark and publishes valid JSON payloads to Kafka topic `chicago-crimes.raw` (`kafka:9092` internal, `localhost:9094` external). The SODA watermark is updated only after Kafka delivery succeeds.
3. The long-running `crime-stream` service (`chicago-crimes --profile standalone stream`) runs PySpark Structured Streaming over Kafka using `foreachBatch` (`process_kafka_microbatch`):
   - Decodes JSON payloads (`KAFKA_JSON_SCHEMA` + permissive `_corrupt_json_record`).
   - Computes deterministic `stream_event_id = sha2(concat_ws('|', topic, partition, offset), 256)` for every Kafka message.
   - Writes malformed payloads to the DLQ table via insert-only Delta `MERGE` on `stream_event_id` (plus optional Kafka DLQ topic `chicago-crimes.dlq`).
   - Writes valid raw payloads to streaming Bronze via insert-only Delta `MERGE` on `stream_event_id`.
   - Applies `run_silver_pipeline`, routes rejected rows to streaming Quarantine via insert-only Delta `MERGE` on `stream_event_id`, and upserts accepted deduplicated records into Silver Delta via `merge_silver_delta` keyed by `ID` (preserving newer `Updated On` rows).
   - Appends a reconciled `StreamingBatchStats` record atomically via temp-file replacement to the metrics JSONL log (deduplicating `(batch_id, offsets_digest)` on replay).

## 4. Failure Handling & Consistency Model

- **Replay Safety**: Structured Streaming persists Kafka offsets in Spark checkpoints (`CHICAGO_CRIMES_STREAM_CHECKPOINT`). If a micro-batch fails partway through `foreachBatch`, Spark replays the exact Kafka offset range on restart.
- **Per-Sink Idempotency vs Cross-Sink Atomicity**: Each sink is individually replay-safe under Delta Lake (`stream_event_id` insert-only `MERGE` for DLQ, Bronze, and Quarantine; `ID` upsert `MERGE` for Silver; temp-file `os.replace` with `(batch_id, offsets_digest)` dedup for metrics JSONL; conditional watermark commit for SODA). However, Spark Structured Streaming cannot wrap multiple Delta tables, local JSONL metrics, and REST watermark files inside a single distributed two-phase commit transaction—crash recovery relies on replay idempotency across sinks rather than global cross-sink atomicity.
- **Airflow Containerized Dispatch**: DAGs run tasks by invoking ephemeral `chicago-crimes-spark:0.2.0` containers over the mounted Docker socket inside `chicago-crimes-platform_chicago-crimes-net`, keeping Airflow webserver/scheduler dependencies decoupled from JVM/PySpark workers.

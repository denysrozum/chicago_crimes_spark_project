# Verification & Benchmark Evidence

All measurements and check results below were observed directly from executions in this repository workspace.

## Phase 3 Timestamp Semantics Rebuild — Quantified Dataset Impact

Pre-existing `data/silver/chicago_crimes` had been materialized under a host-OS `Europe/Kyiv` (`UTC+2`/`UTC+3`) Spark session prior to setting `spark.sql.session.timeZone=UTC`, shifting stored Parquet `Date` epoch values backward by 2–3 hours (`516` early-morning records from `2001-01-01 00:00..01:59` had shifted into `2000-12-31` / `2000-12`). Rebuilding `data/silver/chicago_crimes`, `data/gold`, `artifacts/quality/ge_validation_report.json`, and `artifacts/ml` from `data/bronze/chicago_crimes` under `session_timezone: "UTC"` with explicit `America/Chicago` (`ts`, `ts_utc`, `timestamp_dst_flag`) semantics produced the following verified changes:

| Dataset / Metric | Pre-Rebuild (Legacy Shifted Silver) | Post-Rebuild (Verified `America/Chicago` Calendar) | Impact Explanation |
| --- | --- | --- | --- |
| Silver `input_records` / `accepted_records` | `8,495,400` / `8,495,400` | `8,495,400` / `8,495,400` | Zero row loss; accounting invariant `8,495,400 == 8,495,400 + 0 + 0` preserved |
| Silver `timestamp_dst_flag` breakdown | *(not present)* | `valid_cst`: `2,952,268`, `valid_cdt`: `5,541,217`, `dst_fall_fold_ambiguous`: `1,662`, `dst_spring_gap_nonexistent`: `253` | Exact `ts_utc` populated for `8,493,485` rows; `1,915` ambiguous/nonexistent DST rows flagged with `ts_utc = NULL` |
| Gold `monthly_crime_summary` rows | `303` (`2000-12`..`2026-02`) | `302` (`2001-01`..`2026-02`) | Eliminated spurious `2000-12` month caused by 2-hour `Europe/Kyiv` shift on `2001-01-01 00:00..01:59` rows |
| Gold `daily_citywide_counts` rows & range | `9,169` (`2000-12-31`..`2026-02-06`) | `9,169` (`2001-01-01`..`2026-02-07`) | Restored true Chicago local calendar dates (`2000-12-31` removed; `2026-02-07` late-evening CST rows restored) |
| Gold `daily_district_counts` rows | `201,982` | `201,970` | Exact Chicago midnight boundaries per Police District (`2001-01-01`..`2026-02-07`) |
| ML continuous district-day rows (`22` districts) | `201,564` | `201,551` | Continuous daily grid aligned to Chicago local calendar (`train: 141,086`, `val: 30,228`, `test: 30,237`) |
| Champion `xgboost_regressor` Held-Out Test RMSE / $R^2$ | `6.8346` / `0.6075` | `6.8025` / `0.6089` | Aligning daily boundaries to true Chicago local midnight improved held-out test RMSE, MAE (`5.2300`), and WMAPE (`16.6795%`) |

## Distributed Cluster & Scaling Benchmarks (`s3a://` Delta on MinIO, `100,000` Rows)

Source files: `artifacts/benchmarks/standalone_benchmark.json`, `artifacts/benchmarks/standalone_1worker_2cores.json`, `artifacts/benchmarks/worker_loss_recovery.json`.

| Scenario | Alive Workers / Cores | Shuffle Partitions | Duration (s) | Throughput (rows/s) | Executor Row Split |
| --- | --- | --- | --- | --- | --- |
| Full cluster sweep | 2 workers / 4 cores | 4 | `18.343` | `5,451.5` | `spark-worker-1: 50,001`, `spark-worker-2: 49,999` |
| Full cluster sweep | 2 workers / 4 cores | 8 | `8.465` | `11,813.7` | `spark-worker-1: 50,001`, `spark-worker-2: 49,999` |
| Full cluster sweep | 2 workers / 4 cores | 16 | `6.425` | `15,564.3` | `spark-worker-1: 49,999`, `spark-worker-2: 50,001` |
| Core cap (`cores_max=2`) | 2 alive (`1` allocated) | 16 | `8.577` | `11,659.7` | `spark-worker-2: 100,000` |
| Fault tolerance (`worker-2` stopped) | 1 worker / 2 cores (`worker-2: DEAD`) | 8 | `9.042` | `11,059.8` | `spark-worker-1: 100,000` |

- **Equal-Budget Local vs. Standalone Comparison (`s3a://` Delta on MinIO, `100,000` Rows)**:
  - Source file: `artifacts/benchmarks/equal_budget_local_vs_standalone.json` (both executed inside `chicago-crimes-spark-master` with `4` cores and `3072m` executor/local JVM heap, `write_delta_output=True`):
    - `local[4]` (`3072m`): `p=8` `23.497s` (`4,255.9` rows/s), `p=16` `16.633s` (`6,012.3` rows/s), `spark-master: 100,000`.
    - Standalone `spark://spark-master:7077` (`2` workers x `2` cores x `1536m`): `p=8` `59.141s` (`1,690.9` rows/s; `spark-worker-1: 25,000, spark-worker-2: 75,000`), `p=16` `41.447s` (`2,412.7` rows/s; `spark-worker-1: 12,500, spark-worker-2: 87,500`).
  - **Benchmark Interpretation Note**: On a single Docker host at `100,000` rows, `local[4]` is ~`2.5x` faster at `p=16` (`16.633s` vs `41.447s`) because cross-container shuffle IPC and worker HTTP S3A contention exceed multi-JVM parallelism gains when all containers share the same underlying CPU and MinIO endpoint. Similarly, the multi-sweep Standalone runs in `standalone_benchmark.json` validate distributed worker execution and failover rather than linear compute speedup.

## Streaming & Replay Verification (`apache/kafka-native:3.8.1` + Delta)

Source log: `artifacts/streaming/live_microbatch_metrics.jsonl`, unit replay tests (`tests/unit/test_soda_and_streaming.py`, `tests/unit/test_lakehouse.py`), and live crash/replay integration suite (`tests/integration/test_streaming_live.py`).

| Batch | Trigger & Scenario | Kafka Input | DLQ | Bronze Written | Silver Accepted | Superseded | Quarantined | Delta Merge Effect | Reconciled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0` | Initial 250 live SODA events | `250` | `0` | `250` | `250` | `0` | `0` | Version `0` (`numOutputRows=250`) | `true` |
| `1` | 10 replays + 1 stale `2026-11-01` + 1 missing-ID + 1 malformed JSON after Kafka restart | `13` | `1` | `12` | `10` | `1` | `1` | Version `1` (`Updated=10, Inserted=0`) | `true` |

- **Guarantees Verified**:
  - SODA watermark updates persist only after Kafka `flush()`, tracking `has_more`, `backlog_detected`, `ingestion_lag_seconds`, and `cumulative_records_fetched`.
  - In Delta mode, Bronze, DLQ, and Quarantine writes use deterministic `stream_event_id` insert-only `MERGE` (`merge_insert_only_delta`).
  - Silver Delta `MERGE` (`merge_silver_delta`) enforces monotonic version safety: newer non-null `Updated On` updates target, older `Updated On` is rejected, null-timestamp source never overwrites valid-timestamp target, valid-timestamp source upgrades null-timestamp target, and equal timestamps resolve deterministically via SHA-256 business-payload digest comparison.
  - JSONL metrics deduplicate by `(batch_id, offsets_digest)` via atomic `os.replace`. Separate sink writes remain non-transactional across sinks during a mid-microbatch crash until Spark checkpoint recovery replays the batch.

## Airflow 2.10.5 Orchestration Verification

Full batch regeneration run `manual__hardening_regen_20261114T225022Z` (`chicago_crimes_batch_pipeline`) completed with all 4 tasks (`ensure_minio_bucket`, `sync_silver_delta`, `build_gold_tables`, `run_silver_analytics`) returning `success`. Fresh isolated DAG execution, failure propagation, and incremental Kafka streaming triggers are verified live by `tests/integration/test_airflow_orchestration.py`.

## Quality Validation (`great-expectations==1.25.0`)

Source report: `artifacts/quality/ge_validation_report.json` (`validated_at_utc=2026-11-15T21:31:05.573684+00:00`).

- **Silver** (`100,000` sampled rows): `28/28` expectations passed (`100.0%`).
- **Gold** (`daily_citywide_counts` `9,169` rows: `7/7`; `daily_district_counts` `201,970` rows: `8/8`; `category_distribution` `34` rows: `8/8`).

## ML Forecasting (`mlflow==3.18.0`, `xgboost==3.2.0`, Spark MLlib)

Source report: `artifacts/ml/ml_evaluation_report.json` (`201,551` continuous-calendar district-day rows across `22` dense districts derived from verified `America/Chicago` calendar Gold tables; sparse outlier District `31` excluded via `min_calendar_coverage=0.5` and calendar gaps zero-filled before lag/rolling calculation).

- Splits (zero temporal overlap): `train` `2001-01-08..2018-07-30` (`141,086`), `val` `2018-07-31..2022-05-04` (`30,228`), `test` `2022-05-05..2026-02-07` (`30,237`).

| Model | Framework | Val RMSE | Test RMSE | Test MAE | Test R2 | Test WMAPE (%) |
| --- | --- | --- | --- | --- | --- | --- |
| `seasonal_naive_lag7` | `baseline` | `9.9210` | `9.5328` | `7.2383` | `0.2320` | `23.0842` |
| `rolling_mean_7_baseline` | `baseline` | `7.5490` | `7.2003` | `5.4846` | `0.5619` | `17.4912` |
| `spark_mllib_rf` | `spark_mllib` | `7.2859` | `6.9566` | `5.3390` | `0.5910` | `17.0272` |
| `spark_mllib_gbt` | `spark_mllib` | `7.1733` | `6.8588` | `5.3151` | `0.6024` | `16.9509` |
| `xgboost_regressor` **(Champion by Val RMSE)** | `xgboost` | `7.0473` | `6.8025` | `5.2300` | `0.6089` | `16.6795` |

# Verification & Benchmark Evidence

All measurements and check results below were observed directly from executions in this repository workspace.

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

Source log: `artifacts/streaming/live_microbatch_metrics.jsonl`, unit replay tests (`tests/unit/test_soda_and_streaming.py`), and live crash/replay integration suite (`tests/integration/test_streaming_live.py`).

| Batch | Trigger & Scenario | Kafka Input | DLQ | Bronze Written | Silver Accepted | Superseded | Quarantined | Delta Merge Effect | Reconciled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0` | Initial 250 live SODA events | `250` | `0` | `250` | `250` | `0` | `0` | Version `0` (`numOutputRows=250`) | `true` |
| `1` | 10 replays + 1 stale `2026-11-01` + 1 missing-ID + 1 malformed JSON after Kafka restart | `13` | `1` | `12` | `10` | `1` | `1` | Version `1` (`Updated=10, Inserted=0`) | `true` |

- Note on guarantees: SODA watermark updates persist only after Kafka `flush()`. In Delta mode, Bronze, DLQ, and Quarantine writes use deterministic `stream_event_id` insert-only `MERGE` (`merge_insert_only_delta`), Silver Delta `MERGE` enforces `(ID, Updated On >= target.Updated On)`, and JSONL metrics deduplicate by `(batch_id, offsets_digest)` via atomic `os.replace`. Separate sink writes remain non-transactional across sinks during a mid-microbatch crash until Spark checkpoint recovery replays the batch.

## Airflow 2.10.5 Orchestration Verification

Full batch regeneration run `manual__hardening_regen_20261114T225022Z` (`chicago_crimes_batch_pipeline`) completed with all 4 tasks (`ensure_minio_bucket`, `sync_silver_delta`, `build_gold_tables`, `run_silver_analytics`) returning `success` and regenerated `artifacts/airflow/batch_gold/_gold_manifest.json` matching local `data/gold/_gold_manifest.json` (`9,169` daily, `303` monthly, `201,982` daily-district; `8,495,400` source Silver rows). Fresh isolated DAG execution, failure propagation, and incremental Kafka streaming triggers are verified live by `tests/integration/test_airflow_orchestration.py`.

## Quality Validation (`great-expectations==1.25.0`)

Source report: `artifacts/quality/ge_validation_report.json`.

- **Silver** (`100,000` sampled rows): `28/28` expectations passed (`100.0%`).
- **Gold** (`daily_citywide_counts` `9,169` rows: `7/7`; `daily_district_counts` `201,982` rows: `8/8`; `category_distribution` `34` rows: `8/8`).

## ML Forecasting (`mlflow==3.18.0`, `xgboost==3.2.0`, Spark MLlib)

Source report: `artifacts/ml/ml_evaluation_report.json` (`201,564` continuous-calendar district-day rows across `22` dense districts derived from regenerated UTC Gold tables; sparse outlier District `31` excluded via `min_calendar_coverage=0.5` and calendar gaps zero-filled before lag/rolling calculation).

- Splits (zero temporal overlap): `train` `2001-01-07..2018-07-29` (`141,086`), `val` `2018-07-30..2022-05-03` (`30,228`), `test` `2022-05-04..2026-02-06` (`30,250`).

| Model | Framework | Val RMSE | Test RMSE | Test MAE | Test R2 | Test WMAPE (%) |
| --- | --- | --- | --- | --- | --- | --- |
| `seasonal_naive_lag7` | `baseline` | `9.7193` | `9.5078` | `7.2079` | `0.2403` | `22.9840` |
| `rolling_mean_7_baseline` | `baseline` | `7.4558` | `7.2542` | `5.5167` | `0.5578` | `17.5911` |
| `spark_mllib_rf` | `spark_mllib` | `7.2438` | `7.0398` | `5.3885` | `0.5835` | `17.1823` |
| `spark_mllib_gbt` | `spark_mllib` | `7.0862` | `6.9519` | `5.3548` | `0.5939` | `17.0750` |
| `xgboost_regressor` **(Champion by Val RMSE)** | `xgboost` | `6.9617` | `6.8346` | `5.2601` | `0.6075` | `16.7729` |

# Deployment & Operations Guide

This guide documents local and Docker Compose deployment for the **Chicago Crime Analytics Platform**.

## 1. Prerequisites

### Local Host Runtime
- **Python**: `>=3.11,<3.12` (verified on Python `3.11.9`) managed with [`uv`](https://docs.astral.sh/uv/).
- **Java**: **JDK 17** (`JAVA_HOME` configured; required by PySpark 3.5 and Delta Lake 3.2.0).
- **Optional on Windows**: Local Hadoop bare-filesystem execution is handled automatically in Python (`core/spark.py` attaches its bundled Windows helper JAR only on Windows hosts; Linux environments skip it).

### Containerized Stack (`compose.yaml`)
- **Docker Engine** & **Docker Compose v2**.
- Recommended host allocation for running the full `orchestration` + `streaming` stack simultaneously: **>= 16 GB host RAM** (Airflow + PostgreSQL + Kafka + MinIO + Spark Master + 2 Workers at 1800 MB RAM each) and **>= 20 GB free disk space** (for images, Delta/Parquet datasets, Kafka logs, and Spark shuffle scratch).

## 2. Environment Configuration

Create a local `.env` file from `.env.example` (never commit `.env` to version control):

| Variable | Purpose | Example / Default |
|---|---|---|
| `CHICAGO_CRIMES_PROFILE` | Config profile (`local` or `standalone`) | `local` |
| `CHICAGO_CRIMES_SPARK_MASTER` | Spark master URL in host runs | `local[4]` |
| `CHICAGO_CRIMES_SPARK_DRIVER_MEMORY` | Driver heap memory | `4g` |
| `CHICAGO_CRIMES_SPARK_EXECUTOR_MEMORY` | Executor heap memory | `2g` |
| `CHICAGO_CRIMES_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `MINIO_ENDPOINT` | MinIO S3 endpoint | `http://localhost:9000` (`http://minio:9000` in Compose) |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | MinIO credentials | `minioadmin` / `minioadmin` (override locally for non-ephemeral use) |
| `MINIO_BUCKET` | Target lakehouse bucket | `chicago-crimes` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap address | `localhost:9094` (host) / `kafka:9092` (Compose) |
| `KAFKA_CRIME_TOPIC` | Primary raw crime JSON topic | `chicago-crimes.raw` |
| `KAFKA_DLQ_TOPIC` | Malformed payload DLQ topic | `chicago-crimes.dlq` |
| `SOCRATA_APP_TOKEN` | Optional SODA API v2 app token for higher rate limits | *(unset)* |
| `AIRFLOW_SECRET_KEY` | Airflow webserver session secret | Replace default placeholder before multi-user use |
| `GCP_PROJECT_ID` / `BQ_DATASET` / `GOOGLE_APPLICATION_CREDENTIALS` | Optional BigQuery publish & dbt targets | *(unset; local runs use `--dry-run`)* |

## 3. Compose Services, Profiles & Ports

`compose.yaml` (`name: chicago-crimes-platform`) exposes profile-grouped services on `chicago-crimes-net`:

| Service | Profiles | Exposed Ports | Role |
|---|---|---|---|
| `minio` | `storage`, `spark`, `streaming`, `orchestration` | `9000` (S3 API), `9001` (Console UI) | S3-compatible Delta/Parquet lakehouse storage (`cgr.dev/chainguard/minio:latest`) |
| `minio-init` | `storage`, `spark`, `streaming`, `orchestration` | — | Creates `${MINIO_BUCKET}` on startup |
| `spark-master` | `spark`, `streaming`, `orchestration` | `7077` (RPC), `8080` (Master UI) | Spark Standalone cluster master (`chicago-crimes-spark:0.2.0`) |
| `spark-worker-1` | `spark`, `streaming`, `orchestration` | `8081` (Worker-1 UI) | Standalone worker (2 cores, 1800 MB RAM) |
| `spark-worker-2` | `spark`, `streaming`, `orchestration` | `8082` (Worker-2 UI) | Standalone worker (2 cores, 1800 MB RAM) |
| `spark-runner` | `tools` | — | Ad-hoc CLI runner inside `chicago-crimes-net` |
| `kafka` | `streaming`, `tools`, `orchestration` | `9094` (external host) / `9092` (internal) | Single-broker KRaft Kafka (`apache/kafka-native:3.8.1`) |
| `crime-stream` | `streaming` | — | Continuous Structured Streaming consumer (`chicago-crimes --profile standalone stream`) |
| `postgres` | `orchestration` | `5432` (internal only) | PostgreSQL 16 metadata DB for Airflow |
| `airflow-init` | `orchestration` | — | Runs `airflow db migrate` and bootstraps admin user |
| `airflow` | `orchestration` | `8085` -> `8080` (Web UI) | Airflow Webserver + Scheduler (`LocalExecutor`, `DAGS_ARE_PAUSED_AT_CREATION=false`) |

## 4. Startup & Verification

### Install Local CLI & Dependencies
```bash
uv sync --locked --all-extras
uv run chicago-crimes --help
```

### Validate Compose Configuration
```bash
docker compose --profile orchestration config --quiet
```

### Start Lakehouse + Spark + Kafka + Airflow Stack
```bash
# Build images and start storage, Spark cluster, Kafka, and Airflow orchestration
docker compose --profile orchestration build
docker compose --profile orchestration up -d

# Optionally add the continuous Spark Structured Streaming consumer
docker compose --profile orchestration --profile streaming up -d
```

### Verify Live Endpoints & Cluster Health
1. **Spark Master UI**: `http://localhost:8080` (verify `spark-worker-1` and `spark-worker-2` report `ALIVE` with 4 total cores).
2. **MinIO Console**: `http://localhost:9001` (verify bucket `${MINIO_BUCKET:-chicago-crimes}` exists).
3. **Airflow Web UI**: `http://localhost:8085` (login `${AIRFLOW_ADMIN_USER:-admin}` / `${AIRFLOW_ADMIN_PASSWORD:-admin}`; confirm DAGs `chicago_crimes_batch_pipeline` and `chicago_crimes_incremental_streaming` are registered and unpaused).
4. **Streaming / Ad-Hoc Verification via `tools` profile**:
   ```bash
   docker compose --profile tools run --rm spark-runner \
     python -m chicago_crimes --profile standalone analyze --list
   ```

## 5. Offline dbt Parsing

To validate `dbt/` SQL and YAML definitions without live Google Cloud credentials:

```bash
mkdir -p .cache/dbt_profiles
cp dbt/profiles.yml.example .cache/dbt_profiles/profiles.yml
GCP_PROJECT_ID=local-dry-run BQ_DATASET=chicago_crimes_gold \
  uv run dbt parse --project-dir dbt --profiles-dir .cache/dbt_profiles --no-send-anonymous-usage-stats
```

## 6. Safe Shutdown & Troubleshooting

- **Stop containers cleanly without deleting persistent MinIO / Kafka / Postgres volumes**:
  ```bash
  docker compose --profile orchestration --profile streaming down
  ```
- **Avoid `docker compose down -v`** unless you intend to wipe `minio-data`, `kafka-data`, `postgres-data`, `airflow-logs`, and `spark-logs`.
- **Airflow task containers fail to connect to Docker daemon**: Confirm Docker Desktop / Docker Engine is running and `./` paths in `compose.yaml` bind mounts (`./data`, `./configs`, `./artifacts`, `/var/run/docker.sock`) are accessible to the daemon.
- **Kafka host vs container hostname mismatch**: Use `localhost:9094` when invoking producers/tests directly from the host OS and `kafka:9092` inside `chicago-crimes-net` containers.
- **Windows / WSL2 disk growth**: Docker Desktop `ext4.vhdx` grows over repeated builds and image pulls; reclaim space with `docker system df`, `docker builder prune`, or WSL2 VHDX compaction if host disk free space drops low.

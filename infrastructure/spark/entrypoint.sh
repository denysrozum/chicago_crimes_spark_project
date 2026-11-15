#!/usr/bin/env bash
set -euo pipefail

ROLE="${CHICAGO_CRIMES_SPARK_ROLE:-runner}"
SPARK_HOME="$(python -c 'import pyspark, pathlib; print(pathlib.Path(pyspark.__file__).parent)')"
export SPARK_HOME
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$(command -v python)"

case "${ROLE}" in
  master)
    exec "${SPARK_HOME}/bin/spark-class" \
      org.apache.spark.deploy.master.Master \
      --host 0.0.0.0 \
      --port "${SPARK_MASTER_PORT:-7077}" \
      --webui-port "${SPARK_MASTER_WEBUI_PORT:-8080}"
    ;;
  worker)
    MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
    exec "${SPARK_HOME}/bin/spark-class" \
      org.apache.spark.deploy.worker.Worker \
      "${MASTER_URL}" \
      --cores "${SPARK_WORKER_CORES:-2}" \
      --memory "${SPARK_WORKER_MEMORY:-1800M}" \
      --webui-port "${SPARK_WORKER_WEBUI_PORT:-8081}"
    ;;
  minio-init)
    python - <<'PY'
import os, time
from chicago_crimes.core.config import StorageConfig
from chicago_crimes.storage.minio import ensure_bucket

cfg = StorageConfig(
    backend="s3a",
    s3_endpoint=os.environ.get("MINIO_ENDPOINT", "http://minio:9000"),
    s3_access_key=os.environ.get("MINIO_ROOT_USER", "minioadmin"),
    s3_secret_key=os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin"),
    s3_bucket=os.environ.get("MINIO_BUCKET", "chicago-crimes"),
)
for attempt in range(1, 31):
    try:
        bucket = ensure_bucket(cfg)
        print(f"MinIO bucket ready: {bucket}")
        break
    except Exception as exc:
        if attempt == 30:
            raise
        print(f"Waiting for MinIO ({attempt}/30): {exc}")
        time.sleep(2)
PY
    ;;
  runner)
    exec "$@"
    ;;
  *)
    echo "Unknown CHICAGO_CRIMES_SPARK_ROLE: ${ROLE}" >&2
    exit 1
    ;;
esac

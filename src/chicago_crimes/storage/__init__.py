"""Storage adapters for local Parquet, MinIO S3A, and Delta Lake."""

from chicago_crimes.storage.bigquery import (
    build_bq_ddl,
    build_bq_schema_spec,
    publish_gold_and_silver_to_bigquery,
    sanitize_bq_column_name,
)
from chicago_crimes.storage.delta import (
    delta_history,
    is_delta_table,
    merge_silver_delta,
    read_delta,
    write_delta,
)
from chicago_crimes.storage.lakehouse import read_table, sync_to_delta, write_table
from chicago_crimes.storage.local import (
    is_remote_uri,
    read_parquet,
    write_parquet,
    write_single_csv,
)
from chicago_crimes.storage.minio import (
    bucket_exists,
    build_s3_client,
    ensure_bucket,
    list_objects,
    s3a_uri,
)

__all__ = [
    "bucket_exists",
    "build_bq_ddl",
    "build_bq_schema_spec",
    "build_s3_client",
    "delta_history",
    "ensure_bucket",
    "is_delta_table",
    "is_remote_uri",
    "list_objects",
    "merge_silver_delta",
    "publish_gold_and_silver_to_bigquery",
    "read_delta",
    "read_parquet",
    "read_table",
    "s3a_uri",
    "sanitize_bq_column_name",
    "sync_to_delta",
    "write_delta",
    "write_parquet",
    "write_single_csv",
    "write_table",
]

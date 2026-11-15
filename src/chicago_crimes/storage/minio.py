"""MinIO / S3-compatible object storage helper utilities."""

from __future__ import annotations

from typing import Any

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from chicago_crimes.core.config import StorageConfig
from chicago_crimes.core.logging import get_logger

logger = get_logger("storage.minio")


def s3a_uri(bucket: str, key: str) -> str:
    """Construct a normalized `s3a://bucket/key` URI."""
    clean_bucket = bucket.strip("/")
    clean_key = key.lstrip("/")
    return f"s3a://{clean_bucket}/{clean_key}" if clean_key else f"s3a://{clean_bucket}"


def build_s3_client(storage_cfg: StorageConfig) -> Any:
    """Create a boto3 S3 client pointed at the configured MinIO/S3 endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=storage_cfg.s3_endpoint,
        aws_access_key_id=storage_cfg.s3_access_key,
        aws_secret_access_key=storage_cfg.s3_secret_key,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
        use_ssl=storage_cfg.s3_ssl_enabled,
    )


def bucket_exists(storage_cfg: StorageConfig, bucket_name: str | None = None) -> bool:
    """Check whether the target S3/MinIO bucket exists."""
    client = build_s3_client(storage_cfg)
    bucket = bucket_name or storage_cfg.s3_bucket
    try:
        client.head_bucket(Bucket=bucket)
        return True
    except ClientError:
        return False


def ensure_bucket(storage_cfg: StorageConfig, bucket_name: str | None = None) -> str:
    """Idempotently ensure the MinIO bucket exists and return its name."""
    client = build_s3_client(storage_cfg)
    bucket = bucket_name or storage_cfg.s3_bucket
    try:
        client.head_bucket(Bucket=bucket)
        logger.info("Verified MinIO bucket exists: %s (%s)", bucket, storage_cfg.s3_endpoint)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        logger.info("Created MinIO bucket: %s (%s)", bucket, storage_cfg.s3_endpoint)
    return bucket


def list_objects(
    storage_cfg: StorageConfig,
    prefix: str = "",
    bucket_name: str | None = None,
    max_keys: int = 200,
) -> list[str]:
    """List object keys inside the target MinIO bucket under `prefix`."""
    client = build_s3_client(storage_cfg)
    bucket = bucket_name or storage_cfg.s3_bucket
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    return [str(item["Key"]) for item in response.get("Contents", [])]


def delete_prefix(
    storage_cfg: StorageConfig,
    prefix: str,
    bucket_name: str | None = None,
) -> int:
    """Delete all objects under `prefix` in the MinIO bucket (e.g., ephemeral test paths)."""
    if not prefix or prefix.strip("/") == "":
        raise ValueError("Refusing to delete empty S3 prefix.")
    client = build_s3_client(storage_cfg)
    bucket = bucket_name or storage_cfg.s3_bucket
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)
    return deleted

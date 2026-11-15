"""Batch historical CSV, SODA API, and Kafka ingestion modules."""

from chicago_crimes.ingestion.bronze import (
    BronzeIngestionResult,
    build_bronze_dataframe,
    ingest_csv_to_bronze,
)
from chicago_crimes.ingestion.csv_loader import load_crime_csv, load_raw_csv_for_bronze
from chicago_crimes.ingestion.kafka_producer import (
    KafkaPublishResult,
    ensure_kafka_topics,
    ingest_soda_to_kafka,
    publish_bronze_records_to_kafka,
)
from chicago_crimes.ingestion.soda_client import (
    SodaFetchResult,
    SodaWatermark,
    fetch_soda_incremental,
    load_watermark,
    save_watermark,
    soda_record_to_bronze_dict,
)

__all__ = [
    "BronzeIngestionResult",
    "KafkaPublishResult",
    "SodaFetchResult",
    "SodaWatermark",
    "build_bronze_dataframe",
    "ensure_kafka_topics",
    "fetch_soda_incremental",
    "ingest_csv_to_bronze",
    "ingest_soda_to_kafka",
    "load_crime_csv",
    "load_raw_csv_for_bronze",
    "load_watermark",
    "publish_bronze_records_to_kafka",
    "save_watermark",
    "soda_record_to_bronze_dict",
]

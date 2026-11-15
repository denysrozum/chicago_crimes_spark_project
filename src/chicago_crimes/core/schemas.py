"""Canonical PySpark schemas for Chicago Crime datasets."""

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

RAW_CRIME_SCHEMA = StructType(
    [
        StructField("ID", IntegerType(), True),
        StructField("Case Number", StringType(), True),
        StructField("Date", TimestampType(), True),
        StructField("Block", StringType(), True),
        StructField("IUCR", StringType(), True),
        StructField("Primary Type", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Location Description", StringType(), True),
        StructField("Arrest", BooleanType(), True),
        StructField("Domestic", BooleanType(), True),
        StructField("Beat", IntegerType(), True),
        StructField("District", IntegerType(), True),
        StructField("Ward", IntegerType(), True),
        StructField("Community Area", IntegerType(), True),
        StructField("FBI Code", StringType(), True),
        StructField("X Coordinate", IntegerType(), True),
        StructField("Y Coordinate", IntegerType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Updated On", TimestampType(), True),
        StructField("Latitude", DoubleType(), True),
        StructField("Longitude", DoubleType(), True),
        StructField("Location", StringType(), True),
    ]
)

RAW_CRIME_COLUMNS: tuple[str, ...] = tuple(field.name for field in RAW_CRIME_SCHEMA.fields)

CORRUPT_RECORD_COL = "_corrupt_record"

RAW_CRIME_STRING_SCHEMA = StructType(
    [StructField(field.name, StringType(), True) for field in RAW_CRIME_SCHEMA.fields]
    + [StructField(CORRUPT_RECORD_COL, StringType(), True)]
)

BRONZE_METADATA_COLUMNS: tuple[str, ...] = (
    "ingestion_timestamp",
    "source_id",
    "ingestion_batch_id",
    "corrupt_record",
)

MIN_VALID_YEAR = 2000
CHICAGO_LAT_MIN = 41.5
CHICAGO_LAT_MAX = 42.1
CHICAGO_LON_MIN = -88.0
CHICAGO_LON_MAX = -87.4
DISTRICT_MIN = 1
DISTRICT_MAX = 31
WARD_MIN = 1
WARD_MAX = 50
COMMUNITY_AREA_MIN = 1
COMMUNITY_AREA_MAX = 77

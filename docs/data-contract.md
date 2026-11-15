# Data Contract & Accounting Invariants

## Source System

- **Official Dataset**: City of Chicago Crimes (`https://data.cityofchicago.org/resource/ijzp-q8t2.json`).
- **Live Schema Verification**: Observed `8,679,677` rows (`max_updated_on=2026-11-13T15:48:31.000`). Recent SODA JSON responses omit coordinate fields completely when coordinates are missing rather than serializing `null` values; `soda_client.py` explicitly materializes missing optional keys as `None`.

## Medallion Layers

### Bronze

- Preserves all 22 CSV source columns as **raw strings** without silent dropping or type coercion failures (`PERMISSIVE` mode with `_corrupt_record` -> `corrupt_record`).
- Attaches immutable lineage columns: `ingestion_timestamp`, `source_id`, `ingestion_batch_id`.

### Silver

- Casts types explicitly via `try_cast` without failing jobs on malformed text tokens.
- Validates `ID > 0`, non-null `Date` (`year >= 2001`), valid `Updated On` if present, non-empty `Primary Type`, and coordinate numeric parseability.
- **Zero Synthetic Geographic Imputation**: Coordinates (`Latitude`, `Longitude`, `X Coordinate`, `Y Coordinate`, `Location`) are never replaced with neighborhood or city centroids. Missing or partial/out-of-bounds coordinates are nullified and flagged via `has_valid_geo` and `geo_quality_flag` (`valid`, `missing`, `partial_missing_nullified`, `out_of_bounds_nullified`, `invalid`).
- **Administrative Preservation**: Missing `District` (`1..31`), `Ward` (`1..50`), or `Community Area` (`1..77`) values are preserved with `has_valid_admin` and `admin_quality_flag` rather than dropping incident records.
- **Deterministic Deduplication / Upserts**: Batch Silver resolves duplicate `ID`s by latest `Updated On`; Delta streaming `MERGE` only updates target rows when `source.Updated On >= target.Updated On`.

### Strict Row Accounting Invariant

Every input record across both batch Silver processing and Structured Streaming micro-batches satisfies:

```text
accepted_records + superseded_records + quarantined_records == input_records
```

For Kafka Structured Streaming:

```text
kafka_input_rows == bronze_written_rows + malformed_json_rows
bronze_written_rows == silver_accepted_deduped + silver_superseded + silver_quarantined
```

Verified full-dataset batch Silver accounting (`data/silver/chicago_crimes/_silver_accounting.json`):

| Metric | Observed Count |
| --- | --- |
| `input_records` | `8,495,400` |
| `accepted_records` | `8,495,400` |
| `superseded_records` | `0` |
| `quarantined_records` | `0` |
| `missing_geo_records` (preserved with `has_valid_geo=false`) | `94,765` |
| `out_of_bounds_geo_nullified` | `149` |
| `missing_admin_records` (preserved instead of legacy drop) | `616,418` |
| `out_of_range_admin_nullified` | `76` |

### Gold Analytical Tables

Materialized by `chicago-crimes gold` (`data/gold/_gold_manifest.json`) with documented denominators (`arrest_count / incident_count * 100`, `domestic_count / incident_count * 100`, and valid-geo filtering on spatial cells):

- `daily_citywide_counts` (`9,169` rows)
- `monthly_crime_summary` (`303` rows; includes boundary `2000-12` under `UTC` session timezone on legacy-materialized Silver `Date` values)
- `district_community_summary` (`422` rows)
- `category_distribution` (`34` rows)
- `temporal_hourly_profile` (`2,016` rows)
- `spatial_valid_grid_summary` (`738` rows; built strictly from `has_valid_geo = true`)
- `daily_district_counts` (`201,982` rows; consumed by chronological ML forecasting)

## Analytical Task Registry (`AQ-01`..`AQ-24`)

| ID | Category | Domain Name | Legacy Name |
| --- | --- | --- | --- |
| `AQ-01` | `temporal` | `night_crimes_by_type` | `Q1_D_Night_Crimes` |
| `AQ-02` | `temporal` | `weekend_vs_weekday_distribution` | `Q2_D_Weekend_vs_Weekday` |
| `AQ-03` | `temporal` | `monthly_moving_average_crimes` | `Q6_D_Moving_Average_Crimes` |
| `AQ-04` | `temporal` | `domestic_crimes_by_month` | `Q2_J_Domestic_Monthly` |
| `AQ-05` | `temporal` | `hourly_crime_distribution_citywide` | `Q6_M_Hourly_All_City` |
| `AQ-06` | `temporal` | `daytime_crimes_by_type` | `Q3_V_Day_Crimes` |
| `AQ-07` | `geographic` | `top_theft_districts` | `Q1_J_Top_Theft_Districts` |
| `AQ-08` | `geographic` | `top5_crime_types_locations` | `Q3_J_Top5_Crime_Locations` |
| `AQ-09` | `geographic` | `top_crime_streets` | `Q4_J_Top_Streets` |
| `AQ-10` | `geographic` | `top_crime_type_per_district` | `Q5_J_Top_Crime_Per_District` |
| `AQ-11` | `geographic` | `most_concentrated_blocks_by_district_share` | `Q6_J_Most_Dangerous_Blocks_Share` |
| `AQ-12` | `geographic` | `top_criminal_damage_locations` | `Q1_V_Criminal_Damage_Locations` |
| `AQ-13` | `geographic` | `above_average_crime_beats` | `Q5_V_High_Crime_Beats_Vs_Avg` |
| `AQ-14` | `categories` | `narcotics_vs_assault_by_district` | `Q3_D_Narcotics_vs_Assault_By_District` |
| `AQ-15` | `categories` | `domestic_crimes_by_district` | `Q4_D_Domestic_By_District` |
| `AQ-16` | `categories` | `weekday_vs_weekend_by_district` | `Q3_M_Weekday_Vs_Weekend_By_District` |
| `AQ-17` | `categories` | `domestic_vs_nondomestic_top_streets` | `Q4_M_Domestic_Vs_NonDomestic_Streets` |
| `AQ-18` | `categories` | `top_crimes_in_community_area` | `Q5_M_Crime_Type_Rank_In_Community` |
| `AQ-19` | `categories` | `street_crimes_by_type` | `Q2_V_Street_Crimes` |
| `AQ-20` | `arrests` | `arrest_rate_by_ward` | `Q5_D_Arrest_Ratio_By_Ward` |
| `AQ-21` | `arrests` | `arrests_by_crime_type` | `Q1_M_Arrest_By_Type` |
| `AQ-22` | `arrests` | `arrest_share_by_crime_type_year` | `Q2_M_Arrest_Share_2025` |
| `AQ-23` | `arrests` | `cumulative_arrests_by_year` | `Q4_V_Cumulative_Arrests_By_Year` |
| `AQ-24` | `arrests` | `arrest_success_by_fbi_code` | `Q6_V_FBI_Code_Arrest_Success` |


---

## Raw Input & Streaming Event Contracts

### Raw 22-Column CSV Schema (`RAW_CRIME_SCHEMA`)
All batch CSV inputs are ingested non-destructively as `StringType` columns plus permissive `_corrupt_record` (`RAW_CRIME_STRING_SCHEMA`) before Silver casting:

1. `ID` (`IntegerType` in Silver; required `> 0`)
2. `Case Number` (`StringType`)
3. `Date` (`TimestampType` in Silver; required, `year >= 2000` and `<= current_timestamp() + 7 days`)
4. `Block` (`StringType`)
5. `IUCR` (`StringType`)
6. `Primary Type` (`StringType`; trimmed & uppercased in Silver; required)
7. `Description` (`StringType`)
8. `Location Description` (`StringType`)
9. `Arrest` (`BooleanType` in Silver; parsed from `true/1/t/yes/y` or `false/0/f/no/n`)
10. `Domestic` (`BooleanType` in Silver)
11. `Beat` (`IntegerType`)
12. `District` (`IntegerType` in Silver; valid range `[1, 31]`, otherwise nullified)
13. `Ward` (`IntegerType` in Silver; valid range `[1, 50]`, otherwise nullified)
14. `Community Area` (`IntegerType` in Silver; valid range `[1, 77]`, otherwise nullified)
15. `FBI Code` (`StringType`)
16. `X Coordinate` (`IntegerType` in Silver; nullified when lat/lon out of bounds)
17. `Y Coordinate` (`IntegerType` in Silver; nullified when lat/lon out of bounds)
18. `Year` (`IntegerType` preserved as `source_year` in Silver)
19. `Updated On` (`TimestampType` in Silver; used descending for deterministic deduplication per `ID`)
20. `Latitude` (`DoubleType` in Silver; valid Chicago range `[41.5, 42.1]`)
21. `Longitude` (`DoubleType` in Silver; valid Chicago range `[-88.0, -87.4]`)
22. `Location` (`StringType`)

### Bronze & Streaming Lineage Columns
- **Batch Bronze (`ingestion/bronze.py`)**: appends `ingestion_timestamp`, `source_id`, `ingestion_batch_id`, and `corrupt_record`.
- **Streaming Bronze (`streaming/crime_stream.py`)**: parses Kafka JSON via `KAFKA_JSON_SCHEMA` (`mode=PERMISSIVE`, `_corrupt_json_record`) and adds `kafka_topic`, `kafka_partition`, `kafka_offset`, `kafka_timestamp`, deterministic `stream_event_id` (`sha2(concat_ws('|', topic, partition, offset), 256)`), and `microbatch_id`.
- **Streaming DLQ**: malformed or blank Kafka payloads are routed with `raw_payload`, Kafka coordinates, `error_reason="malformed_json"`, `stream_event_id`, `microbatch_id`, and `quarantined_at_utc`.

## Silver Casting, Quarantine, Quality Flags & Derived Features

- **Safe Casting & Timestamps**: Numeric fields use `try_cast`; `Date` and `Updated On` accept `MM/dd/yyyy hh:mm:ss a`, `MM/dd/yyyy HH:mm:ss`, `yyyy-MM-dd'T'HH:mm:ss.SSS`, `yyyy-MM-dd'T'HH:mm:ss`, `yyyy-MM-dd HH:mm:ss`, and `yyyy-MM-dd`. Spark runs with `spark.sql.session.timeZone=UTC`, whereas historical CSV strings represent Chicago wall-clock timestamps (note boundary records near `2000-12-31T22:00:00Z`).
- **Quarantine Reasons (`quarantine_reason`, `quarantined_at`)**:
  - `corrupt_csv_row`
  - `missing_or_invalid_id`
  - `missing_or_invalid_date`
  - `date_out_of_valid_range`
  - `invalid_updated_on`
  - `missing_primary_type`
  - `invalid_coordinate_format`
  - `out_of_bounds_coordinates` *(optional; only when `--quarantine-oob-geo` is passed)*
- **Non-Destructive Quality Flags**:
  - `has_valid_geo` (`boolean`) and `geo_quality_flag` in `{valid, missing, partial_missing_nullified, out_of_bounds_nullified, invalid}`.
  - `has_valid_admin` (`boolean`) and `admin_quality_flag` in `{valid, partial_or_missing, out_of_range_nullified}`.
- **Derived Enriched Silver Columns**: `ts`, `hour`, `year`, `month`, `day_of_week_num` (`1=Sun..7=Sat`), `day_of_week`, `block_number` (non-imputed in Silver), `direction`, `street_type`, `street_name`, `geo_grid` (`floor(coord * round(1/grid_size))`, default `0.01`), plus `source_year`.
- **Accounting Invariant**: Every Silver run verifies `input_records == accepted_records + superseded_records + quarantined_records`.

## dbt BigQuery Layer Contracts (`dbt/models/`)

- `stg_chicago_crimes_silver` *(view)*: normalizes column naming (`crime_id`, `crime_date`, `incident_timestamp`, `is_arrest`, `is_domestic`, `district_id`, `ward_id`, `community_area_id`, `updated_at`) from source `chicago_crimes_lakehouse.silver_crimes`.
- `mart_daily_citywide_crime_trends` *(table, partitioned by `crime_date`, clustered by `year`, `month`)*: citywide daily counts, arrest/domestic rates, and `rolling_7d_avg_incidents`.
- `mart_district_crime_summary` *(table, clustered by `district_id`)*: district incident totals, valid geo share, arrest rate, and mean valid coordinates.
- `mart_category_arrest_profile` *(table, clustered by `primary_type`)*: category totals, `citywide_share_percent`, `arrest_rate_percent`, and `domestic_rate_percent`.

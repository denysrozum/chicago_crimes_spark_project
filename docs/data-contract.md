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
- **Deterministic Deduplication & Replay-Safe Delta `MERGE`**:
  - **In-batch deduplication (`_rank_by_id`, `_deduplicate_source_batch`)**: Ranks duplicate `ID` rows by `Updated On DESC NULLS LAST` with a deterministic tie-breaker on the SHA-256 digest of normalized business payload columns (`Case Number`, `Date`, `Block`, `IUCR`, `Primary Type`, `Description`, `Location Description`, `Arrest`, `Domestic`, `Beat`, `District`, `Ward`, `Community Area`, `FBI Code`, `X Coordinate`, `Y Coordinate`, `source_year`, `Updated On`, `Latitude`, `Longitude`, `Location`).
  - **Delta Lake `merge_silver_delta()` update policy**:
    1. **Strictly newer source version**: `src.Updated On > target.Updated On` (both non-null) updates the target record.
    2. **Older source version rejected**: `src.Updated On < target.Updated On` is ignored and never overwrites a newer target record.
    3. **Null-timestamp source rejected against valid target**: `src.Updated On IS NULL AND target.Updated On IS NOT NULL` is rejected so missing update timestamps never overwrite valid target versions.
    4. **Valid-timestamp source upgrades null-timestamp target**: `target.Updated On IS NULL AND src.Updated On IS NOT NULL` replaces the unversioned target record.
    5. **Equal-timestamp conflict resolution**: When `src.Updated On <=> target.Updated On` (including both `NULL`), the update proceeds if and only if `_src_payload_digest > _target_payload_digest`, guaranteeing commutativity, determinism, and zero no-op writes on identical replays.
    6. **Safe new-record insertion**: Unmatched `ID` records (including those with `Updated On IS NULL`) are inserted via `whenNotMatchedInsertAll()`.

### Chicago Timestamp & Timezone Semantics (`America/Chicago` vs `UTC`)

The platform explicitly separates six distinct temporal concepts:

1. **Chicago Local Event Time (`Date`, `ts`)**:
   - Raw historical CSV strings (`MM/dd/yyyy hh:mm:ss a`, e.g., `01/01/2001 12:00:00 AM`) and SODA floating timestamps (`yyyy-MM-dd'T'HH:mm:ss.SSS`) represent wall-clock local time in `America/Chicago` (`CST = UTC-6` in winter; `CDT = UTC-5` in summer).
   - Spark sessions run with `spark.sql.session.timeZone=UTC` for deterministic cross-host execution, so `Date` and `ts` store the exact Chicago wall-clock components without host-OS timezone drift.
2. **Derived UTC Event Instant (`ts_utc`) & DST Classification (`timestamp_dst_flag`)**:
   - `ts_utc` converts `ts` from `America/Chicago` wall-clock time to an unambiguous UTC instant via `to_utc_timestamp(ts, 'America/Chicago')` **only** when the local wall-clock time is unambiguous:
     - Winter CST example (`UTC-6`): `2001-01-01 00:00:00` local (`America/Chicago`) -> `2001-01-01T06:00:00Z` (`timestamp_dst_flag = "valid_cst"`).
     - Summer CDT example (`UTC-5`): `2025-07-04 19:30:00` local (`America/Chicago`) -> `2025-07-05T00:30:00Z` (`timestamp_dst_flag = "valid_cdt"`).
   - **DST Spring-Forward Nonexistent Hour (`dst_spring_gap_nonexistent`)**: Local wall-clock values falling inside the spring DST transition gap (e.g., `02:00:00..02:59:59` on the transition Sunday) do not exist in `America/Chicago`. `ts` and local calendar fields are preserved as reported by the source, while `ts_utc` is set to `NULL` rather than inventing a synthetic UTC offset (`253` records in the full dataset).
   - **DST Fall-Back Ambiguous Hour (`dst_fall_fold_ambiguous`)**: Local wall-clock values falling inside the repeated `01:00:00..01:59:59` autumn transition hour occur twice (once in CDT `UTC-5` and once in CST `UTC-6`) without an offset indicator in the source data. `ts` and local calendar fields are preserved, `timestamp_dst_flag` is set to `"dst_fall_fold_ambiguous"`, and `ts_utc` is set to `NULL` (`1,662` records in the full dataset).
3. **Source `Updated On` Semantics**:
   - Represents the upstream Chicago Police Department / SODA record modification timestamp (`America/Chicago` wall-clock floating timestamp), used strictly for version ordering (`Updated On DESC NULLS LAST`) and incremental SODA polling watermarks.
4. **Ingestion & Streaming Timestamps (`ingestion_timestamp`, `kafka_timestamp`, `quarantined_at_utc`)**:
   - Recorded as true UTC instants at the moment of batch ingestion, Kafka broker append, or DLQ/quarantine routing.
5. **Calendar Dates for Historical Aggregation (`crime_date`, `year_month`, `year`, `month`, `hour`, `day_of_week_num`, `day_of_week`)**:
   - Derived directly from Chicago local wall-clock time (`ts`) so midnight and daily/monthly boundaries reflect actual Chicago calendar dates (`2001-01-01` through `2026-02-07`, `302` months `2001-01..2026-02`).
6. **Calendar Features for ML Forecasting (`year`, `month`, `day_of_month`, `day_of_week_num`, `is_weekend`)**:
   - Constructed from the Chicago local calendar date (`crime_date`) in `daily_district_counts` before computing leakage-free causal lags (`lag_1`, `lag_7`) and rolling statistics (`rolling_mean_7`, `rolling_std_7`, `rolling_mean_14`).

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
| `valid_cst` (`timestamp_dst_flag`) | `2,952,268` |
| `valid_cdt` (`timestamp_dst_flag`) | `5,541,217` |
| `dst_fall_fold_ambiguous` (`ts_utc = NULL`) | `1,662` |
| `dst_spring_gap_nonexistent` (`ts_utc = NULL`) | `253` |

### Gold Analytical Tables

Materialized by `chicago-crimes gold` (`data/gold/_gold_manifest.json`) on Chicago local calendar boundaries (`ts`) with documented denominators (`arrest_count / incident_count * 100`, `domestic_count / incident_count * 100`, and valid-geo filtering on spatial cells):

- `daily_citywide_counts` (`9,169` rows; `2001-01-01`..`2026-02-07`)
- `monthly_crime_summary` (`302` rows; `2001-01`..`2026-02`)
- `district_community_summary` (`422` rows)
- `category_distribution` (`34` rows)
- `temporal_hourly_profile` (`2,016` rows)
- `spatial_valid_grid_summary` (`738` rows; built strictly from `has_valid_geo = true`)
- `daily_district_counts` (`201,970` rows; consumed by chronological ML forecasting)

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

- **Safe Casting & Timestamps**: Numeric fields use `try_cast`; `Date` and `Updated On` accept `MM/dd/yyyy hh:mm:ss a`, `MM/dd/yyyy HH:mm:ss`, `yyyy-MM-dd'T'HH:mm:ss.SSS`, `yyyy-MM-dd'T'HH:mm:ss`, `yyyy-MM-dd HH:mm:ss`, and `yyyy-MM-dd`. Spark runs with `spark.sql.session.timeZone=UTC`, preserving Chicago local wall-clock fields (`Date`, `ts`, `year`, `month`, `hour`, `day_of_week_num`, `day_of_week`) while explicitly deriving `ts_utc` (`America/Chicago` -> `UTC`, or `NULL` on spring-gap/fall-fold rows) and `timestamp_dst_flag`.
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
- **Derived Enriched Silver Columns**: `ts`, `ts_utc`, `timestamp_dst_flag`, `hour`, `year`, `month`, `day_of_week_num` (`1=Sun..7=Sat`), `day_of_week`, `block_number` (non-imputed in Silver), `direction`, `street_type`, `street_name`, `geo_grid` (`floor(coord * round(1/grid_size))`, default `0.01`), plus `source_year`.
- **Accounting Invariant**: Every Silver run verifies `input_records == accepted_records + superseded_records + quarantined_records`.

## dbt BigQuery Layer Contracts (`dbt/models/`)

- `stg_chicago_crimes_silver` *(view)*: normalizes column naming (`crime_id`, `crime_date`, `incident_timestamp`, `is_arrest`, `is_domestic`, `district_id`, `ward_id`, `community_area_id`, `updated_at`) from source `chicago_crimes_lakehouse.silver_crimes`.
- `mart_daily_citywide_crime_trends` *(table, partitioned by `crime_date`, clustered by `year`, `month`)*: citywide daily counts, arrest/domestic rates, and `rolling_7d_avg_incidents`.
- `mart_district_crime_summary` *(table, clustered by `district_id`)*: district incident totals, valid geo share, arrest rate, and mean valid coordinates.
- `mart_category_arrest_profile` *(table, clustered by `primary_type`)*: category totals, `citywide_share_percent`, `arrest_rate_percent`, and `domestic_rate_percent`.

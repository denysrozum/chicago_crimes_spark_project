{{ config(materialized='view') }}

with source_silver as (
    select *
    from {{ source('chicago_crimes_lakehouse', 'silver_crimes') }}
)

select
    cast(id as int64) as crime_id,
    case_number,
    timestamp(date) as incident_timestamp,
    date(timestamp(date)) as crime_date,
    cast(extract(year from timestamp(date)) as int64) as year,
    cast(extract(month from timestamp(date)) as int64) as month,
    cast(extract(hour from timestamp(date)) as int64) as hour_of_day,
    trim(upper(primary_type)) as primary_type,
    description as crime_description,
    location_description,
    cast(arrest as bool) as is_arrest,
    cast(domestic as bool) as is_domestic,
    cast(district as int64) as district_id,
    cast(ward as int64) as ward_id,
    cast(community_area as int64) as community_area_id,
    fbi_code,
    cast(latitude as float64) as latitude,
    cast(longitude as float64) as longitude,
    cast(has_valid_geo as bool) as has_valid_geo,
    geo_quality_flag,
    cast(has_valid_admin as bool) as has_valid_admin,
    admin_quality_flag,
    timestamp(updated_on) as updated_at
from source_silver
where id is not null
  and date is not null
  and primary_type is not null

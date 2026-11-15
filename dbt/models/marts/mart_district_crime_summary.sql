{{
    config(
        materialized='table',
        cluster_by=['district_id']
    )
}}

select
    district_id,
    count(*) as incident_count,
    sum(case when is_arrest then 1 else 0 end) as arrest_count,
    sum(case when is_domestic then 1 else 0 end) as domestic_count,
    sum(case when has_valid_geo then 1 else 0 end) as valid_geo_count,
    safe_divide(sum(case when is_arrest then 1 else 0 end), count(*)) * 100.0 as arrest_rate_percent,
    safe_divide(sum(case when has_valid_geo then 1 else 0 end), count(*)) * 100.0 as valid_geo_share_percent,
    avg(case when has_valid_geo then latitude end) as mean_valid_latitude,
    avg(case when has_valid_geo then longitude end) as mean_valid_longitude
from {{ ref('stg_chicago_crimes_silver') }}
where district_id is not null
group by district_id

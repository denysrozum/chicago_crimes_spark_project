{{
    config(
        materialized='table',
        partition_by={'field': 'crime_date', 'data_type': 'date'},
        cluster_by=['year', 'month']
    )
}}

with daily_agg as (
    select
        crime_date,
        year,
        month,
        count(*) as incident_count,
        sum(case when is_arrest then 1 else 0 end) as arrest_count,
        sum(case when is_domestic then 1 else 0 end) as domestic_count,
        sum(case when has_valid_geo then 1 else 0 end) as valid_geo_count
    from {{ ref('stg_chicago_crimes_silver') }}
    group by crime_date, year, month
)

select
    crime_date,
    year,
    month,
    incident_count,
    arrest_count,
    domestic_count,
    valid_geo_count,
    safe_divide(arrest_count, incident_count) * 100.0 as arrest_rate_percent,
    safe_divide(domestic_count, incident_count) * 100.0 as domestic_rate_percent,
    avg(incident_count) over (
        order by crime_date
        rows between 6 preceding and current row
    ) as rolling_7d_avg_incidents
from daily_agg

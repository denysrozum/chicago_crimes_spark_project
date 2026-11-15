{{
    config(
        materialized='table',
        cluster_by=['primary_type']
    )
}}

with category_totals as (
    select
        primary_type,
        count(*) as incident_count,
        sum(case when is_arrest then 1 else 0 end) as arrest_count,
        sum(case when is_domestic then 1 else 0 end) as domestic_count
    from {{ ref('stg_chicago_crimes_silver') }}
    group by primary_type
),
grand_total as (
    select sum(incident_count) as all_incidents from category_totals
)

select
    c.primary_type,
    c.incident_count,
    c.arrest_count,
    c.domestic_count,
    safe_divide(c.incident_count, g.all_incidents) * 100.0 as citywide_share_percent,
    safe_divide(c.arrest_count, c.incident_count) * 100.0 as arrest_rate_percent,
    safe_divide(c.domestic_count, c.incident_count) * 100.0 as domestic_rate_percent
from category_totals c
cross join grand_total g

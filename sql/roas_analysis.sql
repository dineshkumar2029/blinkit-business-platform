-- ============================================================================
-- SQL Query: Master Analytical View (Marketing Spend vs. Sales Revenue)
-- Project: AI-Powered Blinkit Business Decision Platform
-- Description:
--   Aggregates transactional orders data to a daily level, joins it with
--   daily marketing performance data, and calculates daily ROAS (Return on Ad Spend).
--   Includes logic to handle granularity mismatch, prevent division by zero,
--   and flag underperforming campaigns where ROAS drops below 2.0x.
-- ============================================================================

WITH Daily_Sales AS (
    -- Step A: Squash transactional orders (thousands of rows) into one row per day
    SELECT 
        CAST(order_date AS DATE) AS order_day,
        COUNT(order_id) AS total_orders,
        SUM(order_total) AS total_revenue,
        -- Calculate the average delivery delay in minutes
        -- Delay is positive if actual_time > promised_time
        ROUND(
            AVG(
                CASE 
                    WHEN actual_time > promised_time THEN EXTRACT(EPOCH FROM (actual_time - promised_time)) / 60.0
                    ELSE 0.0
                END
            )::numeric, 
            2
        ) AS avg_delay_minutes,
        -- Calculate delay rate (percentage of orders that were late)
        ROUND(
            (COUNT(CASE WHEN actual_time > promised_time THEN 1 END) * 100.0 / COUNT(order_id))::numeric,
            2
        ) AS delay_rate_pct
    FROM orders
    GROUP BY CAST(order_date AS DATE)
),

Daily_Marketing AS (
    -- Step B: Aggregate marketing spend and impressions by day across all channels
    SELECT 
        date AS marketing_day,
        SUM(spend) AS total_spend,
        SUM(impressions) AS total_impressions
    FROM marketing_performance
    GROUP BY date
)

-- Step C: Join the daily transactional sales data with time-series marketing data
SELECT 
    COALESCE(m.marketing_day, s.order_day) AS business_date,
    COALESCE(s.total_orders, 0) AS total_orders,
    COALESCE(s.total_revenue, 0.0) AS total_revenue,
    COALESCE(m.total_spend, 0.0) AS total_spend,
    COALESCE(m.total_impressions, 0) AS total_impressions,
    
    -- Calculate ROAS: Revenue / Spend. Handle zero spend to avoid division-by-zero errors.
    CASE 
        WHEN COALESCE(m.total_spend, 0.0) = 0.0 AND COALESCE(s.total_revenue, 0.0) > 0.0 THEN 999.99 -- Represents "Infinite ROAS" (Organic)
        WHEN COALESCE(m.total_spend, 0.0) = 0.0 THEN 0.0
        ELSE ROUND((s.total_revenue / m.total_spend)::numeric, 2)
    END AS roas,
    
    -- Calculate average delay minutes and late order rate
    COALESCE(s.avg_delay_minutes, 0.0) AS avg_delay_minutes,
    COALESCE(s.delay_rate_pct, 0.0) AS delay_rate_pct,
    
    -- Business Alert logic: If ROAS drops below 2.0x, trigger alert
    CASE 
        WHEN COALESCE(m.total_spend, 0.0) = 0.0 THEN 'Organic / No Ad Spend'
        WHEN (s.total_revenue / m.total_spend) < 2.0 THEN '⚠ Loss-Making Campaign (ROAS < 2.0x)'
        ELSE '🟢 Profitable Campaign'
    END AS campaign_status
FROM Daily_Marketing m
FULL OUTER JOIN Daily_Sales s 
    ON m.marketing_day = s.order_day
ORDER BY business_date DESC;

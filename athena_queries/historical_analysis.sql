-- ── 1. ALL BTC DATA FOR A GIVEN DATE ─────────────────────────────────────────
SELECT *
FROM btc_raw
WHERE date = '2024-01-15'
ORDER BY timestamp ASC;


-- ── 2. DAILY PRICE SUMMARY ────────────────────────────────────────────────────
SELECT
    date,
    MIN(CAST(low AS DOUBLE))        AS day_low,
    MAX(CAST(high AS DOUBLE))       AS day_high,
    AVG(CAST(close AS DOUBLE))      AS avg_close,
    SUM(CAST(volume AS DOUBLE))     AS total_volume,
    COUNT(*)                        AS tick_count
FROM btc_raw
GROUP BY date
ORDER BY date DESC;


-- ── 3. ALL DETECTED ANOMALIES ─────────────────────────────────────────────────
SELECT
    date,
    timestamp,
    close,
    zscore
FROM btc_raw
WHERE is_anomaly = true
ORDER BY ABS(CAST(zscore AS DOUBLE)) DESC;


-- ── 4. HOURLY AVERAGE CLOSE PRICE ────────────────────────────────────────────
SELECT
    date,
    SUBSTR(timestamp, 12, 2)        AS hour,
    AVG(CAST(close AS DOUBLE))      AS avg_close,
    COUNT(*)                        AS ticks
FROM btc_raw
GROUP BY date, SUBSTR(timestamp, 12, 2)
ORDER BY date DESC, hour ASC;


-- ── 5. ROLLING ZSCORE EXTREMES ───────────────────────────────────────────────
SELECT
    timestamp,
    close,
    zscore
FROM btc_raw
WHERE zscore != 'N/A'
ORDER BY ABS(CAST(zscore AS DOUBLE)) DESC
LIMIT 20;


-- ── 6. PULL TRAINING DATA FOR ML MODEL ───────────────────────────────────────
SELECT
    CAST(close AS DOUBLE)                           AS close,
    CAST(volume AS DOUBLE)                          AS volume,
    CAST(high AS DOUBLE) - CAST(low AS DOUBLE)      AS price_range,
    CAST(open AS DOUBLE)                            AS open,
    CAST(high AS DOUBLE)                            AS high,
    CAST(low AS DOUBLE)                             AS low
FROM btc_raw
WHERE close != '0'
  AND volume != '0'
ORDER BY timestamp ASC;
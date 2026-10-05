-- Healthcare Patient Analytics
-- SQL analysis examples for the UCI Diabetes 130-US Hospitals dataset


-- 1. Overall 30-day readmission rate
SELECT
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmitted_within_30_days,
    ROUND(
        100.0 * SUM(readmitted_30d) / COUNT(*),
        2
    ) AS readmission_rate_pct
FROM patient_encounters;


-- 2. Readmission rate by age group
SELECT
    age,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmitted_within_30_days,
    ROUND(
        100.0 * AVG(readmitted_30d),
        2
    ) AS readmission_rate_pct
FROM patient_encounters
GROUP BY age
ORDER BY readmission_rate_pct DESC;


-- 3. Readmission rate by length of stay
SELECT
    time_in_hospital,
    COUNT(*) AS encounters,
    ROUND(
        100.0 * AVG(readmitted_30d),
        2
    ) AS readmission_rate_pct
FROM patient_encounters
GROUP BY time_in_hospital
ORDER BY time_in_hospital;


-- 4. Readmission rate by prior healthcare utilization
SELECT
    prior_utilization,
    COUNT(*) AS encounters,
    ROUND(
        100.0 * AVG(readmitted_30d),
        2
    ) AS readmission_rate_pct
FROM patient_encounters
GROUP BY prior_utilization
ORDER BY prior_utilization;


-- 5. High-utilization patient segment
SELECT
    CASE
        WHEN prior_utilization = 0
            THEN 'No prior utilization'
        WHEN prior_utilization BETWEEN 1 AND 4
            THEN 'Low utilization'
        WHEN prior_utilization BETWEEN 5 AND 9
            THEN 'Moderate utilization'
        ELSE 'High utilization'
    END AS utilization_segment,
    COUNT(*) AS encounters,
    ROUND(
        100.0 * AVG(readmitted_30d),
        2
    ) AS readmission_rate_pct
FROM patient_encounters
GROUP BY
    CASE
        WHEN prior_utilization = 0
            THEN 'No prior utilization'
        WHEN prior_utilization BETWEEN 1 AND 4
            THEN 'Low utilization'
        WHEN prior_utilization BETWEEN 5 AND 9
            THEN 'Moderate utilization'
        ELSE 'High utilization'
    END
ORDER BY readmission_rate_pct DESC;

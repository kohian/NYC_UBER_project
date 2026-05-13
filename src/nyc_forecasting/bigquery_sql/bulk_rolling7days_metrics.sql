CREATE TABLE IF NOT EXISTS `nyc-uber-494107.nyc_forecasting.prediction_metrics_rolling_7d` (
  metric_date DATE,
  window_start_timestamp TIMESTAMP,
  window_end_timestamp TIMESTAMP,
  model_version STRING,
  num_predictions INT64,
  mae FLOAT64,
  rmse FLOAT64,
  mape_excluding_zero_actuals FLOAT64
);

TRUNCATE TABLE `nyc-uber-494107.nyc_forecasting.prediction_metrics_rolling_7d`;

INSERT INTO `nyc-uber-494107.nyc_forecasting.prediction_metrics_rolling_7d`

WITH max_data AS (
  SELECT
    MAX(target_timestamp) AS max_target_timestamp
  FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors`
),

windows AS (
  SELECT DISTINCT
    target_timestamp AS window_start_timestamp,
    TIMESTAMP_ADD(target_timestamp, INTERVAL 167 HOUR) AS window_end_timestamp
  FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors`
  WHERE EXTRACT(HOUR FROM target_timestamp) = 0
),

filtered_windows AS(
  SELECT w.*
  FROM windows w
  CROSS JOIN max_data m
  WHERE window_end_timestamp <= m.max_target_timestamp
)

SELECT
    DATE(w.window_end_timestamp) AS metric_date,
    w.window_start_timestamp,
    w.window_end_timestamp,
    e.model_version AS model_version,
    COUNT(*) AS num_predictions,
    AVG(e.absolute_error) AS mae,
    SQRT(AVG(e.squared_error)) AS rmse,
    AVG(
      SAFE_DIVIDE(
        e.absolute_error,
        NULLIF(e.actual_demand, 0)
      )
    ) AS mape_excluding_zero_actuals

FROM filtered_windows w

JOIN `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` e
    ON e.target_timestamp
    BETWEEN w.window_start_timestamp
    AND w.window_end_timestamp

GROUP BY
    metric_date,
    w.window_start_timestamp,
    w.window_end_timestamp,
    model_version;
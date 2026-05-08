CREATE OR REPLACE VIEW `nyc-uber-494107.nyc_forecasting.vw_prediction_metrics_by_hour` AS

WITH joined AS (
  SELECT
    p.forecast_run_timestamp,
    p.target_timestamp,
    p.PULocationID,
    a.demand AS actual_demand,
    p.predicted_demand,
    p.model_version,
    p.predicted_demand - a.demand AS error,
    ABS(p.predicted_demand - a.demand) AS absolute_error,
    POW(p.predicted_demand - a.demand, 2) AS squared_error
  FROM `nyc-uber-494107.nyc_forecasting.hourly_demand_predictions` p
  JOIN `nyc-uber-494107.nyc_forecasting.hourly_demand_actuals` a
    ON p.target_timestamp = a.hour
   AND p.PULocationID = a.PULocationID
)

SELECT
  model_version,
  target_timestamp,
  COUNT(*) AS num_zones,
  AVG(absolute_error) AS mae,
  SQRT(AVG(squared_error)) AS rmse,
  AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals
FROM joined
GROUP BY model_version, target_timestamp
ORDER BY target_timestamp;
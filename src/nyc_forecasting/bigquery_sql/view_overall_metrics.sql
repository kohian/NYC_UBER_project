CREATE OR REPLACE VIEW `nyc-uber-494107.nyc_forecasting.vw_overall_metrics` AS

SELECT
  model_version,
  COUNT(*) AS num_predictions,

  AVG(actual_demand) AS avg_actual_demand,
  AVG(predicted_demand) AS avg_predicted_demand,

  AVG(raw_error) AS mean_error,
  AVG(absolute_error) AS mae,
  SQRT(AVG(squared_error)) AS rmse,

  AVG(
    SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))
  ) AS mape_excluding_zero_actuals

FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` 
GROUP BY model_version;
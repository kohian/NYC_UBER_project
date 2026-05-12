CREATE OR REPLACE VIEW `nyc-uber-494107.nyc_forecasting.vw_latest_prediction_metrics_rolling_7d` AS

SELECT *
FROM `nyc-uber-494107.nyc_forecasting.prediction_metrics_rolling_7d`
QUALIFY ROW_NUMBER() OVER (
  ORDER BY window_end_timestamp DESC
) = 1;
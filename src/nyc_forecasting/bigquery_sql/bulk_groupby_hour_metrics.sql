CREATE TABLE IF NOT EXISTS `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_hour` (
    model_version STRING, 
    target_timestamp TIMESTAMP,
    num_zones INT64, 
    mae FLOAT64,
    rmse FLOAT64,
    mape_excluding_zero_actuals FLOAT64
);

TRUNCATE TABLE `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_hour`;

INSERT INTO `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_hour`

SELECT
    model_version,
    target_timestamp,
    COUNT(*) AS num_zones,
    AVG(absolute_error) AS mae,
    SQRT(AVG(squared_error)) AS rmse,
    AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals
FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` 
GROUP BY model_version, target_timestamp
ORDER BY target_timestamp;

CREATE TABLE IF NOT EXISTS `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_day` (
    model_version STRING, 
    date DATE,
    num_predictions INT64, 
    mae FLOAT64,
    rmse FLOAT64,
    mape_excluding_zero_actuals FLOAT64
);

TRUNCATE TABLE `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_day`;

INSERT INTO `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_day`

SELECT
    model_version,
    DATE(target_timestamp) as date,
    COUNT(*) AS num_predictions,
    AVG(absolute_error) AS mae,
    SQRT(AVG(squared_error)) AS rmse,
    AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals
FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` 
GROUP BY model_version, date
ORDER BY date;
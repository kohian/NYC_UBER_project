-- CREATE TABLE IF NOT EXISTS `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_zone` (
--     model_version STRING, 
--     PULocationID STRING,
--     num_predictions INT64, 
--     mae FLOAT64,
--     rmse FLOAT64,
--     mape_excluding_zero_actuals FLOAT64
-- );

-- TRUNCATE TABLE `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_zone`;

-- INSERT INTO `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_zone`


CREATE OR REPLACE VIEW `nyc-uber-494107.nyc_forecasting.vw_prediction_metrics_groupby_zone` AS

SELECT
    model_version,
    PULocationID,
    COUNT(*) AS num_predictions,
    AVG(absolute_error) AS mae,
    SQRT(AVG(squared_error)) AS rmse,
    AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals,
    AVG(actual_demand) as average_actual_demand
FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` 
GROUP BY model_version, PULocationID
ORDER BY PULocationID;

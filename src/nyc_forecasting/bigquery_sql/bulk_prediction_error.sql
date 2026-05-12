CREATE TABLE IF NOT EXISTS `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` (
    forecast_run_timestamp TIMESTAMP,
    target_timestamp TIMESTAMP,
    PULocationID STRING,
    actual_demand FLOAT64,
    predicted_demand FLOAT64,
    model_version STRING,
    raw_error FLOAT64,
    absolute_error FLOAT64,
    squared_error FLOAT64
);

TRUNCATE TABLE `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors`;

INSERT INTO `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors`

SELECT
    p.forecast_run_timestamp,
    p.target_timestamp,
    p.PULocationID,
    a.demand AS actual_demand,
    p.predicted_demand,
    p.model_version,
    p.predicted_demand - a.demand AS raw_error,
    ABS(p.predicted_demand - a.demand) AS absolute_error,
    POW(p.predicted_demand - a.demand, 2) AS squared_error
    FROM `nyc-uber-494107.nyc_forecasting.hourly_demand_predictions` p
    INNER JOIN `nyc-uber-494107.nyc_forecasting.hourly_demand_actuals` a
    ON p.target_timestamp = a.hour
    AND p.PULocationID = a.PULocationID;



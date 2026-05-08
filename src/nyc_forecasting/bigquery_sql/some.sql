MERGE `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` e

USING (
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
) src

ON e.target_timestamp = src.target_timestamp
AND e.PULocationID = src.PULocationID
AND e.model_version = src.model_version

WHEN NOT MATCHED THEN
  INSERT (
    forecast_run_timestamp,
    target_timestamp,
    PULocationID,
    actual_demand,
    predicted_demand,
    model_version,
    error,
    absolute_error,
    squared_error
  )
  VALUES (
    src.forecast_run_timestamp,
    src.target_timestamp,
    src.PULocationID,
    src.actual_demand,
    src.predicted_demand,
    src.model_version,
    src.error,
    src.absolute_error,
    src.squared_error
  );
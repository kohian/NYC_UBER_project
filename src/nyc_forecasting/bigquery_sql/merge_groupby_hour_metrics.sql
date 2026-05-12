MERGE `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_hour` e

USING (
  SELECT
    model_version,
    target_timestamp,
    COUNT(*) AS num_zones,
    AVG(absolute_error) AS mae,
    SQRT(AVG(squared_error)) AS rmse,
    AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals
  FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` 
  WHERE target_timestamp = @target_timestamp
    AND model_version = @model_version
  GROUP BY model_version, target_timestamp
  -- ORDER BY target_timestamp
) src

ON e.target_timestamp = src.target_timestamp
AND e.model_version = src.model_version

WHEN NOT MATCHED THEN
  INSERT (
    model_version,
    target_timestamp,
    num_zones,
    mae,
    rmse,
    mape_excluding_zero_actuals
  )
  VALUES (
    src.model_version,
    src.target_timestamp,
    src.num_zones,
    src.mae,
    src.rmse,
    src.mape_excluding_zero_actuals
  );
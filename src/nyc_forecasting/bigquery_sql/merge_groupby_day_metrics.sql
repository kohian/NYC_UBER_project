MERGE `nyc-uber-494107.nyc_forecasting.prediction_metrics_groupby_day` e

USING (
  SELECT
    model_version,
    DATE(target_timestamp) as date,
    COUNT(*) AS num_predictions,
    AVG(absolute_error) AS mae,
    SQRT(AVG(squared_error)) AS rmse,
    AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals
  FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` 
  WHERE DATE(target_timestamp) = DATE(@target_timestamp)
    AND model_version = @model_version
  GROUP BY model_version, date
  -- ORDER BY target_timestamp
) src

ON e.date = src.date
AND e.model_version = src.model_version

WHEN MATCHED THEN UPDATE SET
  num_predictions = src.num_predictions,
  mae = src.mae,
  rmse = src.rmse,
  mape_excluding_zero_actuals = src.mape_excluding_zero_actuals

WHEN NOT MATCHED THEN
  INSERT (
    model_version,
    date,
    num_predictions,
    mae,
    rmse,
    mape_excluding_zero_actuals
  )
  VALUES (
    src.model_version,
    src.date,
    src.num_predictions,
    src.mae,
    src.rmse,
    src.mape_excluding_zero_actuals
  );
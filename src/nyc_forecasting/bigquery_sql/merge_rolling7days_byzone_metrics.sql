MERGE `nyc-uber-494107.nyc_forecasting.prediction_metrics_rolling_7d_byzone` t

USING (

  WITH latest_window AS (
    SELECT
      TIMESTAMP_SUB(MAX(target_timestamp), INTERVAL 167 HOUR) AS window_start_timestamp,
      MAX(target_timestamp) AS window_end_timestamp
    FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors`
  ),

  windowed AS (
    SELECT
      e.*,
      w.window_start_timestamp,
      w.window_end_timestamp
    FROM `nyc-uber-494107.nyc_forecasting.hourly_prediction_errors` e
    CROSS JOIN latest_window w
    WHERE e.target_timestamp BETWEEN w.window_start_timestamp AND w.window_end_timestamp
  )

  SELECT
    DATE(window_end_timestamp) AS metric_date,
    PULocationID,
    -- CURRENT_TIMESTAMP() AS metric_run_timestamp,
    window_start_timestamp,
    -- TIMESTAMP_SUB(MAX(target_timestamp), INTERVAL 7 DAY) AS window_start_timestamp,
    window_end_timestamp,
    model_version,
    COUNT(*) AS num_predictions,
    AVG(absolute_error) AS mae,
    SQRT(AVG(squared_error)) AS rmse,
    AVG(SAFE_DIVIDE(absolute_error, NULLIF(actual_demand, 0))) AS mape_excluding_zero_actuals
  FROM windowed
  GROUP BY
    metric_date,
    PULocationID,
    window_start_timestamp,
    window_end_timestamp,
    model_version
) src

ON t.metric_date = src.metric_date
AND t.PULocationID = src.PULocationID
AND t.model_version = src.model_version

WHEN MATCHED THEN UPDATE SET
  -- metric_run_timestamp = src.metric_run_timestamp,
  window_start_timestamp = src.window_start_timestamp,
  window_end_timestamp = src.window_end_timestamp,
  num_predictions = src.num_predictions,
  mae = src.mae,
  rmse = src.rmse,
  mape_excluding_zero_actuals = src.mape_excluding_zero_actuals

WHEN NOT MATCHED THEN INSERT (
  metric_date,
  -- metric_run_timestamp,
  PULocationID,
  window_start_timestamp,
  window_end_timestamp,
  model_version,
  num_predictions,
  mae,
  rmse,
  mape_excluding_zero_actuals
)
VALUES (
  src.metric_date,
  -- src.metric_run_timestamp,
  src.PULocationID,
  src.window_start_timestamp,
  src.window_end_timestamp,
  src.model_version,
  src.num_predictions,
  src.mae,
  src.rmse,
  src.mape_excluding_zero_actuals
);
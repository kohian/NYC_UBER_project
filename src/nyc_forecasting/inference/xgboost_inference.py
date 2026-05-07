import pandas as pd
from google.cloud import bigquery

from nyc_forecasting.inference.config import XGBoostInferConfig, BigQueryConfig
from nyc_forecasting.inference.bigquery_io import load_dataframe_to_bigquery, load_latest_actuals_from_bigquery, prediction_already_exists

from nyc_forecasting.core.data import make_full_panel

from nyc_forecasting.core.features import (
    make_time_features_only,
    transform_wide_frame,
)
from nyc_forecasting.core.tree_tabular import make_selected_lag_inference_row

from nyc_forecasting.core.artifacts import (
    load_joblib_object_from_gcs,
    load_json_from_gcs,
)


def main() -> None:

    model_cfg = XGBoostInferConfig()
    bigquery_config = BigQueryConfig()

    model_version = model_cfg.model_version
    actuals_table = bigquery_config.demand_actuals_table
    predictions_table = bigquery_config.demand_predictions_table
####################################################################

    client = bigquery.Client(project=bigquery_config.project_id)

    print("Loading inference artifacts...")
    model = load_joblib_object_from_gcs(model_cfg.model_path)
    scaler = load_joblib_object_from_gcs(model_cfg.scaler_path)
    zone_names = load_json_from_gcs(model_cfg.zone_names_path)

    # Need enough history for the largest feature window
    # lookback_hours = max(model_cfg.selected_lags) + bigquery_config.database_buffer        

    print("Loading latest actuals from BigQuery...")

    actuals_df = load_latest_actuals_from_bigquery(
        client=client,
        actuals_table=actuals_table,
        lookback_hours=max(model_cfg.selected_lags),
    )

    if actuals_df.empty:
        raise ValueError("No actuals found in BigQuery.")

    zone_names = [int(z) for z in zone_names]

    actuals_df["hour"] = pd.to_datetime(actuals_df["hour"])
    actuals_df["PULocationID"] = actuals_df["PULocationID"].astype("int32")
    actuals_df["demand"] = actuals_df["demand"].astype("float32")

    wide_df = make_full_panel(actuals_df)
    # then enforce training schema
    wide_df = wide_df.reindex(columns=zone_names).fillna(0)

    latest_actual_hour = wide_df.index.max()
    target_timestamp = latest_actual_hour + pd.Timedelta(hours=model_cfg.horizon)

    print(f"Latest actual hour: {latest_actual_hour}")
    print(f"Target timestamp: {target_timestamp}")

    if prediction_already_exists(
        client=client,
        predictions_table=predictions_table,
        target_timestamp=target_timestamp,
        model_version=model_version,
    ):
        print(
            f"Prediction already exists for {target_timestamp} "
            f"and model_version={model_version}. Skipping."
        )
        return

    print("Building inference features...")
    scaled_wide = transform_wide_frame(wide_df, scaler)

    max_lag = max(model_cfg.selected_lags)

    # check scaled_wide is enough
    if len(scaled_wide) < max_lag: # strictly less than for single inference
        raise ValueError(
            f"Not enough rows for inference. "
            f"Need EQUAL or more than max_lag={max_lag}, got {len(scaled_wide)}."
        )

    # Only one time feature needed
    target_time_features = make_time_features_only(
        pd.DatetimeIndex([target_timestamp])
    )


    X_infer = make_selected_lag_inference_row(
        demand_array=scaled_wide.to_numpy(dtype="float32"),
        target_time_features=target_time_features,
        use_time_features=model_cfg.use_time_features,
        lags=model_cfg.selected_lags,
    )        


    print("Predicting...")
    pred_scaled = model.predict(X_infer)
    pred_real = scaler.inverse_transform(pred_scaled)

    # Optional safety: demand should not be negative
    # pred_real = pred_real.clip(min=0)


    assert pred_real.shape[1] == len(zone_names), (
        f"Prediction output has {pred_real.shape[1]} zones, "
        f"but zone_names has {len(zone_names)} zones."
    )

    pred_df = pd.DataFrame({
        "forecast_run_timestamp": latest_actual_hour,
        "target_timestamp": target_timestamp,
        "PULocationID": zone_names,
        "predicted_demand": pred_real[0],
        "model_version": model_version,
    })

    pred_df["PULocationID"] = pred_df["PULocationID"].astype(str)
    pred_df["predicted_demand"] = pred_df["predicted_demand"].astype("float64")

    print("Writing predictions to BigQuery...")

    predictions_schema = [
        bigquery.SchemaField("forecast_run_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("target_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("PULocationID", "STRING"),
        bigquery.SchemaField("predicted_demand", "FLOAT"),
        bigquery.SchemaField("model_version", "STRING"),
    ]

    load_dataframe_to_bigquery(
        client = client,
        df = pred_df,
        table_id = predictions_table,
        schema = predictions_schema,
        write_disposition = "WRITE_APPEND",
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()
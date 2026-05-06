import pandas as pd
from google.cloud import bigquery

from nyc_forecasting.inference.config import XGBoostInferConfig, BigQueryConfig
from nyc_forecasting.inference.bigquery_io import load_dataframe_to_bigquery, load_all_actuals_from_bigquery

from nyc_forecasting.core.data import make_full_panel

from nyc_forecasting.core.features import (
    make_time_features_only,
    transform_wide_frame,
)
from nyc_forecasting.core.tree_tabular import (
    make_selected_lag_tabular,
)

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

    actuals_df = load_all_actuals_from_bigquery(
        client=client,
        actuals_table=actuals_table,
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

    print("Building inference features...")
    scaled_wide = transform_wide_frame(wide_df, scaler)

    max_lag = max(model_cfg.selected_lags)

    if len(scaled_wide) <= max_lag:
        raise ValueError(
            f"Not enough rows for inference. "
            f"Need more than max_lag={max_lag}, got {len(scaled_wide)}."
        )

    time_features = make_time_features_only(scaled_wide.index)

    X_tab, _ = make_selected_lag_tabular(
        demand_array=scaled_wide.to_numpy(dtype="float32"),
        time_features=time_features,
        use_time_features=model_cfg.use_time_features,
        lags=model_cfg.selected_lags,
        horizon = model_cfg.horizon
    )        


    print("Predicting...")
    pred_scaled = model.predict(X_tab)
    pred_real = scaler.inverse_transform(pred_scaled)

    # Optional safety: demand should not be negative
    # pred_real = pred_real.clip(min=0)

    max_lag = max(model_cfg.selected_lags)
    target_timestamps = wide_df.index[max_lag + model_cfg.horizon - 1:]

    assert pred_real.shape[0] == len(target_timestamps), (
        f"Prediction output has {pred_real.shape[0]} timestamps, "
        f"but target_timestamps has {len(target_timestamps)} timestamps."
    )

    assert pred_real.shape[1] == len(zone_names), (
        f"Prediction output has {pred_real.shape[1]} zones, "
        f"but zone_names has {len(zone_names)} zones."
    )

    # pred_real is wide: rows = timestamps, columns = zones
    pred_wide_df = pd.DataFrame(
        pred_real,
        index=target_timestamps,
        columns=zone_names,
    )

    pred_wide_df.index.name = "target_timestamp"

    # Convert wide predictions to long BigQuery format
    pred_df = (
        pred_wide_df
        .reset_index()
        .melt(
            id_vars="target_timestamp",
            var_name="PULocationID",
            value_name="predicted_demand",
        )
    )

    # Since this is simulated batch inference,
    # pretend the forecast was made 1 hour before the target.
    pred_df["forecast_run_timestamp"] = (
        pred_df["target_timestamp"] - pd.Timedelta(hours=model_cfg.horizon)
    )
    pred_df["model_version"] = model_version

    # Optional but recommended: reorder columns to match BigQuery table
    pred_df = pred_df[
        [
            "forecast_run_timestamp",
            "target_timestamp",
            "PULocationID",
            "predicted_demand",
            "model_version",
        ]
    ]

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
        write_disposition = "WRITE_TRUNCATE",
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()
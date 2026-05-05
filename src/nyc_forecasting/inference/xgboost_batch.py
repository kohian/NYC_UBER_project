# src/nyc_forecasting/pipelines/inference_xgboost.py

import json
from datetime import datetime, timezone

import gcsfs
import joblib
import pandas as pd
from google.cloud import bigquery

from nyc_forecasting.core.config import DataConfig, XGBoostConfig
from nyc_forecasting.core.features import (
    add_time_features,
    make_time_features_only,
    transform_wide_frame,
)
from nyc_forecasting.core.tree_tabular import (
    make_tabular_windows,
    make_selected_lag_tabular,
)

from nyc_forecasting.core.artifacts import (
    load_joblib_object_from_gcs,
    load_json_from_gcs,
)


def make_wide_actuals(
    df: pd.DataFrame,
    zone_names: list,
) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = pd.to_datetime(df["hour"])
    df["PULocationID"] = df["PULocationID"].astype(type(zone_names[0]))

    wide_df = (
        df.pivot_table(
            index="hour",
            columns="PULocationID",
            values="demand",
            aggfunc="sum",
        )
        .sort_index()
    )

    # Critical: force same zones and same column order as training
    wide_df = wide_df.reindex(columns=zone_names).fillna(0)

    return wide_df


def build_single_inference_row(
    wide_df: pd.DataFrame,
    scaler,
    data_cfg: DataConfig,
    model_cfg: XGBoostConfig,
    zone_names: list,
):
    scaled_wide = transform_wide_frame(wide_df, scaler)

    if model_cfg.mode == "full":
        if len(scaled_wide) < data_cfg.input_len:
            raise ValueError(
                f"Not enough rows for inference. "
                f"Need at least {data_cfg.input_len}, got {len(scaled_wide)}."
            )

        X = add_time_features(scaled_wide)
        dummy_y = scaled_wide.to_numpy(dtype="float32")

        X_tab, _ = make_tabular_windows(
            X=X,
            y=dummy_y,
            input_len=data_cfg.input_len,
        )

    else:
        max_lag = max(model_cfg.selected_lags)

        if len(scaled_wide) <= max_lag:
            raise ValueError(
                f"Not enough rows for inference. "
                f"Need more than max_lag={max_lag}, got {len(scaled_wide)}."
            )

        time_features = make_time_features_only(scaled_wide.index)

        X_tab, _, _ = make_selected_lag_tabular(
            demand_array=scaled_wide.to_numpy(dtype="float32"),
            time_features=time_features,
            use_time_features=model_cfg.use_time_features,
            lags=model_cfg.selected_lags,
            zone_names=zone_names,
        )

    # Last row predicts the next available target step
    return X_tab[-1:]


def prediction_already_exists(
    client: bigquery.Client,
    predictions_table: str,
    target_timestamp: pd.Timestamp,
    model_version: str,
) -> bool:
    query = f"""
    SELECT COUNT(*) AS cnt
    FROM `{predictions_table}`
    WHERE target_timestamp = @target_timestamp
      AND model_version = @model_version
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "target_timestamp",
                "TIMESTAMP",
                target_timestamp.to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    )

    result = client.query(query, job_config=job_config).to_dataframe()
    return int(result.loc[0, "cnt"]) > 0



def main() -> None:
    data_cfg = DataConfig()
    model_cfg = XGBoostConfig()

    # Change this to the run_id you want to use
    model_version = "2026-05-05_120000"

    project_id = "nyc-uber-494107"
    dataset_id = "nyc_forecasting"

    actuals_table = f"{project_id}.{dataset_id}.hourly_demand_actuals"
    predictions_table = f"{project_id}.{dataset_id}.hourly_demand_predictions"

    base_path = f"{model_cfg.model_path.rstrip('/')}/{model_version}/inference"

    model_path = f"{base_path}/xgboost_model.joblib"
    scaler_path = f"{base_path}/scaler.joblib"
    zone_names_path = f"{base_path}/zone_names.json"

    client = bigquery.Client(project=project_id)

    print("Loading inference artifacts...")
    model = load_joblib_from_gcs(model_path)
    scaler = load_joblib_from_gcs(scaler_path)
    zone_names = load_json_from_gcs(zone_names_path)

    # Need enough history for the largest feature window
    if model_cfg.mode == "full":
        lookback_hours = data_cfg.input_len + 5
    else:
        lookback_hours = max(model_cfg.selected_lags) + 5

    print("Loading latest actuals from BigQuery...")
    actuals_df = load_latest_actuals_from_bigquery(
        client=client,
        actuals_table=actuals_table,
        lookback_hours=lookback_hours,
    )

    if actuals_df.empty:
        raise ValueError("No actuals found in BigQuery.")

    wide_df = make_wide_actuals(actuals_df, zone_names)

    latest_actual_hour = wide_df.index.max()
    target_timestamp = latest_actual_hour + pd.Timedelta(hours=1)

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
    X_infer = build_single_inference_row(
        wide_df=wide_df,
        scaler=scaler,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        zone_names=zone_names,
    )

    print("Predicting...")
    pred_scaled = model.predict(X_infer)
    pred_real = scaler.inverse_transform(pred_scaled)

    # Optional safety: demand should not be negative
    pred_real = pred_real.clip(min=0)

    forecast_run_timestamp = datetime.now(timezone.utc)

    pred_df = pd.DataFrame({
        "forecast_run_timestamp": forecast_run_timestamp,
        "target_timestamp": target_timestamp,
        "PULocationID": zone_names,
        "predicted_demand": pred_real[0],
        "model_version": model_version,
    })

    pred_df["PULocationID"] = pred_df["PULocationID"].astype("int64")
    pred_df["predicted_demand"] = pred_df["predicted_demand"].astype("float64")

    print("Writing predictions to BigQuery...")
    write_predictions_to_bigquery(
        client=client,
        pred_df=pred_df,
        predictions_table=predictions_table,
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()
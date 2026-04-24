import mlflow
import xgboost as xgb

from datetime import datetime
from dataclasses import asdict
from sklearn.multioutput import MultiOutputRegressor

from nyc_forecasting.core.config import DataConfig, XGBoostConfig
from nyc_forecasting.core.data import (
    load_monthly_files,
    select_files,
    make_full_panel,
    split_wide_by_month,
    make_raw_targets,
)
from nyc_forecasting.core.features import (
    add_time_features,
    fit_demand_scaler,
    transform_wide_frame,
    make_time_features_only,
)    
from nyc_forecasting.core.metrics import (
    calculate_regression_metrics,
    print_metric_summary,
)
from nyc_forecasting.core.artifacts import (
    save_config_to_gcs,
    save_results_to_gcs,
    save_joblib_object_to_gcs,
)
from nyc_forecasting.core.torch_seed import set_seed

from nyc_forecasting.core.tree_tabular import make_tabular_windows, make_selected_lag_tabular


def main() -> None:
    # -----------------------------
    # 1. Load config
    # -----------------------------
    data_cfg = DataConfig()
    model_cfg = XGBoostConfig()

    # -----------------------------
    # 2. Set up MLflow
    # -----------------------------
    # We use Databricks-hosted MLflow for experiment tracking.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/iankoh@ymail.com/NYC_UBER_DEMAND")

    # Generate a single run_id to align:
    # - MLflow run name
    # - GCS artifact folder
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    with mlflow.start_run(run_name=f"xgboost_{run_id}"):
        # -----------------------------
        # 3. Reproducibility
        # -----------------------------
        set_seed(42)

        # -----------------------------
        # 4. Load processed monthly files
        # -----------------------------
        all_files = select_files(
            data_cfg.processed_dest,
            data_cfg.train_start,
            data_cfg.test_end,
        )
        print("All files:", all_files)

        combined_df = load_monthly_files(all_files)

        # Build full hour × zone matrix
        wide_df = make_full_panel(combined_df)

        # -----------------------------
        # 5. Split into train / val / test
        # -----------------------------
        train_wide = split_wide_by_month(wide_df, data_cfg.train_start, data_cfg.train_end)
        val_wide = split_wide_by_month(wide_df, data_cfg.val_start, data_cfg.val_end)
        test_wide = split_wide_by_month(wide_df, data_cfg.test_start, data_cfg.test_end)

        # -----------------------------
        # 6. Fit scaler on TRAIN only
        # -----------------------------
        demand_scaler = fit_demand_scaler(train_wide)

        train_scaled = transform_wide_frame(train_wide, demand_scaler)
        val_scaled = transform_wide_frame(val_wide, demand_scaler)
        test_scaled = transform_wide_frame(test_wide, demand_scaler)

        # -----------------------------
        # 7. Build features
        # -----------------------------
        # Same feature engineering as LSTM for fair comparison
        X_train = add_time_features(train_scaled)
        X_val = add_time_features(val_scaled)
        X_test = add_time_features(test_scaled)

        # Targets are scaled demand values
        y_train = train_scaled.to_numpy(dtype="float32")
        y_val = val_scaled.to_numpy(dtype="float32")
        y_test = test_scaled.to_numpy(dtype="float32")

        print("BEFORE MAKE TABULAR")
        # -----------------------------
        # 8. Convert sequences into tabular windows
        # -----------------------------

        if model_cfg.mode == "full":
            # LSTM uses [samples, seq_len, features]
            # XGBoost needs [samples, flattened_features]
            X_train_tab, y_train_tab = make_tabular_windows(X_train, y_train, data_cfg.input_len)
            X_val_tab, y_val_tab = make_tabular_windows(X_val, y_val, data_cfg.input_len)
            X_test_tab, y_test_tab = make_tabular_windows(X_test, y_test, data_cfg.input_len)

        else:
            # time features (only)
            train_time = make_time_features_only(train_scaled.index)
            val_time = make_time_features_only(val_scaled.index)
            test_time = make_time_features_only(test_scaled.index)

            X_train_tab, y_train_tab = make_selected_lag_tabular(
                demand_array=y_train,
                time_features=train_time,
                lags=model_cfg.selected_lags,
            )

            X_val_tab, y_val_tab = make_selected_lag_tabular(
                demand_array=y_val,
                time_features=val_time,
                lags=model_cfg.selected_lags,
            )

            X_test_tab, y_test_tab = make_selected_lag_tabular(
                demand_array=y_test,
                time_features=test_time,
                lags=model_cfg.selected_lags,
            )

        print("AFTER MAKE TABULAR")
        # Runtime-derived dimensions
        input_size = X_train.shape[-1]        # features per timestep
        num_targets = y_train.shape[-1]       # number of zones predicted
        flat_input_size = X_train_tab.shape[-1]

        # -----------------------------
        # 9. Build model
        # -----------------------------
        # MultiOutputRegressor wraps one XGBRegressor per target column.
        # This keeps the output shape similar to the LSTM pipeline.
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=model_cfg.n_estimators,
                max_depth=model_cfg.max_depth,
                learning_rate=model_cfg.learning_rate,
                subsample=model_cfg.subsample,
                colsample_bytree=model_cfg.colsample_bytree,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=1,
            )
        )

        # -----------------------------
        # 10. Log parameters to MLflow
        # -----------------------------
        mlflow.log_params({
            "model_type": "xgboost",
            "input_len": data_cfg.input_len,
            "output_len": data_cfg.output_len,
            "input_size_per_timestep": input_size,
            "flat_input_size": flat_input_size,
            "num_targets": num_targets,
            "n_estimators": model_cfg.n_estimators,
            "max_depth": model_cfg.max_depth,
            "learning_rate": model_cfg.learning_rate,
            "subsample": model_cfg.subsample,
            "colsample_bytree": model_cfg.colsample_bytree,
            "xgboost_mode": model_cfg.mode,
            "selected_lags": model_cfg.selected_lags,
        })

        print("BEFORE TRAINING")
        print("Num targets:", y_train_tab.shape[1])
        # -----------------------------
        # 11. Train model
        # -----------------------------
        # First baseline: fit on train split only.
        # You can later use val split for tuning.
        model.fit(X_train_tab, y_train_tab)
        print("AFTER TRAINING")
        # -----------------------------
        # 12. Create artifact path
        # -----------------------------
        base_path = f"{model_cfg.model_path.rstrip('/')}/{run_id}"

        # -----------------------------
        # 13. Save artifacts
        # -----------------------------
        # Save model
        save_joblib_object_to_gcs(model, base_path, "xgboost_model.joblib")

        # Save scaler
        save_joblib_object_to_gcs(demand_scaler, base_path, "scaler.joblib")

        # Save config
        model_config = asdict(model_cfg)
        model_config.update({
            "input_size_per_timestep": input_size,
            "flat_input_size": flat_input_size,
            "num_targets": num_targets,
            "model_type": "xgboost",
        })

        run_config = {
            "model_config": model_config,
            "data_config": asdict(data_cfg),
        }

        save_config_to_gcs(run_config, base_path, "run_config.json")

        # -----------------------------
        # 14. Predict on test set
        # -----------------------------
        pred_scaled = model.predict(X_test_tab)

        # Convert predictions back to raw demand units
        pred_real = demand_scaler.inverse_transform(pred_scaled)

        # Align raw next-step targets from original unscaled test data
        if model_cfg.mode == "full":        
            y_real = make_raw_targets(test_wide, input_len=data_cfg.input_len)
        else:
            max_lag = max(model_cfg.selected_lags)
            y_real = make_raw_targets(test_wide, input_len=max_lag )

        # Sanity check
        assert pred_real.shape == y_real.shape

        # -----------------------------
        # 15. Compute metrics
        # -----------------------------
        results = calculate_regression_metrics(
            preds_raw=pred_real,
            targets_raw=y_real,
            zone_names=wide_df.columns.tolist(),
            mape_mode="exclude",
            epsilon=1e-1,
        )

        print_metric_summary(results, mape_mode="exclude")

        # -----------------------------
        # 16. Log metrics to MLflow
        # -----------------------------
        mlflow.log_metrics({
            "overall_mae": float(results["overall_mae"]),
            "overall_rmse": float(results["overall_rmse"]),
            "overall_mape": float(results["overall_mape"]),
        })

        # Add tags so MLflow run is easy to connect to GCS artifacts
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("artifact_path", base_path)

        # -----------------------------
        # 17. Save results
        # -----------------------------
        save_results_to_gcs(
            results,
            base_path=base_path,
            metadata={
                "run_id": run_id,
                "model_type": "xgboost",
            },
        )


if __name__ == "__main__":
    main()
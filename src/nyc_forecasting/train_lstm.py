import torch
from torch.utils.data import DataLoader
from datetime import datetime

from dataclasses import asdict
import mlflow

from nyc_forecasting.core.config import DataConfig, LSTMConfig
from nyc_forecasting.core.torch_dataset import SequenceDataset
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
)
from nyc_forecasting.core.metrics import (
    calculate_regression_metrics,
    print_metric_summary,
)
from nyc_forecasting.core.artifacts import (
    save_config_to_gcs,
    save_results_to_gcs,
    save_joblib_object_to_gcs,
    save_torch_state_dict_to_gcs,
)
from nyc_forecasting.core.lstm_class import DemandLSTM
from nyc_forecasting.core.lstm_functions import evaluate, fit_lstm, predict
from nyc_forecasting.core.torch_seed import set_seed


def main() -> None:
    # -----------------------------
    # 1. Load config
    # -----------------------------
    # DataConfig: data locations + train/val/test date ranges
    # LSTMConfig: model/training hyperparameters
    data_cfg = DataConfig()
    model_cfg = LSTMConfig()

    # Start mlflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/iankoh@ymail.com/NYC_UBER_DEMAND")

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with mlflow.start_run(run_name=f"lstm_{run_id}"):
        # -----------------------------
        # 2. Reproducibility + device
        # -----------------------------
        set_seed(42)

        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        # -----------------------------
        # 3. Load processed monthly files
        # -----------------------------
        # Build list of processed parquet files from train start to test end
        all_files = select_files(
            data_cfg.processed_dest,
            data_cfg.train_start,
            data_cfg.test_end,
        )
        print("All files:", all_files)

        # Read all months into one long DataFrame
        combined_df = load_monthly_files(all_files)

        # Build full hour-zone wide matrix:
        # index = hour, columns = zone IDs, values = demand
        wide_df = make_full_panel(combined_df)

        # -----------------------------
        # 4. Split into train / val / test by month ranges
        # -----------------------------
        train_wide = split_wide_by_month(wide_df, data_cfg.train_start, data_cfg.train_end)
        val_wide = split_wide_by_month(wide_df, data_cfg.val_start, data_cfg.val_end)
        test_wide = split_wide_by_month(wide_df, data_cfg.test_start, data_cfg.test_end)

        # -----------------------------
        # 5. Fit scaler on TRAIN only
        # -----------------------------
        # This avoids leakage from val/test into training preprocessing
        demand_scaler = fit_demand_scaler(train_wide)

        # Apply the same scaler to all splits
        train_scaled = transform_wide_frame(train_wide, demand_scaler)
        val_scaled = transform_wide_frame(val_wide, demand_scaler)
        test_scaled = transform_wide_frame(test_wide, demand_scaler)

        # -----------------------------
        # 6. Build model inputs
        # -----------------------------
        # Add calendar/time-based features to each split
        X_train = add_time_features(train_scaled)
        X_val = add_time_features(val_scaled)
        X_test = add_time_features(test_scaled)

        # Targets are next-step scaled demand values for all zones
        y_train = train_scaled.to_numpy(dtype="float32")
        y_val = val_scaled.to_numpy(dtype="float32")
        y_test = test_scaled.to_numpy(dtype="float32")

        # -----------------------------
        # 7. Build PyTorch datasets + loaders
        # -----------------------------
        # SequenceDataset creates sliding windows:
        # X: past input_len steps
        # y: next output_len step(s)
        train_ds = SequenceDataset(
            X_train,
            y_train,
            input_len=data_cfg.input_len,
            output_len=data_cfg.output_len,
        )
        val_ds = SequenceDataset(
            X_val,
            y_val,
            input_len=data_cfg.input_len,
            output_len=data_cfg.output_len,
        )
        test_ds = SequenceDataset(
            X_test,
            y_test,
            input_len=data_cfg.input_len,
            output_len=data_cfg.output_len,
        )

        # shuffle=False keeps order deterministic and is okay for this setup
        train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False)

        # -----------------------------
        # 8. Infer model input/output sizes from one batch
        # -----------------------------
        xb, yb = next(iter(train_loader))
        input_size = xb.shape[-1]   # number of input features per timestep
        num_targets = yb.shape[-1]  # number of output zones

        # -----------------------------
        # 9. Build model
        # -----------------------------
        model = DemandLSTM(
            input_size=input_size,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers,
            num_targets=num_targets,
            dropout=model_cfg.dropout,
        ).to(device)

        # -----------------------------
        # LOG PARAMS
        # -----------------------------
        mlflow.log_params({
            "hidden_size": model_cfg.hidden_size,
            "num_layers": model_cfg.num_layers,
            "dropout": model_cfg.dropout,
            "learning_rate": model_cfg.learning_rate,
            "num_epochs": model_cfg.num_epochs,
            "batch_size": data_cfg.batch_size,
            "input_len": data_cfg.input_len,
            "output_len": data_cfg.output_len,
            "input_size": input_size,
            "num_targets": num_targets,
        })

        # -----------------------------
        # 10. Create run/version folder
        # -----------------------------
        # Use timestamp to version this run in GCS

        base_path = f"{model_cfg.model_path.rstrip('/')}/{run_id}"

        # -----------------------------
        # 11. Train model
        # -----------------------------
        # fit_lstm returns:
        # - model loaded with best validation weights
        # - training history
        # - best_state_dict for saving
        model, _history, best_state_dict = fit_lstm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=model_cfg.learning_rate,
            num_epochs=model_cfg.num_epochs,
        )

        # -----------------------------
        # 12. Save artifacts
        # -----------------------------
        # Save best model weights
        save_torch_state_dict_to_gcs(
            best_state_dict,
            base_path,
        )

        # Save fitted scaler
        save_joblib_object_to_gcs(demand_scaler, base_path, "scaler.joblib")

        model_config = asdict(model_cfg)

        model_config.update({
            "input_size": input_size,
            "num_targets": num_targets,
            "device_used": device,
        })

        run_config = {
            "model_config": model_config,
            "data_config": asdict(data_cfg),
        }

        save_config_to_gcs(
            run_config,
            base_path,
            "run_config.json"
        )
        # -----------------------------
        # 13. Evaluate on scaled test loader
        # -----------------------------
        # This is test loss in scaled space
        test_loss = evaluate(model, test_loader, torch.nn.MSELoss(), device)
        print(f"Test loss: {test_loss:.6f}")

        # -----------------------------
        # 14. Predict on test set
        # -----------------------------
        pred_scaled, _y_scaled = predict(model, test_loader, device)

        # Convert predictions back to raw demand units
        pred_real = demand_scaler.inverse_transform(pred_scaled)

        # Build aligned raw targets directly from the original unscaled test_wide
        y_real = make_raw_targets(test_wide, input_len=data_cfg.input_len)

        # Sanity check alignment
        assert pred_real.shape == y_real.shape

        # -----------------------------
        # 15. Compute regression metrics on raw scale
        # -----------------------------
        results = calculate_regression_metrics(
            preds_raw=pred_real,
            targets_raw=y_real,
            zone_names=wide_df.columns.tolist(),
            mape_mode="exclude",
            epsilon=1e-1,
        )

        # Print summary to console
        print_metric_summary(results, mape_mode="exclude")

        # -----------------------------
        # 16. Save metrics/results artifacts
        # -----------------------------

        #ALSO LOG TO MLFLOW
        mlflow.log_metrics({
            "test_loss_scaled": float(test_loss),
            "overall_mae": float(results["overall_mae"]),
            "overall_rmse": float(results["overall_rmse"]),
            "overall_mape": float(results["overall_mape"]),
        })

        save_results_to_gcs(
            results,
            base_path=base_path,
            metadata={
                "test_loss_scaled": float(test_loss),
                "run_id": run_id,
        },
        )


if __name__ == "__main__":
    main()
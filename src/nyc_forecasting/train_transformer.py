import torch
from torch.utils.data import DataLoader
from datetime import datetime
from dataclasses import asdict
import mlflow

from nyc_forecasting.core.config import DataConfig, TransformerConfig
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
from nyc_forecasting.core.transformer_class import DemandTransformer
from nyc_forecasting.core.torch_functions import evaluate, fit_torch_model, predict
from nyc_forecasting.core.torch_seed import set_seed


def main() -> None:
    # -----------------------------
    # 1. Load configs
    # -----------------------------
    data_cfg = DataConfig()
    model_cfg = TransformerConfig()

    # -----------------------------
    # 2. MLflow setup
    # -----------------------------
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/iankoh@ymail.com/NYC_UBER_DEMAND")

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    with mlflow.start_run(run_name=f"transformer_{run_id}"):

        # -----------------------------
        # 3. Set seed + device
        # -----------------------------
        set_seed(42)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        # -----------------------------
        # 4. Load data from GCS
        # -----------------------------
        all_files = select_files(
            data_cfg.processed_dest,
            data_cfg.train_start,
            data_cfg.test_end,
        )
        print("All files:", all_files)

        combined_df = load_monthly_files(all_files)

        # -----------------------------
        # 5. Build wide panel
        # -----------------------------
        wide_df = make_full_panel(combined_df)

        # -----------------------------
        # 6. Train / Val / Test split
        # -----------------------------
        train_wide = split_wide_by_month(wide_df, data_cfg.train_start, data_cfg.train_end)
        val_wide = split_wide_by_month(wide_df, data_cfg.val_start, data_cfg.val_end)
        test_wide = split_wide_by_month(wide_df, data_cfg.test_start, data_cfg.test_end)

        # -----------------------------
        # 7. Fit scaler on train only
        # -----------------------------
        demand_scaler = fit_demand_scaler(train_wide)

        # -----------------------------
        # 8. Scale datasets
        # -----------------------------
        train_scaled = transform_wide_frame(train_wide, demand_scaler)
        val_scaled = transform_wide_frame(val_wide, demand_scaler)
        test_scaled = transform_wide_frame(test_wide, demand_scaler)

        # -----------------------------
        # 9. Add time features
        # -----------------------------
        X_train = add_time_features(train_scaled)
        X_val = add_time_features(val_scaled)
        X_test = add_time_features(test_scaled)

        # -----------------------------
        # 10. Convert targets to numpy
        # -----------------------------
        y_train = train_scaled.to_numpy(dtype="float32")
        y_val = val_scaled.to_numpy(dtype="float32")
        y_test = test_scaled.to_numpy(dtype="float32")

        # -----------------------------
        # 11. Build sequence datasets
        # -----------------------------
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

        # -----------------------------
        # 12. DataLoaders
        # -----------------------------
        train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False)

        # -----------------------------
        # 13. Infer input/output sizes
        # -----------------------------
        xb, yb = next(iter(train_loader))
        input_size = xb.shape[-1]
        num_targets = yb.shape[-1]

        # -----------------------------
        # 14. Initialize Transformer
        # -----------------------------
        model = DemandTransformer(
            input_size=input_size,
            d_model=model_cfg.d_model,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            num_targets=num_targets,
            dropout=model_cfg.dropout,
        ).to(device)

        # -----------------------------
        # 15. Log MLflow params
        # -----------------------------
        mlflow.log_params({
            "model_type": "transformer",
            "d_model": model_cfg.d_model,
            "num_heads": model_cfg.num_heads,
            "num_layers": model_cfg.num_layers,
            "dim_feedforward": model_cfg.dim_feedforward,
            "dropout": model_cfg.dropout,
            "learning_rate": model_cfg.learning_rate,
            "num_epochs": model_cfg.num_epochs,
            "batch_size": data_cfg.batch_size,
            "input_len": data_cfg.input_len,
            "output_len": data_cfg.output_len,
            "input_size": input_size,
            "num_targets": num_targets,
            "device_used": device,
        })

        base_path = f"{model_cfg.model_path.rstrip('/')}/{run_id}"

        # -----------------------------
        # 16. Train model
        # -----------------------------
        model, _history, best_state_dict = fit_torch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=model_cfg.learning_rate,
            num_epochs=model_cfg.num_epochs,
        )

        # -----------------------------
        # 17. Save artifacts
        # -----------------------------
        save_torch_state_dict_to_gcs(best_state_dict, base_path)
        save_joblib_object_to_gcs(demand_scaler, base_path,  "scaler.joblib")

        model_config = asdict(model_cfg)
        model_config.update({
            "input_size": input_size,
            "num_targets": num_targets,
            "device_used": device,
            "model_type": "transformer",
        })

        run_config = {
            "model_config": model_config,
            "data_config": asdict(data_cfg),
        }

        save_config_to_gcs(run_config, base_path, "run_config.json")

        # -----------------------------
        # 18. Evaluate on test set
        # -----------------------------
        test_loss = evaluate(model, test_loader, torch.nn.MSELoss(), device)
        print(f"Test loss: {test_loss:.6f}")

        # -----------------------------
        # 19. Predict + inverse scale
        # -----------------------------
        pred_scaled, _y_scaled = predict(model, test_loader, device)
        pred_real = demand_scaler.inverse_transform(pred_scaled)

        # -----------------------------
        # 20. Align raw targets
        # -----------------------------
        y_real = make_raw_targets(test_wide, input_len=data_cfg.input_len)

        assert pred_real.shape == y_real.shape

        # -----------------------------
        # 21. Compute metrics
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
        # 22. Log metrics to MLflow
        # -----------------------------
        mlflow.log_metrics({
            "test_loss_scaled": float(test_loss),
            "overall_mae": float(results["overall_mae"]),
            "overall_rmse": float(results["overall_rmse"]),
            "overall_mape": float(results["overall_mape"]),
        })

        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("artifact_path", base_path)
        mlflow.set_tag("model_type", "transformer")

        # -----------------------------
        # 23. Save results to GCS
        # -----------------------------
        save_results_to_gcs(
            results,
            base_path=base_path,
            metadata={
                "test_loss_scaled": float(test_loss),
                "run_id": run_id,
                "model_type": "transformer",
            },
        )


if __name__ == "__main__":
    main()
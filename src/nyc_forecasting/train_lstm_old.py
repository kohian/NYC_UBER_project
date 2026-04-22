# from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datetime import datetime

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
    save_results_to_gcs,
    save_scaler_to_gcs,
    save_torch_state_dict_to_gcs,
)

from nyc_forecasting.core.lstm_class import DemandLSTM
from nyc_forecasting.core.lstm_functions import evaluate, fit_lstm, predict
from nyc_forecasting.core.torch_seed import set_seed


def main() -> None:
    data_cfg = DataConfig()
    model_cfg = LSTMConfig()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    all_files = select_files(data_cfg.processed_dest, data_cfg.train_start, data_cfg.test_end)
    print("All files:", all_files)

    combined_df = load_monthly_files(all_files)
    wide_df = make_full_panel(combined_df)

    train_wide = split_wide_by_month(wide_df, data_cfg.train_start, data_cfg.train_end)
    val_wide = split_wide_by_month(wide_df, data_cfg.val_start, data_cfg.val_end)
    test_wide = split_wide_by_month(wide_df, data_cfg.test_start, data_cfg.test_end)

    demand_scaler = fit_demand_scaler(train_wide)
    train_scaled = transform_wide_frame(train_wide, demand_scaler)
    val_scaled = transform_wide_frame(val_wide, demand_scaler)
    test_scaled = transform_wide_frame(test_wide, demand_scaler)

    X_train = add_time_features(train_scaled)
    X_val = add_time_features(val_scaled)
    X_test = add_time_features(test_scaled)

    y_train = train_scaled.to_numpy(dtype="float32")
    y_val = val_scaled.to_numpy(dtype="float32")
    y_test = test_scaled.to_numpy(dtype="float32")

    train_ds = SequenceDataset(X_train, y_train, input_len=data_cfg.input_len, output_len=data_cfg.output_len)
    val_ds = SequenceDataset(X_val, y_val, input_len=data_cfg.input_len, output_len=data_cfg.output_len)
    test_ds = SequenceDataset(X_test, y_test, input_len=data_cfg.input_len, output_len=data_cfg.output_len)

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False)

    xb, yb = next(iter(train_loader))
    input_size = xb.shape[-1]
    num_targets = yb.shape[-1]

    model = DemandLSTM(
        input_size=input_size,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        num_targets=num_targets,
        dropout=model_cfg.dropout,
    ).to(device)

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    base_path = f"{model_cfg.model_path.rstrip('/')}/{run_id}"

    model, history, best_state_dict = fit_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=model_cfg.learning_rate,
        num_epochs=model_cfg.num_epochs,
        # best_model_path=best_model_path,
    )

    save_torch_state_dict_to_gcs(
        best_state_dict,
        base_path,
    )

    save_scaler_to_gcs(
        demand_scaler,
        base_path,
    )

    test_loss = evaluate(model, test_loader, torch.nn.MSELoss(), device)
    print(f"Test loss: {test_loss:.6f}")

    pred_scaled, _y_scaled = predict(model, test_loader, device)
    pred_real = demand_scaler.inverse_transform(pred_scaled)
    y_real = make_raw_targets(test_wide, input_len=data_cfg.input_len)

    assert pred_real.shape == y_real.shape

    results = calculate_regression_metrics(
        preds_raw=pred_real,
        targets_raw=y_real,
        zone_names=wide_df.columns.tolist(),
        mape_mode="exclude",
        epsilon=1e-1,
    )
    print_metric_summary(results, mape_mode="exclude")

    save_results_to_gcs(
        results,
        base_path=base_path
    )

if __name__ == "__main__":
    main()

# """Starter script for tree-based models using the same shared prep steps.

# This keeps file loading, panel creation, splitting, scaling, and calendar features
# shared with the LSTM pipeline. The main difference is that sequential windows are
# flattened into 2D tabular rows.
# """

# from pathlib import Path

# import joblib
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor

# from nyc_forecasting.core.config import DataConfig, XGBoostConfig

# from nyc_forecasting.core.data import (
#     load_monthly_files,
#     select_files,
#     ym_to_exclusive_end_ts,
#     ym_to_start_ts,
#     make_full_panel, 
#     split_wide_by_time, 
#     fit_demand_scaler, 
#     transform_wide_frame,
# )

# from nyc_forecasting.core.tree_tabular import make_tabular_next_step_dataset
# from nyc_forecasting.core.features import add_time_features
# from nyc_forecasting.core.metrics import (
#     calculate_regression_metrics,
#     print_metric_summary,
# )
# from nyc_forecasting.core.torch_seed import set_seed


# def main() -> None:
#     data_cfg = DataConfig()
#     model_cfg = XGBoostConfig()
#     set_seed(model_cfg.random_state)

#     all_files = select_files(data_cfg.data_dir, data_cfg.train_start, data_cfg.test_end)
#     combined_df = load_monthly_files(all_files)
#     _, wide_df = make_full_panel(combined_df)

#     train_wide, val_wide, test_wide = split_wide_by_time(
#         wide_df=wide_df,
#         train_start_ts=ym_to_start_ts(data_cfg.train_start),
#         train_end_ts=ym_to_exclusive_end_ts(data_cfg.train_end),
#         val_start_ts=ym_to_start_ts(data_cfg.val_start),
#         val_end_ts=ym_to_exclusive_end_ts(data_cfg.val_end),
#         test_start_ts=ym_to_start_ts(data_cfg.test_start),
#         test_end_ts=ym_to_exclusive_end_ts(data_cfg.test_end),
#     )

#     demand_scaler = fit_demand_scaler(train_wide)
#     train_scaled = transform_wide_frame(train_wide, demand_scaler)
#     val_scaled = transform_wide_frame(val_wide, demand_scaler)
#     test_scaled = transform_wide_frame(test_wide, demand_scaler)

#     X_train = add_time_features(train_scaled)
#     X_val = add_time_features(val_scaled)
#     X_test = add_time_features(test_scaled)

#     y_train = train_scaled.to_numpy(dtype="float32")
#     y_val = val_scaled.to_numpy(dtype="float32")
#     y_test = test_scaled.to_numpy(dtype="float32")

#     X_train_tab, y_train_tab = make_tabular_next_step_dataset(X_train, y_train, input_len=data_cfg.input_len)
#     X_val_tab, y_val_tab = make_tabular_next_step_dataset(X_val, y_val, input_len=data_cfg.input_len)
#     X_test_tab, _y_test_tab = make_tabular_next_step_dataset(X_test, y_test, input_len=data_cfg.input_len)

#     base_model = XGBRegressor(
#         n_estimators=model_cfg.n_estimators,
#         max_depth=model_cfg.max_depth,
#         learning_rate=model_cfg.learning_rate,
#         subsample=model_cfg.subsample,
#         colsample_bytree=model_cfg.colsample_bytree,
#         objective="reg:squarederror",
#         random_state=model_cfg.random_state,
#         tree_method="hist",
#     )
#     model = MultiOutputRegressor(base_model)
#     model.fit(X_train_tab, y_train_tab)

#     pred_scaled = model.predict(X_test_tab)
#     pred_real = demand_scaler.inverse_transform(pred_scaled)
#     y_real = test_wide.to_numpy(dtype="float32")[data_cfg.input_len :]

#     results = calculate_regression_metrics(
#         preds_raw=pred_real,
#         targets_raw=y_real,
#         zone_names=wide_df.columns.tolist(),
#         mape_mode="exclude",
#         epsilon=1e-1,
#     )
#     print_metric_summary(results, mape_mode="exclude")

#     model_cfg.model_dir.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, model_cfg.model_dir / "xgb_multioutput.joblib")


# if __name__ == "__main__":
#     main()

import xgboost as xgb

from datetime import datetime
from dataclasses import asdict
from sklearn.multioutput import MultiOutputRegressor

from nyc_forecasting.training.config import DataConfig, XGBoostConfig
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
    save_tree_feature_importance_to_gcs,
)
from nyc_forecasting.core.torch_seed import set_seed

from nyc_forecasting.core.tree_tabular import make_tabular_windows, make_selected_lag_tabular, compute_feature_importance


def main() -> None:

    # -----------------------------
    # 0. config
    # -----------------------------
    data_cfg = DataConfig()
    model_cfg = XGBoostConfig()
    set_seed(42)

    # -----------------------------

    # 1. Load all data up to production cutoff
    all_files = select_files(
        data_cfg.processed_dest,
        data_cfg.train_start,
        data_cfg.test_end,
    )

    combined_df = load_monthly_files(all_files)
    wide_df = make_full_panel(combined_df)
    zone_names = wide_df.columns.tolist()

    # 2. Fit scaler on ALL available historical data
    demand_scaler = fit_demand_scaler(wide_df)
    wide_scaled = transform_wide_frame(wide_df, demand_scaler)

    # 3. Build targets
    y_all = wide_scaled.to_numpy(dtype="float32")

    # 4. Build time features
    time_features = make_time_features_only(wide_scaled.index)

    # 5. Build XGBoost tabular data
    X_all_tab, y_all_tab, feature_names = make_selected_lag_tabular(
        demand_array=y_all,
        time_features=time_features,
        use_time_features=model_cfg.use_time_features,
        lags=model_cfg.selected_lags,
        zone_names=zone_names,
    )

    # 6. Train
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
        )
    )

    model.fit(X_all_tab, y_all_tab)

    # 7. Save artifacts

    # Save config
    model_config = asdict(model_cfg)
    model_config.update({
        # "input_size_per_timestep": input_size,
        "flat_input_size": flat_input_size,
        "num_targets": num_targets,
        "model_type": "xgboost",
    })

    run_config = {
        "model_config": model_config,
        "data_config": asdict(data_cfg),
    }


    # NOT DONE HERE
    base_path ="?????"

    save_joblib_object_to_gcs(model, base_path, "xgboost_model.joblib")
    save_joblib_object_to_gcs(demand_scaler, base_path, "scaler.joblib")
    save_json_to_gcs(zone_names,base_path, "zone_names.json")
    save_config_to_gcs(run_config, base_path, "run_config.json")



    if __name__ == "__main__":
    main()
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipeConfig:
    pipe_src: str = "gs://raw-nyc/processed"
    batch_pipe_start: str = "2025-01"
    batch_pipe_end: str = "2025-01"
    single_pipe_start_end: str = "2025-02"


@dataclass(frozen=True)
class BigQueryConfig:

    project_id: str = "nyc-uber-494107"
    dataset: str = "nyc_forecasting"
    demand_actuals_table: str = f"{project_id}.{dataset}.hourly_demand_actuals"    
    demand_predictions_table: str = f"{project_id}.{dataset}.hourly_demand_predictions"       
    database_buffer: int = 5 


@dataclass(frozen=True)
class XGBoostInferConfig:

    model_version: str = "2026-05-05_150458"
    model_path: str = f"gs://raw-nyc/models/xgboost/{model_version}/inference/xgboost_model.joblib"
    scaler_path: str = f"gs://raw-nyc/models/xgboost/{model_version}/inference/scaler.joblib"
    zone_names_path: str = f"gs://raw-nyc/models/xgboost/{model_version}/inference/zone_names.json"


    horizon: int = 1          # next hour prediction
    use_time_features: bool = True

    # select_lags: bool = True  # true = select fale = sequence

    # # Full Mode
    # input_len: int = 24

    # Select Mode
    selected_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 24])




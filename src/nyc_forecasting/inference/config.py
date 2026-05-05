from dataclasses import dataclass

@dataclass(frozen=True)
class PipeConfig:
    pipe_src: str = "gs://raw-nyc/processed"
    pipe_start: str = "2025-01"
    pipe_end: str = "2025-01"

    bq_table_id: str = "nyc-uber-494107.nyc_forecasting.hourly_demand_actuals"



@dataclass(frozen=True)
class XGBoostInferConfig:
    model_path: str = "gs://raw-nyc/models/xgboost/2026-05-04_211251/xgboost_model.joblib"
    scaler_path: str = "gs://raw-nyc/models/xgboost/2026-05-04_211251/scaler.joblib"

    input_table: str = ""         # BigQuery or GCS source
    output_table: str = ""        # where predictions go

    selected_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 24])

    horizon: int = 1          # next hour prediction
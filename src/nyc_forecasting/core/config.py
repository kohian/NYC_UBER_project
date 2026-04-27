from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    # File I/O
    raw_source: str = "gs://raw-nyc/raw"
    processed_dest: str = "gs://raw-nyc/processed"
    zone_lookup_path: str = "gs://raw-nyc/raw/taxi_zone_lookup.csv"
    process_start: str = "2024-07"
    process_end: str = "2024-12"

     # Filter
    columns: tuple[str, str, str] = ("hvfhs_license_num", "request_datetime", "PULocationID")
    keep_license: str | None = "HV0003"
    borough: str = "Manhattan"

    # Duration
    train_start: str = "2024-07"
    train_end: str = "2024-10"
    val_start: str = "2024-11"
    val_end: str = "2024-11"
    test_start: str = "2024-12"
    test_end: str = "2024-12"

    # Sequencing
    input_len: int = 24
    output_len: int = 1
    batch_size: int = 256


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    num_epochs: int = 50
    model_path: str = "gs://raw-nyc/models/lstm"


@dataclass(frozen=True)
class XGBoostConfig:
    model_path: str = "gs://raw-nyc/models/xgboost"

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    random_state: int = 42
    n_jobs: int = -1
    objective: str = "reg:squarederror"

    mode: str = "select" # full or select
    selected_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 24])

    # use_early_stopping: bool = True
    # early_stopping_rounds: int = 5


@dataclass(frozen=True)
class TransformerConfig:
    model_path: str = "gs://raw-nyc/models/transformer"

    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

    learning_rate: float = 1e-3
    num_epochs: int = 50
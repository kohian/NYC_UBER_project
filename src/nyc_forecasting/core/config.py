from dataclasses import dataclass




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
    # best_model_path: Path = Path("artifacts/checkpoints/best_lstm.pt")
    model_path: str = "gs://raw-nyc/models/lstm"


# @dataclass(frozen=True)
# class XGBoostConfig:
#     n_estimators: int = 300
#     max_depth: int = 8
#     learning_rate: float = 0.05
#     subsample: float = 0.8
#     colsample_bytree: float = 0.8
#     random_state: int = 42
#     model_dir: Path = Path("artifacts/models/xgboost")

# NYC Taxi Demand Forecasting

This project builds an end-to-end machine learning pipeline to forecast hourly ride demand across NYC taxi zones using historical high-volume for-hire vehicle trip data.

The goal is not only to train a forecasting model, but to build a production-style ML system that covers data processing, model training, batch inference, BigQuery storage, monitoring, and future simulation of fleet allocation decisions.

## Project Overview

The system follows this flow:

```text
Raw trip data
→ hourly demand aggregation
→ model training
→ saved model artifacts
→ BigQuery actuals table
→ batch / single-hour inference
→ BigQuery predictions table
→ monitoring and error analysis
→ future fleet allocation simulation
```

This project is designed to demonstrate practical ML engineering, data pipeline design, and operational thinking around forecasting systems.

## What This Project Does

- Processes raw NYC trip data into hourly demand per pickup zone
- Builds full hourly zone panels for time-series modeling
- Trains forecasting models for next-hour demand prediction
- Supports LSTM, Transformer, and XGBoost-style tabular lag models
- Tracks training runs and metrics with MLflow
- Saves trained models, scalers, zone order, configs, and metrics to GCS
- Uses Docker containerization for reproducible training and inference execution
- Loads actual demand data into BigQuery
- Runs inference using saved model artifacts
- Writes predictions back to BigQuery
- Supports monitoring through prediction error tables and metric views
- Provides a foundation for fleet allocation simulation

## High-Level Architecture

```text
Data Processing
    Raw monthly parquet files
    → hourly demand by PULocationID
    → processed parquet files in GCS

Training
    Processed demand data
    → full hour × zone matrix
    → feature engineering
    → model training
    → evaluation
    → MLflow experiment tracking
    → artifacts saved to GCS

Inference
    BigQuery actuals table
    → latest demand history
    → saved scaler + model + zone_names
    → prediction
    → BigQuery predictions table

Monitoring
    actuals + predictions
    → prediction error table
    → metrics views
    → dashboard / analysis
```

## Data Design

### Actuals Table

```text
hour
PULocationID
demand
```

### Predictions Table

```text
forecast_run_timestamp
target_timestamp
PULocationID
predicted_demand
model_version
```

### Prediction Error Table

Planned / recommended monitoring table:

```text
forecast_run_timestamp
target_timestamp
PULocationID
actual_demand
predicted_demand
model_version
error
absolute_error
squared_error
```

This table can be updated incrementally using a BigQuery `MERGE` query.

## Key Design Decision: Long vs Wide Format

The project uses both long and wide formats depending on the task.

### BigQuery Format: Long

```text
hour | PULocationID | demand
```

This is better for:

- SQL queries
- joins
- dashboards
- monitoring
- scalable storage
- zone-level filtering

### Model Format: Wide

```text
hour | zone_1 | zone_2 | zone_3 | ...
```

This is better for:

- scaler fitting
- sequence models
- tabular lag features
- multi-output prediction

The pipeline converts between these formats as needed.

## Models

### LSTM

The LSTM model uses sliding windows of historical demand and time features.

```text
past input_len hours → next-hour demand for all zones
```

This is useful for sequence modeling and temporal pattern learning.

### Transformer

The Transformer model is included as an additional sequence-modeling approach for demand forecasting.

Like the LSTM, it is designed to use historical time windows, but it can model temporal relationships using attention instead of recurrent layers.

### XGBoost Selected-Lag Model

The XGBoost model uses explicit lag features instead of flattened full sequences.

Example lags:

```text
lag_1
lag_2
lag_3
lag_24
```

The model also uses target timestamp time features such as:

```text
hour_sin
hour_cos
dow_sin
dow_cos
month_sin
month_cos
is_weekend
is_holiday
```

This design was chosen because tree-based models work well with explicit tabular features, and target-time features are easier to reason about than flattened historical time features.

## Feature Engineering

The project includes:

- Hour-of-day cyclical encoding
- Day-of-week cyclical encoding
- Month cyclical encoding
- Weekend flag
- Holiday flag
- Selected demand lags
- Standard scaling of demand values

The scaler is fit on training data only and reused during inference.

## Experiment Tracking

Training runs are tracked with MLflow.

MLflow is used to log:

- model type
- selected lag features
- model hyperparameters
- number of targets
- input feature size
- overall MAE
- overall RMSE
- overall MAPE
- run metadata such as run ID and artifact path

The project uses MLflow to make training runs easier to compare and to connect model results with saved GCS artifacts.

The saved artifacts remain in GCS, while MLflow provides experiment tracking and run-level metadata.

## Artifact Saving

Each training run saves artifacts to GCS.

Example structure:

```text
gs://raw-nyc/models/xgboost/<run_id>/
    inference/
        xgboost_model.joblib
        scaler.joblib
        zone_names.json

    config/
        run_config.json

    results/
        summary.json
        per_zone.csv
        top_bottom.csv
        predictions.npz

    feature_importance/
        feature_importance.csv
        lag_importance.csv
        zone_importance.csv
        type_importance_summary.csv
```

Important inference artifacts:

- `xgboost_model.joblib`
- `scaler.joblib`
- `zone_names.json`

The saved `zone_names.json` ensures that inference uses the exact same zone column order as training.

## Containerization

The project includes Docker-based containerization so the source code can run consistently across local and cloud environments.

Containerization helps standardize:

- Python dependencies
- package installation
- training entrypoints
- inference entrypoints
- environment setup

This does not automatically mean training is performed in the cloud. The same container can be run locally or on a cloud service such as Cloud Run Jobs, Vertex AI, or GKE.

A practical production setup would separate training and inference workloads.

### Training container

The training container is responsible for:

- loading processed data
- training models
- evaluating model performance
- logging MLflow metrics
- saving model artifacts to GCS

Example entrypoint:

```bash
python -m nyc_forecasting.training.train_xgboost
```

### Inference container

The inference container is responsible for:

- loading saved model artifacts
- reading latest actuals from BigQuery
- building inference features
- writing predictions back to BigQuery

Example entrypoint:

```bash
python -m nyc_forecasting.inference.xgboost_inference
```

Training and inference can share the same base image, but they should have separate commands or Docker targets because they serve different operational purposes.

## Project Structure

```text
src/nyc_forecasting/
    core/
        data.py
        features.py
        tree_tabular.py
        torch_dataset.py
        lstm_class.py
        lstm_functions.py
        metrics.py
        artifacts.py
        config.py
        torch_seed.py

    training/
        train_lstm.py
        train_xgboost.py
        train_transformer.py

    inference/
        config.py
        bigquery_io.py
        batch_pipe.py
        single_pipe.py
        xgboost_batch.py
        xgboost_inference.py

    run_preprocessing.py

Dockerfile
```

## Training Pipeline

### LSTM Training

```bash
python -m nyc_forecasting.training.train_lstm
```

### Transformer Training

```bash
python -m nyc_forecasting.training.train_transformer
```

### XGBoost Training

```bash
python -m nyc_forecasting.training.train_xgboost
```

The XGBoost pipeline:

1. Loads processed monthly demand files
2. Builds a full hourly zone panel
3. Splits data by month
4. Fits scaler on training data only
5. Builds selected-lag tabular features
6. Trains a multi-output XGBoost model
7. Evaluates predictions on raw demand values
8. Logs experiment metadata and metrics with MLflow
9. Saves model, scaler, zone order, config, metrics, and feature importance to GCS

## BigQuery Pipeline

### Load Batch Actuals

```bash
python -m nyc_forecasting.inference.batch_pipe
```

This loads a selected month of processed hourly demand data into BigQuery.

Typical use case:

```text
processed parquet → BigQuery actuals table
```

### Load Single Hour Actuals

```bash
python -m nyc_forecasting.inference.single_pipe
```

This simulates hourly data arrival by appending the next timestamp of actual demand into BigQuery.

Typical use case:

```text
load latest available hour → append to actuals table
```

### Batch Inference

```bash
python -m nyc_forecasting.inference.xgboost_batch
```

This runs inference across all available actuals in BigQuery and writes predictions to the predictions table.

Typical use case:

```text
historical actuals → batch predictions
```

### Single-Hour Inference

```bash
python -m nyc_forecasting.inference.xgboost_inference
```

This loads the latest actual demand history from BigQuery, builds one inference row, predicts the next target timestamp, and appends predictions to BigQuery.

Typical use case:

```text
latest actuals → next-hour prediction
```

## Monitoring Design

The monitoring layer is intended to compare predictions against actual demand.

Recommended flow:

```text
actuals table
+ predictions table
→ prediction error table
→ metric views
→ dashboard
```

Useful metrics:

- MAE
- RMSE
- MAPE excluding zero actuals
- error by zone
- error by hour
- latest actual timestamp
- latest prediction timestamp
- missing prediction checks
- model version comparison

## Example BigQuery Metric Views

Planned views:

```text
vw_prediction_metrics_by_hour
vw_prediction_metrics_by_zone
vw_prediction_metrics_by_model
```

These views can support a Looker Studio dashboard for monitoring model performance and data freshness.

## Future Simulation Extension

A planned extension is to use the forecast output for a fleet allocation simulation.

The simulation would answer:

```text
Given predicted demand, how many vehicles should be allocated to each zone?
Given actual demand, how much demand would be fulfilled?
What fleet size is needed to maintain a target fulfillment rate?
```

Potential simulation metrics:

- fulfillment rate
- unfulfilled demand
- driver utilization
- idle vehicle hours
- repositioning count
- fleet size sensitivity
- baseline vs forecast allocation

This would connect the ML forecast to an operational business outcome.

## Configuration

Configuration is defined with dataclasses.

### `DataConfig`

Controls:

- raw data paths
- processed data paths
- train/test date ranges
- input/output lengths
- preprocessing settings

### `XGBoostConfig`

Controls:

- model parameters
- selected lags
- use of time features
- forecast horizon
- artifact path

### `BigQueryConfig`

Controls:

- GCP project ID
- BigQuery dataset
- actuals table
- predictions table

## How to Run

### 1. Install package locally

```bash
pip install -e .
```

Install PyTorch separately if using GPU training.

### 2. Run preprocessing

```bash
python -m nyc_forecasting.run_preprocessing
```

### 3. Train XGBoost model

```bash
python -m nyc_forecasting.training.train_xgboost
```

### 4. Load actuals to BigQuery

```bash
python -m nyc_forecasting.inference.batch_pipe
```

### 5. Run batch inference

```bash
python -m nyc_forecasting.inference.xgboost_batch
```

### 6. Run single-hour pipeline

```bash
python -m nyc_forecasting.inference.single_pipe
python -m nyc_forecasting.inference.xgboost_inference
```

### 7. Run with Docker

Build the container image:

```bash
docker build -t nyc-forecasting .
```

Run training:

```bash
docker run --rm nyc-forecasting python -m nyc_forecasting.training.train_xgboost
```

Run inference:

```bash
docker run --rm nyc-forecasting python -m nyc_forecasting.inference.xgboost_inference
```

In a cloud environment, the same image can be used as the base for scheduled training or inference jobs.

## Current Status

Implemented:

- Raw trip processing
- Hourly demand aggregation
- Full hour-zone panel creation
- LSTM training pipeline
- Transformer training pipeline
- XGBoost selected-lag training pipeline
- MLflow experiment tracking
- GCS artifact saving
- Docker containerization
- BigQuery actuals loading
- Batch XGBoost inference
- Single-hour XGBoost inference
- Prediction table output
- Feature importance export

## Key Lessons

This project demonstrates:

- End-to-end ML pipeline design
- Time-series split discipline
- Feature consistency between training and inference
- Long-format warehouse design and wide-format model design
- Saved artifact management for reproducibility
- BigQuery-based inference output and monitoring
- Practical tradeoffs between model complexity and operational clarity

## Project Goal

This project is a production-style ML forecasting system built around a simple but realistic use case:

```text
Forecast hourly demand
→ store predictions
→ monitor performance
→ support operational decisions
```

The focus is on building practical ML engineering habits, not just training a model in a notebook.
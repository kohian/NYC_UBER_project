# NYC Taxi Demand Forecasting

This project builds a production-style machine learning pipeline to forecast hourly ride demand across NYC taxi zones using historical high-volume for-hire vehicle trip data.

The project focuses on the full ML system, not just model training:

```text
Raw trip data
→ hourly demand aggregation
→ model training
→ MLflow experiment tracking
→ GCS model artifacts
→ BigQuery actuals table
→ Cloud Run inference
→ BigQuery predictions table
→ BigQuery monitoring metrics
→ future fleet allocation simulation
```

The target is completed trip volume, used as a proxy for realized demand. The dataset does not include unfulfilled requests, cancelled trips, or suppressed demand.

## What This Project Does

- Processes raw NYC trip data into hourly pickup-zone demand
- Builds full hourly zone panels for time-series modeling
- Trains LSTM, Transformer, and XGBoost forecasting models
- Tracks experiments and metrics with MLflow
- Saves trained models, scalers, configs, predictions, feature importance, and metrics to GCS
- Builds and pushes separate training and inference Docker images to Google Artifact Registry
- Runs the training image on Vertex AI Training
- Runs the inference image on Cloud Run
- Stores actuals, predictions, and monitoring metrics in BigQuery
- Supports batch backfills and simulated single-hour pipeline execution
- Provides a foundation for future fleet allocation simulation and RL-based matching experiments

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
    Cloud Run inference container
    → load latest actuals from BigQuery
    → load saved model, scaler, and zone order from GCS
    → predict next-hour demand
    → write predictions to BigQuery

Monitoring
    actuals + predictions
    → BigQuery prediction error table
    → hourly, daily, rolling 7-day, and zone-level metrics
    → dashboard-ready monitoring tables/views
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

```text
forecast_run_timestamp
target_timestamp
PULocationID
actual_demand
predicted_demand
model_version
raw_error
absolute_error
squared_error
```

The prediction error table is updated incrementally using BigQuery `MERGE` queries.

## Key Design Decision: Long vs Wide Format

The project uses both long and wide data formats.

### BigQuery Format: Long

```text
hour | PULocationID | demand
```

This is better for:

- SQL joins
- dashboarding
- filtering by zone or time
- monitoring tables
- scalable warehouse storage

### Model Format: Wide

```text
hour | zone_1 | zone_2 | zone_3 | ...
```

This is better for:

- scaler fitting
- sequence models
- tabular lag features
- multi-output forecasting

The pipeline converts between long and wide formats depending on the task.

## Models

### LSTM

The LSTM model uses sliding windows of historical demand and time features.

```text
past input_len hours → next-hour demand for all zones
```

### Transformer

The Transformer model is included as an additional sequence-modeling approach.

Like the LSTM, it uses historical time windows, but models temporal relationships using attention instead of recurrent layers.

### XGBoost Selected-Lag Model

The XGBoost model uses explicit lag features instead of flattened full sequences.

Example lags:

```text
lag_1
lag_2
lag_3
lag_24
```

It also uses target timestamp time features:

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

From experimentation, XGBoost produced the strongest results, so the current Cloud Run inference path uses the XGBoost model. LSTM and Transformer inference paths are not implemented yet.

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

MLflow logs:

- model type
- selected lag features
- model hyperparameters
- number of output targets
- input feature size
- overall MAE
- overall RMSE
- overall MAPE
- run ID
- GCS artifact path

Model artifacts remain in GCS, while MLflow is used to compare runs and connect metrics to saved artifacts.

## Artifact Saving

Training produces metrics and artifacts in GCS.

Example XGBoost artifact structure:

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

The saved `zone_names.json` ensures that inference uses the same zone column order as training.

## Containerization and Cloud Deployment

The project uses Docker multi-stage builds with separate training and inference targets.

### Training Image

The training image is pushed to Google Artifact Registry and run on Vertex AI Training.

It can train different model types by changing the CLI command:

```bash
python -m nyc_forecasting.training.train_xgboost
python -m nyc_forecasting.training.train_lstm
python -m nyc_forecasting.training.train_transformer
```

The training image is responsible for:

- loading processed data from GCS
- training models
- evaluating model performance
- logging metrics to MLflow
- saving model artifacts and reports to GCS

### Inference Image

The inference image is pushed to Google Artifact Registry and run on Cloud Run.

It currently supports two main scripts.

#### 1. Single-hour data pipeline

```bash
python -m nyc_forecasting.inference.single_pipe
```

This script simulates one new hour of demand data arriving into BigQuery.

It:

- loads the next hour of processed demand data
- appends it to the BigQuery actuals table
- runs BigQuery metric updates for the newly available actual timestamp

#### 2. XGBoost single-hour inference

```bash
python -m nyc_forecasting.inference.xgboost_inference
```

This script:

- loads latest actual demand history from BigQuery
- loads the saved XGBoost model, scaler, and zone order from GCS
- builds one inference row
- predicts the next target timestamp
- appends predictions to BigQuery

There are also batch scripts used to load historical actuals and run historical batch inference before the simulated hourly pipeline begins.

## BigQuery Monitoring

The project includes BigQuery SQL for monitoring and metric calculation.

Metrics include:

- MAE
- RMSE
- MAPE excluding zero actuals
- hourly metrics
- daily metrics
- rolling 7-day metrics
- rolling 7-day by-zone metrics
- zone-level metrics
- latest rolling 7-day view
- overall metrics view

This monitoring layer supports dashboarding and model performance tracking over time.

## Project Structure

```text
src/nyc_forecasting/
    core/
        data.py
        features.py
        tree_tabular.py
        torch_dataset.py
        lstm_class.py
        transformer_class.py
        torch_functions.py
        metrics.py
        artifacts.py
        torch_artifacts.py
        torch_seed.py

    training/
        config.py
        run_preprocessing.py
        train_lstm.py
        train_xgboost.py
        train_transformer.py
        train_production_xgboost.py

    inference/
        config.py
        bigquery_io.py
        batch_pipe.py
        single_pipe.py
        xgboost_batch.py
        xgboost_inference.py

    bigquery_sql/
        run_bigquery_sql.py
        merge_prediction_error.sql
        merge_groupby_hour_metrics.sql
        merge_groupby_day_metrics.sql
        merge_rolling7days_metrics.sql
        merge_rolling7days_byzone_metrics.sql
        view_overall_metrics.sql
        view_groupby_zone_metrics.sql
        view_latest_prediction_metrics_rolling_7d.sql
        bulk_prediction_error.sql
        bulk_groupby_hour_metrics.sql
        bulk_groupby_day_metrics.sql
        bulk_rolling7days_metrics.sql
        bulk_rolling7days_byzone_metrics.sql

Dockerfile
pyproject.toml
requirements_training.txt
requirements_inference.txt
```

## Training Pipeline

### Run preprocessing

```bash
python -m nyc_forecasting.training.run_preprocessing
```

### Train XGBoost

```bash
python -m nyc_forecasting.training.train_xgboost
```

### Train LSTM

```bash
python -m nyc_forecasting.training.train_lstm
```

### Train Transformer

```bash
python -m nyc_forecasting.training.train_transformer
```

The XGBoost training pipeline:

1. Loads processed monthly demand files
2. Builds a full hourly zone panel
3. Splits data by month
4. Fits scaler on training data only
5. Builds selected-lag tabular features
6. Trains a multi-output XGBoost model
7. Evaluates predictions on raw demand values
8. Logs experiment metadata and metrics with MLflow
9. Saves model artifacts, config, metrics, predictions, and feature importance to GCS

## Inference and Monitoring Pipeline

### Historical setup

Batch scripts are used to initialize history before running the simulated hourly pipeline.

```bash
python -m nyc_forecasting.inference.batch_pipe
python -m nyc_forecasting.inference.xgboost_batch
```

### Simulated hourly flow

```bash
python -m nyc_forecasting.inference.single_pipe
python -m nyc_forecasting.inference.xgboost_inference
```

The intended hourly flow is:

```text
1. Load actual demand for hour T into BigQuery
2. Calculate metrics for predictions that targeted hour T
3. Run inference to predict demand for hour T+1
4. Write predictions for hour T+1 to BigQuery
```

This mirrors a real forecasting loop where metrics are only calculated after the actual demand for a predicted timestamp becomes available.

## Docker Images

Build the training image:

```bash
docker build --target training -t nyc-forecast-training .
```

Build the inference image:

```bash
docker build --target inference -t nyc-forecast-inference .
```

The GitHub Actions workflow builds and pushes both images to Google Artifact Registry:

```text
nyc-forecast-training
nyc-forecast-inference
```

The training image is used for Vertex AI Training, while the inference image is used for Cloud Run execution.

## Configuration

Configuration is managed with dataclasses.

### Training Config

Controls:

- raw and processed GCS paths
- train, validation, and test date ranges
- model hyperparameters
- selected lags
- forecast horizon
- artifact paths

### Inference Config

Controls:

- BigQuery project, dataset, and table names
- model version
- model artifact paths
- selected lags
- forecast horizon
- batch and single-hour pipeline dates

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
- Dockerized training and inference images
- GitHub Actions image build and push to Artifact Registry
- Vertex AI compatible training container
- Cloud Run compatible inference container
- BigQuery actuals loading
- Batch XGBoost inference
- Single-hour XGBoost inference
- BigQuery prediction output
- BigQuery prediction error and monitoring metric updates
- Feature importance export

## Dashboard

A Looker Studio / Data Studio dashboard is used to visualize model performance and monitoring outputs from BigQuery.

Dashboard details and screenshots will be added later.

## Future Work

- Add Vertex AI hyperparameter tuning
- Improve model comparison workflow across XGBoost, LSTM, and Transformer models
- Add production inference support for LSTM and Transformer if future results justify it
- Build a fleet allocation simulation using predicted demand
- Compare baseline allocation against forecast-driven allocation
- Explore RL-based matching or repositioning as a future extension
- Improve cloud orchestration for the full hourly pipeline
- Add stronger monitoring and alerting around model and data freshness

## Key Lessons

This project demonstrates:

- End-to-end ML pipeline design
- Time-series split discipline
- Feature consistency between training and inference
- Long-format warehouse design and wide-format model design
- MLflow experiment tracking
- GCS artifact management
- Dockerized training and inference workflows
- Vertex AI and Cloud Run deployment patterns
- BigQuery-based prediction storage and monitoring
- Practical tradeoffs between model complexity and operational clarity

## Project Goal

This project is a production-style ML forecasting system built around a realistic operational use case:

```text
Forecast hourly demand
→ store predictions
→ monitor performance
→ support future operational decisions
```

The focus is on practical ML engineering habits, reproducible cloud workflows, and connecting model output to downstream monitoring and decision-making.
# NYC Uber Demand Forecasting

This project builds an end-to-end machine learning pipeline to forecast hourly ride demand across NYC taxi zones using historical Uber trip data.

The focus is not just on modeling, but on building a clean, reproducible ML pipeline with proper data processing, training, evaluation, and artifact tracking.

## What this project does

- Processes raw NYC Uber trip data into hourly demand per zone  
- Builds a time-series dataset with calendar-based features  
- Trains an LSTM model to predict next-step demand  
- Evaluates predictions using MAE, RMSE, and MAPE  
- Saves all artifacts (model, scaler, config, metrics) to GCS  

## Pipeline Overview

1. Data Processing  
   - Load monthly parquet files  
   - Filter relevant zones (e.g. Manhattan)  
   - Aggregate to hourly demand  
   - Build a full time × zone panel  

2. Train / Validation / Test Split  
   - Split by time ranges (monthly boundaries)  

3. Feature Engineering  
   - Calendar features (hour, day, weekend, holidays)  
   - Standard scaling (fit on train only)  

4. Dataset Construction  
   - Sliding window sequences for time-series learning  

5. Model Training  
   - LSTM model (PyTorch)  
   - Validation-based model selection  

6. Evaluation  
   - Predictions converted back to raw demand  
   - Metrics computed per zone and overall  

7. Artifact Saving (GCS)  
   - Model weights  
   - Scaler  
   - Config (data + model)  
   - Metrics + predictions  

## Project Structure

src/nyc_forecasting/
    core/
        data.py          # data loading, panel building, splitting
        features.py      # scaling + time features
        torch_dataset.py # sequence dataset
        lstm_class.py    # model definition
        lstm_functions.py# train/eval/predict
        metrics.py       # evaluation logic
        artifacts.py     # saving to GCS

train_lstm.py           # end-to-end training pipeline
run_preprocessing.py    # raw → processed data pipeline

## Model

- LSTM (PyTorch)  
- Input: past input_len hours of features  
- Output: next-step demand for all zones  

## Saved Artifacts

Each training run is versioned:

gs://<bucket>/models/<run_id>/
    best_model.pt
    scaler.joblib
    run_config.json
    summary.json
    per_zone.csv
    top_bottom.csv
    predictions.npz

## Configuration

Configuration is defined via dataclasses:

- DataConfig: data paths, splits, preprocessing  
- LSTMConfig: model and training parameters  

A combined run_config.json is saved for reproducibility.

## How to run

1. Setup environment

pip install -e .

(Install CUDA-enabled PyTorch separately if using GPU)

2. Run preprocessing

python -m nyc_forecasting.run_preprocessing

3. Train model

python -m nyc_forecasting.train_lstm

## Metrics

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- MAPE (Mean Absolute Percentage Error)  

Metrics are computed on raw demand values, not scaled data.

## Project Goal

This project demonstrates:

- End-to-end ML pipeline design  
- Clean modular code structure  
- Proper handling of time-series data  
- Reproducible experiments with saved artifacts  
- Practical MLOps patterns (GCS storage, versioned runs)  

## Future Improvements

- Add XGBoost or tree-based baseline  
- Add experiment tracking (MLflow or Weights & Biases)  
- Add model monitoring (data drift, prediction drift)  
- Deploy inference API (FastAPI or Cloud Run)
- Dashboard and Simulation 
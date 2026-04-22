# NYC Forecasting Modular Refactor

This refactor splits the original notebook-exported script into shared modules so you can:
- reuse the same data prep for LSTM, XGBoost, and other models
- keep evaluation logic in one place
- keep model-specific code small

## Structure

- `nyc_forecasting/config/settings.py` — central configs
- `nyc_forecasting/data/` — file loading, panel building, splitting, scaling, datasets
- `nyc_forecasting/features/calendar.py` — shared calendar feature engineering
- `nyc_forecasting/models/lstm.py` — LSTM model definition
- `nyc_forecasting/training/lstm_train.py` — train/eval/predict loop for PyTorch
- `nyc_forecasting/metrics/regression.py` — MAE/RMSE/MAPE evaluation
- `train_lstm.py` — end-to-end LSTM entry point
- `train_xgboost.py` — end-to-end XGBoost starter entry point

## Why this split works

The original script mixes together:
- environment setup
- data loading
- feature engineering
- sequence creation
- model code
- training
- evaluation

That is fine in a notebook, but painful once you want fair model comparisons.

This refactor keeps these shared steps common across models:
1. select parquet files
2. load combined data
3. build full hour-zone panel
4. split train/val/test by time
5. fit scaler on train only
6. add shared calendar features
7. generate model-specific training arrays

## Important notes

- The XGBoost script is a solid starter, not a finished model-selection pipeline.
- In Colab, save checkpoints locally first, then copy to Drive if needed.
- Evaluation uses raw test targets directly (`test_wide[input_len:]`) rather than inverse-transforming scaled targets again.

import numpy as np


def make_tabular_windows(
    X: np.ndarray,
    y: np.ndarray,
    input_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sequential data into tabular windows for models like XGBoost.

    X shape: [T, F]
    y shape: [T, num_targets]

    Returns
    -------
    X_tab : [T - input_len, input_len * F]
    y_tab : [T - input_len, num_targets]
    """
    X_list = []
    y_list = []

    for i in range(len(X) - input_len):
        x_window = X[i : i + input_len]          # [input_len, F]
        target = y[i + input_len]                # [num_targets]

        X_list.append(x_window.reshape(-1))      # flatten to 1D
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def make_selected_lag_tabular(
    demand_array: np.ndarray,
    time_features: np.ndarray,
    lags: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build tabular data for XGBoost using selected demand lags plus current time features.

    Parameters
    ----------
    demand_array : np.ndarray
        Scaled demand array of shape [T, num_targets]
    time_features : np.ndarray
        Time/calendar feature array of shape [T, num_time_features]
        These should represent the CURRENT timestep features only.
    lags : list[int]
        Selected lag steps to include, e.g. [1, 2, 3, 24, 168]

    Returns
    -------
    X_tab : np.ndarray
        Shape [N, num_targets * len(lags) + num_time_features]
    y_tab : np.ndarray
        Shape [N, num_targets]

    Notes
    -----
    For each time t, X contains:
    - demand at t-lag for each lag in `lags`
    - current time features at t

    y is demand at time t.
    """
    max_lag = max(lags)

    X_list = []
    y_list = []

    for t in range(max_lag, len(demand_array)):
        lagged_parts = [demand_array[t - lag] for lag in lags]
        lagged_flat = np.concatenate(lagged_parts, axis=0)

        current_time_feats = time_features[t]

        x_row = np.concatenate([lagged_flat, current_time_feats], axis=0)
        y_row = demand_array[t]

        X_list.append(x_row)
        y_list.append(y_row)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )
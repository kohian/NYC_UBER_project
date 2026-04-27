import numpy as np
import pandas as pd



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


# def make_selected_lag_tabular(
#     demand_array: np.ndarray,
#     time_features: np.ndarray,
#     lags: list[int],
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Build tabular data for XGBoost using selected demand lags plus current time features.

#     Parameters
#     ----------
#     demand_array : np.ndarray
#         Scaled demand array of shape [T, num_targets]
#     time_features : np.ndarray
#         Time/calendar feature array of shape [T, num_time_features]
#         These should represent the CURRENT timestep features only.
#     lags : list[int]
#         Selected lag steps to include, e.g. [1, 2, 3, 24, 168]

#     Returns
#     -------
#     X_tab : np.ndarray
#         Shape [N, num_targets * len(lags) + num_time_features]
#     y_tab : np.ndarray
#         Shape [N, num_targets]

#     Notes
#     -----
#     For each time t, X contains:
#     - demand at t-lag for each lag in `lags`
#     - current time features at t

#     y is demand at time t.
#     """
#     max_lag = max(lags)

#     X_list = []
#     y_list = []

#     for t in range(max_lag, len(demand_array)):
#         lagged_parts = [demand_array[t - lag] for lag in lags]
#         lagged_flat = np.concatenate(lagged_parts, axis=0)

#         current_time_feats = time_features[t]

#         x_row = np.concatenate([lagged_flat, current_time_feats], axis=0)
#         y_row = demand_array[t]

#         X_list.append(x_row)
#         y_list.append(y_row)

#     return (
#         np.array(X_list, dtype=np.float32),
#         np.array(y_list, dtype=np.float32),
#     )


def make_selected_lag_tabular(
    demand_array: np.ndarray,
    time_features: pd.DataFrame,
    lags: list[int],
    zone_names: list[str] | list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build tabular XGBoost features using selected demand lags + time features.

    Returns:
        X_tab: shape (num_samples, num_features)
        y_tab: shape (num_samples, num_zones)
        feature_names: names matching columns of X_tab
    """

    if len(lags) == 0:
        raise ValueError("lags must contain at least one lag value.")

    if min(lags) <= 0:
        raise ValueError("All lags must be positive integers, e.g. [1, 2, 24].")

    if len(demand_array) != len(time_features):
        raise ValueError(
            f"demand_array and time_features must have same length. "
            f"Got {len(demand_array)} and {len(time_features)}."
        )

    num_zones = demand_array.shape[1]

    if len(zone_names) != num_zones:
        raise ValueError(
            f"zone_names length must match number of demand columns. "
            f"Got {len(zone_names)} zone names but demand_array has {num_zones} zones."
        )

    max_lag = max(lags)

    feature_names = []

    # Lag demand features
    for lag in lags:
        for zone_name in zone_names:
            feature_names.append(f"lag_{lag}_zone_{zone_name}")

    # Time features
    feature_names.extend(time_features.columns.tolist())

    X_rows = []
    y_rows = []

    time_values = time_features.to_numpy(dtype="float32")

    for t in range(max_lag, len(demand_array)):
        row = []

        for lag in lags:
            row.extend(demand_array[t - lag])

        row.extend(time_values[t])

        X_rows.append(row)
        y_rows.append(demand_array[t])

    X_tab = np.array(X_rows, dtype="float32")
    y_tab = np.array(y_rows, dtype="float32")

    return X_tab, y_tab, feature_names




def compute_feature_importance(model, feature_names):
    """
    Compute feature importance from MultiOutput XGBoost model.

    Returns:
        fi_df: feature-level importance
        lag_importance: aggregated by lag
        zone_importance: aggregated by zone
        type_importance: temporal vs spatial
    """

    # Collect importance from each target model
    feature_importances = np.array([
        est.feature_importances_
        for est in model.estimators_
    ])

    # Average across targets
    mean_importance = feature_importances.mean(axis=0)

    # Base dataframe
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_importance
    }).sort_values("importance", ascending=False)

    # ---- Extract structure ----
    fi_df["lag"] = fi_df["feature"].str.extract(r"(lag_\d+)")
    fi_df["zone"] = fi_df["feature"].str.extract(r"zone_(\d+)")

    def classify_feature(f):
        if "lag_" in f:
            return "spatial"
        # elif "hour" in f or "dow" in f:
        #     return "temporal"
        else:
            return "other"

    fi_df["type"] = fi_df["feature"].apply(classify_feature)

    # ---- Aggregations ----
    lag_importance = (
        fi_df.groupby("lag")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    zone_importance = (
        fi_df.groupby("zone")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    # type_importance = (
    #     fi_df.groupby("type")["importance"]
    #     .sum()
    #     .sort_values(ascending=False)
    # )

    type_summary = (
        fi_df.groupby("type")["importance"]
        .agg(
            sum="sum",
            mean="mean",
            count="count"
        )
        .sort_values(by="sum", ascending=False)
    )

    return fi_df, lag_importance, zone_importance, type_summary
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NYC_HOLIDAYS = pd.to_datetime([
    # ---- 2023 ----
    "2023-01-02",  # New Year (observed)
    "2023-01-16",
    "2023-02-20",
    "2023-05-29",
    "2023-06-19",
    "2023-07-04",
    "2023-09-04",
    "2023-10-09",
    "2023-11-07",
    "2023-11-11",
    "2023-11-23",
    "2023-12-25",

    # ---- 2024 ----
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-10-14",
    "2024-11-05",
    "2024-11-11",
    "2024-11-28",
    "2024-12-25",

    # ---- 2025 ----
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-10-13",
    "2025-11-04",
    "2025-11-11",
    "2025-11-27",
    "2025-12-25",

    # ---- 2026 ----
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",  # July 4 observed (falls on Saturday)
    "2026-09-07",
    "2026-10-12",
    "2026-11-03",
    "2026-11-11",
    "2026-11-26",
    "2026-12-25",
])



def add_time_features(
    wide_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex | None = None,
    include_month: bool = True,
) -> np.ndarray:
    time_index = wide_df.index
    holiday_dates = NYC_HOLIDAYS if holiday_dates is None else holiday_dates
    holiday_dates = set(pd.DatetimeIndex(holiday_dates).normalize())

    hour_of_day = time_index.hour.values
    day_of_week = time_index.dayofweek.values

    hour_sin = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7).astype(np.float32)
    is_weekend = (day_of_week >= 5).astype(np.float32)

    dates_only = pd.DatetimeIndex(time_index.normalize())
    is_holiday = dates_only.isin(holiday_dates).astype(np.float32)

    extra_cols = [hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday]

    if include_month:
        month_of_year = time_index.month.values
        month_sin = np.sin(2 * np.pi * (month_of_year - 1) / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * (month_of_year - 1) / 12).astype(np.float32)
        extra_cols.extend([month_sin, month_cos])

    demand_matrix = wide_df.to_numpy(dtype=np.float32)
    extra_feats = np.column_stack(extra_cols).astype(np.float32)
    return np.concatenate([demand_matrix, extra_feats], axis=1).astype(np.float32)


def fit_demand_scaler(train_wide: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_wide)
    return scaler


def transform_wide_frame(wide_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(wide_df),
        index=wide_df.index,
        columns=wide_df.columns,
    )


# def make_time_features_only(index: pd.DatetimeIndex) -> np.ndarray:
#     """
#     Create time-based features from a DatetimeIndex.

#     Returns:
#     --------
#     np.ndarray of shape [T, num_features]
#     """

#     hour = index.hour
#     dow = index.dayofweek

#     hour_sin = np.sin(2 * np.pi * hour / 24)
#     hour_cos = np.cos(2 * np.pi * hour / 24)

#     dow_sin = np.sin(2 * np.pi * dow / 7)
#     dow_cos = np.cos(2 * np.pi * dow / 7)

#     return np.stack(
#         [hour_sin, hour_cos, dow_sin, dow_cos],
#         axis=1,
#     ).astype("float32")


def make_time_features_only(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour
    dow = index.dayofweek

    df = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=index)

    return df.astype("float32")


def make_time_features_only(index: pd.DatetimeIndex) -> pd.DataFrame:
    holiday_dates = set(pd.DatetimeIndex(NYC_HOLIDAYS).normalize())

    hour = index.hour
    dow = index.dayofweek
    month = index.month

    df = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "is_weekend": (dow >= 5).astype("float32"),
        "is_holiday": pd.DatetimeIndex(index.normalize())
            .isin(holiday_dates)
            .astype("float32"),
    }, index=index)

    return df.astype("float32")
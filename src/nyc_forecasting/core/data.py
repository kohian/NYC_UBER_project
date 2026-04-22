from __future__ import annotations

import pandas as pd
import numpy as np


def ym_to_start_ts(ym: str) -> pd.Timestamp:
    """
    Convert 'YYYY-MM' -> start-of-month timestamp.

    Example:
        '2024-07' -> Timestamp('2024-07-01 00:00:00')
    """
    return pd.Timestamp(f"{ym}-01 00:00:00")


def ym_to_exclusive_end_ts(ym: str) -> pd.Timestamp:
    """
    Convert 'YYYY-MM' -> start of next month timestamp.

    Useful for filtering with:
        start <= ts < end
    """
    ts = ym_to_start_ts(ym)
    return ts + pd.offsets.MonthBegin(1)


def generate_year_month_list(start_ym: str, end_ym: str) -> list[str]:
    """
    Generate inclusive list of YYYY-MM strings.

    Example:
        generate_year_month_list("2024-01", "2024-03")
        -> ["2024-01", "2024-02", "2024-03"]
    """
    dates = pd.date_range(
        start=ym_to_start_ts(start_ym),
        end=ym_to_start_ts(end_ym),
        freq="MS",
    )
    return [d.strftime("%Y-%m") for d in dates]


def build_monthly_raw_path(base: str, year_month: str) -> str:
    """
    Build raw parquet path for one month.

    Example:
        base = "Data"
        -> "Data/fhvhv_tripdata_2024-01.parquet"

        base = "gs://my-bucket/raw"
        -> "gs://my-bucket/raw/fhvhv_tripdata_2024-01.parquet"
    """
    return f"{base.rstrip('/')}/fhvhv_tripdata_{year_month}.parquet"


def build_processed_path(base: str, year_month: str) -> str:
    """
    Build processed parquet path for one month.

    Example:
        base = "Data/processed/hourly_demand"
        -> "Data/processed/hourly_demand/hourly_demand_2024-01.parquet"

        base = "gs://my-bucket/processed/hourly_demand"
        -> "gs://my-bucket/processed/hourly_demand/hourly_demand_2024-01.parquet"
    """
    return f"{base.rstrip('/')}/hourly_demand_{year_month}.parquet"


def select_files(data_dir: str, start_ym: str, end_ym: str) -> list[str]:
    """
    Build explicit list of processed monthly parquet paths
    for the requested inclusive YYYY-MM range.

    Works for both local paths and gs:// paths.
    """
    year_month_list = generate_year_month_list(start_ym, end_ym)
    return [build_processed_path(data_dir, ym) for ym in year_month_list]


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV from local path or gs:// path.

    For gs:// paths, make sure gcsfs is installed.
    """
    return pd.read_csv(path, **kwargs)


def read_parquet(path: str, columns: list[str] | None = None, **kwargs) -> pd.DataFrame:
    """
    Read parquet from local path or gs:// path.

    For gs:// paths, make sure gcsfs is installed.
    """
    return pd.read_parquet(path, columns=columns, engine="pyarrow", **kwargs)


def write_parquet(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Write parquet to local path or gs:// path.

    For gs:// paths, pandas/gcsfs handles the write.
    """
    if not path.startswith("gs://"):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False, engine="pyarrow", **kwargs)


def load_monthly_files(files: list[str]) -> pd.DataFrame:
    """
    Load monthly processed parquet files and combine into one DataFrame.

    Expected columns:
        - hour
        - PULocationID
        - demand
    """
    if not files:
        raise ValueError("No files selected.")

    dfs = []
    for f in files:
        df = read_parquet(f)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["hour"] = pd.to_datetime(out["hour"])
    out["PULocationID"] = out["PULocationID"].astype("int32")
    out["demand"] = out["demand"].astype("float32")

    out = out.sort_values(["hour", "PULocationID"]).reset_index(drop=True)
    return out


def get_borough_zone_ids(zone_lookup: pd.DataFrame, borough: str) -> set[int]:
    """
    Return LocationIDs for the requested borough.
    """
    return set(
        zone_lookup.loc[zone_lookup["Borough"] == borough, "LocationID"]
        .astype("int32")
        .tolist()
    )


def process_monthly_hourly_demand(
    df: pd.DataFrame,
    allowed_zone_ids: set[int],
    year_month: str,
    keep_license: str | None = None,
) -> pd.DataFrame:
    """
    Process one raw monthly trip DataFrame into hourly demand by pickup zone.

    Input expected columns:
        - hvfhs_license_num
        - request_datetime
        - PULocationID

    Output columns:
        - hour
        - PULocationID
        - demand
    """
    year, month = map(int, year_month.split("-"))

    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)

    if keep_license is not None:
        df = df[df["hvfhs_license_num"] == keep_license]

    df = df.dropna(subset=["request_datetime", "PULocationID"])
    df["request_datetime"] = pd.to_datetime(df["request_datetime"])
    df["PULocationID"] = df["PULocationID"].astype("int32")

    df = df[(df["request_datetime"] >= start) & (df["request_datetime"] < end)]
    df = df[df["PULocationID"].isin(allowed_zone_ids)].copy()

    df["hour"] = df["request_datetime"].dt.floor("h")

    hourly = (
        df.groupby(["hour", "PULocationID"])
        .size()
        .reset_index(name="demand")
        .sort_values(["hour", "PULocationID"])
        .reset_index(drop=True)
    )

    return hourly


def make_full_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a full hour-zone panel and return wide demand matrix.

    Input columns:
        - hour
        - PULocationID
        - demand

    Output:
        wide_df with:
        - index = hour
        - columns = PULocationID
        - values = demand
    """
    all_hours = pd.date_range(
        start=df["hour"].min(),
        end=df["hour"].max(),
        freq="H",
    )
    all_zones = np.sort(df["PULocationID"].unique())

    full_index = pd.MultiIndex.from_product(
        [all_hours, all_zones],
        names=["hour", "PULocationID"],
    )

    full_df = (
        df.set_index(["hour", "PULocationID"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    wide_df = (
        full_df.pivot(index="hour", columns="PULocationID", values="demand")
        .sort_index()
        .fillna(0)
    )

    return wide_df


def split_wide_by_month(
    wide_df: pd.DataFrame,
    start_ym: str,
    end_ym: str,
) -> pd.DataFrame:
    """
    Slice a wide hourly DataFrame by inclusive YYYY-MM range.
    """
    start_ts = ym_to_start_ts(start_ym)
    end_ts = ym_to_exclusive_end_ts(end_ym)
    return wide_df[(wide_df.index >= start_ts) & (wide_df.index < end_ts)]


def make_raw_targets(
    wide_df: pd.DataFrame,
    input_len: int,
) -> np.ndarray:
    """
    Build raw next-step targets aligned with sequence predictions.

    Assumes:
        - input window length = input_len
        - target = next row after window
        - stride = 1

    Output shape:
        [T - input_len, num_zones]
    """
    return wide_df.to_numpy(dtype=np.float32)[input_len:]
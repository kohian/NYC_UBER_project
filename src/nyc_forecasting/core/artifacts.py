from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import gcsfs
import torch


def save_torch_state_dict_to_gcs(
    state_dict: dict,
    base_path: str,
    filename: str = "best_model.pt",
) -> str:
    """
    Save a PyTorch state_dict to GCS.

    Parameters
    ----------
    state_dict : dict
        PyTorch model state_dict
    base_path : str
        GCS directory, e.g. gs://bucket/models/run_001
    filename : str
        Output filename

    Returns
    -------
    str
        Full saved path
    """
    fs = gcsfs.GCSFileSystem()
    path = f"{base_path.rstrip('/')}/{filename}"

    with fs.open(path, "wb") as f:
        torch.save(state_dict, f)

    print(f"Saved model state_dict to {path}")
    return path


def load_torch_state_dict_from_gcs(path: str, map_location: str | None = None) -> dict:
    """
    Load a PyTorch state_dict from GCS.
    """
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "rb") as f:
        return torch.load(f, map_location=map_location)


def save_scaler_to_gcs(
    scaler,
    base_path: str,
    filename: str = "scaler.joblib",
) -> str:
    """
    Save a fitted scaler to GCS.
    """
    fs = gcsfs.GCSFileSystem()
    path = f"{base_path.rstrip('/')}/{filename}"

    with fs.open(path, "wb") as f:
        joblib.dump(scaler, f)

    print(f"Saved scaler to {path}")
    return path


def load_scaler_from_gcs(path: str):
    """
    Load a scaler from GCS.
    """
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "rb") as f:
        return joblib.load(f)


def build_top_bottom_df(results: dict) -> pd.DataFrame:
    """
    Combine best/worst metric tables into one DataFrame.
    """
    return pd.concat(
        [
            results["worst_5_rmse"].assign(type="worst_rmse"),
            results["best_5_rmse"].assign(type="best_rmse"),
            results["worst_5_mae"].assign(type="worst_mae"),
            results["best_5_mae"].assign(type="best_mae"),
            results["worst_5_mape"].assign(type="worst_mape"),
            results["best_5_mape"].assign(type="best_mape"),
        ],
        ignore_index=True,
    )


def save_results_to_gcs(
    results: dict,
    base_path: str,
    metadata: dict | None = None,
) -> None:
    """
    Save evaluation results to GCS.

    Files created:
    - summary.json
    - per_zone.csv
    - top_bottom.csv
    - predictions.npz
    """
    fs = gcsfs.GCSFileSystem()
    base_path = base_path.rstrip("/")

    summary = {
        "overall_mae": float(results["overall_mae"]),
        "overall_rmse": float(results["overall_rmse"]),
        "overall_mape": float(results["overall_mape"]),
    }

    if metadata is not None:
        summary.update(metadata)

    with fs.open(f"{base_path}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with fs.open(f"{base_path}/per_zone.csv", "w") as f:
        results["per_zone_df"].to_csv(f, index=False)

    top_bottom_df = build_top_bottom_df(results)
    with fs.open(f"{base_path}/top_bottom.csv", "w") as f:
        top_bottom_df.to_csv(f, index=False)

    with fs.open(f"{base_path}/predictions.npz", "wb") as f:
        np.savez(
            f,
            preds=results["preds_raw"],
            targets=results["targets_raw"],
        )

    print(f"Saved results to {base_path}")
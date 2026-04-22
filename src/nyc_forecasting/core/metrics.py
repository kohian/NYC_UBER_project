import json
import numpy as np
import pandas as pd
import gcsfs



def calculate_regression_metrics(
    preds_raw: np.ndarray,
    targets_raw: np.ndarray,
    zone_names=None,
    mape_mode: str = "exclude",
    epsilon: float = 1e-8,
) -> dict:
    if preds_raw.shape != targets_raw.shape:
        raise ValueError(
            f"Shape mismatch: preds.shape={preds_raw.shape}, targets.shape={targets_raw.shape}"
        )

    if preds_raw.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays [N, num_zones], got preds.ndim={preds_raw.ndim}"
        )

    if mape_mode not in {"exclude", "epsilon"}:
        raise ValueError(f"mape_mode must be 'exclude' or 'epsilon', got {mape_mode!r}")

    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    num_zones = preds_raw.shape[1]

    if zone_names is None:
        zone_names = [f"zone_{i}" for i in range(num_zones)]

    if len(zone_names) != num_zones:
        raise ValueError(
            f"len(zone_names)={len(zone_names)} does not match num_zones={num_zones}"
        )

    errors = preds_raw - targets_raw
    abs_errors = np.abs(errors)
    sq_errors = errors**2

    overall_mae = abs_errors.mean()
    overall_rmse = np.sqrt(sq_errors.mean())

    if mape_mode == "exclude":
        valid_mape_mask = targets_raw != 0
        abs_pct_errors = np.full_like(targets_raw, np.nan, dtype=np.float64)
        abs_pct_errors[valid_mape_mask] = (
            abs_errors[valid_mape_mask] / np.abs(targets_raw[valid_mape_mask])
        ) * 100.0
        overall_mape = np.nanmean(abs_pct_errors)
        mape_per_zone = np.nanmean(abs_pct_errors, axis=0)
    else:
        denom = np.maximum(np.abs(targets_raw), epsilon)
        abs_pct_errors = (abs_errors / denom) * 100.0
        overall_mape = abs_pct_errors.mean()
        mape_per_zone = abs_pct_errors.mean(axis=0)

    mae_per_zone = abs_errors.mean(axis=0)
    rmse_per_zone = np.sqrt(sq_errors.mean(axis=0))

    per_zone_df = pd.DataFrame({
        "zone": zone_names,
        "mae": mae_per_zone,
        "rmse": rmse_per_zone,
        "mape": mape_per_zone,
    })

    worst_5_rmse = per_zone_df.sort_values("rmse", ascending=False).head(5).reset_index(drop=True)
    best_5_rmse = per_zone_df.sort_values("rmse", ascending=True).head(5).reset_index(drop=True)
    worst_5_mae = per_zone_df.sort_values("mae", ascending=False).head(5).reset_index(drop=True)
    best_5_mae = per_zone_df.sort_values("mae", ascending=True).head(5).reset_index(drop=True)

    per_zone_mape_valid = per_zone_df.dropna(subset=["mape"]) if mape_mode == "exclude" else per_zone_df
    worst_5_mape = per_zone_mape_valid.sort_values("mape", ascending=False).head(5).reset_index(drop=True)
    best_5_mape = per_zone_mape_valid.sort_values("mape", ascending=True).head(5).reset_index(drop=True)

    return {
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_mape": overall_mape,
        "per_zone_df": per_zone_df,
        "worst_5_rmse": worst_5_rmse,
        "best_5_rmse": best_5_rmse,
        "worst_5_mae": worst_5_mae,
        "best_5_mae": best_5_mae,
        "worst_5_mape": worst_5_mape,
        "best_5_mape": best_5_mape,
        "preds_raw": preds_raw,
        "targets_raw": targets_raw,
    }

def print_metric_summary(results: dict, mape_mode: str | None = None) -> None:
    print("=" * 50)
    print(f"Overall MAE  : {results['overall_mae']:.4f}")
    print(f"Overall RMSE : {results['overall_rmse']:.4f}")

    if "overall_mape" in results:
        if np.isnan(results["overall_mape"]):
            print("Overall MAPE : NaN (no valid targets)")
        else:
            print(f"Overall MAPE : {results['overall_mape']:.2f}%")

    if mape_mode:
        print(f"(MAPE mode: {mape_mode})")

    print("=" * 50)
    print("\nWorst 5 zones by RMSE")
    print(results["worst_5_rmse"].to_string(index=False))
    print("\nBest 5 zones by RMSE")
    print(results["best_5_rmse"].to_string(index=False))
    print("\nWorst 5 zones by MAE")
    print(results["worst_5_mae"].to_string(index=False))
    print("\nBest 5 zones by MAE")
    print(results["best_5_mae"].to_string(index=False))

    if "worst_5_mape" in results and "best_5_mape" in results:
        print("\nWorst 5 zones by MAPE")
        print(results["worst_5_mape"].to_string(index=False))
        print("\nBest 5 zones by MAPE")
        print(results["best_5_mape"].to_string(index=False))



# def save_results_to_gcs(
#     results: dict,
#     base_path: str,
# ) -> None:
#     """
#     Save model evaluation results to GCS.

#     Files created:
#     - summary.json
#     - per_zone.csv
#     - top_bottom.csv
#     - predictions.npz
#     """

#     fs = gcsfs.GCSFileSystem()

#     # -----------------------------
#     # 1. Summary JSON
#     # -----------------------------
#     summary = {
#         "overall_mae": float(results["overall_mae"]),
#         "overall_rmse": float(results["overall_rmse"]),
#         "overall_mape": float(results["overall_mape"]),
#     }

#     with fs.open(f"{base_path}/summary.json", "w") as f:
#         json.dump(summary, f, indent=2)

#     # -----------------------------
#     # 2. Per-zone CSV
#     # -----------------------------
#     with fs.open(f"{base_path}/per_zone.csv", "w") as f:
#         results["per_zone_df"].to_csv(f, index=False)

#     # -----------------------------
#     # 3. Top/bottom CSV
#     # -----------------------------
#     top_bottom_df = pd.concat([
#         results["worst_5_rmse"].assign(label="worst_rmse"),
#         results["best_5_rmse"].assign(label="best_rmse"),
#         results["worst_5_mae"].assign(label="worst_mae"),
#         results["best_5_mae"].assign(label="best_mae"),
#         results["worst_5_mape"].assign(label="worst_mape"),
#         results["best_5_mape"].assign(label="best_mape"),
#     ], ignore_index=True)

#     with fs.open(f"{base_path}/top_bottom.csv", "w") as f:
#         top_bottom_df.to_csv(f, index=False)

#     # -----------------------------
#     # 4. Predictions (npz)
#     # -----------------------------
#     with fs.open(f"{base_path}/predictions.npz", "wb") as f:
#         np.savez(
#             f,
#             preds=results["preds_raw"],
#             targets=results["targets_raw"],
#         )

#     print(f"Results saved to: {base_path}")
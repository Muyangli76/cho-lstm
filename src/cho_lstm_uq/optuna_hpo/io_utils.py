from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def read_val_metrics(outdir: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Read validation CSVs produced by the training script.

    Expects:
        - VAL_xab_flat.csv: rows named {"X_gL", "Ab_gL"} with columns R2 / MAE / RMSE.
        - VAL_pools_flat.csv: per-pool metrics; macro averages will be computed.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]:
            - metrics: per-target (e.g., X_gL, Ab_gL) metrics
            - metrics_pools: macro-averaged pool metrics
    """
    xab_csv = outdir / "VAL_xab_flat.csv"
    pools_csv = outdir / "VAL_pools_flat.csv"

    if not xab_csv.exists():
        raise FileNotFoundError(f"Missing {xab_csv}")
    if not pools_csv.exists():
        raise FileNotFoundError(f"Missing {pools_csv}")

    df_xab = pd.read_csv(xab_csv)
    df_pools = pd.read_csv(pools_csv)

    metrics: Dict[str, float] = {}

    # Tolerate both 'name' and 'target' column labels
    name_col = (
        "name"
        if "name" in df_xab.columns
        else ("target" if "target" in df_xab.columns else None)
    )
    if name_col is None:
        raise KeyError("VAL_xab_flat.csv must contain a 'name' or 'target' column")

    for _, row in df_xab.iterrows():
        name = str(row.get(name_col, "")).strip()
        for k in ["R2", "MAE", "RMSE"]:
            if k in row:
                metrics[f"{k}_{name}"] = float(row[k])

    metrics_pools = {
        "macro_MAE_pools": float(df_pools["MAE"].mean()),
        "macro_RMSE_pools": float(df_pools["RMSE"].mean()),
        "macro_R2_pools": float(df_pools["R2"].mean()),
    }

    return metrics, metrics_pools

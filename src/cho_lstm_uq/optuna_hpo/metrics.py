# optuna_hpo/metrics.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import math
import pandas as pd

def _safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")

def read_val_metrics(trial_outdir: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Reads the two CSVs your trainer writes:
      - VAL_xab_flat.csv   (columns: name, R2, MAE, RMSE) with names "X_gL", "Ab_gL"
      - VAL_pools_flat.csv (columns: name, R2, MAE, RMSE) for each pool; we take macro means

    Returns:
      (xab_dict, pools_dict)
      xab_dict keys:   R2_X_gL, R2_Ab_gL, MAE_X_gL, MAE_Ab_gL, RMSE_X_gL, RMSE_Ab_gL
      pools_dict keys: macro_R2_pools, macro_MAE_pools, macro_RMSE_pools
    """
    trial_outdir = Path(trial_outdir)
    p_xab   = trial_outdir / "VAL_xab_flat.csv"
    p_pools = trial_outdir / "VAL_pools_flat.csv"

    xab: Dict[str, float] = {}
    if p_xab.exists():
        dfx = pd.read_csv(p_xab)
        def get_for(name: str, col: str) -> float:
            s = dfx.loc[dfx["name"] == name, col]
            return _safe_float(s.iloc[0]) if not s.empty else float("nan")

        xab = {
            "R2_X_gL":   get_for("X_gL",  "R2"),
            "R2_Ab_gL":  get_for("Ab_gL", "R2"),
            "MAE_X_gL":  get_for("X_gL",  "MAE"),
            "MAE_Ab_gL": get_for("Ab_gL", "MAE"),
            "RMSE_X_gL": get_for("X_gL",  "RMSE"),
            "RMSE_Ab_gL":get_for("Ab_gL", "RMSE"),
        }

    pools: Dict[str, float] = {}
    if p_pools.exists():
        dfp = pd.read_csv(p_pools)
        if not dfp.empty:
            pools = {
                "macro_R2_pools":   _safe_float(dfp["R2"].mean()),
                "macro_MAE_pools":  _safe_float(dfp["MAE"].mean()),
                "macro_RMSE_pools": _safe_float(dfp["RMSE"].mean()),
            }
        else:
            pools = {"macro_R2_pools": float("nan"),
                     "macro_MAE_pools": float("nan"),
                     "macro_RMSE_pools": float("nan")}
    return xab, pools

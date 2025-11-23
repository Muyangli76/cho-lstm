"""
Stage A — resample/clean the raw simulator export to a 1-hour grid,
recompute auxiliaries, and write a *_Cleaned.csv.
"""

from __future__ import annotations
import os
from typing import Dict
import numpy as np
import pandas as pd

A_GLC = 6.0  # mmol-C per mmol glucose
B_CO2_BY_SCALE = {"2L": 1.80, "50L": 1.00, "2000L": 0.70}
B_CO2_DEFAULT = 1.80


def _get_B_co2(scale: str) -> float:
    if pd.isna(scale):
        return B_CO2_DEFAULT
    s = str(scale).replace(" ", "")
    return float(B_CO2_BY_SCALE.get(s, B_CO2_DEFAULT))


def resample_run(g: pd.DataFrame, has_ctr_meas: bool) -> pd.DataFrame:
    g = g.sort_values("time").reset_index(drop=True).copy()
    run_id = str(g["batch_name"].iloc[0])
    scale = str(g["scale"].iloc[0])

    # 1) 1-hour grid
    t0 = float(np.floor(g["time"].min()))
    t1 = float(np.ceil(g["time"].max()))
    t_grid = np.arange(t0, t1 + 1.0, 1.0)

    # 2) Linear interpolation on key columns
    cols_interp = [
        "V", "pCO2", "feed_glucose", "qCO2",
        "C_X", "C_glc", "C_lac", "C_Ab", "C_X_raw_cells_per_ml",
    ]
    if has_ctr_meas:
        cols_interp.append("CTR_mmolC_L_h")

    out = pd.DataFrame({"time_h": t_grid})
    t_src = g["time"].astype(float).values
    for c in cols_interp:
        out[c] = np.interp(t_grid, t_src, g[c].astype(float).values)

    # 3) IDs & constants
    out["run_id"] = run_id
    out["scale"] = scale
    out["B_CO2_eff"] = _get_B_co2(scale)

    # 4) Derived ops
    out["V_L"] = out["V"] / 1000.0

    V_L = out["V_L"].to_numpy()
    V_prev = np.roll(V_L, 1)
    V_prev[0] = V_L[0]
    vvd_per_day = 24.0 * (V_L - V_prev) / np.maximum(V_prev, 1e-12)
    vvd_per_day[0] = 0.0
    out["vvd_per_day"] = vvd_per_day
    out["Fin_over_V_1ph"] = vvd_per_day / 24.0

    out["CinC_mmolC_L"] = A_GLC * out["feed_glucose"]
    out["DIC_mmolC_L"] = out["B_CO2_eff"] * out["pCO2"]

    t = out["time_h"].to_numpy()
    DIC = out["DIC_mmolC_L"].to_numpy()
    out["dDIC_dt"] = np.gradient(DIC, t)

    cells_per_mL = out["C_X_raw_cells_per_ml"].to_numpy()
    out["ProdCO2_mmolC_L_h"] = out["qCO2"].to_numpy() * cells_per_mL * 1000.0

    out["Dilution_DIC"] = out["Fin_over_V_1ph"] * out["DIC_mmolC_L"]
    out["CTR_calc_mmolC_L_h"] = (
        out["ProdCO2_mmolC_L_h"] - out["dDIC_dt"] - out["Dilution_DIC"]
    )

    if has_ctr_meas:
        out["CTR_meas_mmolC_L_h"] = out["CTR_mmolC_L_h"]
        out["R_C_balance"] = (
            out["dDIC_dt"]
            - out["Fin_over_V_1ph"] * (out["CinC_mmolC_L"] - out["DIC_mmolC_L"])
            - out["ProdCO2_mmolC_L_h"]
            + out["CTR_meas_mmolC_L_h"]
        )
    else:
        out["R_C_balance"] = (
            out["dDIC_dt"]
            - out["Fin_over_V_1ph"] * (out["CinC_mmolC_L"] - out["DIC_mmolC_L"])
            - out["ProdCO2_mmolC_L_h"]
            + out["CTR_calc_mmolC_L_h"]
        )

    base_cols = [
        "run_id", "scale", "time_h",
        "C_X", "C_glc", "C_lac", "C_Ab", "pCO2", "C_X_raw_cells_per_ml",
        "V_L", "vvd_per_day", "Fin_over_V_1ph",
        "feed_glucose", "CinC_mmolC_L",
        "DIC_mmolC_L", "dDIC_dt",
        "ProdCO2_mmolC_L_h", "Dilution_DIC",
        "R_C_balance", "CTR_calc_mmolC_L_h",
    ]
    if has_ctr_meas:
        base_cols.insert(base_cols.index("CTR_calc_mmolC_L_h") + 1, "CTR_meas_mmolC_L_h")

    return out[base_cols].sort_values("time_h").reset_index(drop=True)


def resample_csv(input_csv: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df_raw = pd.read_csv(input_csv)
    required = {
        "time", "V", "pCO2", "feed_glucose", "qCO2",
        "C_X", "C_glc", "C_lac", "C_Ab", "C_X_raw_cells_per_ml",
        "batch_name", "scale",
    }
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_ctr_meas = "CTR_mmolC_L_h" in df_raw.columns
    parts = [resample_run(g, has_ctr_meas) for _, g in df_raw.groupby("batch_name")]
    df_out = pd.concat(parts, ignore_index=True)

    base = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv = os.path.join(out_dir, f"{base}_Cleaned.csv")
    df_out.to_csv(out_csv, index=False)
    return out_csv

# cho_lstm_uq/prep/resample.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


# =========================== Config & helpers ===========================

@dataclass(frozen=True)
class Config:
    A_GLC: float = 6.0  # mmol-C per mmol glucose
    B_CO2_BY_SCALE: Dict[str, float] = None
    B_CO2_DEFAULT: float = 1.80

    def __post_init__(self):
        # dataclass default for dict
        if self.B_CO2_BY_SCALE is None:
            object.__setattr__(self, "B_CO2_BY_SCALE",
                {"2L": 1.80, "50L": 1.00, "2000L": 0.70})

    def get_B_co2(self, scale: str) -> float:
        if scale is None or (isinstance(scale, float) and pd.isna(scale)):
            return self.B_CO2_DEFAULT
        s = str(scale).replace(" ", "")
        return float(self.B_CO2_BY_SCALE.get(s, self.B_CO2_DEFAULT))


REQUIRED_COLS = {
    "time", "V", "pCO2", "feed_glucose", "qCO2",
    "C_X", "C_glc", "C_lac", "C_Ab", "C_X_raw_cells_per_ml",
    "batch_name", "scale",
}


# =========================== Core logic ===========================

def resample_run(g: pd.DataFrame, cfg: Config, has_ctr_meas: bool) -> pd.DataFrame:
    """
    Resample one run to 1 h grid and compute derived carbonate/CTR terms.
    """
    g = g.sort_values("time").reset_index(drop=True).copy()
    run_id = str(g["batch_name"].iloc[0])
    scale  = str(g["scale"].iloc[0])

    # 1) 1-hour grid
    t0 = float(np.floor(g["time"].min()))
    t1 = float(np.ceil(g["time"].max()))
    t_grid = np.arange(t0, t1 + 1.0, 1.0)

    # 2) Linear interpolation on key columns
    cols_interp = [
        "V","pCO2","feed_glucose","qCO2",
        "C_X","C_glc","C_lac","C_Ab","C_X_raw_cells_per_ml"
    ]
    if has_ctr_meas:
        cols_interp.append("CTR_mmolC_L_h")  # independent measurement if present

    g_interp = pd.DataFrame({"time_h": t_grid})
    t_src = g["time"].astype(float).values
    for c in cols_interp:
        g_interp[c] = np.interp(t_grid, t_src, g[c].astype(float).values)

    # 3) IDs & constants
    g_interp["run_id"] = run_id
    g_interp["scale"]  = scale
    g_interp["B_CO2_eff"] = cfg.get_B_co2(scale)

    # 4) Derived ops
    # Volume (mL -> L)
    g_interp["V_L"] = g_interp["V"] / 1000.0

    # VVD (per day) and Fin/V (1/h)
    V_L = g_interp["V_L"].to_numpy()
    V_prev = np.roll(V_L, 1); V_prev[0] = V_L[0]
    vvd_per_day = 24.0 * (V_L - V_prev) / np.maximum(V_prev, 1e-12)
    vvd_per_day[0] = 0.0
    g_interp["vvd_per_day"] = vvd_per_day
    g_interp["Fin_over_V_1ph"] = vvd_per_day / 24.0

    # Carbon in feed (mmol-C/L)
    g_interp["CinC_mmolC_L"] = cfg.A_GLC * g_interp["feed_glucose"]

    # DIC from pCO2 and buffer capacity
    g_interp["DIC_mmolC_L"] = g_interp["B_CO2_eff"] * g_interp["pCO2"]

    # d(DIC)/dt (mmol-C/L/h)
    t = g_interp["time_h"].to_numpy()
    DIC = g_interp["DIC_mmolC_L"].to_numpy()
    g_interp["dDIC_dt"] = np.gradient(DIC, t)

    # CO2 production (mmol-C/L/h)
    cells_per_mL = g_interp["C_X_raw_cells_per_ml"].to_numpy()
    g_interp["ProdCO2_mmolC_L_h"] = g_interp["qCO2"].to_numpy() * cells_per_mL * 1000.0  # per L

    # Dilution term (Fin/V * DIC)
    g_interp["Dilution_DIC"] = g_interp["Fin_over_V_1ph"] * g_interp["DIC_mmolC_L"]

    # CTR_calc from balance
    g_interp["CTR_calc_mmolC_L_h"] = (
        g_interp["ProdCO2_mmolC_L_h"] - g_interp["dDIC_dt"] - g_interp["Dilution_DIC"]
    )

    if has_ctr_meas:
        g_interp["CTR_meas_mmolC_L_h"] = g_interp["CTR_mmolC_L_h"]
        # Residual using measured CTR
        g_interp["R_C_balance"] = (
            g_interp["dDIC_dt"]
            - g_interp["Fin_over_V_1ph"] * (g_interp["CinC_mmolC_L"] - g_interp["DIC_mmolC_L"])
            - g_interp["ProdCO2_mmolC_L_h"]
            + g_interp["CTR_meas_mmolC_L_h"]
        )
    else:
        # Diagnostic residual if no independent CTR is available
        g_interp["R_C_balance"] = (
            g_interp["dDIC_dt"]
            - g_interp["Fin_over_V_1ph"] * (g_interp["CinC_mmolC_L"] - g_interp["DIC_mmolC_L"])
            - g_interp["ProdCO2_mmolC_L_h"]
            + g_interp["CTR_calc_mmolC_L_h"]
        )

    # 5) Tidy columns
    base_cols = [
        "run_id","scale","time_h",
        "C_X","C_glc","C_lac","C_Ab","pCO2","C_X_raw_cells_per_ml",
        "V_L","vvd_per_day","Fin_over_V_1ph",
        "feed_glucose","CinC_mmolC_L",
        "DIC_mmolC_L","dDIC_dt",
        "ProdCO2_mmolC_L_h","Dilution_DIC",
        "R_C_balance","CTR_calc_mmolC_L_h"
    ]
    if has_ctr_meas:
        base_cols.insert(base_cols.index("CTR_calc_mmolC_L_h")+1, "CTR_meas_mmolC_L_h")

    out = g_interp[base_cols].copy()
    return out.sort_values("time_h").reset_index(drop=True)


def resample_all_runs(df_raw: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    """
    Validate columns, detect measured CTR, resample every run, and return one DataFrame.
    """
    cfg = cfg or Config()

    missing = REQUIRED_COLS - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_ctr_meas = "CTR_mmolC_L_h" in df_raw.columns

    parts = [resample_run(g, cfg, has_ctr_meas) for _, g in df_raw.groupby("batch_name")]
    return pd.concat(parts, ignore_index=True)


# =========================== Notebook/CLI wrappers ===========================

def resample_csv(input_csv: str | Path,
                 out_dir: Optional[str | Path] = None,
                 out_csv: Optional[str | Path] = None,
                 cfg: Optional[Config] = None) -> pd.DataFrame:
    """
    Read a CSV, resample all runs, optionally save, and return the DataFrame.
    """
    input_csv = Path(input_csv)
    df_raw = pd.read_csv(input_csv)
    df_out = resample_all_runs(df_raw, cfg=cfg)

    if out_dir is not None and out_csv is None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = input_csv.stem
        out_csv = out_dir / f"{base}_Cleaned.csv"

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)

    return df_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Resample CHO runs to 1h grid and compute carbonate/CTR terms.")
    p.add_argument("--in", dest="inp", required=True, help="Input CSV")
    p.add_argument("--out", dest="out", help="Output CSV (optional). If omitted but --out_dir is given, will derive name.")
    p.add_argument("--out_dir", dest="out_dir", help="Output directory (optional).")
    args = p.parse_args()
    df = resample_csv(args.inp, out_dir=args.out_dir, out_csv=args.out)
    if args.out or args.out_dir:
        print("✅ Saved:", args.out or (Path(args.out_dir) / (Path(args.inp).stem + "_Cleaned.csv")))
    print(df.head(8))

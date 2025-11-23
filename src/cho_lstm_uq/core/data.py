from typing import List, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# canonical feature groups
OBS_POOLS = ["GlcC","LacC","DIC_mmolC_L","BioC","ProdC"]
DRIVING   = ["Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h"]
AUX_FEATS = ["pCO2","V_L","vvd_per_day"]

# -------- helpers (light placeholders) --------
def guess_units_is_gL(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return (not s.empty) and (float(s.mean()) > 5.0)

def carbonize_df(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Convert raw columns to carbon pools and drivers.
    NOTE: this is a minimal stub—port your working logic here.
    """
    df = df.copy()
    # TODO: port full glucose/lactate/biomass/product logic
    # create safe placeholders so downstream code can import
    for c in OBS_POOLS:
        if c not in df: df[c] = np.nan
    for c in DRIVING:
        if c not in df: df[c] = np.nan
    if "time_h" not in df and "time" in df: df = df.rename(columns={"time":"time_h"})
    if "run_id" not in df and "batch_name" in df: df["run_id"] = df["batch_name"].astype(str)
    df = df.sort_values(["run_id","time_h"])
    df["dt"] = df.groupby("run_id")["time_h"].diff().fillna(0.0).clip(lower=1.0)
    return df

def get_aux_cols_present(df: pd.DataFrame) -> List[str]:
    return [c for c in AUX_FEATS if c in df.columns]

def build_scalers(train_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    mu = train_df[cols].mean()
    sd = train_df[cols].std().replace(0, 1.0)
    return mu, sd

def apply_scaler(df: pd.DataFrame, cols: List[str], mu, sd):
    out = df.copy(); out[cols] = (out[cols] - mu) / sd; return out

def train_val_split(df: pd.DataFrame, val_frac: float = 0.2):
    runs = sorted(df["run_id"].unique())
    n_val = max(1, int(len(runs) * val_frac))
    val_runs = set(runs[-n_val:])
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, list(val_runs)

# --------------- dataset ---------------
class Seq2SeqDataset(Dataset):
    """
    Yields (enc_scaled, dec_scaled, y_next_sc, y_prev_raw, flows_raw, xab_next, run_id)
    This is a thin shell; port your sampling logic later.
    """
    def __init__(self, df_scaled: pd.DataFrame, df_raw: pd.DataFrame,
                 obs_pools: List[str], driving: List[str], aux_cols: List[str],
                 t_in: int, t_out: int, stride: int = 1):
        self.obs = obs_pools; self.drv = driving; self.aux = aux_cols
        self.t_in = t_in; self.t_out = t_out
        self.samples = []
        # TODO: port your sliding-window sample builder here

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

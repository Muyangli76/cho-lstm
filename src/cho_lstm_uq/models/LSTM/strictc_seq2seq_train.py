# strictc_seq2seq_train.py
# Single-file Strict-C Seq2Seq (train + UQ + learning curve)

from __future__ import annotations
import os, json, argparse, random
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================= Stage B: Training =======================
OBS_POOLS = ["GlcC","LacC","DIC_mmolC_L","BioC","ProdC"]   # mmol-C/L
DRIVING   = ["Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h"]
AUX_FEATS = ["pCO2","V_L","vvd_per_day"]

# ---------- reasonable defaults (edit if you have project values) ----------
DEFAULTS = dict(
    SEED=1337, EPOCHS=40, BATCH_TR=64, BATCH_EVAL=128, T_IN=24, T_OUT=24,
    TF_START=1.0, TF_END=0.5, CLIP=1.0, PATIENCE=6, VAL_FRAC=0.2,
    LAMBDA_MB=1.0, GAMMA_RES=0.1, LAMBDA_XAB=0.5, LAMBDA_CONS=0.2,
    HIDDEN=128, LAYERS=2, DROPOUT=0.1, LC_EPOCHS=6,
    HET_XAB=True, LOGV_MIN=-10.0, LOGV_MAX=3.0, SOFTPLUS_VAR=True,
    NONNEG_W=1e-3, ENSEMBLE_K=5, ENSEMBLE_BOOTSTRAP_FRAC=0.85,

    # chemistry/units (adjust to your project constants)
    MW_GLC=180.156, MW_LAC=90.078,
    CARBON_PER_GLC=6.0, CARBON_PER_LAC=3.0,        # mmol-C / mmol
    DEFAULT_gDCW_PER_CELL=3.0e-9,                  # g/cell (example)
    MMOLC_PER_G_BIOMASS=41.7,                      # ≈0.5 gC/g * 1000/12
    MMOLC_PER_G_MAB=39.0,                          # rough; OK as soft prior
)

# --------- helpers ----------
def guess_units_is_gL(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return False
    return float(s.mean()) > 5.0

def carbonize_df(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()

    # Glucose -> mmol C/L
    if guess_units_is_gL(df["C_glc"]):
        df["GlcC"] = (df["C_glc"].astype(float) / cfg.MW_GLC) * 1000.0 * cfg.CARBON_PER_GLC
    else:
        df["GlcC"] = df["C_glc"].astype(float) * cfg.CARBON_PER_GLC

    # Lactate -> mmol C/L
    if guess_units_is_gL(df["C_lac"]):
        df["LacC"] = (df["C_lac"].astype(float) / cfg.MW_LAC) * 1000.0 * cfg.CARBON_PER_LAC
    else:
        df["LacC"] = df["C_lac"].astype(float) * cfg.CARBON_PER_LAC

    # Biomass → mmol C/L; keep X_gL
    if "C_X_raw_cells_per_ml" in df.columns and df["C_X_raw_cells_per_ml"].notna().any():
        cells_per_L = df["C_X_raw_cells_per_ml"].astype(float) * 1e3
        X_gL = cells_per_L * cfg.DEFAULT_gDCW_PER_CELL
    else:
        X_gL = df["C_X"].astype(float) if df["C_X"].astype(float).mean() < 5.0 \
               else df["C_X"].astype(float) * 1e6 * cfg.DEFAULT_gDCW_PER_CELL
    df["X_gL"] = X_gL
    df["BioC"] = X_gL * cfg.MMOLC_PER_G_BIOMASS

    # Product → mmol C/L; keep Ab_gL (assume g/L)
    df["Ab_gL"] = df["C_Ab"].astype(float)
    df["ProdC"] = df["Ab_gL"] * cfg.MMOLC_PER_G_MAB

    # --- DRIVING aliases (unified CTR_mmolC_L_h) ---
    if "CTR_mmolC_L_h" not in df.columns:
        for cand in ["CTR_meas_mmolC_L_h", "CTR_calc_mmolC_L_h", "ProdCO2_mmolC_L_h"]:
            if cand in df.columns:
                df["CTR_mmolC_L_h"] = pd.to_numeric(df[cand], errors="coerce")
                break

    for c in ["Fin_over_V_1ph", "CinC_mmolC_L"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["run_id","time_h"])
    df["dt"] = df.groupby("run_id")["time_h"].diff().fillna(0.0)
    df.loc[df["dt"] <= 0, "dt"] = 1.0
    df = df.replace([np.inf, -np.inf], np.nan)

    need = OBS_POOLS + DRIVING
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after aliasing: {missing}. Present: {list(df.columns)}")

    df = df.dropna(subset=need)
    return df

def get_aux_cols_present(df: pd.DataFrame):
    return [c for c in AUX_FEATS if c in df.columns]

def build_scalers(train_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    mu = train_df[cols].mean()
    sd = train_df[cols].std().replace(0, 1.0)
    return mu, sd

def apply_scaler(df: pd.DataFrame, cols: List[str], mu, sd):
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

# -------------------- Dataset (drivers in decoder) --------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, df_scaled: pd.DataFrame, df_raw: pd.DataFrame,
                 obs_pools: List[str], driving: List[str], aux_cols: List[str],
                 t_in: int, t_out: int, stride: int = 1):
        self.obs = obs_pools; self.drv = driving; self.aux = aux_cols
        self.t_in = t_in; self.t_out = t_out; self.samples = []
        use_cols = self.obs + self.drv + self.aux

        for rid, gs in df_scaled.groupby("run_id"):
            gs = gs.sort_values("time_h").reset_index(drop=True)
            gr = df_raw[df_raw["run_id"] == rid].sort_values("time_h").reset_index(drop=True)
            if len(gs) < (t_in + t_out):
                continue

            Xs = gs[use_cols].values.astype(np.float32)
            Ys_sc = gs[self.obs].values.astype(np.float32)
            Yr_raw = gr[self.obs].values.astype(np.float32)

            dt   = gr["dt"].values.astype(np.float32)
            FinV = gr["Fin_over_V_1ph"].values.astype(np.float32)
            CinC = gr["CinC_mmolC_L"].values.astype(np.float32)
            CTR  = gr["CTR_mmolC_L_h"].values.astype(np.float32)
            X_gL = gr["X_gL"].values.astype(np.float32) if "X_gL" in gr else np.zeros(len(gr), np.float32)
            Ab_gL= gr["Ab_gL"].values.astype(np.float32) if "Ab_gL" in gr else np.zeros(len(gr), np.float32)

            T = len(gs)
            for t0 in range(0, T - t_in - t_out + 1, stride):
                t1, t2 = t0 + t_in, t0 + t_in + t_out
                enc_scaled = Xs[t0:t1]
                dec_scaled = Xs[t1:t2]
                y_next_sc  = Ys_sc[t1:t2]
                y_prev_raw = Yr_raw[t1-1:t2-1]
                flows_raw  = np.column_stack([dt[t1:t2], FinV[t1:t2], CinC[t1:t2], CTR[t1:t2]]).astype(np.float32)
                xab_next   = np.column_stack([X_gL[t1:t2], Ab_gL[t1:t2]]).astype(np.float32)
                self.samples.append((enc_scaled, dec_scaled, y_next_sc, y_prev_raw, flows_raw, xab_next, str(rid)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        es, ds, y, yprev, flows, xab, rid = self.samples[idx]
        return (torch.from_numpy(es), torch.from_numpy(ds), torch.from_numpy(y),
                torch.from_numpy(yprev), torch.from_numpy(flows),
                torch.from_numpy(xab), rid)

# --------------------- Model ---------------------
class StrictCSeq2Seq(nn.Module):
    def __init__(self, in_dim: int, out_pools: int, n_drv_aux: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 het_xab: bool = True, logv_min: float = -10.0, logv_max: float = 3.0,
                 softplus_var: bool = False):
        super().__init__()
        self.out_pools = out_pools
        self.n_drv_aux = n_drv_aux
        self.het_xab = het_xab
          # bounds for log-variance if used
        self.logv_min = logv_min
        self.logv_max = logv_max
        self.softplus_var = softplus_var

        self.enc = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec = nn.LSTM(out_pools + n_drv_aux, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head_pools = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_pools))
        self.head_cres  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        if het_xab:
            self.head_xab_mu = nn.Linear(hidden, 2)
            self.head_xab_lv = nn.Linear(hidden, 2)
        else:
            self.head_xab = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def _stabilize_logv(self, lv: torch.Tensor) -> torch.Tensor:
        if self.softplus_var:
            v = F.softplus(lv) + 1e-6
            return torch.log(v)
        return torch.clamp(lv, self.logv_min, self.logv_max)

    def forward(self, enc_all_sc: torch.Tensor, dec_all_sc: torch.Tensor,
                y_tf_sc: torch.Tensor, tf_ratio: float = 1.0):
        B, T_out = dec_all_sc.size(0), dec_all_sc.size(1)
        _, (h, c) = self.enc(enc_all_sc)

        y_prev_sc = y_tf_sc[:, 0, :].clone()
        pools_out, cres_out, xab_mu_out, xab_lv_out, xab_det_out = [], [], [], [], []

        for t in range(T_out):
            drv_aux_sc = dec_all_sc[:, t, -(self.n_drv_aux):]
            dec_in = torch.cat([y_prev_sc, drv_aux_sc], dim=-1).unsqueeze(1)
            z, (h, c) = self.dec(dec_in, (h, c))
            z = z.squeeze(1)
            y_t_sc = self.head_pools(z)
            c_t    = self.head_cres(z).squeeze(-1)
            pools_out.append(y_t_sc)
            cres_out.append(c_t)

            if self.het_xab:
                mu_t = self.head_xab_mu(z)
                lv_t = self._stabilize_logv(self.head_xab_lv(z))
                xab_mu_out.append(mu_t)
                xab_lv_out.append(lv_t)
            else:
                xab_det_out.append(self.head_xab(z))

            use_tf = (torch.rand(B, device=z.device) < tf_ratio).float().unsqueeze(-1)
            y_prev_sc = use_tf * y_tf_sc[:, t, :] + (1.0 - use_tf) * y_t_sc

        pools_sc = torch.stack(pools_out, dim=1)
        cres     = torch.stack(cres_out,  dim=1)
        if self.het_xab:
            mu = torch.stack(xab_mu_out, dim=1)
            lv = torch.stack(xab_lv_out, dim=1)
            return pools_sc, cres, (mu, lv)
        else:
            xab = torch.stack(xab_det_out, dim=1)
            return pools_sc, cres, xab

# --------------------- Physics loss helper ----------------------
def carbon_closure_eps_seq(p_next_raw: torch.Tensor, p_prev_raw: torch.Tensor,
                           flows_raw: torch.Tensor, cres: torch.Tensor) -> torch.Tensor:
    """
    ΔC_total (liquid pools) = dt * [ (Fin/V)*(CinC - DIC_mid) - CTR ]
    Residual ε = (ΔC_total) - RHS  + c_res   → should be ~0
    flows_raw[..., :] = [dt, Fin_over_V_1ph, CinC_mmolC_L, CTR_mmolC_L_h]
    OBS_POOLS order: ["GlcC","LacC","DIC_mmolC_L","BioC","ProdC"]
    """
    dt, FinV, CinC, CTR = flows_raw[..., 0], flows_raw[..., 1], flows_raw[..., 2], flows_raw[..., 3]
    d_acc = (p_next_raw - p_prev_raw).sum(dim=-1)
    DIC_mid = 0.5 * (p_next_raw[..., 2] + p_prev_raw[..., 2])
    eps = d_acc - dt * FinV * (CinC - DIC_mid) + dt * CTR + cres
    return eps

# ------------------ Evaluation (TF=1.0 OR open-loop=0.0) -----------------
@torch.no_grad()
def eval_seq_metrics(loader, model, inv_scale_pools, obs_names, xab_names,
                     device, het_xab, mu_xab=None, sd_xab=None, tf_ratio: float = 1.0):
    model.eval()
    Ys_t, Ys_p, Xa_t, Xa_p = [], [], [], []
    for batch in loader:
        enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _rid = batch
        enc_sc = enc_sc.to(device); dec_sc = dec_sc.to(device); y_sc = y_sc.to(device)

        out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=tf_ratio)

        if het_xab:
            pools_sc, _, (mu_sc, lv_sc) = out
            pools_raw_pred = inv_scale_pools(pools_sc)
            pools_raw_true = inv_scale_pools(y_sc)
            mu_raw = mu_sc * sd_xab + mu_xab  # back to raw units
            Ys_t.append(pools_raw_true.cpu().numpy()); Ys_p.append(pools_raw_pred.cpu().numpy())
            Xa_t.append(xab_next.numpy()); Xa_p.append(mu_raw.cpu().numpy())
        else:
            pools_sc, _, xab_hat = out
            pools_raw_pred = inv_scale_pools(pools_sc)
            pools_raw_true = inv_scale_pools(y_sc)
            Ys_t.append(pools_raw_true.cpu().numpy()); Ys_p.append(pools_raw_pred.cpu().numpy())
            Xa_t.append(xab_next.numpy()); Xa_p.append(xab_hat.cpu().numpy())

    Yp_true = np.concatenate(Ys_t, axis=0)
    Yp_pred = np.concatenate(Ys_p, axis=0)
    Xa_true = np.concatenate(Xa_t, axis=0)
    Xa_pred = np.concatenate(Xa_p, axis=0)

    def flat_metrics(y_true, y_pred, names):
        ys = y_true.reshape(-1, y_true.shape[-1]); yp = y_pred.reshape(-1, y_true.shape[-1])
        out = []
        for i, name in enumerate(names):
            out.append({"name": name,
                        "RMSE": float(mean_squared_error(ys[:, i], yp[:, i], squared=False)),
                        "MAE":  float(mean_absolute_error(ys[:, i], yp[:, i])),
                        "R2":   float(r2_score(ys[:, i], yp[:, i]))})
        return out

    return flat_metrics(Yp_true, Yp_pred, obs_names), flat_metrics(Xa_true, Xa_pred, xab_names)

# --------------------------- Train One --------------------------
def train_one(df_train, df_val, feat_cols, obs_mu_t, obs_sd_t, cfg):
    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING
    ds_tr = Seq2SeqDataset(apply_scaler(df_train, feat_cols, cfg._mu_all, cfg._sd_all),
                           df_train[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                           cfg.T_IN, cfg.T_OUT, stride=1)
    ds_va = Seq2SeqDataset(apply_scaler(df_val, feat_cols, cfg._mu_all, cfg._sd_all),
                           df_val[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                           cfg.T_IN, cfg.T_OUT, stride=1)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_TR, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.BATCH_EVAL, shuffle=False)

    in_dim = len(feat_cols); n_drv_aux = len(DRIVING) + len(cfg._aux_cols)
    model = StrictCSeq2Seq(in_dim, out_pools=len(OBS_POOLS), n_drv_aux=n_drv_aux,
                           hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
                           het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
                           logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)
    mse = nn.MSELoss()

    mu_xab = torch.tensor(df_train[["X_gL","Ab_gL"]].mean().values, device=DEVICE, dtype=torch.float32)
    sd_xab = torch.tensor(df_train[["X_gL","Ab_gL"]].std().replace(0,1.0).values, device=DEVICE, dtype=torch.float32)
    def inv_scale_pools(y_sc): return obs_mu_t + obs_sd_t * y_sc

    best_val = float('inf'); patience = 0
    mb_warm_ep = max(1, int(0.4 * cfg.EPOCHS))
    nonneg_w = float(getattr(cfg, "NONNEG_W", 1e-3))

    for ep in range(1, cfg.EPOCHS+1):
        tf_ratio = cfg.TF_END + (cfg.TF_START - cfg.TF_END) * max(0.0, 1.0 - (ep-1)/(cfg.EPOCHS*0.4))
        lam_mb = cfg.LAMBDA_MB * min(1.0, ep / mb_warm_ep)

        model.train()
        for batch in dl_tr:
            enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _rid = batch
            enc_sc = enc_sc.to(DEVICE); dec_sc = dec_sc.to(DEVICE); y_sc = y_sc.to(DEVICE)
            yprev_raw = yprev_raw.to(DEVICE); flows_raw = flows_raw.to(DEVICE); xab_next = xab_next.to(DEVICE)

            out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=tf_ratio)
            if cfg.HET_XAB:
                pools_sc, cres_t, (mu_sc, lv_sc) = out
                lv_sc = (torch.log(torch.nn.functional.softplus(lv_sc) + 1e-6) if cfg.SOFTPLUS_VAR
                         else torch.clamp(lv_sc, cfg.LOGV_MIN, cfg.LOGV_MAX))

                var_sc = torch.exp(lv_sc).clamp(min=1e-8, max=1e8)

                p_next_raw = inv_scale_pools(pools_sc)
                eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                loss_state = mse(pools_sc, y_sc)

                xab_sc = (xab_next - mu_xab) / sd_xab
                nll = 0.5 * (lv_sc + (xab_sc - mu_sc)**2 / var_sc)
                loss_xab = nll.mean()
                mu_raw = mu_sc * sd_xab + mu_xab
                X_pred_gL  = mu_raw[..., 0]
                Ab_pred_gL = mu_raw[..., 1]
            else:
                pools_sc, cres_t, xab_hat = out
                p_next_raw = inv_scale_pools(pools_sc)
                eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                loss_state = mse(pools_sc, y_sc)
                loss_xab = mse(xab_hat, xab_next)
                X_pred_gL, Ab_pred_gL = xab_hat[..., 0], xab_hat[..., 1]

            BioC_pred  = p_next_raw[..., 3]
            ProdC_pred = p_next_raw[..., 4]
            cons_bio = mse(BioC_pred,  X_pred_gL * cfg.MMOLC_PER_G_BIOMASS)
            cons_ab  = mse(ProdC_pred, Ab_pred_gL * cfg.MMOLC_PER_G_MAB)
            loss_cons = cons_bio + cons_ab

            loss_mb  = eps.abs().mean()
            loss_res = (cres_t**2).mean()
            loss_nonneg = (p_next_raw.clamp(max=0.0)**2).mean() * nonneg_w

            loss = (loss_state
                    + lam_mb * loss_mb
                    + cfg.GAMMA_RES * loss_res
                    + cfg.LAMBDA_XAB * loss_xab
                    + cfg.LAMBDA_CONS * loss_cons
                    + loss_nonneg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
            opt.step()

        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                enc_sc, dec_sc, y_sc, *_ = batch
                enc_sc = enc_sc.to(DEVICE); dec_sc = dec_sc.to(DEVICE); y_sc = y_sc.to(DEVICE)
                pools_sc, *_ = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                tot += mse(pools_sc, y_sc).item(); n += 1
        v = tot / max(1, n)
        if v < best_val - 1e-6:
            best_val = v; patience = 0
            best_state = {k: v.cpu().clone() if torch.is_tensor(v) else v
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                break

    if 'best_state' in locals():
        model.load_state_dict(best_state)

    if cfg.HET_XAB:
        scal = {"mu_xab": mu_xab.detach().cpu().numpy().tolist(),
                "sd_xab": sd_xab.detach().cpu().numpy().tolist()}
        with open(os.path.join(cfg.TRAIN_OUTDIR, "scaler_xab.json"), "w") as f:
            json.dump(scal, f)

    dl_va_eval = DataLoader(ds_va, batch_size=cfg.BATCH_EVAL, shuffle=False)
    pools_tf, xab_tf = eval_seq_metrics(
        dl_va_eval, model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"],
        DEVICE, cfg.HET_XAB, mu_xab=mu_xab, sd_xab=sd_xab, tf_ratio=1.0
    )
    return model, pools_tf, xab_tf, ds_tr, ds_va, mu_xab, sd_xab, (obs_mu_t, obs_sd_t)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_val_split(df, val_frac=0.2):
    runs = sorted(df["run_id"].unique())
    n_val = max(1, int(len(runs) * val_frac))
    val_runs = set(runs[-n_val:])
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, list(val_runs)

# --------------------------- run_train --------------------------
def run_train(cfg):
    Path(cfg.TRAIN_OUTDIR).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.SEED)

    print(">>> Strict-C Seq2Seq — single-file trainer")
    df = pd.read_csv(cfg.TRAIN_INPUT_CSV)
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"] = df["batch_name"].astype(str)
    df = carbonize_df(df, cfg)

    sizes = df.groupby("run_id").size()
    need_len = cfg.T_IN + cfg.T_OUT
    keep = sizes.index[sizes >= need_len]
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)

    aux_cols = get_aux_cols_present(df)
    feat_cols = OBS_POOLS + DRIVING + aux_cols
    print(f"Aux present: {aux_cols}")

    df_tr, df_va, val_runs = train_val_split(df, val_frac=cfg.VAL_FRAC)
    mu_all, sd_all = build_scalers(df_tr, feat_cols)
    mu_obs, sd_obs = mu_all[OBS_POOLS], sd_all[OBS_POOLS]
    cfg._mu_all, cfg._sd_all, cfg._aux_cols = mu_all, sd_all, aux_cols

    cols_to_cast = list({*feat_cols, "X_gL", "Ab_gL", *OBS_POOLS, *DRIVING} & set(df.columns))
    df[cols_to_cast] = df[cols_to_cast].astype(np.float32)

    obs_mu_t = torch.tensor(mu_obs.values, dtype=torch.float32, device=DEVICE)
    obs_sd_t = torch.tensor(sd_obs.values, dtype=torch.float32, device=DEVICE)

    model, pools_tf, xab_tf, ds_tr, ds_va, mu_xab, sd_xab, (obs_mu_t, obs_sd_t) = \
        train_one(df_tr, df_va, feat_cols, obs_mu_t, obs_sd_t, cfg)

    pd.DataFrame(pools_tf).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_pools_TF.csv"), index=False)
    pd.DataFrame(xab_tf).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_xab_TF.csv"), index=False)

    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING
    ds_va_open = Seq2SeqDataset(apply_scaler(df_va, feat_cols, mu_all, sd_all),
                                df_va[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                                cfg.T_IN, cfg.T_OUT, stride=1)
    dl_va_open = DataLoader(ds_va_open, batch_size=cfg.BATCH_EVAL, shuffle=False)
    def inv_scale(y_sc): return obs_mu_t + obs_sd_t * y_sc
    pools_ol, xab_ol = eval_seq_metrics(
        dl_va_open, model, inv_scale, OBS_POOLS, ["X_gL","Ab_gL"],
        DEVICE, cfg.HET_XAB, mu_xab=mu_xab, sd_xab=sd_xab, tf_ratio=0.0
    )
    pd.DataFrame(pools_ol).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_pools_OL.csv"), index=False)
    pd.DataFrame(xab_ol).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_xab_OL.csv"), index=False)

    torch.save(model.state_dict(), os.path.join(cfg.TRAIN_OUTDIR, "strictC_seq2seq_50L_XAb.pt"))
    with open(os.path.join(cfg.TRAIN_OUTDIR, "scaler_features.json"), "w") as f:
        json.dump({"mu_all": mu_all.to_dict(), "sd_all": sd_all.to_dict(), "feat_cols": feat_cols}, f)

    print("Saved to:", cfg.TRAIN_OUTDIR)

    # ----------------- Deep Ensemble UQ (X/Ab) -----------------
    uq_dir = Path(cfg.TRAIN_OUTDIR) / "UQ"
    uq_dir.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_K = int(getattr(cfg, "ENSEMBLE_K", 5))
    BOOT_FRAC  = float(getattr(cfg, "ENSEMBLE_BOOTSTRAP_FRAC", 0.85))

    ds_va2 = Seq2SeqDataset(apply_scaler(df_va, feat_cols, mu_all, sd_all),
                            df_va[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                            cfg.T_IN, cfg.T_OUT, stride=1)
    dl_va_eval = DataLoader(ds_va2, batch_size=cfg.BATCH_EVAL, shuffle=False)

    member_paths = []

    def train_member_from_scratch(seed: int, df_tr_member: pd.DataFrame):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        obs_mu_t_local = torch.tensor(mu_all[OBS_POOLS].values, dtype=torch.float32, device=DEVICE)
        obs_sd_t_local = torch.tensor(sd_all[OBS_POOLS].values, dtype=torch.float32, device=DEVICE)
        cfg._mu_all, cfg._sd_all = mu_all, sd_all
        model_m, *_ = train_one(df_tr_member, df_va, feat_cols, obs_mu_t_local, obs_sd_t_local, cfg)
        return model_m

    all_runs = df_tr["run_id"].unique()
    n_boot   = max(1, int(len(all_runs) * BOOT_FRAC))

    for k in range(ENSEMBLE_K):
        print(f"[ensemble] training member {k+1}/{ENSEMBLE_K} …")
        boot_runs = np.random.choice(all_runs, size=n_boot, replace=True)
        df_tr_boot = df_tr[df_tr["run_id"].isin(boot_runs)].reset_index(drop=True)
        model_k = train_member_from_scratch(seed=cfg.SEED + 1000 + k, df_tr_member=df_tr_boot)
        pth = uq_dir / f"member_{k:02d}.pt"
        torch.save(model_k.state_dict(), pth)
        member_paths.append(pth)

    @torch.no_grad()
    def ensemble_predict_to_csv(member_paths, dl, out_csv):
        members = []
        for pth in member_paths:
            m = StrictCSeq2Seq(
                in_dim=len(feat_cols), out_pools=len(OBS_POOLS),
                n_drv_aux=len(DRIVING)+len(cfg._aux_cols),
                hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
                het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
                logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR
            ).to(DEVICE)
            sd = torch.load(pth, map_location=DEVICE)
            m.load_state_dict(sd, strict=True)
            m.eval()
            members.append(m)

        rows = []
        for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, rid in dl:
            enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)
            MU_list, VAR_list = [], []
            for m in members:
                _, _, (mu_sc, lv_sc) = m(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                MU_list.append(mu_sc.cpu().numpy())
                VAR_list.append(torch.exp(lv_sc).cpu().numpy())

            MU  = np.stack(MU_list)       # (S,B,T,2)
            VAR = np.stack(VAR_list)      # (S,B,T,2)
            mu_ens   = MU.mean(0) * sd_xab.cpu().numpy() + mu_xab.cpu().numpy()
            var_epi  = MU.var(0, ddof=1) * (sd_xab.cpu().numpy()**2)
            var_alea = VAR.mean(0)        * (sd_xab.cpu().numpy()**2)
            var_tot  = var_epi + var_alea
            Y        = xab_next.numpy()

            B, T_out = mu_ens.shape[0], mu_ens.shape[1]
            rids = [rid] if isinstance(rid, str) else list(rid)
            for b in range(B):
                rid_b = rids[b] if isinstance(rids, list) else rid
                for t in range(T_out):
                    rows.append({
                        "run_id": str(rid_b), "t_idx": int(t),
                        "true_X": float(Y[b,t,0]), "true_Ab": float(Y[b,t,1]),
                        "mu_X": float(mu_ens[b,t,0]), "mu_Ab": float(mu_ens[b,t,1]),
                        "var_ale_X": float(var_alea[b,t,0]), "var_ale_Ab": float(var_alea[b,t,1]),
                        "var_epi_X": float(var_epi[b,t,0]),   "var_epi_Ab": float(var_epi[b,t,1]),
                        "var_tot_X": float(var_tot[b,t,0]),   "var_tot_Ab": float(var_tot[b,t,1]),
                    })

        dfE = pd.DataFrame(rows)
        dfE.to_csv(out_csv, index=False)
        print(f"[ensemble] saved → {out_csv}")

        for tag, tcol, mu_col in [("X_gL","true_X","mu_X"), ("Ab_gL","true_Ab","mu_Ab")]:
            mae  = mean_absolute_error(dfE[tcol], dfE[mu_col])
            rmse = mean_squared_error(dfE[tcol], dfE[mu_col], squared=False)
            r2   = r2_score(dfE[tcol], dfE[mu_col])
            print(f"{tag}: MAE {mae:.3f} | RMSE {rmse:.3f} | R² {r2:.3f}")

    out_ens = (Path(cfg.TRAIN_OUTDIR) / "UQ" / "VAL_UQ_ensemble.csv").as_posix()
    ensemble_predict_to_csv(member_paths, dl_va_eval, out_ens)

    # ----------------- Learning Curve -----------------
    print("\n>>> Learning curve by added runs (validation TF & OL metrics)")
    ds_va2 = Seq2SeqDataset(apply_scaler(df_va, feat_cols, mu_all, sd_all),
                            df_va[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                            cfg.T_IN, cfg.T_OUT, stride=1)
    dl_va_lc = DataLoader(ds_va2, batch_size=cfg.BATCH_EVAL, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

    def inv_scale_lc(y_sc: torch.Tensor) -> torch.Tensor:
        return obs_mu_t + obs_sd_t * y_sc

    def tiny_train(model, dl, mu_xab=None, sd_xab=None, epochs=cfg.LC_EPOCHS, lr=3e-3, tf_start=1.0, tf_end=0.6):
        mse = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        model.train()
        for ep in range(1, epochs+1):
            tf_ratio = tf_end + (tf_start - tf_end) * max(0.0, 1.0 - (ep-1)/max(1, epochs*0.5))
            for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl:
                enc_sc = enc_sc.to(DEVICE); dec_sc = dec_sc.to(DEVICE); y_sc = y_sc.to(DEVICE)
                yprev_raw = yprev_raw.to(DEVICE); flows_raw = flows_raw.to(DEVICE); xab_next = xab_next.to(DEVICE)

                out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=tf_ratio)
                if cfg.HET_XAB:
                    pools_sc, cres_t, (mu_sc, lv_sc) = out
                    lv_sc = torch.clamp(lv_sc, cfg.LOGV_MIN, cfg.LOGV_MAX)
                    p_next_raw = inv_scale_lc(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                    loss_state = mse(pools_sc, y_sc)
                    xab_sc = (xab_next - mu_xab) / sd_xab
                    nll = 0.5 * (lv_sc + (xab_sc - mu_sc)**2 / torch.exp(lv_sc))
                    loss_xab = nll.mean()
                    mu_raw = mu_sc * sd_xab + mu_xab
                    X_pred_gL, Ab_pred_gL = mu_raw[...,0], mu_raw[...,1]
                else:
                    pools_sc, cres_t, xab_hat = out
                    p_next_raw = inv_scale_lc(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                    loss_state = mse(pools_sc, y_sc)
                    loss_xab = mse(xab_hat, xab_next)
                    X_pred_gL, Ab_pred_gL = xab_hat[...,0], xab_hat[...,1]

                cons_bio = mse(p_next_raw[...,3], X_pred_gL * cfg.MMOLC_PER_G_BIOMASS)
                cons_ab  = mse(p_next_raw[...,4], Ab_pred_gL * cfg.MMOLC_PER_G_MAB)
                loss = (loss_state
                        + 0.5*cfg.LAMBDA_MB*eps.abs().mean()
                        + 0.5*cfg.GAMMA_RES*(cres_t**2).mean()
                        + 0.2*cfg.LAMBDA_XAB*loss_xab
                        + 0.2*cfg.LAMBDA_CONS*(cons_bio + cons_ab))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
                opt.step()

    all_train_runs = sorted(df_tr["run_id"].unique())
    rows_lc = []

    init_model = StrictCSeq2Seq(
        in_dim=len(feat_cols), out_pools=len(OBS_POOLS),
        n_drv_aux=len(DRIVING)+len(cfg._aux_cols),
        hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
        het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
        logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR
    ).to(DEVICE)
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)
    init_state = deepcopy(init_model.state_dict())

    for k in range(1, len(all_train_runs)+1):
        use_runs = set(all_train_runs[:k])
        sub_tr = df_tr[df_tr["run_id"].isin(use_runs)].reset_index(drop=True)

        sub_tr_s = apply_scaler(sub_tr, feat_cols, mu_all, sd_all)
        sub_tr_r = sub_tr[["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING].copy()

        ds_k = Seq2SeqDataset(sub_tr_s, sub_tr_r, OBS_POOLS, DRIVING, cfg._aux_cols,
                              cfg.T_IN, cfg.T_OUT, stride=1)
        dl_k = DataLoader(ds_k, batch_size=cfg.BATCH_TR, shuffle=True, drop_last=True,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))

        mu_xab_k = torch.tensor(sub_tr[["X_gL","Ab_gL"]].mean().values, dtype=torch.float32, device=DEVICE)
        sd_xab_k = torch.tensor(sub_tr[["X_gL","Ab_gL"]].std().replace(0,1.0).values, dtype=torch.float32, device=DEVICE)

        model_k = StrictCSeq2Seq(
            in_dim=len(feat_cols), out_pools=len(OBS_POOLS),
            n_drv_aux=len(DRIVING)+len(cfg._aux_cols),
            hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
            het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
            logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR
        ).to(DEVICE)
        model_k.load_state_dict(init_state)

        tiny_train(model_k, dl_k,
                   mu_xab=(mu_xab_k if cfg.HET_XAB else None),
                   sd_xab=(sd_xab_k if cfg.HET_XAB else None),
                   epochs=max(6, int(getattr(cfg, "LC_EPOCHS", 6))),
                   lr=3e-3, tf_start=1.0, tf_end=0.6)

        pools_tf_k, xab_tf_k = eval_seq_metrics(
            dl_va_lc, model_k, inv_scale_lc, OBS_POOLS, ["X_gL","Ab_gL"],
            DEVICE, cfg.HET_XAB, mu_xab=mu_xab_k, sd_xab=sd_xab_k, tf_ratio=1.0
        )
        pools_ol_k, xab_ol_k = eval_seq_metrics(
            dl_va_lc, model_k, inv_scale_lc, OBS_POOLS, ["X_gL","Ab_gL"],
            DEVICE, cfg.HET_XAB, mu_xab=mu_xab_k, sd_xab=sd_xab_k, tf_ratio=0.0
        )

        def grab(mets, name): return float(next(d["R2"] for d in mets if d["name"] == name))
        rows_lc.append({
            "k_runs": k,
            "runs_used_json": json.dumps(sorted(list(use_runs))),
            "TF_R2_avg_pools": float(np.mean([d["R2"] for d in pools_tf_k])),
            "TF_R2_X_gL": grab(xab_tf_k, "X_gL"),
            "TF_R2_Ab_gL": grab(xab_tf_k, "Ab_gL"),
            "OL_R2_avg_pools": float(np.mean([d["R2"] for d in pools_ol_k])),
            "OL_R2_X_gL": grab(xab_ol_k, "X_gL"),
            "OL_R2_Ab_gL": grab(xab_ol_k, "Ab_gL"),
        })

        print(f"  k={k:02d} | TF R²[X]={rows_lc[-1]['TF_R2_X_gL']:.3f}  TF R²[Ab]={rows_lc[-1]['TF_R2_Ab_gL']:.3f} "
              f"| OL R²[X]={rows_lc[-1]['OL_R2_X_gL']:.3f}  OL R²[Ab]={rows_lc[-1]['OL_R2_Ab_gL']:.3f}")

    lc_csv = os.path.join(cfg.TRAIN_OUTDIR, "LC_added_runs_metrics_TF_OL.csv")
    pd.DataFrame(rows_lc).to_csv(lc_csv, index=False)
    print("Saved learning-curve table →", lc_csv)

# ---------------------------- CLI ------------------------------
class C: pass
def make_cfg_from_args(args):
    c = C()
    for k, v in DEFAULTS.items(): setattr(c, k, v)
    for k, v in vars(args).items():
        if v is None: continue
        setattr(c, k, v)
    return c

def main():
    p = argparse.ArgumentParser(description="Strict-C Seq2Seq (single-file)")
    p.add_argument("--TRAIN_INPUT_CSV", type=str, required=True)
    p.add_argument("--TRAIN_OUTDIR", type=str, required=True)
    # hparams
    for k in ["SEED","EPOCHS","BATCH_TR","BATCH_EVAL","T_IN","T_OUT","TF_START","TF_END","CLIP","PATIENCE",
              "LAMBDA_MB","GAMMA_RES","LAMBDA_XAB","LAMBDA_CONS","HIDDEN","LAYERS","DROPOUT","LC_EPOCHS","VAL_FRAC"]:
        p.add_argument(f"--{k}", type=type(DEFAULTS[k]))
    # uncertainty flags
    p.add_argument("--HET_XAB", type=bool)
    p.add_argument("--LOGV_MIN", type=float)
    p.add_argument("--LOGV_MAX", type=float)
    p.add_argument("--SOFTPLUS_VAR", type=bool)
    # ensemble knobs
    p.add_argument("--ENSEMBLE_K", type=int)
    p.add_argument("--ENSEMBLE_BOOTSTRAP_FRAC", type=float)

    args = p.parse_args()
    cfg = make_cfg_from_args(args)
    run_train(cfg)

if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        sys.argv = [""]  # allow notebook run
    main()

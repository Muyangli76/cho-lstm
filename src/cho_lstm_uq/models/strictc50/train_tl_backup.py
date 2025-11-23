#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse, json, math, os, sys, random, copy
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    # Compatible with older sklearn (no squared= argument)
    return mean_squared_error(y_true, y_pred) ** 0.5

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# -------------------------
# Utilities
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

OBS_POOLS = ["GlcC","LacC","DIC_mmolC_L","BioC","ProdC"]
DRIVING   = ["Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h"]
AUX_FEATS = ["pCO2","V_L","vvd_per_day"]  # if present

def guess_units_is_gL(s: pd.Series)->bool:
    x = pd.to_numeric(s, errors='coerce').dropna()
    return (not x.empty) and (float(x.mean()) > 5.0)

def get_aux_cols_present(df): 
    return [c for c in AUX_FEATS if c in df.columns]

def build_scalers(train_df, cols):
    mu = train_df[cols].mean()
    sd = train_df[cols].std().replace(0,1.0)
    return mu, sd

def apply_scaler(df, cols, mu, sd):
    out=df.copy()
    out[cols] = (out[cols]-mu)/sd
    return out

# This is a problem, and this will not work for 50L code since not all inpyt data were used
# for the trianing
#def train_val_split(df, val_frac=0.2):
   # runs = sorted(df["run_id"].unique()); n_val=max(1,int(len(runs)*val_frac))
     #val_runs=set(runs[-n_val:])
    #tr=df[~df["run_id"].isin(val_runs)].copy()
    #va=df[df["run_id"].isin(val_runs)].copy()
    #return tr, va, list(val_runs)


# -------------------------
# Carbonization
# -------------------------

def carbonize_df(df: pd.DataFrame, args)->pd.DataFrame:
    df = df.copy()

    # Glucose -> mmol C/L
    if "C_glc" in df:
        if guess_units_is_gL(df["C_glc"]):
            df["GlcC"] = (df["C_glc"].astype(float)/args.MW_GLC)*1000.0*args.CARBON_PER_GLC
        else:
            df["GlcC"] = df["C_glc"].astype(float)*args.CARBON_PER_GLC

    # Lactate -> mmol C/L
    if "C_lac" in df:
        if guess_units_is_gL(df["C_lac"]):
            df["LacC"] = (df["C_lac"].astype(float)/args.MW_LAC)*1000.0*args.CARBON_PER_LAC
        else:
            df["LacC"] = df["C_lac"].astype(float)*args.CARBON_PER_LAC

    # Biomass → mmol C/L; keep X_gL
    if "C_X_raw_cells_per_ml" in df.columns and df["C_X_raw_cells_per_ml"].notna().any():
        cells_per_L = df["C_X_raw_cells_per_ml"].astype(float)*1e3
        X_gL = cells_per_L*args.DEFAULT_gDCW_PER_CELL
    elif "C_X" in df:
        X_gL = df["C_X"].astype(float) if df["C_X"].astype(float).mean()<5.0 else df["C_X"].astype(float)*1e6*args.DEFAULT_gDCW_PER_CELL
    else:
        X_gL = 0.0
    df["X_gL"] = X_gL
    df["BioC"] = df["X_gL"]*args.MMOLC_PER_G_BIOMASS

    # Product → mmol C/L; keep Ab_gL (assume g/L)
    if "C_Ab" in df:
        df["Ab_gL"] = df["C_Ab"].astype(float)
    elif "Ab_gL" not in df:
        df["Ab_gL"] = 0.0
    df["ProdC"] = df["Ab_gL"]*args.MMOLC_PER_G_MAB

    # Sort + dt
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"]=df["batch_name"].astype(str)
    if "time_h" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time":"time_h"})
    df = df.sort_values(["run_id","time_h"])
    df["dt"]=df.groupby("run_id")["time_h"].diff().fillna(0.0)
    df.loc[df["dt"]<=0,"dt"]=1.0

    # Basic engineering if missing
    if "V_L" not in df and "V" in df: df["V_L"]=df["V"]/1000.0
    if "vvd_per_day" not in df and "V_L" in df:
        V=df["V_L"].to_numpy(); Vp=np.roll(V,1); Vp[0]=V[0]
        vvd=24.0*(V - Vp)/np.maximum(Vp,1e-12); vvd[0]=0.0; df["vvd_per_day"]=vvd
    if "Fin_over_V_1ph" not in df and "vvd_per_day" in df: df["Fin_over_V_1ph"]=df["vvd_per_day"]/24.0
    if "CinC_mmolC_L" not in df and "feed_glucose" in df: df["CinC_mmolC_L"]=6.0*df["feed_glucose"]
    if "DIC_mmolC_L"  not in df and "pCO2" in df: df["DIC_mmolC_L"]=1.0*df["pCO2"]


    # --- in carbonize_df(...) add a CTR alias before "Require core cols" ---
    # Alias: CTR measured vs calculated
    if "CTR_mmolC_L_h" not in df.columns and "CTR_calc_mmolC_L_h" in df.columns:
        df["CTR_mmolC_L_h"] = df["CTR_calc_mmolC_L_h"].astype(float)

    # DIC from pCO2 if available (keep as-is, but guard)
    if "DIC_mmolC_L" not in df.columns and "pCO2" in df.columns:
        df["DIC_mmolC_L"] = 1.0 * pd.to_numeric(df["pCO2"], errors="coerce")

    # Core drivers best-effort
    if "Fin_over_V_1ph" not in df.columns and "vvd_per_day" in df.columns:
        df["Fin_over_V_1ph"] = pd.to_numeric(df["vvd_per_day"], errors="coerce") / 24.0
    if "CinC_mmolC_L" not in df.columns and "feed_glucose" in df.columns:
        df["CinC_mmolC_L"] = 6.0 * pd.to_numeric(df["feed_glucose"], errors="coerce")

    # Require core cols that truly must exist for the model
    need = ["GlcC","LacC","DIC_mmolC_L","BioC","ProdC",  # pools (derived above)
            "Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h"]  # drivers
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in need if c in df.columns])

    # Require core cols
    need = OBS_POOLS + DRIVING
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=[c for c in need if c in df.columns])
    return df


# -------------------------
# Dataset
# -------------------------

class Seq2SeqDataset(Dataset):
    def __init__(self, df_scaled, df_raw, obs_pools, driving, aux_cols, t_in, t_out, stride=1):
        self.obs=obs_pools; self.drv=driving; self.aux=aux_cols
        self.t_in=t_in; self.t_out=t_out
        self.samples=[]
        use_cols = self.obs + self.drv + self.aux

        for rid, gs in df_scaled.groupby("run_id"):
            gs = gs.sort_values("time_h").reset_index(drop=True)
            gr = df_raw[df_raw["run_id"]==rid].sort_values("time_h").reset_index(drop=True)
            if len(gs) < (t_in+t_out): continue

            Xs    = gs[use_cols].values.astype(np.float32)
            Ys_sc = gs[self.obs].values.astype(np.float32)
            Yr_raw= gr[self.obs].values.astype(np.float32)
            dt  = gr["dt"].values.astype(np.float32)
            FinV= gr["Fin_over_V_1ph"].values.astype(np.float32)
            CinC= gr["CinC_mmolC_L"].values.astype(np.float32)
            CTR = gr["CTR_mmolC_L_h"].values.astype(np.float32)
            X_gL= gr["X_gL"].values.astype(np.float32)
            Ab_gL=gr["Ab_gL"].values.astype(np.float32)

            T=len(gs)
            for t0 in range(0, T - t_in - t_out + 1, stride):
                t1=t0+t_in; t2=t1+t_out
                enc_scaled = Xs[t0:t1]
                dec_scaled = Xs[t1:t2]   # carries drivers+aux to decoder
                y_next_sc  = Ys_sc[t1:t2]
                y_prev_raw = Yr_raw[t1-1:t2-1]
                flows_raw  = np.column_stack([dt[t1:t2], FinV[t1:t2], CinC[t1:t2], CTR[t1:t2]]).astype(np.float32)
                xab_next   = np.column_stack([X_gL[t1:t2], Ab_gL[t1:t2]]).astype(np.float32)
                self.samples.append((enc_scaled, dec_scaled, y_next_sc, y_prev_raw, flows_raw, xab_next, rid))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        es, ds, y, yprev, flows, xab, rid = self.samples[idx]
        return (torch.from_numpy(es), torch.from_numpy(ds),
                torch.from_numpy(y), torch.from_numpy(yprev),
                torch.from_numpy(flows), torch.from_numpy(xab), rid)


# -------------------------
# Model
# -------------------------

class StrictCSeq2Seq(nn.Module):
    def __init__(self, in_dim, out_pools, n_drv_aux, hidden=128, layers=2, dropout=0.1,
                 het_xab=True, softplus_var=True, logv_min=-10.0, logv_max=3.0):
        super().__init__()
        self.out_pools = out_pools
        self.n_drv_aux = n_drv_aux
        self.het_xab = het_xab
        self.softplus_var = softplus_var
        self.logv_min = logv_min
        self.logv_max = logv_max

        self.enc = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers>1 else 0.0)
        self.dec = nn.LSTM(out_pools + n_drv_aux, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers>1 else 0.0)
        # head_pools translates the decoder to the biochemical pools
        # head_cres translates the decoder to the mass balance conserved
        self.head_pools = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, out_pools))
        self.head_cres  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(),
                                        nn.Linear(hidden//2, 1))
        if het_xab:
            self.head_xab_mu = nn.Linear(hidden, 2)
            self.head_xab_lv = nn.Linear(hidden, 2)
        else:
            self.head_xab    = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                             nn.Linear(hidden, 2))

    def _stabilize_logv(self, raw_lv):
        if self.softplus_var:
            v = F.softplus(raw_lv) + 1e-6
            return torch.log(v)
        return torch.clamp(raw_lv, self.logv_min, self.logv_max)

    def forward(self, enc_all_sc, dec_all_sc, y_tf_sc=None, tf_ratio=1.0):
        B, T_out = dec_all_sc.size(0), dec_all_sc.size(1)
        _, (h, c) = self.enc(enc_all_sc)
        if y_tf_sc is None:
            raise ValueError("Provide y_tf_sc for teacher-forcing schedule.")
        y_prev_sc = y_tf_sc[:,0,:].clone()
        pools_out, cres_out, xab_mu_out, xab_lv_out, xab_det_out = [], [], [], [], []
        for t in range(T_out):
            drv_aux_sc = dec_all_sc[:, t, -(self.n_drv_aux):]
            dec_in = torch.cat([y_prev_sc, drv_aux_sc], dim=-1).unsqueeze(1)
            z, (h, c) = self.dec(dec_in, (h, c)); z=z.squeeze(1)
            y_t = self.head_pools(z)
            c_t = self.head_cres(z).squeeze(-1)
            pools_out.append(y_t); cres_out.append(c_t)
            if self.het_xab:
                xab_mu_out.append(self.head_xab_mu(z))
                xab_lv_out.append(self._stabilize_logv(self.head_xab_lv(z)))
            else:
                xab_det_out.append(self.head_xab(z))
            # strict/scheduled TF (strict=1.0 for TL)
            use_tf = (torch.rand(B, device=z.device) < tf_ratio).float().unsqueeze(-1)
            y_prev_sc = use_tf * y_tf_sc[:, t, :] + (1.0 - use_tf) * y_t

        pools_sc = torch.stack(pools_out,1)
        cres     = torch.stack(cres_out,1)
        if self.het_xab:
            mu = torch.stack(xab_mu_out,1)
            lv = torch.stack(xab_lv_out,1)
            return pools_sc, cres, (mu, lv)
        else:
            xab = torch.stack(xab_det_out,1)
            return pools_sc, cres, xab


# -------------------------
# Physics & metrics
# -------------------------

def carbon_closure_eps_seq(p_next_raw, p_prev_raw, flows_raw, cres):
    dt, FinV, CinC, CTR = flows_raw[...,0], flows_raw[...,1], flows_raw[...,2], flows_raw[...,3]
    d_acc = (p_next_raw - p_prev_raw).sum(dim=-1) + cres
    C_in  = dt * FinV * CinC
    C_gas = dt * CTR
    C_out = 0.0
    return d_acc - C_in + C_out + C_gas

@torch.no_grad()
def eval_tf_metrics(loader, model, inv_scale_pools, obs_names, xab_names, device,
                    het_xab=False, mu_xab=None, sd_xab=None):
    model.eval()
    Yp_t, Yp_p, Xa_t, Xa_p, R = [], [], [], [], []
    for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, rid in loader:
        enc_sc, dec_sc, y_sc = enc_sc.to(device), dec_sc.to(device), y_sc.to(device)
        out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
        if het_xab:
            pools_sc, _, (mu_sc, lv_sc) = out
            pools_raw_pred = inv_scale_pools(pools_sc)
            pools_raw_true = inv_scale_pools(y_sc)
            mu_raw = mu_sc * sd_xab + mu_xab
            Yp_t.append(pools_raw_true.cpu().numpy())
            Yp_p.append(pools_raw_pred.cpu().numpy())
            Xa_t.append(xab_next.numpy())
            Xa_p.append(mu_raw.cpu().numpy())
        else:
            pools_sc, _, xab_hat = out
            pools_raw_pred = inv_scale_pools(pools_sc)
            pools_raw_true = inv_scale_pools(y_sc)
            Yp_t.append(pools_raw_true.cpu().numpy())
            Yp_p.append(pools_raw_pred.cpu().numpy())
            Xa_t.append(xab_next.numpy())
            Xa_p.append(xab_hat.cpu().numpy())
        R.extend(list(rid))

    Yp_t = np.concatenate(Yp_t, 0)
    Yp_p = np.concatenate(Yp_p, 0)
    Xa_t = np.concatenate(Xa_t, 0)
    Xa_p = np.concatenate(Xa_p, 0)

    def chan_flat(y_true, y_pred, names):
        ys = y_true.reshape(-1, y_true.shape[-1])
        yp = y_pred.reshape(-1, y_true.shape[-1])
        out = []
        for i, nm in enumerate(names):
            out.append({
                "name": nm,
                "RMSE": float(rmse(ys[:, i], yp[:, i])),   # <- uses rmse helper
                "MAE":  float(mean_absolute_error(ys[:, i], yp[:, i])),
                "R2":   float(r2_score(ys[:, i], yp[:, i])),
            })
        return out

    def chan_by_h(y_true, y_pred, names):
        N, T, C = y_true.shape
        rows = []
        for t in range(T):
            yt = y_true[:, t, :]
            yp = y_pred[:, t, :]
            for i, nm in enumerate(names):
                rows.append({
                    "horizon": t+1,
                    "name": nm,
                    "RMSE": float(rmse(yt[:, i], yp[:, i])),   # <- also rmse
                    "MAE":  float(mean_absolute_error(yt[:, i], yp[:, i])),
                    "R2":   float(r2_score(yt[:, i], yp[:, i])),
                })
        return rows

    pools_flat = chan_flat(Yp_t, Yp_p, obs_names)
    xab_flat   = chan_flat(Xa_t, Xa_p, xab_names)
    pools_by_h = chan_by_h(Yp_t, Yp_p, obs_names)
    xab_by_h   = chan_by_h(Xa_t, Xa_p, xab_names)

    # per-run summary (flattened over horizon)
    per_run_rows = []
    run_ids = np.array(R)
    for rid in np.unique(run_ids):
        m = (run_ids == rid)
        ytp, ypp = Yp_t[m], Yp_p[m]
        xat, xap = Xa_t[m], Xa_p[m]
        ys = ytp.reshape(-1, ytp.shape[-1])
        yp = ypp.reshape(-1, ypp.shape[-1])
        xs = xat.reshape(-1, xat.shape[-1])
        xp = xap.reshape(-1, xap.shape[-1])
        for i, nm in enumerate(obs_names):
            per_run_rows.append({
                "run_id": rid,
                "name": nm,
                "RMSE": float(rmse(ys[:, i], yp[:, i])),   # <- ys/xs here, not yt
                "MAE":  float(mean_absolute_error(ys[:, i], yp[:, i])),
                "R2":   float(r2_score(ys[:, i], yp[:, i])),
            })
        for i, nm in enumerate(xab_names):
            per_run_rows.append({
                "run_id": rid,
                "name": nm,
                "RMSE": float(rmse(xs[:, i], xp[:, i])),
                "MAE":  float(mean_absolute_error(xs[:, i], xp[:, i])),
                "R2":   float(r2_score(xs[:, i], xp[:, i])),
            })
    return pools_flat, xab_flat, pools_by_h, xab_by_h, per_run_rows

# -------------------------
# Training stages
# -------------------------

def run_tl_and_save(args):
    set_seed(args.SEED)

    out = Path(args.OUTDIR_TL)
    out.mkdir(parents=True, exist_ok=True)

    # Load + carbonize
    df = pd.read_csv(args.TRAIN_INPUT_CSV)
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"] = df["batch_name"].astype(str)
    df = carbonize_df(df, args)

    # ---- Feature selection (aux are optional) ----
    aux_cols = get_aux_cols_present(df)  # picks any of ["pCO2","V_L","vvd_per_day"] that exist

    FEATURE_ORDER = [
        # required pools (5)
        "GlcC", "LacC", "DIC_mmolC_L", "BioC", "ProdC",
        # required drivers (3)
        "Fin_over_V_1ph", "CinC_mmolC_L", "CTR_mmolC_L_h",
        # optional aux (0–3)
        "pCO2", "V_L", "vvd_per_day",
    ]

    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]

    # enforce only the truly required ones
    required = {
        "GlcC", "LacC", "DIC_mmolC_L", "BioC", "ProdC",
        "Fin_over_V_1ph", "CinC_mmolC_L", "CTR_mmolC_L_h",
    }
    missing_req = sorted(required - set(feat_cols))
    if missing_req:
        raise RuntimeError(f"Missing required features: {missing_req}. Found: {feat_cols}")

    # ---------- keep your length filter ----------
    sizes = df.groupby("run_id").size(); need_len = args.T_IN + args.T_OUT
    keep = sizes.index[sizes >= need_len]
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No runs meet minimum length T_IN+T_OUT.")

    # ---------- choose split strategy ----------
    from cho_lstm_uq.utils.data_split import (
        train_val_split_basic,
        train_val_split_pairs,
        train_val_split_reps_by_index,
        train_val_split_doe_holdout,
        train_val_split_auto,
    )

    def _has_col(df, name): 
        return name in df.columns

    def _any_rep_suffix(df, col):
        if not _has_col(df, col): 
            return False
        return df[col].astype(str).str.contains(r"_rep\d+$", regex=True).any()

    def _looks_like_pairs(df):
        # True if there exists any value ending with _repN and every DOE appears to have exactly 2 reps
        col = "run_id" if _has_col(df, "run_id") else ("batch_name" if _has_col(df, "batch_name") else None)
        if col is None: 
            return False
        if not _any_rep_suffix(df, col):
            return False
        # quick heuristic: per-DOE unique rep counts are mostly 2
        s = df[col].astype(str)
        cores = s.str.replace(r"_rep\d+$", "", regex=True)
        rep_counts = df.assign(core=cores)["core"].value_counts()
        # if most cores have 2 occurrences, assume pairs
        return (rep_counts.median() == 2)

    def _looks_like_threeplus(df):
        col = "run_id" if _has_col(df, "run_id") else ("batch_name" if _has_col(df, "batch_name") else None)
        if col is None: 
            return False
        if not _any_rep_suffix(df, col):
            return False
        s = df[col].astype(str)
        cores = s.str.replace(r"_rep\d+$", "", regex=True)
        rep_counts = df.assign(core=cores)["core"].value_counts()
        return (rep_counts.median() >= 3)

    if args.SPLIT_STRATEGY == "basic":
        split_fn = lambda d: train_val_split_basic(d, getattr(args, "VAL_FRAC", 0.2))

    elif args.SPLIT_STRATEGY == "pairs":
        split_fn = lambda d: train_val_split_pairs(d, val_frac=0.5, random_state=args.SEED)

    elif args.SPLIT_STRATEGY == "rep_index":
        # e.g., for 2L with 3 reps per DoE, hold out all _rep3 as validation
        rep_idx = getattr(args, "REP_VAL_IDX", 3)
        split_fn = lambda d: train_val_split_reps_by_index(d, rep_val_index=rep_idx)

    elif args.SPLIT_STRATEGY == "doe_holdout":
        split_fn = lambda d: train_val_split_doe_holdout(d, val_frac_doe=0.25, random_state=args.SEED)

    elif args.SPLIT_STRATEGY == "auto":
        split_fn = lambda d: train_val_split_auto(
            d,
            default_pairs_val_frac=0.5,   # same behavior you intended
            rep_val_index=getattr(args, "REP_VAL_IDX", 3),
            random_state=args.SEED,
        )
    else:
    # default to auto if an unknown strategy is passed
        split_fn = lambda d: train_val_split_auto(
            d,
            default_pairs_val_frac=0.5,
            rep_val_index=getattr(args, "REP_VAL_IDX", 3),
            random_state=args.SEED,
        )

    # ---------- split ----------
    df_tr, df_va, val_runs = split_fn(df)
    if len(val_runs) == 0:
        raise RuntimeError("Validation split is empty.")
    print(f"[split] strategy={args.SPLIT_STRATEGY} | VAL runs: {val_runs}")

    # ---------- then keep your scaling as before ----------
    mu_all, sd_all = build_scalers(df_tr, feat_cols)
    mu_obs, sd_obs = mu_all[OBS_POOLS], sd_all[OBS_POOLS]
    obs_mu_t = torch.tensor(mu_obs.values, dtype=torch.float32, device=DEVICE)
    obs_sd_t = torch.tensor(sd_obs.values, dtype=torch.float32, device=DEVICE)
    def inv_scale_pools(y_sc): return obs_mu_t + obs_sd_t * y_sc


    df_tr_s = apply_scaler(df_tr, feat_cols, mu_all, sd_all)
    df_va_s = apply_scaler(df_va, feat_cols, mu_all, sd_all)

    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING
    ds_tr = Seq2SeqDataset(df_tr_s, df_tr[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT, stride=1)
    ds_va = Seq2SeqDataset(df_va_s, df_va[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT, stride=1)
    if len(ds_va) == 0:
        raise RuntimeError("Validation dataset produced 0 sequences.")

    dl_tr = DataLoader(ds_tr, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False)

    # Build model
    in_dim=len(feat_cols); n_drv_aux=len(DRIVING)+len(aux_cols)  # expect 6
    model = StrictCSeq2Seq(in_dim, out_pools=len(OBS_POOLS), n_drv_aux=n_drv_aux,
                           hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
                           het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
                           logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX).to(DEVICE)

    # Warm start
    theta0={}
    if os.path.isfile(args.TWO_L_WEIGHTS):
        sd=torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        theta0 = {k: v.clone().detach().to(DEVICE) for k, v in model.state_dict().items() if k in sd}
        print("Loaded 2L checkpoint | missing:", len(missing), "unexpected:", len(unexpected))
    else:
        print("WARNING: 2L weights not found; training from scratch.")

    mse = nn.MSELoss()

    # heteroscedastic scaler for X/Ab (z-space)
    if args.HET_XAB:
        mu_xab = torch.tensor(df_tr[["X_gL","Ab_gL"]].mean().values, device=DEVICE, dtype=torch.float32)
        sd_xab = torch.tensor(df_tr[["X_gL","Ab_gL"]].std().replace(0,1.0).values, device=DEVICE, dtype=torch.float32)

    def l2sp_term():
        if not args.L2SP_ALPHA or len(theta0)==0: return 0.0
        pen=0.0
        for (n,p) in model.named_parameters():
            if (not p.requires_grad) or (n not in theta0) or n.endswith(".bias"): continue
            pen = pen + torch.sum((p - theta0[n].to(p.device))**2)
        return args.L2SP_ALPHA * pen

    def train_stage(epochs, lr, lamb_mb, lamb_cons):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
        best_val=float('inf'); patience=0
        for ep in range(1, epochs+1):
            model.train()
            for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_tr:
                enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)
                yprev_raw, flows_raw, xab_next = yprev_raw.to(DEVICE), flows_raw.to(DEVICE), xab_next.to(DEVICE)

                out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)  # strict TF
                if args.HET_XAB:
                    pools_sc, cres_t, (mu_sc, lv_sc) = out
                    lv_sc = model._stabilize_logv(lv_sc)
                    # physics
                    p_next_raw = inv_scale_pools(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                    # losses
                    loss_state = mse(pools_sc, y_sc)
                    z = (xab_next - mu_xab) / sd_xab
                    nll = 0.5 * (lv_sc + (z - mu_sc)**2 / torch.exp(lv_sc))
                    loss_xab = nll.mean()
                    # consistency
                    BioC_pred  = p_next_raw[..., 3]
                    ProdC_pred = p_next_raw[..., 4]
                    mu_raw = mu_sc * sd_xab + mu_xab
                    X_pred_gL  = mu_raw[..., 0]
                    Ab_pred_gL = mu_raw[..., 1]
                else:
                    pools_sc, cres_t, xab_hat = out
                    p_next_raw = inv_scale_pools(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                    loss_state = mse(pools_sc, y_sc)
                    loss_xab = mse(xab_hat, xab_next)
                    BioC_pred  = p_next_raw[..., 3]
                    ProdC_pred = p_next_raw[..., 4]
                    X_pred_gL  = xab_hat[..., 0]
                    Ab_pred_gL = xab_hat[..., 1]

                cons_bio = mse(BioC_pred,  X_pred_gL * args.MMOLC_PER_G_BIOMASS)
                cons_ab  = mse(ProdC_pred, Ab_pred_gL * args.MMOLC_PER_G_MAB)
                loss_cons = cons_bio + cons_ab

                ramp = 0.5 + 0.5 * min(1.0, ep / max(1,epochs))
                loss_mb  = eps.abs().mean()
                loss_res = (cres_t**2).mean()
                loss = (loss_state
                        + args.LAMBDA_XAB*loss_xab
                        + lamb_mb*loss_mb
                        + args.GAMMA_RES*loss_res
                        + (lamb_cons*ramp)*loss_cons
                        + l2sp_term())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.CLIP)
                opt.step()

            # early stop via val state loss (TF=1.0)
            model.eval(); vtot=0.0; vn=0
            with torch.no_grad():
                for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_va:
                    enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)
                    pools_sc, *_ = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                    vtot += mse(pools_sc, y_sc).item(); vn+=1
            val = vtot/max(1,vn)
            if val < best_val - 1e-6:
                best_val=val; patience=0
                best_state={k:(v.detach().cpu().clone() if torch.is_tensor(v) else v) for k,v in model.state_dict().items()}
            else:
                patience+=1
                if patience>=args.PATIENCE: break
        if 'best_state' in locals(): model.load_state_dict(best_state)

    # Stage A: heads + top decoder
    for p in model.parameters(): p.requires_grad=False
    for p in model.head_pools.parameters(): p.requires_grad=True
    for p in model.head_cres.parameters():  p.requires_grad=True
    if args.HET_XAB:
        for p in model.head_xab_mu.parameters(): p.requires_grad=True
        for p in model.head_xab_lv.parameters(): p.requires_grad=True
    else:
        for p in model.head_xab.parameters():    p.requires_grad=True
    L = model.dec.num_layers - 1
    for name,p in model.dec.named_parameters():
        if f"_l{L}" in name: p.requires_grad=True
    train_stage(args.FREEZE_EPOCHS_HEAD, args.LR_STAGE_A, args.LAMBDA_MB*0.5, args.LAMBDA_CONS*0.5)

    # Stage B: + top encoder
    for name,p in model.enc.named_parameters():
        if f"_l{L}" in name: p.requires_grad=True
    train_stage(args.FREEZE_EPOCHS_TOP, args.LR_STAGE_B, args.LAMBDA_MB, args.LAMBDA_CONS)

    # Stage C: short polish (all)
    for p in model.parameters(): p.requires_grad=True
    train_stage(args.POLISH_EPOCHS, args.LR_STAGE_C, args.LAMBDA_MB, args.LAMBDA_CONS)

    # Final VAL metrics (teacher-forced)
    if args.HET_XAB:
        pf, xf, pbh, xbh, pr = eval_tf_metrics(
            dl_va, model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"], DEVICE,
            het_xab=True, mu_xab=mu_xab, sd_xab=sd_xab
        )
    else:
        pf, xf, pbh, xbh, pr = eval_tf_metrics(
            dl_va, model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"], DEVICE
        )
    print("VAL R2 (X,Ab):", {d["name"]: d["R2"] for d in xf})

    # Save VAL metric CSVs + checkpoint + feature scaler + (optional) xab scaler
    pd.DataFrame(pf).to_csv(out/"VAL_pools_flat.csv", index=False)
    pd.DataFrame(xf).to_csv(out/"VAL_xab_flat.csv", index=False)
    pd.DataFrame(pbh).to_csv(out/"VAL_pools_by_h.csv", index=False)
    pd.DataFrame(xbh).to_csv(out/"VAL_xab_by_h.csv", index=False)
    pd.DataFrame(pr).to_csv(out/"VAL_per_run_flat.csv", index=False)

    torch.save(model.state_dict(), out/"strictC_seq2seq_50L_XAb.pt")
    with open(out/"scaler_features.json","w") as f:
        json.dump({"mu_all": mu_all.to_dict(), "sd_all": sd_all.to_dict(), "feat_cols": feat_cols}, f)
    if args.HET_XAB:
        with open(out/"scaler_xab.json","w") as f:
            json.dump({"mu_xab": mu_xab.detach().cpu().numpy().tolist(),
                       "sd_xab": sd_xab.detach().cpu().numpy().tolist()}, f)

    # === VAL summaries (old-schema one-row) + Learning Curve table ===
    write_val_summaries_and_lc(
        model=model,
        dl_va=dl_va,
        inv_scale_pools=inv_scale_pools,
        args=args,
        df_tr=df_tr,
        df_va=df_va,
        feat_cols=feat_cols,
        aux_cols=aux_cols,
        out=out,
        val_runs=val_runs,
        train_stage=train_stage,
        mu_xab=(mu_xab if args.HET_XAB else None),
        sd_xab=(sd_xab if args.HET_XAB else None),
    )

    # -------- Optional: UQ ensemble (writes OUTDIR_TL/UQ/ensemble_val_predictions_with_uncert.csv) --------
    UQ_OUTDIR = out / "UQ"
    UQ_OUTDIR.mkdir(parents=True, exist_ok=True)

    # save snapshot for consistency (already saved above, but harmless if repeated)
    torch.save(model.state_dict(), out / "strictC_seq2seq_50L_XAb.pt")

    # tiny ensemble for variance file (kept small to avoid runtime bloat)
    SEEDS = [0, 1, 2]      # adjust if you want bigger ensemble
    TOPUP_EPOCHS = 0       # keep 0 for speed
    TOPUP_LR = 3e-4

    # for rebuilding models from TL checkpoint
    n_drv_aux = len(DRIVING) + len(aux_cols)
    def build_model_from_tl(strict=True):
        m = StrictCSeq2Seq(in_dim=len(feat_cols), out_pools=len(OBS_POOLS), n_drv_aux=n_drv_aux,
                           hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
                           het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
                           logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX).to(DEVICE)
        sd = torch.load(out / "strictC_seq2seq_50L_XAb.pt", map_location=DEVICE)
        m.load_state_dict(sd, strict=strict)
        return m

    seed_dirs = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        mdl = build_model_from_tl(strict=True)
        if TOPUP_EPOCHS > 0:
            opt = torch.optim.AdamW(mdl.parameters(), lr=TOPUP_LR, weight_decay=1e-6)
            for ep in range(1, TOPUP_EPOCHS + 1):
                mdl.train()
                for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in DataLoader(
                        ds_tr, batch_size=args.BATCH_TR, shuffle=True, drop_last=True):
                    enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)
                    pools_sc, *_ = mdl(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                    loss = mse(pools_sc, y_sc)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(mdl.parameters(), args.CLIP); opt.step()
        sdir = UQ_OUTDIR / f"seed{s}"
        sdir.mkdir(parents=True, exist_ok=True)
        torch.save(mdl.state_dict(), sdir / "best.pt")
        seed_dirs.append(sdir)

    # Only produce UQ CSV if heteroscedastic head is enabled (runner will read it if present)
    if args.HET_XAB:
        ensemble_predict(
            seed_dirs=seed_dirs,
            ds_va=dl_va,
            feat_cols=feat_cols,
            n_drv_aux=n_drv_aux,
            mu_xab=mu_xab,
            sd_xab=sd_xab,
            args=args,
            UQ_OUTDIR=UQ_OUTDIR,
        )

@torch.no_grad()
def ensemble_predict(
    seed_dirs: List[Path],
    ds_va: DataLoader,
    feat_cols: List[str],
    n_drv_aux: int,
    mu_xab: torch.Tensor,
    sd_xab: torch.Tensor,
    args,
    UQ_OUTDIR: Path,
):
    """Writes ensemble UQ CSV (in raw g/L units for X, Ab)."""
    assert args.HET_XAB, "ensemble_predict is only called for heteroscedastic runs."

    mu_xab_t = mu_xab.to(DEVICE)
    sd_xab_t = sd_xab.to(DEVICE)

    models = []
    for sdir in seed_dirs:
        m = StrictCSeq2Seq(
            in_dim=len(feat_cols),
            out_pools=len(OBS_POOLS),
            n_drv_aux=n_drv_aux,
            hidden=args.HIDDEN,
            layers=args.LAYERS,
            dropout=args.DROPOUT,
            het_xab=args.HET_XAB,
            softplus_var=args.SOFTPLUS_VAR,
            logv_min=args.LOGV_MIN,
            logv_max=args.LOGV_MAX,
        ).to(DEVICE)
        m.load_state_dict(torch.load(sdir / "best.pt", map_location=DEVICE), strict=True)
        m.eval()
        models.append(m)

    rows = []
    va_loader = DataLoader(ds_va.dataset, batch_size=args.BATCH_EVAL, shuffle=False)

    for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, rid in va_loader:
        enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)

        MU_raw_list, VAR_ale_raw_list = [], []

        for m in models:
            _, _, (mu_sc, lv_sc) = m(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
            lv_sc = m._stabilize_logv(lv_sc)

            # z-space → raw
            mu_raw = mu_sc * sd_xab_t + mu_xab_t                 # g/L
            var_ale_raw = torch.exp(lv_sc) * (sd_xab_t ** 2)     # g^2/L^2

            MU_raw_list.append(mu_raw.detach().cpu().numpy())
            VAR_ale_raw_list.append(var_ale_raw.detach().cpu().numpy())

        MU  = np.stack(MU_raw_list)       # (S,B,T,2)
        VAR = np.stack(VAR_ale_raw_list)  # (S,B,T,2)

        mu_ens   = MU.mean(0)
        var_epi  = MU.var(0, ddof=1) if MU.shape[0] > 1 else np.zeros_like(MU[0])
        var_alea = VAR.mean(0)
        var_tot  = var_epi + var_alea

        Y       = xab_next.numpy()
        rid_arr = np.array(rid)
        B, T = mu_ens.shape[:2]
        for b in range(B):
            for t in range(T):
                rows.append({
                    "run_id":     str(rid_arr[b]),
                    "t_idx":      int(t),
                    "true_X":     float(Y[b, t, 0]),
                    "true_Ab":    float(Y[b, t, 1]),
                    "mu_X":       float(mu_ens[b, t, 0]),
                    "mu_Ab":      float(mu_ens[b, t, 1]),
                    "var_ale_X":  float(var_alea[b, t, 0]),
                    "var_ale_Ab": float(var_alea[b, t, 1]),
                    "var_epi_X":  float(var_epi[b, t, 0]),
                    "var_epi_Ab": float(var_epi[b, t, 1]),
                    "var_tot_X":  float(var_tot[b, t, 0]),
                    "var_tot_Ab": float(var_tot[b, t, 1]),
                })

    dfE = pd.DataFrame(rows)
    out_csv = UQ_OUTDIR / "ensemble_val_predictions_with_uncert.csv"
    dfE.to_csv(out_csv, index=False)
    print(f"[ensemble] saved → {out_csv}")


@torch.no_grad()
def _eval_r2_summary(
    model: nn.Module,
    dl_va: DataLoader,
    inv_scale_pools,           # maps scaled pools -> raw pools (tensor op)
    het_xab: bool,
    mu_xab: Optional[torch.Tensor],
    sd_xab: Optional[torch.Tensor],
    device: str = DEVICE,
    tf_ratio: float = 1.0,     # 1.0 = TF, ~0 = OL
) -> Dict[str, float]:
    """
    Returns dict with:
      R2_avg_pools, R2_X_gL, R2_Ab_gL
    Evaluates across all VAL batches. Pools are averaged across the 5 obs pools.
    """
    model.eval()
    y_pools_all, yhat_pools_all = [], []
    yX_all, yXhat_all = [], []
    yAb_all, yAbhat_all = [], []

    for batch in dl_va:
        if len(batch) < 6:
            raise RuntimeError("VAL dataloader must return (enc, dec, y_sc, yprev_raw, flows_raw, xab_next, ...)")
        enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next = batch[:6]

        enc_sc = enc_sc.to(device)
        dec_sc = dec_sc.to(device)
        y_sc   = y_sc.to(device)
        xab_next = xab_next.to(device)

        out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=tf_ratio)

        if het_xab:
            pools_sc, _, (mu_sc, lv_sc) = out
            mu_raw = mu_sc * sd_xab + mu_xab     # raw g/L
            xab_hat_raw = mu_raw
        else:
            pools_sc, _, xab_hat_raw = out

        y_pools_all.append((inv_scale_pools(y_sc)).detach().cpu())
        yhat_pools_all.append((inv_scale_pools(pools_sc)).detach().cpu())

        yX_all.append(xab_next[..., 0].detach().cpu())
        yAb_all.append(xab_next[..., 1].detach().cpu())
        yXhat_all.append(xab_hat_raw[..., 0].detach().cpu())
        yAbhat_all.append(xab_hat_raw[..., 1].detach().cpu())

    Yp   = torch.cat(y_pools_all, dim=0).numpy()
    Yph  = torch.cat(yhat_pools_all, dim=0).numpy()
    yX   = torch.cat(yX_all, dim=0).numpy()
    yXh  = torch.cat(yXhat_all, dim=0).numpy()
    yAb  = torch.cat(yAb_all, dim=0).numpy()
    yAbh = torch.cat(yAbhat_all, dim=0).numpy()

    def flat(a): return a.reshape(-1)

    r2_pools = [r2_score(flat(Yp[..., k]), flat(Yph[..., k])) for k in range(Yp.shape[-1])]
    R2_avg_pools = float(np.mean(r2_pools)) if len(r2_pools) else float("nan")
    R2_X_gL  = float(r2_score(flat(yX),  flat(yXh)))
    R2_Ab_gL = float(r2_score(flat(yAb), flat(yAbh)))

    return dict(R2_avg_pools=R2_avg_pools, R2_X_gL=R2_X_gL, R2_Ab_gL=R2_Ab_gL)

def write_val_summaries_and_lc(
    model: nn.Module,
    dl_va: DataLoader,
    inv_scale_pools,
    args,
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    feat_cols: List[str],
    aux_cols: List[str],
    out: Path,
    val_runs: List[str],
    train_stage,  # kept for compatibility with caller (used in main TL, not in LC anymore)
    mu_xab: Optional[torch.Tensor] = None,
    sd_xab: Optional[torch.Tensor] = None,
):
    """Writes (1) VAL_k_summary.csv and (2) LC_added_runs_metrics_TF_OL.csv."""

    # --- One-row old-schema summary (TF vs OL) ---
    tf_scores = _eval_r2_summary(
        model=model,
        dl_va=dl_va,
        inv_scale_pools=inv_scale_pools,
        het_xab=bool(args.HET_XAB),
        mu_xab=mu_xab,
        sd_xab=sd_xab,
        device=DEVICE,
        tf_ratio=1.0,
    )
    ol_scores = _eval_r2_summary(
        model=model,
        dl_va=dl_va,
        inv_scale_pools=inv_scale_pools,
        het_xab=bool(args.HET_XAB),
        mu_xab=mu_xab,
        sd_xab=sd_xab,
        device=DEVICE,
        tf_ratio=1e-6,
    )

    k_runs = len(sorted(df_tr["run_id"].astype(str).unique()))
    runs_used_json = json.dumps(sorted(list(val_runs)))  # VAL runs

    row = {
        "k_runs": k_runs,
        "runs_used_json": runs_used_json,
        "TF_R2_avg_pools": tf_scores["R2_avg_pools"],
        "TF_R2_X_gL":      tf_scores["R2_X_gL"],
        "TF_R2_Ab_gL":     tf_scores["R2_Ab_gL"],
        "OL_R2_avg_pools": ol_scores["R2_avg_pools"],
        "OL_R2_X_gL":      ol_scores["R2_X_gL"],
        "OL_R2_Ab_gL":     ol_scores["R2_Ab_gL"],
    }

    out_summary = out / "VAL_k_summary.csv"
    pd.DataFrame([row]).to_csv(out_summary, index=False)
    print(f"[summary] wrote old-schema row → {out_summary}")

    row_ext = dict(row)
    row_ext.update({
        "split_strategy": args.SPLIT_STRATEGY,
        "seed": args.SEED,
        "T_IN": args.T_IN,
        "T_OUT": args.T_OUT,
        "timestamp": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "outdir": str(out),
    })
    lc_snapshot = out / "LC_snapshot_metrics_TF_OL.csv"
    df_row = pd.DataFrame([row_ext])
    if lc_snapshot.exists():
        df_row.to_csv(lc_snapshot, mode="a", header=False, index=False)
    else:
        df_row.to_csv(lc_snapshot, index=False)
    print(f"[summary] appended → {lc_snapshot}")

    # --- L2SP reference for LC: snapshot of 2L weights (same idea as main TL) ---
    theta0_lc = None
    if getattr(args, "TWO_L_WEIGHTS", None) and Path(args.TWO_L_WEIGHTS).exists():
        ref_model = StrictCSeq2Seq(
            in_dim=len(feat_cols),
            out_pools=len(OBS_POOLS),
            n_drv_aux=len(DRIVING) + len(aux_cols),
            hidden=args.HIDDEN,
            layers=args.LAYERS,
            dropout=args.DROPOUT,
            het_xab=args.HET_XAB,
            softplus_var=args.SOFTPLUS_VAR,
            logv_min=args.LOGV_MIN,
            logv_max=args.LOGV_MAX,
        ).to(DEVICE)
        sd_ref = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
        missing, unexpected = ref_model.load_state_dict(sd_ref, strict=False)
        print(f"[LC] built theta0_lc from 2L | missing={len(missing)} unexpected={len(unexpected)}")

        theta0_lc = {
            k: v.clone().detach().to(DEVICE)
            for k, v in ref_model.state_dict().items()
            if k in sd_ref
        }
    else:
        print("[LC] theta0_lc disabled (no TWO_L_WEIGHTS or file missing)")

    # ----------------- Generic LC training helper (k-specific) -----------------
    def _train_stage_k(
        model_k: nn.Module,
        dl_tr_k: DataLoader,
        dl_va_k: DataLoader,
        inv_scale_k,
        mu_xab_k: Optional[torch.Tensor],
        sd_xab_k: Optional[torch.Tensor],
        epochs: int,
        lr: float,
        lamb_mb: float,
        lamb_cons: float,
        theta0_lc=None,
    ):
        """
        LC-only training stage:
          - Uses k-specific loaders and scalers
          - Same loss recipe as main TL (MB + X/Ab + CONS + RES)
          - Optional L2SP toward theta0_lc (2L reference) if provided
        """
        mse = nn.MSELoss()

        def l2sp_term_k():
            if (not getattr(args, "L2SP_ALPHA", 0.0)) or (theta0_lc is None):
                return 0.0
            pen = 0.0
            for name, p in model_k.named_parameters():
                if (not p.requires_grad) or (name not in theta0_lc) or name.endswith(".bias"):
                    continue
                pen = pen + torch.sum((p - theta0_lc[name].to(p.device)) ** 2)
            return args.L2SP_ALPHA * pen

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_k.parameters()),
            lr=lr,
            weight_decay=1e-6,
        )
        best_val = float("inf")
        patience = 0

        for ep in range(1, epochs + 1):
            model_k.train()
            for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_tr_k:
                enc_sc    = enc_sc.to(DEVICE)
                dec_sc    = dec_sc.to(DEVICE)
                y_sc      = y_sc.to(DEVICE)
                yprev_raw = yprev_raw.to(DEVICE)
                flows_raw = flows_raw.to(DEVICE)
                xab_next  = xab_next.to(DEVICE)

                out = model_k(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)  # strict TF

                if args.HET_XAB:
                    assert mu_xab_k is not None and sd_xab_k is not None, "mu_xab_k/sd_xab_k required for HET_XAB."
                    pools_sc, cres_t, (mu_sc, lv_sc) = out
                    lv_sc = model_k._stabilize_logv(lv_sc)

                    # physics
                    p_next_raw = inv_scale_k(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)

                    # state loss
                    loss_state = mse(pools_sc, y_sc)

                    # heteroscedastic X/Ab NLL in z-space
                    z = (xab_next - mu_xab_k) / sd_xab_k
                    nll = 0.5 * (lv_sc + (z - mu_sc) ** 2 / torch.exp(lv_sc))
                    loss_xab = nll.mean()

                    BioC_pred  = p_next_raw[..., 3]
                    ProdC_pred = p_next_raw[..., 4]
                    mu_raw = mu_sc * sd_xab_k + mu_xab_k
                    X_pred_gL  = mu_raw[..., 0]
                    Ab_pred_gL = mu_raw[..., 1]
                else:
                    pools_sc, cres_t, xab_hat = out
                    p_next_raw = inv_scale_k(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)

                    loss_state = mse(pools_sc, y_sc)
                    loss_xab = mse(xab_hat, xab_next)

                    BioC_pred  = p_next_raw[..., 3]
                    ProdC_pred = p_next_raw[..., 4]
                    X_pred_gL  = xab_hat[..., 0]
                    Ab_pred_gL = xab_hat[..., 1]

                cons_bio = mse(BioC_pred,  X_pred_gL * args.MMOLC_PER_G_BIOMASS)
                cons_ab  = mse(ProdC_pred, Ab_pred_gL * args.MMOLC_PER_G_MAB)
                loss_cons = cons_bio + cons_ab

                ramp = 0.5 + 0.5 * min(1.0, ep / max(1, epochs))
                loss_mb  = eps.abs().mean()
                loss_res = (cres_t ** 2).mean()

                loss = (
                    loss_state
                    + args.LAMBDA_XAB * loss_xab
                    + lamb_mb * loss_mb
                    + args.GAMMA_RES * loss_res
                    + (lamb_cons * ramp) * loss_cons
                    + l2sp_term_k()
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_k.parameters()), args.CLIP)
                opt.step()

            # Early stopping on TF state loss (using k-specific VAL)
            model_k.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_va_k:
                    enc_sc = enc_sc.to(DEVICE)
                    dec_sc = dec_sc.to(DEVICE)
                    y_sc   = y_sc.to(DEVICE)
                    pools_sc, *_ = model_k(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                    vtot += mse(pools_sc, y_sc).item()
                    vn   += 1
            val = vtot / max(1, vn)

            if val < best_val - 1e-6:
                best_val = val
                patience = 0
                best_state = {
                    k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                    for k, v in model_k.state_dict().items()
                }
            else:
                patience += 1
                if patience >= args.PATIENCE:
                    break

        if "best_state" in locals():
            model_k.load_state_dict(best_state)

    # ----------------- Learning Curve (production-faithful) -----------------
    print("\n>>> Learning curve by added 50L runs (production-equivalent recipe)")

    all_train_runs = sorted(df_tr["run_id"].astype(str).unique().tolist())
    rows_lc = []

    # Save base init to reuse same random weights across k
    torch.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    base_init = StrictCSeq2Seq(
        in_dim=len(feat_cols),
        out_pools=len(OBS_POOLS),
        n_drv_aux=len(DRIVING) + len(aux_cols),
        hidden=args.HIDDEN,
        layers=args.LAYERS,
        dropout=args.DROPOUT,
        het_xab=args.HET_XAB,
        softplus_var=args.SOFTPLUS_VAR,
        logv_min=args.LOGV_MIN,
        logv_max=args.LOGV_MAX,
    ).to(DEVICE)
    init_state = copy.deepcopy(base_init.state_dict())

    raw_cols = ["run_id", "time_h", "dt", "X_gL", "Ab_gL"] + OBS_POOLS + DRIVING

    for k in range(1, len(all_train_runs) + 1):
        use_runs = set(all_train_runs[:k])
        sub_tr = df_tr[df_tr["run_id"].astype(str).isin(use_runs)].reset_index(drop=True)

        # Rebuild scalers on this k-subset
        mu_all_k, sd_all_k = build_scalers(sub_tr, feat_cols)
        mu_obs_k, sd_obs_k = mu_all_k[OBS_POOLS], sd_all_k[OBS_POOLS]
        obs_mu_t_k = torch.tensor(mu_obs_k.values, dtype=torch.float32, device=DEVICE)
        obs_sd_t_k = torch.tensor(sd_obs_k.values, dtype=torch.float32, device=DEVICE)
        inv_scale_k = lambda y_sc: obs_mu_t_k + obs_sd_t_k * y_sc

        sub_tr_s  = apply_scaler(sub_tr, feat_cols, mu_all_k, sd_all_k)
        df_va_s_k = apply_scaler(df_va,  feat_cols, mu_all_k, sd_all_k)

        ds_tr_k = Seq2SeqDataset(
            sub_tr_s,
            sub_tr[raw_cols].copy(),
            OBS_POOLS,
            DRIVING,
            aux_cols,
            args.T_IN,
            args.T_OUT,
        )
        ds_va_k = Seq2SeqDataset(
            df_va_s_k,
            df_va[raw_cols].copy(),
            OBS_POOLS,
            DRIVING,
            aux_cols,
            args.T_IN,
            args.T_OUT,
        )
        dl_tr_k = DataLoader(ds_tr_k, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
        dl_va_k = DataLoader(ds_va_k, batch_size=args.BATCH_EVAL, shuffle=False)

        # heteroscedastic norm for LC loss
        if args.HET_XAB:
            mu_xab_k = torch.tensor(
                sub_tr[["X_gL", "Ab_gL"]].mean().values,
                dtype=torch.float32,
                device=DEVICE,
            )
            sd_xab_k = torch.tensor(
                sub_tr[["X_gL", "Ab_gL"]].std().replace(0, 1.0).values,
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            mu_xab_k = sd_xab_k = None

        # Fresh model init
        model_k = StrictCSeq2Seq(
            in_dim=len(feat_cols),
            out_pools=len(OBS_POOLS),
            n_drv_aux=len(DRIVING) + len(aux_cols),
            hidden=args.HIDDEN,
            layers=args.LAYERS,
            dropout=args.DROPOUT,
            het_xab=args.HET_XAB,
            softplus_var=args.SOFTPLUS_VAR,
            logv_min=args.LOGV_MIN,
            logv_max=args.LOGV_MAX,
        ).to(DEVICE)
        model_k.load_state_dict(init_state)

        # Optional warm-start from 2L
        if getattr(args, "TWO_L_WEIGHTS", None) and Path(args.TWO_L_WEIGHTS).exists():
            sd_tl = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
            model_k.load_state_dict(sd_tl, strict=False)

        # Run the same Stage A/B/C schedule as production, but via LC-specific helper
        # Stage A: heads + top decoder
        for p in model_k.parameters():
            p.requires_grad = False
        for p in model_k.head_pools.parameters():
            p.requires_grad = True
        for p in model_k.head_cres.parameters():
            p.requires_grad = True
        if args.HET_XAB:
            for p in model_k.head_xab_mu.parameters():
                p.requires_grad = True
            for p in model_k.head_xab_lv.parameters():
                p.requires_grad = True
        else:
            for p in model_k.head_xab.parameters():
                p.requires_grad = True
        L = model_k.dec.num_layers - 1
        for name, p in model_k.dec.named_parameters():
            if f"_l{L}" in name:
                p.requires_grad = True
        _train_stage_k(
            model_k,
            dl_tr_k,
            dl_va_k,
            inv_scale_k,
            mu_xab_k,
            sd_xab_k,
            epochs=args.FREEZE_EPOCHS_HEAD,
            lr=args.LR_STAGE_A,
            lamb_mb=args.LAMBDA_MB * 0.5,
            lamb_cons=args.LAMBDA_CONS * 0.5,
            theta0_lc=theta0_lc,
        )

        # Stage B: + top encoder
        for name, p in model_k.enc.named_parameters():
            if f"_l{L}" in name:
                p.requires_grad = True
        _train_stage_k(
            model_k,
            dl_tr_k,
            dl_va_k,
            inv_scale_k,
            mu_xab_k,
            sd_xab_k,
            epochs=args.FREEZE_EPOCHS_TOP,
            lr=args.LR_STAGE_B,
            lamb_mb=args.LAMBDA_MB,
            lamb_cons=args.LAMBDA_CONS,
            theta0_lc=theta0_lc,
        )

        # Stage C: short polish (all)
        for p in model_k.parameters():
            p.requires_grad = True
        _train_stage_k(
            model_k,
            dl_tr_k,
            dl_va_k,
            inv_scale_k,
            mu_xab_k,
            sd_xab_k,
            epochs=args.POLISH_EPOCHS,
            lr=args.LR_STAGE_C,
            lamb_mb=args.LAMBDA_MB,
            lamb_cons=args.LAMBDA_CONS,
            theta0_lc=theta0_lc,
        )

        # Evaluate (TF & OL) with k-specific scalers + VAL loader
        tf_scores_k = _eval_r2_summary(
            model=model_k,
            dl_va=dl_va_k,
            inv_scale_pools=inv_scale_k,
            het_xab=bool(args.HET_XAB),
            mu_xab=mu_xab_k,
            sd_xab=sd_xab_k,
            device=DEVICE,
            tf_ratio=1.0,
        )
        ol_scores_k = _eval_r2_summary(
            model=model_k,
            dl_va=dl_va_k,
            inv_scale_pools=inv_scale_k,
            het_xab=bool(args.HET_XAB),
            mu_xab=mu_xab_k,
            sd_xab=sd_xab_k,
            device=DEVICE,
            tf_ratio=1e-6,
        )

        rows_lc.append({
            "k_runs": k,
            "runs_used_json": json.dumps(sorted(list(use_runs))),
            "TF_R2_avg_pools": tf_scores_k["R2_avg_pools"],
            "TF_R2_X_gL":      tf_scores_k["R2_X_gL"],
            "TF_R2_Ab_gL":     tf_scores_k["R2_Ab_gL"],
            "OL_R2_avg_pools": ol_scores_k["R2_avg_pools"],
            "OL_R2_X_gL":      ol_scores_k["R2_X_gL"],
            "OL_R2_Ab_gL":     ol_scores_k["R2_Ab_gL"],
        })

        print(
            f"  k={k:02d} | TF R²[X]={tf_scores_k['R2_X_gL']:.3f}  TF R²[Ab]={tf_scores_k['R2_Ab_gL']:.3f} "
            f"| OL R²[X]={ol_scores_k['R2_X_gL']:.3f}  OL R²[Ab]={ol_scores_k['R2_Ab_gL']:.3f}"
        )

    lc_csv = out / "LC_added_runs_metrics_TF_OL.csv"
    pd.DataFrame(rows_lc).to_csv(lc_csv, index=False)
    print(f"Saved learning-curve table → {lc_csv}")


# -------------------------
# CLI
# -------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Strict-C TL trainer (2L→50L) — compatible with Optuna runner")

    # Paths (must match runner)
    ap.add_argument("--TRAIN_INPUT_CSV", type=str, required=True)
    ap.add_argument("--TWO_L_WEIGHTS",   type=str, required=True)
    ap.add_argument("--OUTDIR_TL",       type=str, required=True)

    # Chemistry / constants
    ap.add_argument("--MMOLC_PER_G_BIOMASS", type=float, default=39.6)
    ap.add_argument("--MMOLC_PER_G_MAB",     type=float, default=43.8)
    ap.add_argument("--MW_GLC",  type=float, default=180.156)
    ap.add_argument("--MW_LAC",  type=float, default=90.078)
    ap.add_argument("--CARBON_PER_GLC", type=float, default=6.0)
    ap.add_argument("--CARBON_PER_LAC", type=float, default=3.0)
    ap.add_argument("--DEFAULT_gDCW_PER_CELL", type=float, default=2.8e-12)

    # Model / training knobs (must match runner)
    # in build_argparser()
    ap.add_argument("--SPLIT_STRATEGY",
                type=str, default="auto",
                choices=["auto","basic","pairs","rep_index","doe_holdout"])
    ap.add_argument("--REP_VAL_IDX", type=int, default=3,
                help="When using rep_index (or auto with ≥3 reps), which _repN to place in VAL.")
    ap.add_argument("--SEED", type=int, default=42)
    ap.add_argument("--T_IN", type=int, default=72)
    ap.add_argument("--T_OUT", type=int, default=24)
    ap.add_argument("--HIDDEN", type=int, default=128)
    ap.add_argument("--LAYERS", type=int, default=2)
    ap.add_argument("--DROPOUT", type=float, default=0.1)
    ap.add_argument("--CLIP", type=float, default=1.0)
    ap.add_argument("--BATCH_TR", type=int, default=64)
    ap.add_argument("--BATCH_EVAL", type=int, default=128)
    ap.add_argument("--PATIENCE", type=int, default=8)

    # Loss weights
    ap.add_argument("--LAMBDA_MB",   type=float, default=0.05)
    ap.add_argument("--GAMMA_RES",   type=float, default=2e-5)
    ap.add_argument("--LAMBDA_XAB",  type=float, default=1.0)
    ap.add_argument("--LAMBDA_CONS", type=float, default=0.2)

    # TL schedule
    ap.add_argument("--FREEZE_EPOCHS_HEAD", type=int, default=3)
    ap.add_argument("--FREEZE_EPOCHS_TOP",  type=int, default=4)
    ap.add_argument("--LR_STAGE_A", type=float, default=3e-4)
    ap.add_argument("--LR_STAGE_B", type=float, default=3e-4)
    ap.add_argument("--LR_STAGE_C", type=float, default=5e-5)
    ap.add_argument("--L2SP_ALPHA", type=float, default=1e-4)

    # Uncertainty head
    ap.add_argument("--HET_XAB", action="store_true", default=False)
    ap.add_argument("--SOFTPLUS_VAR", action="store_true", default=False)
    ap.add_argument("--LOGV_MIN", type=float, default=-10.0)
    ap.add_argument("--LOGV_MAX", type=float, default=3.0)

    # Learning curve (disabled during HPO; kept for compatibility)
    ap.add_argument("--LC_ORDERS", type=int, default=5)
    ap.add_argument("--LC_EPOCHS", type=int, default=0)

    # TL polish epochs (runner passes POLISH_EPOCHS)
    ap.add_argument("--POLISH_EPOCHS", type=int, default=2)

    # Validation fraction (kept configurable)
    ap.add_argument("--VAL_FRAC", type=float, default=0.2)

    return ap

def main():
    try:
        ap = build_argparser()
        args = ap.parse_args()

        # Resolve to absolute paths early
        args.TRAIN_INPUT_CSV = str(Path(args.TRAIN_INPUT_CSV).resolve())
        args.TWO_L_WEIGHTS   = str(Path(args.TWO_L_WEIGHTS).resolve())
        args.OUTDIR_TL       = str(Path(args.OUTDIR_TL).resolve())

        # Sanity checks
        if not Path(args.TRAIN_INPUT_CSV).exists():
            raise FileNotFoundError(f"TRAIN_INPUT_CSV not found: {args.TRAIN_INPUT_CSV}")
        if not Path(args.TWO_L_WEIGHTS).exists():
            print(f"WARNING: TWO_L_WEIGHTS not found: {args.TWO_L_WEIGHTS} — training from scratch.")

        run_tl_and_save(args)
        sys.exit(0)
    except Exception as e:
        # Minimal error surface for Optuna runner; full trace is in stderr
        print(f"[trainer] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

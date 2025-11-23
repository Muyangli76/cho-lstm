from __future__ import annotations
"""
StrictC50 — Train-from-scratch LSTM with optional UQ, plus a *consistent* learning-curve (LC)
implementation where each k uses:
  • the same fixed validation set,
  • an independent model re-init (same random seed & init snapshot),
  • its own scalers by default (optionally global with --LC_USE_GLOBAL_SCALER),
  • optional warm-start from TWO_L_WEIGHTS (off by default for LC, can be enabled).
Also writes:
  - VAL_* metrics CSVs (flat, by_h, per_run)
  - Old-schema one-row TF/OL summary (VAL_k_summary.csv)
  - LC table (LC_added_runs_metrics_TF_OL.csv)
  - UQ ensemble CSVs (ensemble_val_predictions_with_uncert.csv, ensemble_val_picp_by_h.csv)

Assumes your package layout provides:
  carbonize_df, build_scalers, apply_scaler, Seq2SeqDataset,
  OBS_POOLS, DRIVING, AUX_FEATS, get_aux_cols_present,
  split helpers under cho_lstm_uq.utils.data_split,
  StrictCSeq2Seq model,
  carbon_closure_eps_seq physics,
  eval_tf_metrics metric table generator (TF path).
"""

import argparse, json, copy, os, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from cho_lstm_uq.models.strictc50.data import (
    carbonize_df, build_scalers, apply_scaler, Seq2SeqDataset,
    OBS_POOLS, DRIVING, AUX_FEATS, get_aux_cols_present,
)
from cho_lstm_uq.models.strictc50.model import StrictCSeq2Seq
from cho_lstm_uq.models.strictc50.physics import carbon_closure_eps_seq
from cho_lstm_uq.models.strictc50.metrics import eval_tf_metrics
from cho_lstm_uq.utils.data_split import (
    train_val_split_basic,
    train_val_split_pairs,
    train_val_split_reps_by_index,
    train_val_split_doe_holdout,
    train_val_split_auto,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Utilities
# -------------------------

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_argparser():
    ap = argparse.ArgumentParser(description="StrictC50 — Train LSTM from scratch (with optional UQ) + LC")
    # I/O
    ap.add_argument("--TRAIN_INPUT_CSV", type=str, required=True)
    ap.add_argument("--OUTDIR_TL",       type=str, required=True)
    ap.add_argument("--TWO_L_WEIGHTS",   type=str, default="",
                    help="Optional warm-start checkpoint. For LC, disabled by default unless --LC_WARM_START_TL is set.")
    # Splits
    ap.add_argument("--SPLIT_STRATEGY", type=str, default="auto",
                    choices=["auto","basic","pairs","rep_index","doe_holdout"]) 
    ap.add_argument("--VAL_FRAC", type=float, default=0.2)
    ap.add_argument("--REP_VAL_IDX", type=int, default=3)

    # Core hparams
    ap.add_argument("--SEED", type=int, default=42)
    ap.add_argument("--T_IN", type=int, default=72)
    ap.add_argument("--T_OUT", type=int, default=24)
    ap.add_argument("--HIDDEN", type=int, default=128)
    ap.add_argument("--LAYERS", type=int, default=2)
    ap.add_argument("--DROPOUT", type=float, default=0.1)
    ap.add_argument("--CLIP", type=float, default=1.0)
    ap.add_argument("--BATCH_TR", type=int, default=64)
    ap.add_argument("--BATCH_EVAL", type=int, default=128)
    ap.add_argument("--EPOCHS", type=int, default=20,
                    help="Train-from-scratch epochs for the single full-data run.")
    ap.add_argument("--LR", type=float, default=3e-4)
    ap.add_argument("--PATIENCE", type=int, default=8)

    # Physics / loss weights
    ap.add_argument("--LAMBDA_MB", type=float, default=0.05)
    ap.add_argument("--GAMMA_RES", type=float, default=2e-5)
    ap.add_argument("--LAMBDA_XAB", type=float, default=1.0)
    ap.add_argument("--LAMBDA_CONS", type=float, default=0.2)

    # UQ head
    ap.add_argument("--HET_XAB", action="store_true", default=False)
    ap.add_argument("--SOFTPLUS_VAR", action="store_true", default=False)
    ap.add_argument("--LOGV_MIN", type=float, default=-10.0)
    ap.add_argument("--LOGV_MAX", type=float, default=3.0)

    # Chemistry
    ap.add_argument("--MMOLC_PER_G_BIOMASS", type=float, default=39.6)
    ap.add_argument("--MMOLC_PER_G_MAB",     type=float, default=43.8)
    ap.add_argument("--MW_GLC",  type=float, default=180.156)
    ap.add_argument("--MW_LAC",  type=float, default=90.078)
    ap.add_argument("--CARBON_PER_GLC", type=float, default=6.0)
    ap.add_argument("--CARBON_PER_LAC", type=float, default=3.0)
    ap.add_argument("--DEFAULT_gDCW_PER_CELL", type=float, default=2.8e-12)

    # Learning-curve controls
    ap.add_argument("--LC_ENABLED", action="store_true", default=False)
    ap.add_argument("--LC_USE_GLOBAL_SCALER", action="store_true", default=False,
                    help="If set, LC uses scalers from *all* train runs instead of per-k scalers.")
    ap.add_argument("--LC_WARM_START_TL", action="store_true", default=False,
                    help="If set, also warm-start LC models from TWO_L_WEIGHTS.")
    ap.add_argument("--LC_MAX_K", type=int, default=0,
                    help="Optional cap on k (0 = all runs). Useful for quick debugging.")
    ap.add_argument("--LC_EPOCHS", type=int, default=12,
                    help="Epochs per-k in LC (keep modest to control runtime).")
    ap.add_argument("--LC_LR", type=float, default=3e-4)

    return ap


# -------------------------
# Data / split helpers
# -------------------------

def do_split(df: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if args.SPLIT_STRATEGY == "basic":
        return train_val_split_basic(df, args.VAL_FRAC)
    if args.SPLIT_STRATEGY == "pairs":
        return train_val_split_pairs(df, val_frac=0.5, random_state=args.SEED)
    if args.SPLIT_STRATEGY == "rep_index":
        return train_val_split_reps_by_index(df, rep_val_index=args.REP_VAL_IDX)
    if args.SPLIT_STRATEGY == "doe_holdout":
        return train_val_split_doe_holdout(df, val_frac_doe=0.25, random_state=args.SEED)
    # auto
    return train_val_split_auto(df, default_pairs_val_frac=0.5, rep_val_index=args.REP_VAL_IDX, random_state=args.SEED)


def build_loaders(df_tr, df_va, feat_cols, aux_cols, args):
    mu_all, sd_all = build_scalers(df_tr, feat_cols)
    mu_obs, sd_obs = mu_all[OBS_POOLS], sd_all[OBS_POOLS]
    obs_mu_t = torch.tensor(mu_obs.values, dtype=torch.float32, device=DEVICE)
    obs_sd_t = torch.tensor(sd_obs.values, dtype=torch.float32, device=DEVICE)
    inv_scale_pools = lambda y_sc: obs_mu_t + obs_sd_t * y_sc

    df_tr_s = apply_scaler(df_tr, feat_cols, mu_all, sd_all)
    df_va_s = apply_scaler(df_va, feat_cols, mu_all, sd_all)

    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING
    ds_tr = Seq2SeqDataset(df_tr_s, df_tr[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
    ds_va = Seq2SeqDataset(df_va_s, df_va[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)

    if len(ds_va) == 0:
        raise RuntimeError("Validation dataset produced 0 sequences.")

    dl_tr = DataLoader(ds_tr, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False)

    # For heteroscedastic X/Ab normalization
    mu_xab = torch.tensor(df_tr[["X_gL","Ab_gL"]].mean().values, device=DEVICE, dtype=torch.float32)
    sd_xab = torch.tensor(df_tr[["X_gL","Ab_gL"]].std().replace(0,1.0).values, device=DEVICE, dtype=torch.float32)

    return dl_tr, dl_va, inv_scale_pools, (mu_all, sd_all), (mu_xab, sd_xab)


# -------------------------
# Eval helpers (TF vs OL)
# -------------------------
@torch.no_grad()
def _eval_r2_summary(
    model: nn.Module,
    dl_va: DataLoader,
    inv_scale_pools,
    het_xab: bool,
    mu_xab: Optional[torch.Tensor],
    sd_xab: Optional[torch.Tensor],
    tf_ratio: float,
) -> Dict[str, float]:
    model.eval()
    y_pools_all, yhat_pools_all = [], []
    yX_all, yXhat_all = [], []
    yAb_all, yAbhat_all = [], []

    for batch in dl_va:
        enc_sc, dec_sc, y_sc, *rest = batch
        enc_sc = enc_sc.to(DEVICE); dec_sc = dec_sc.to(DEVICE); y_sc = y_sc.to(DEVICE)
        xab_next = rest[2].to(DEVICE) if len(rest) >= 3 else None
        out = model(enc_sc, dec_sc, y_tf_sc=(y_sc if tf_ratio > 0 else None), tf_ratio=tf_ratio)
        if het_xab:
            pools_sc, _, (mu_sc, lv_sc) = out
            xab_hat_raw = mu_sc * sd_xab + mu_xab
        else:
            pools_sc, _, xab_hat_raw = out
        y_pools_all.append(inv_scale_pools(y_sc).detach().cpu())
        yhat_pools_all.append(inv_scale_pools(pools_sc).detach().cpu())
        yX_all.append(xab_next[...,0].detach().cpu()); yXhat_all.append(xab_hat_raw[...,0].detach().cpu())
        yAb_all.append(xab_next[...,1].detach().cpu()); yAbhat_all.append(xab_hat_raw[...,1].detach().cpu())

    Yp  = torch.cat(y_pools_all, 0).numpy(); Yph = torch.cat(yhat_pools_all, 0).numpy()
    yX  = torch.cat(yX_all, 0).numpy();     yXh = torch.cat(yXhat_all, 0).numpy()
    yAb = torch.cat(yAb_all, 0).numpy();    yAbh= torch.cat(yAbhat_all, 0).numpy()

    def flat(a): return a.reshape(-1)
    r2_pools = [r2_score(flat(Yp[...,k]), flat(Yph[...,k])) for k in range(Yp.shape[-1])]
    R2_avg_pools = float(np.mean(r2_pools)) if r2_pools else float("nan")
    return {
        "R2_avg_pools": R2_avg_pools,
        "R2_X_gL": float(r2_score(flat(yX), flat(yXh))),
        "R2_Ab_gL": float(r2_score(flat(yAb), flat(yAbh))),
    }


# -------------------------
# Train one model (from scratch)
# -------------------------

def train_one_model(model: StrictCSeq2Seq, dl_tr: DataLoader, dl_va: DataLoader,
                    inv_scale_pools, args, mu_xab: torch.Tensor, sd_xab: torch.Tensor) -> None:
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=1e-6)
    best = float("inf"); pat=0

    for ep in range(1, args.EPOCHS+1):
        model.train()
        for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_tr:
            enc_sc = enc_sc.to(DEVICE); dec_sc = dec_sc.to(DEVICE); y_sc = y_sc.to(DEVICE)
            yprev_raw = yprev_raw.to(DEVICE); flows_raw = flows_raw.to(DEVICE); xab_next = xab_next.to(DEVICE)

            outputs = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)  # strict TF during training
            if args.HET_XAB:
                pools_sc, cres_t, (mu_sc, lv_sc) = outputs
                lv_sc = model._stabilize_logv(lv_sc)
                z = (xab_next - mu_xab) / sd_xab
                nll = 0.5 * (lv_sc + (z - mu_sc)**2 / torch.exp(lv_sc))
                loss_xab = nll.mean()
                mu_raw = mu_sc * sd_xab + mu_xab
                X_pred_gL = mu_raw[...,0]; Ab_pred_gL = mu_raw[...,1]
            else:
                pools_sc, cres_t, xab_hat = outputs
                loss_xab = mse(xab_hat, xab_next)
                X_pred_gL = xab_hat[...,0]; Ab_pred_gL = xab_hat[...,1]

            p_next_raw = inv_scale_pools(pools_sc)
            loss_state = mse(pools_sc, y_sc)
            eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
            cons_bio = mse(p_next_raw[...,3],  X_pred_gL * args.MMOLC_PER_G_BIOMASS)
            cons_ab  = mse(p_next_raw[...,4], Ab_pred_gL * args.MMOLC_PER_G_MAB)
            loss_mb  = eps.abs().mean()
            loss_res = (cres_t**2).mean()
            loss = (loss_state
                    + args.LAMBDA_XAB*loss_xab
                    + args.LAMBDA_MB*loss_mb
                    + args.GAMMA_RES*loss_res
                    + args.LAMBDA_CONS*(cons_bio + cons_ab))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.CLIP)
            opt.step()

        # Early-stop on TF val MSE of pools
        model.eval(); tot=0.0; n=0
        with torch.no_grad():
            for enc_sc, dec_sc, y_sc, *_ in dl_va:
                enc_sc=enc_sc.to(DEVICE); dec_sc=dec_sc.to(DEVICE); y_sc=y_sc.to(DEVICE)
                pools_sc, *_ = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                tot += mse(pools_sc, y_sc).item(); n+=1
        val = tot/max(1,n)
        if val < best - 1e-6:
            best = val; pat=0; best_state = copy.deepcopy(model.state_dict())
        else:
            pat+=1
            if pat>=args.PATIENCE: break

    if 'best_state' in locals():
        model.load_state_dict(best_state)


# -------------------------
# UQ ensemble logging (VAL)
# -------------------------
@torch.no_grad()
def log_uq_ensemble(args, out_dir: Path, base_model: StrictCSeq2Seq, dl_va: DataLoader,
                    mu_xab: Optional[torch.Tensor], sd_xab: Optional[torch.Tensor], feat_cols: List[str], n_drv_aux: int):
    uq_dir = out_dir/"UQ"; uq_dir.mkdir(parents=True, exist_ok=True)

    # Save the base checkpoint
    ckpt = out_dir/"strictC_seq2seq_50L_XAb.pt"
    torch.save(base_model.state_dict(), ckpt)

    # Build S copies with identical arch
    def _fresh():
        m = StrictCSeq2Seq(
            in_dim=len(feat_cols), out_pools=len(OBS_POOLS), n_drv_aux=n_drv_aux,
            hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
            het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
            logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX
        ).to(DEVICE)
        sd = torch.load(ckpt, map_location=DEVICE)
        m.load_state_dict(sd, strict=True)
        m.eval(); return m

    SEEDS = [0,1,2]
    models = []
    for s in SEEDS:
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
        models.append(_fresh())

    rows = []
    for batch in dl_va:
        enc_sc, dec_sc, y_sc, *rest = batch
        enc_sc=enc_sc.to(DEVICE); dec_sc=dec_sc.to(DEVICE); y_sc=y_sc.to(DEVICE)
        xab_next = rest[2].to(DEVICE) if len(rest)>=3 else None
        run_ids  = rest[-1] if len(rest)>=4 else None

        MU, VAR = [], []
        for m in models:
            if args.HET_XAB:
                _, _, (mu_sc, lv_sc) = m(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                lv_sc = m._stabilize_logv(lv_sc)
                mu_raw = mu_sc * sd_xab + mu_xab
                var_ale_raw = torch.exp(lv_sc) * (sd_xab**2)
            else:
                _, _, xab_hat = m(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                mu_raw = xab_hat; var_ale_raw = torch.zeros_like(xab_hat)
            MU.append(mu_raw.detach().cpu().numpy()); VAR.append(var_ale_raw.detach().cpu().numpy())
        MU = np.stack(MU); VAR=np.stack(VAR)
        mu_ens=MU.mean(0); var_epi=MU.var(0, ddof=1) if MU.shape[0]>1 else np.zeros_like(mu_ens)
        var_ale=VAR.mean(0); var_tot=var_epi+var_ale

        Y = xab_next.detach().cpu().numpy()
        B,T,_ = mu_ens.shape
        rid = [str(r) for r in run_ids] if run_ids is not None else [f"val_{i}" for i in range(B)]
        for b in range(B):
            for t in range(T):
                rows.append({
                    "run_id": rid[b], "t_idx": int(t),
                    "true_X": float(Y[b,t,0]), "true_Ab": float(Y[b,t,1]),
                    "mu_X": float(mu_ens[b,t,0]), "mu_Ab": float(mu_ens[b,t,1]),
                    "var_ale_X": float(var_ale[b,t,0]), "var_ale_Ab": float(var_ale[b,t,1]),
                    "var_epi_X": float(var_epi[b,t,0]), "var_epi_Ab": float(var_epi[b,t,1]),
                    "var_tot_X": float(var_tot[b,t,0]), "var_tot_Ab": float(var_tot[b,t,1]),
                })

    df_unc = pd.DataFrame(rows)
    f_csv = uq_dir/"ensemble_val_predictions_with_uncert.csv"
    df_unc.to_csv(f_csv, index=False)
    print(f"[UQ] saved → {f_csv}")

    if not df_unc.empty:
        # PICP@95 + mean band width per horizon
        for col in ("var_tot_X","var_tot_Ab"):
            df_unc[f"std_{col[-1]}"] = np.sqrt(np.clip(df_unc[col].values, 0.0, None))
        df_unc["lo95_X"]  = df_unc["mu_X"]  - 1.96*df_unc["std_X"]
        df_unc["hi95_X"]  = df_unc["mu_X"]  + 1.96*df_unc["std_X"]
        df_unc["lo95_Ab"] = df_unc["mu_Ab"] - 1.96*df_unc["std_b"] if "std_b" in df_unc else df_unc["mu_Ab"]
        df_unc["hi95_Ab"] = df_unc["mu_Ab"] + 1.96*df_unc["std_b"] if "std_b" in df_unc else df_unc["mu_Ab"]
        # Fix label names (std_Ab)
        if "std_b" in df_unc:
            df_unc.rename(columns={"std_b":"std_Ab"}, inplace=True)
            df_unc["lo95_Ab"] = df_unc["mu_Ab"] - 1.96*df_unc["std_Ab"]
            df_unc["hi95_Ab"] = df_unc["mu_Ab"] + 1.96*df_unc["std_Ab"]

        df_unc["hit95_X"]  = ((df_unc["true_X"] >= df_unc["lo95_X"]) & (df_unc["true_X"] <= df_unc["hi95_X"]))
        df_unc["hit95_Ab"] = ((df_unc["true_Ab"]>= df_unc["lo95_Ab"])& (df_unc["true_Ab"]<= df_unc["hi95_Ab"]))

        by_h = df_unc.groupby("t_idx").agg(
            PICP95_X=("hit95_X","mean"), PICP95_Ab=("hit95_Ab","mean"),
            loX=("lo95_X","mean"), hiX=("hi95_X","mean"), loAb=("lo95_Ab","mean"), hiAb=("hi95_Ab","mean"),
        ).reset_index()
        by_h["mean_band_X"]  = by_h["hiX"]  - by_h["loX"]
        by_h["mean_band_Ab"] = by_h["hiAb"] - by_h["loAb"]
        by_h.drop(columns=["loX","hiX","loAb","hiAb"], inplace=True)
        f_picp = uq_dir/"ensemble_val_picp_by_h.csv"
        by_h.to_csv(f_picp, index=False)
        print(f"[UQ] saved → {f_picp}")


# -------------------------
# Learning-curve (independent k)
# -------------------------

def run_learning_curve(df_tr, df_va, feat_cols, aux_cols, args, out_dir: Path):
    print("\n>>> Learning curve by added runs (independent model per-k)")
    all_runs = sorted(df_tr["run_id"].astype(str).unique().tolist())
    Kmax = args.LC_MAX_K if args.LC_MAX_K and args.LC_MAX_K>0 else len(all_runs)

    # Base init snapshot for repeatable re-init across k
    set_seed(args.SEED)
    base = StrictCSeq2Seq(
        in_dim=len(feat_cols), out_pools=len(OBS_POOLS), n_drv_aux=len(DRIVING)+len(aux_cols),
        hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
        het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
        logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX
    ).to(DEVICE)
    init_state = copy.deepcopy(base.state_dict())

    rows = []
    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING

    # Optionally precompute global scalers (across *all* train runs)
    if args.LC_USE_GLOBAL_SCALER:
        mu_all_g, sd_all_g = build_scalers(df_tr, feat_cols)

    for k in range(1, Kmax+1):
        use = set(all_runs[:k])
        sub_tr = df_tr[df_tr["run_id"].astype(str).isin(use)].reset_index(drop=True)

        # Scalers: per-k (default) or global
        if args.LC_USE_GLOBAL_SCALER:
            mu_all_k, sd_all_k = mu_all_g, sd_all_g
        else:
            mu_all_k, sd_all_k = build_scalers(sub_tr, feat_cols)
        mu_obs_k, sd_obs_k = mu_all_k[OBS_POOLS], sd_all_k[OBS_POOLS]
        obs_mu_t_k = torch.tensor(mu_obs_k.values, dtype=torch.float32, device=DEVICE)
        obs_sd_t_k = torch.tensor(sd_obs_k.values, dtype=torch.float32, device=DEVICE)
        inv_scale_k = lambda y_sc: obs_mu_t_k + obs_sd_t_k * y_sc

        sub_tr_s  = apply_scaler(sub_tr, feat_cols, mu_all_k, sd_all_k)
        df_va_s_k = apply_scaler(df_va,  feat_cols, mu_all_k, sd_all_k)
        ds_tr_k = Seq2SeqDataset(sub_tr_s, sub_tr[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
        ds_va_k = Seq2SeqDataset(df_va_s_k, df_va[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
        dl_tr_k = DataLoader(ds_tr_k, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
        dl_va_k = DataLoader(ds_va_k, batch_size=args.BATCH_EVAL, shuffle=False)

        # X/Ab normalization
        mu_xab_k = torch.tensor(sub_tr[["X_gL","Ab_gL"]].mean().values, dtype=torch.float32, device=DEVICE)
        sd_xab_k = torch.tensor(sub_tr[["X_gL","Ab_gL"]].std().replace(0,1.0).values, dtype=torch.float32, device=DEVICE)

        # Fresh model from identical init
        model_k = StrictCSeq2Seq(
            in_dim=len(feat_cols), out_pools=len(OBS_POOLS), n_drv_aux=len(DRIVING)+len(aux_cols),
            hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
            het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
            logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX
        ).to(DEVICE)
        model_k.load_state_dict(init_state)

        # Optional warm-start from TL for LC
        if args.LC_WARM_START_TL and args.TWO_L_WEIGHTS and Path(args.TWO_L_WEIGHTS).exists():
            sd_tl = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE); model_k.load_state_dict(sd_tl, strict=False)

        # Train for LC with its own small schedule
        E = args.LC_EPOCHS; LR = args.LC_LR
        mse = nn.MSELoss(); opt = torch.optim.AdamW(model_k.parameters(), lr=LR, weight_decay=1e-6)
        best=float('inf'); pat=0
        for ep in range(1, E+1):
            model_k.train()
            for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_tr_k:
                enc_sc=enc_sc.to(DEVICE); dec_sc=dec_sc.to(DEVICE); y_sc=y_sc.to(DEVICE)
                yprev_raw=yprev_raw.to(DEVICE); flows_raw=flows_raw.to(DEVICE); xab_next=xab_next.to(DEVICE)
                outputs = model_k(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                if args.HET_XAB:
                    pools_sc, cres_t, (mu_sc, lv_sc) = outputs
                    lv_sc = model_k._stabilize_logv(lv_sc)
                    z=(xab_next - mu_xab_k)/sd_xab_k
                    nll=0.5*(lv_sc + (z - mu_sc)**2/torch.exp(lv_sc))
                    loss_xab = nll.mean()
                    mu_raw = mu_sc*sd_xab_k+mu_xab_k
                    X_pred_gL=mu_raw[...,0]; Ab_pred_gL=mu_raw[...,1]
                else:
                    pools_sc, cres_t, xab_hat= outputs
                    loss_xab = mse(xab_hat, xab_next)
                    X_pred_gL=xab_hat[...,0]; Ab_pred_gL=xab_hat[...,1]
                p_raw = inv_scale_k(pools_sc)
                loss_state = mse(pools_sc, y_sc)
                eps = carbon_closure_eps_seq(p_raw, yprev_raw, flows_raw, cres_t)
                cons_bio = mse(p_raw[...,3],  X_pred_gL*args.MMOLC_PER_G_BIOMASS)
                cons_ab  = mse(p_raw[...,4], Ab_pred_gL*args.MMOLC_PER_G_MAB)
                loss = (loss_state + args.LAMBDA_XAB*loss_xab + args.LAMBDA_MB*eps.abs().mean()
                        + args.GAMMA_RES*(cres_t**2).mean() + args.LAMBDA_CONS*(cons_bio+cons_ab))
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model_k.parameters(), args.CLIP); opt.step()
            # early-stop on TF val
            model_k.eval(); tot=0.0; n=0
            with torch.no_grad():
                for enc_sc, dec_sc, y_sc, *_ in dl_va_k:
                    enc_sc=enc_sc.to(DEVICE); dec_sc=dec_sc.to(DEVICE); y_sc=y_sc.to(DEVICE)
                    pools_sc,*_ = model_k(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                    tot += mse(pools_sc,y_sc).item(); n+=1
            val = tot/max(1,n)
            if val < best - 1e-6:
                best=val; pat=0; best_state = copy.deepcopy(model_k.state_dict())
            else:
                pat+=1
                if pat>=max(3, args.PATIENCE//2): break
        if 'best_state' in locals(): model_k.load_state_dict(best_state)

        # Evaluate TF & OL
        tf_scores = _eval_r2_summary(model_k, dl_va_k, inv_scale_k, args.HET_XAB, mu_xab_k, sd_xab_k, tf_ratio=1.0)
        ol_scores = _eval_r2_summary(model_k, dl_va_k, inv_scale_k, args.HET_XAB, mu_xab_k, sd_xab_k, tf_ratio=1e-6)
        rows.append({
            "k_runs": k,
            "runs_used_json": json.dumps(sorted(list(use))),
            "TF_R2_avg_pools": tf_scores["R2_avg_pools"],
            "TF_R2_X_gL": tf_scores["R2_X_gL"],
            "TF_R2_Ab_gL": tf_scores["R2_Ab_gL"],
            "OL_R2_avg_pools": ol_scores["R2_avg_pools"],
            "OL_R2_X_gL": ol_scores["R2_X_gL"],
            "OL_R2_Ab_gL": ol_scores["R2_Ab_gL"],
        })
        print(f"  k={k:02d} | TF R²[X]={tf_scores['R2_X_gL']:.3f}  TF R²[Ab]={tf_scores['R2_Ab_gL']:.3f} | "
              f"OL R²[X]={ol_scores['R2_X_gL']:.3f}  OL R²[Ab]={ol_scores['R2_Ab_gL']:.3f}")

    lc_csv = out_dir/"LC_added_runs_metrics_TF_OL.csv"
    pd.DataFrame(rows).to_csv(lc_csv, index=False)
    print(f"Saved learning-curve table → {lc_csv}")


# -------------------------
# Main training flow
# -------------------------

def run_train(args) -> int:
    set_seed(args.SEED)
    out = Path(args.OUTDIR_TL); out.mkdir(parents=True, exist_ok=True)

    # Load & carbonize
    df = pd.read_csv(args.TRAIN_INPUT_CSV)
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"] = df["batch_name"].astype(str)
    df = carbonize_df(df, args)

    # Feature selection
    aux_cols = get_aux_cols_present(df)
    FEATURE_ORDER = [
        # pools (5)
        "GlcC","LacC","DIC_mmolC_L","BioC","ProdC",
        # drivers (3)
        "Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h",
        # AUX (0–3)
        "pCO2","V_L","vvd_per_day",
    ]
    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]
    required = set(["GlcC","LacC","DIC_mmolC_L","BioC","ProdC","Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h"])
    miss = sorted(required - set(feat_cols))
    if miss: raise RuntimeError(f"Missing required features: {miss}. Found: {feat_cols}")

    # Length filter
    need_len = args.T_IN + args.T_OUT
    sizes = df.groupby("run_id").size()
    keep = sizes.index[sizes >= need_len]
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)
    if df.empty: raise RuntimeError("No runs meet T_IN+T_OUT.")

    # Split
    df_tr, df_va, val_runs = do_split(df, args)
    if len(val_runs)==0 or df_va.empty:
        unique_runs = sorted(df["run_id"].astype(str).unique())
        if len(unique_runs) < 2:
            raise RuntimeError("Validation split empty and <2 runs; add data or reduce T_IN/T_OUT.")
        forced = [unique_runs[-1]]
        df_tr = df[~df["run_id"].isin(forced)].copy()
        df_va = df[df["run_id"].isin(forced)].copy()
        val_runs = forced
        print("[split] fallback applied: forced last run into VAL →", val_runs)
    print(f"[split] strategy={args.SPLIT_STRATEGY} | VAL runs: {val_runs}")

    # Loaders/scalers
    dl_tr, dl_va, inv_scale_pools, (mu_all, sd_all), (mu_xab, sd_xab) = build_loaders(df_tr, df_va, feat_cols, aux_cols, args)

    # Model (from scratch; no freeze stages)
    model = StrictCSeq2Seq(
        in_dim=len(feat_cols), out_pools=len(OBS_POOLS), n_drv_aux=len(DRIVING)+len(aux_cols),
        hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
        het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
        logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX
    ).to(DEVICE)

    # Optional warm-start for the full-data run (does not affect LC unless flag set there)
    if args.TWO_L_WEIGHTS and Path(args.TWO_L_WEIGHTS).exists():
        sd = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)

    # Train
    train_one_model(model, dl_tr, dl_va, inv_scale_pools, args, mu_xab, sd_xab)

    # Final VAL metrics
    if args.HET_XAB:
        pf, xf, pbh, xbh, pr = eval_tf_metrics(dl_va, model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"], DEVICE,
                                               het_xab=True, mu_xab=mu_xab, sd_xab=sd_xab)
    else:
        pf, xf, pbh, xbh, pr = eval_tf_metrics(dl_va, model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"], DEVICE)

    pd.DataFrame(pf).to_csv(out/"VAL_pools_flat.csv", index=False)
    pd.DataFrame(xf).to_csv(out/"VAL_xab_flat.csv", index=False)
    pd.DataFrame(pbh).to_csv(out/"VAL_pools_by_h.csv", index=False)
    pd.DataFrame(xbh).to_csv(out/"VAL_xab_by_h.csv", index=False)
    pd.DataFrame(pr).to_csv(out/"VAL_per_run_flat.csv", index=False)

    # Save checkpoint + scalers
    torch.save(model.state_dict(), out/"strictC_seq2seq_50L_XAb.pt")
    with open(out/"scaler_features.json","w") as f:
        json.dump({"mu_all": mu_all.to_dict(), "sd_all": sd_all.to_dict(), "feat_cols": feat_cols}, f)
    if args.HET_XAB:
        with open(out/"scaler_xab.json","w") as f:
            json.dump({"mu_xab": mu_xab.detach().cpu().numpy().tolist(),
                       "sd_xab": sd_xab.detach().cpu().numpy().tolist()}, f)

    # One-row TF vs OL summary (old schema)
    tf_scores = _eval_r2_summary(model, dl_va, inv_scale_pools, args.HET_XAB, mu_xab if args.HET_XAB else None,
                                 sd_xab if args.HET_XAB else None, tf_ratio=1.0)
    ol_scores = _eval_r2_summary(model, dl_va, inv_scale_pools, args.HET_XAB, mu_xab if args.HET_XAB else None,
                                 sd_xab if args.HET_XAB else None, tf_ratio=1e-6)
    row = {
        "k_runs": len(sorted(df_tr["run_id"].astype(str).unique())),
        "runs_used_json": json.dumps(sorted(list(val_runs))),  # VAL runs
        "TF_R2_avg_pools": tf_scores["R2_avg_pools"],
        "TF_R2_X_gL": tf_scores["R2_X_gL"],
        "TF_R2_Ab_gL": tf_scores["R2_Ab_gL"],
        "OL_R2_avg_pools": ol_scores["R2_avg_pools"],
        "OL_R2_X_gL": ol_scores["R2_X_gL"],
        "OL_R2_Ab_gL": ol_scores["R2_Ab_gL"],
        "split_strategy": args.SPLIT_STRATEGY,
        "seed": args.SEED,
        "T_IN": args.T_IN,
        "T_OUT": args.T_OUT,
        "timestamp": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "outdir": str(out),
    }
    out_summary = out/"VAL_k_summary.csv"
    pd.DataFrame([row]).to_csv(out_summary, index=False)
    print(f"[summary] wrote old-schema row → {out_summary}")

    snap = out/"LC_snapshot_metrics_TF_OL.csv"
    if snap.exists(): pd.DataFrame([row]).to_csv(snap, mode="a", header=False, index=False)
    else: pd.DataFrame([row]).to_csv(snap, index=False)
    print(f"[summary] appended → {snap}")

    # UQ ensemble outputs (only if head enabled)
    if args.HET_XAB:
        log_uq_ensemble(args, out, model, dl_va, mu_xab, sd_xab, feat_cols, len(DRIVING)+len(aux_cols))

    # Learning curve
    if args.LC_ENABLED:
        run_learning_curve(df_tr, df_va, feat_cols, aux_cols, args, out)

    print(f"[train] Saved artifacts to {out}")
    return 0


def main():
    ap = build_argparser(); args = ap.parse_args()
    # Resolve paths
    args.TRAIN_INPUT_CSV = str(Path(args.TRAIN_INPUT_CSV).resolve())
    args.OUTDIR_TL       = str(Path(args.OUTDIR_TL).resolve())
    if args.TWO_L_WEIGHTS: args.TWO_L_WEIGHTS = str(Path(args.TWO_L_WEIGHTS).resolve())
    ec = run_train(args)
    raise SystemExit(ec)


if __name__ == "__main__":
    main()

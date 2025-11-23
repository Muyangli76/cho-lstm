#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse, json, math, os, sys, random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------------------------------------------
# Device / seeds
# -------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# -------------------------------------------------------------------
# Feature definitions (aligned with your seq2seq file)
# -------------------------------------------------------------------

OBS_POOLS = ["GlcC","LacC","DIC_mmolC_L","BioC","ProdC"]
DRIVING   = ["Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h"]
AUX_FEATS = ["pCO2","V_L","vvd_per_day"]  # if present

def guess_units_is_gL(s: pd.Series)->bool:
    x = pd.to_numeric(s, errors='coerce').dropna()
    return (not x.empty) and (float(x.mean()) > 5.0)

def get_aux_cols_present(df: pd.DataFrame)->List[str]:
    return [c for c in AUX_FEATS if c in df.columns]

def build_scalers(train_df: pd.DataFrame, cols: List[str]):
    mu = train_df[cols].mean()
    sd = train_df[cols].std().replace(0, 1.0)
    return mu, sd

def apply_scaler(df: pd.DataFrame, cols: List[str], mu: pd.Series, sd: pd.Series):
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

# -------------------------------------------------------------------
# Carbonization (copied / simplified from your seq2seq file)
# -------------------------------------------------------------------

def carbonize_df(df: pd.DataFrame, args) -> pd.DataFrame:
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
        cells_per_L = df["C_X_raw_cells_per_ml"].astype(float) * 1e3
        X_gL = cells_per_L * args.DEFAULT_gDCW_PER_CELL
    elif "C_X" in df:
        # same heuristic as your seq2seq
        x = df["C_X"].astype(float)
        if x.mean() < 5.0:
            X_gL = x
        else:
            X_gL = x * 1e6 * args.DEFAULT_gDCW_PER_CELL
    else:
        X_gL = 0.0
    df["X_gL"] = X_gL
    df["BioC"] = df["X_gL"] * args.MMOLC_PER_G_BIOMASS

    # Product → mmol C/L; keep Ab_gL (assume g/L)
    if "C_Ab" in df:
        df["Ab_gL"] = df["C_Ab"].astype(float)
    elif "Ab_gL" not in df:
        df["Ab_gL"] = 0.0
    df["ProdC"] = df["Ab_gL"] * args.MMOLC_PER_G_MAB

    # Id columns + dt
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"] = df["batch_name"].astype(str)
    if "time_h" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "time_h"})
    df = df.sort_values(["run_id", "time_h"])
    df["dt"] = df.groupby("run_id")["time_h"].diff().fillna(0.0)
    df.loc[df["dt"] <= 0, "dt"] = 1.0

    # basic engineering
    if "V_L" not in df and "V" in df:
        df["V_L"] = df["V"] / 1000.0
    if "vvd_per_day" not in df and "V_L" in df:
        V = df["V_L"].to_numpy()
        Vp = np.roll(V, 1); Vp[0] = V[0]
        vvd = 24.0 * (V - Vp) / np.maximum(Vp, 1e-12)
        vvd[0] = 0.0
        df["vvd_per_day"] = vvd
    if "Fin_over_V_1ph" not in df and "vvd_per_day" in df:
        df["Fin_over_V_1ph"] = df["vvd_per_day"] / 24.0
    if "CinC_mmolC_L" not in df and "feed_glucose" in df:
        df["CinC_mmolC_L"] = 6.0 * df["feed_glucose"]
    if "DIC_mmolC_L" not in df and "pCO2" in df:
        df["DIC_mmolC_L"] = 1.0 * df["pCO2"]

    # Alias for CTR
    if "CTR_mmolC_L_h" not in df.columns and "CTR_calc_mmolC_L_h" in df.columns:
        df["CTR_mmolC_L_h"] = df["CTR_calc_mmolC_L_h"].astype(float)

    # Final required columns
    need = OBS_POOLS + DRIVING
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in need if c in df.columns])
    return df

# -------------------------------------------------------------------
# Dataset: vanilla LSTM (many-to-one, final horizon)
# -------------------------------------------------------------------

class VanillaLSTMDataset(Dataset):
    """
    Each sample:
      - input: scaled features [T_IN, in_dim]
      - target: [2] = (X_gL, Ab_gL) at horizon t = t0 + T_IN + T_OUT - 1
    We slide over time within each run with stride=1 (like your seq2seq).
    """
    def __init__(self, df_scaled: pd.DataFrame, df_raw: pd.DataFrame,
                 feat_cols: List[str], t_in: int, t_out: int, stride: int = 1):
        self.feat_cols = feat_cols
        self.t_in = t_in
        self.t_out = t_out
        self.samples = []

        for rid, gs in df_scaled.groupby("run_id"):
            gs = gs.sort_values("time_h").reset_index(drop=True)
            gr = df_raw[df_raw["run_id"] == rid].sort_values("time_h").reset_index(drop=True)

            if len(gs) < (t_in + t_out):
                continue

            X_all = gs[feat_cols].values.astype(np.float32)
            Xg = gr["X_gL"].values.astype(np.float32)
            Ab = gr["Ab_gL"].values.astype(np.float32)

            T = len(gs)
            for t0 in range(0, T - t_in - t_out + 1, stride):
                t1 = t0 + t_in
                t2 = t1 + t_out
                x_seq = X_all[t0:t1]                  # [T_IN, in_dim]
                # target at last horizon step:
                y_tgt = np.array([Xg[t2 - 1], Ab[t2 - 1]], dtype=np.float32)
                self.samples.append((x_seq, y_tgt, str(rid)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_seq, y_tgt, rid = self.samples[idx]
        return (torch.from_numpy(x_seq), torch.from_numpy(y_tgt), rid)

# -------------------------------------------------------------------
# Model: vanilla many-to-one LSTM regressor for X & Ab
# -------------------------------------------------------------------

class VanillaLSTMRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 2)  # -> [X_gL, Ab_gL]

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T_IN, in_dim]
        out, (h, c) = self.lstm(x_seq)
        # use last time step of top layer
        h_last = out[:, -1, :]  # [B, hidden]
        y_hat = self.head(h_last)  # [B, 2]
        return y_hat

# -------------------------------------------------------------------
# Train / eval
# -------------------------------------------------------------------

def eval_metrics(dl_va: DataLoader, model: nn.Module, device: str = DEVICE) -> Dict[str, float]:
    model.eval()
    Ys, Yh = [], []
    R = []
    with torch.no_grad():
        for x_seq, y_tgt, rid in dl_va:
            x_seq = x_seq.to(device)
            y_tgt = y_tgt.to(device)
            y_hat = model(x_seq)
            Ys.append(y_tgt.cpu().numpy())
            Yh.append(y_hat.cpu().numpy())
            R.extend(list(rid))

    if len(Ys) == 0:
        return dict(R2_X_gL=float("nan"), R2_Ab_gL=float("nan"))

    Ys = np.concatenate(Ys, axis=0)  # [N, 2]
    Yh = np.concatenate(Yh, axis=0)  # [N, 2]

    r2_x  = float(r2_score(Ys[:, 0], Yh[:, 0]))
    r2_ab = float(r2_score(Ys[:, 1], Yh[:, 1]))
    rmse_x  = float(rmse(Ys[:, 0], Yh[:, 0]))
    rmse_ab = float(rmse(Ys[:, 1], Yh[:, 1]))

    print(f"[VAL] R² X_gL={r2_x:.3f}  Ab_gL={r2_ab:.3f} | RMSE X={rmse_x:.3f} Ab={rmse_ab:.3f}")
    return dict(R2_X_gL=r2_x, R2_Ab_gL=r2_ab,
                RMSE_X_gL=rmse_x, RMSE_Ab_gL=rmse_ab)

# -------------------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------------------

def run_baseline(args) -> None:
    set_seed(args.SEED)

    out = Path(args.OUTDIR)
    out.mkdir(parents=True, exist_ok=True)

    # ----------------- Load + carbonize -----------------
    df = pd.read_csv(args.TRAIN_INPUT_CSV)
    df = carbonize_df(df, args)

    # Min length filter
    sizes = df.groupby("run_id").size()
    need_len = args.T_IN + args.T_OUT
    keep = sizes.index[sizes >= need_len]
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No runs meet minimum length T_IN+T_OUT.")

    # ----------------- Split (reuse your split helpers) -----------------
    from cho_lstm_uq.utils.data_split import (
        train_val_split_basic,
        train_val_split_pairs,
        train_val_split_reps_by_index,
        train_val_split_doe_holdout,
        train_val_split_auto,
    )

    if args.SPLIT_STRATEGY == "basic":
        split_fn = lambda d: train_val_split_basic(d, getattr(args, "VAL_FRAC", 0.2))
    elif args.SPLIT_STRATEGY == "pairs":
        split_fn = lambda d: train_val_split_pairs(d, val_frac=0.5, random_state=args.SEED)
    elif args.SPLIT_STRATEGY == "rep_index":
        split_fn = lambda d: train_val_split_reps_by_index(d, rep_val_index=getattr(args, "REP_VAL_IDX", 3))
    elif args.SPLIT_STRATEGY == "doe_holdout":
        split_fn = lambda d: train_val_split_doe_holdout(d, val_frac_doe=0.25, random_state=args.SEED)
    elif args.SPLIT_STRATEGY == "auto":
        split_fn = lambda d: train_val_split_auto(
            d,
            default_pairs_val_frac=0.5,
            rep_val_index=getattr(args, "REP_VAL_IDX", 3),
            random_state=args.SEED,
        )
    else:
        # fallback to auto
        split_fn = lambda d: train_val_split_auto(
            d,
            default_pairs_val_frac=0.5,
            rep_val_index=getattr(args, "REP_VAL_IDX", 3),
            random_state=args.SEED,
        )

    df_tr, df_va, val_runs = split_fn(df)
    if len(val_runs) == 0:
        raise RuntimeError("Validation split is empty.")
    print(f"[split] strategy={args.SPLIT_STRATEGY} | VAL runs: {val_runs}")

    # ----------------- Features + scalers -----------------
    aux_cols = get_aux_cols_present(df)
    FEATURE_ORDER = [
        # pools
        "GlcC", "LacC", "DIC_mmolC_L", "BioC", "ProdC",
        # drivers
        "Fin_over_V_1ph", "CinC_mmolC_L", "CTR_mmolC_L_h",
        # aux
        "pCO2", "V_L", "vvd_per_day",
    ]
    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]

    required = {
        "GlcC","LacC","DIC_mmolC_L","BioC","ProdC",
        "Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h",
    }
    missing_req = sorted(required - set(feat_cols))
    if missing_req:
        raise RuntimeError(f"Missing required features: {missing_req}. Found: {feat_cols}")

    mu_all, sd_all = build_scalers(df_tr, feat_cols)
    df_tr_s = apply_scaler(df_tr, feat_cols, mu_all, sd_all)
    df_va_s = apply_scaler(df_va, feat_cols, mu_all, sd_all)

    raw_cols = ["run_id", "time_h", "dt", "X_gL", "Ab_gL"] + OBS_POOLS + DRIVING

    ds_tr = VanillaLSTMDataset(df_tr_s, df_tr[raw_cols].copy(), feat_cols,
                               t_in=args.T_IN, t_out=args.T_OUT, stride=1)
    ds_va = VanillaLSTMDataset(df_va_s, df_va[raw_cols].copy(), feat_cols,
                               t_in=args.T_IN, t_out=args.T_OUT, stride=1)

    if len(ds_va) == 0:
        raise RuntimeError("Validation dataset produced 0 sequences.")

    dl_tr = DataLoader(ds_tr, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False)

    print(f"[data] train samples={len(ds_tr)} | val samples={len(ds_va)}")

    # ----------------- Build model -----------------
    model = VanillaLSTMRegressor(
        in_dim=len(feat_cols),
        hidden=args.HIDDEN,
        layers=args.LAYERS,
        dropout=args.DROPOUT,
    ).to(DEVICE)

    # (Optional) vanilla TL from 2L weights if you ever train a matching LSTM on 2L.
    # For now, we just leave this as a hook — safe even if file doesn't exist.
    if getattr(args, "TWO_L_WEIGHTS", None) and Path(args.TWO_L_WEIGHTS).exists():
        try:
            sd = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[warm-start] Loaded 2L weights | missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[warm-start] Could not load TWO_L_WEIGHTS ({e}); training from scratch.")
    else:
        print("[warm-start] TWO_L_WEIGHTS not provided / missing; training from scratch.")

    # ----------------- Train -----------------
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=1e-6)

    best_val = float("inf")
    best_state = None
    patience = 0

    for ep in range(1, args.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_seq, y_tgt, _ in dl_tr:
            x_seq = x_seq.to(DEVICE)
            y_tgt = y_tgt.to(DEVICE)

            y_hat = model(x_seq)
            loss = mse(y_hat, y_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.CLIP)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_tr = total_loss / max(1, n_batches)
        print(f"[epoch {ep:03d}] train MSE={avg_tr:.5f}")

        # VAL MSE for early stopping
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for x_seq, y_tgt, _ in dl_va:
                x_seq = x_seq.to(DEVICE)
                y_tgt = y_tgt.to(DEVICE)
                y_hat = model(x_seq)
                vtot += mse(y_hat, y_tgt).item()
                vn += 1
        vavg = vtot / max(1, vn)
        print(f"           val   MSE={vavg:.5f}")

        if vavg < best_val - 1e-6:
            best_val = vavg
            patience = 0
            best_state = {k: v.detach().cpu().clone()
                          if torch.is_tensor(v) else v
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.PATIENCE:
                print("[early stop] patience reached.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----------------- Final VAL metrics + save -----------------
    scores = eval_metrics(dl_va, model, DEVICE)

    # Save checkpoint + scalers + summary
    ckpt_path = out / "vanilla_lstm_XAb.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[save] checkpoint → {ckpt_path}")

    with open(out / "scaler_features.json", "w") as f:
        json.dump({"mu_all": mu_all.to_dict(),
                   "sd_all": sd_all.to_dict(),
                   "feat_cols": feat_cols}, f)

    row = {
        "split_strategy": args.SPLIT_STRATEGY,
        "seed": args.SEED,
        "T_IN": args.T_IN,
        "T_OUT": args.T_OUT,
        "R2_X_gL": scores["R2_X_gL"],
        "R2_Ab_gL": scores["R2_Ab_gL"],
        "RMSE_X_gL": scores["RMSE_X_gL"],
        "RMSE_Ab_gL": scores["RMSE_Ab_gL"],
    }
    pd.DataFrame([row]).to_csv(out / "VAL_summary_vanilla_XAb.csv", index=False)
    print(f"[save] VAL summary → {out/'VAL_summary_vanilla_XAb.csv'}")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Vanilla LSTM baseline for X_gL & Ab_gL (many-to-one).")

    # Paths
    ap.add_argument("--TRAIN_INPUT_CSV", type=str, required=True,
                    help="50L cleaned / carbonizable CSV.")
    ap.add_argument("--OUTDIR", type=str, required=True,
                    help="Output directory for checkpoint + metrics.")
    ap.add_argument("--TWO_L_WEIGHTS", type=str, default="",
                    help="(Optional) 2L LSTM weights for warm-start (if you ever train them).")

    # Chemistry / constants (match your TL script)
    ap.add_argument("--MMOLC_PER_G_BIOMASS", type=float, default=39.6)
    ap.add_argument("--MMOLC_PER_G_MAB",     type=float, default=43.8)
    ap.add_argument("--MW_GLC",  type=float, default=180.156)
    ap.add_argument("--MW_LAC",  type=float, default=90.078)
    ap.add_argument("--CARBON_PER_GLC", type=float, default=6.0)
    ap.add_argument("--CARBON_PER_LAC", type=float, default=3.0)
    ap.add_argument("--DEFAULT_gDCW_PER_CELL", type=float, default=2.8e-12)

    # Split / model hyperparams
    ap.add_argument("--SPLIT_STRATEGY",
                    type=str, default="doe_holdout",
                    choices=["auto","basic","pairs","rep_index","doe_holdout"])
    ap.add_argument("--REP_VAL_IDX", type=int, default=3)
    ap.add_argument("--VAL_FRAC", type=float, default=0.25)

    ap.add_argument("--SEED", type=int, default=42)
    ap.add_argument("--T_IN", type=int, default=48)
    ap.add_argument("--T_OUT", type=int, default=12)

    ap.add_argument("--HIDDEN", type=int, default=128)
    ap.add_argument("--LAYERS", type=int, default=2)
    ap.add_argument("--DROPOUT", type=float, default=0.10)
    ap.add_argument("--CLIP", type=float, default=1.0)
    ap.add_argument("--BATCH_TR", type=int, default=64)
    ap.add_argument("--BATCH_EVAL", type=int, default=128)
    ap.add_argument("--EPOCHS", type=int, default=40)
    ap.add_argument("--PATIENCE", type=int, default=8)
    ap.add_argument("--LR", type=float, default=5e-4)

    return ap

def main():
    try:
        ap = build_argparser()
        args = ap.parse_args()

        args.TRAIN_INPUT_CSV = str(Path(args.TRAIN_INPUT_CSV).resolve())
        args.OUTDIR          = str(Path(args.OUTDIR).resolve())
        if args.TWO_L_WEIGHTS:
            args.TWO_L_WEIGHTS = str(Path(args.TWO_L_WEIGHTS).resolve())

        if not Path(args.TRAIN_INPUT_CSV).exists():
            raise FileNotFoundError(f"TRAIN_INPUT_CSV not found: {args.TRAIN_INPUT_CSV}")

        run_baseline(args)
        sys.exit(0)
    except Exception as e:
        print(f"[vanilla_lstm] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

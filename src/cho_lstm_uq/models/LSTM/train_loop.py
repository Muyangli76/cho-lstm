"""
Train loop + UQ ensemble + learning-curve driver.
Run:
  python -m cho_lstm_uq.models.LSTM.train_loop --TRAIN_INPUT_CSV <cleaned.csv> --TRAIN_OUTDIR <outdir>
"""
from __future__ import annotations
import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cho_lstm_uq.models.LSTM import seq2seq_C as s2c

OBS_POOLS   = s2c.OBS_POOLS
DRIVING     = s2c.DRIVING
AUX_FEATS   = s2c.AUX_FEATS
carbonize_df = s2c.carbonize_df
build_scalers = s2c.build_scalers
apply_scaler  = s2c.apply_scaler
get_aux_cols_present = s2c.get_aux_cols_present
Seq2SeqDataset = s2c.Seq2SeqDataset
StrictCSeq2Seq = s2c.StrictCSeq2Seq
carbon_closure_eps_seq = s2c.carbon_closure_eps_seq
eval_seq_metrics       = s2c.eval_seq_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------ Defaults ------------------------
DEFAULTS = dict(
    # data
    TRAIN_INPUT_CSV="",
    TRAIN_OUTDIR="runs/strictC_2L",
    VAL_FRAC=0.2,
    # model
    HIDDEN=128, LAYERS=2, DROPOUT=0.1,
    T_IN=12, T_OUT=12,
    HET_XAB=True, LOGV_MIN=-10.0, LOGV_MAX=3.0, SOFTPLUS_VAR=True,
    # training
    SEED=42, EPOCHS=60, LC_EPOCHS=8,
    BATCH_TR=64, BATCH_EVAL=128,
    TF_START=1.0, TF_END=0.6,
    CLIP=1.0, PATIENCE=10,
    LAMBDA_MB=0.8, GAMMA_RES=0.1, LAMBDA_XAB=1.0, LAMBDA_CONS=0.5,
    NONNEG_W=1e-3,
    # chemistry constants
    MW_GLC=180.156, MW_LAC=90.078,
    CARBON_PER_GLC=6.0, CARBON_PER_LAC=3.0,
    DEFAULT_gDCW_PER_CELL=3.0e-12,
    MMOLC_PER_G_BIOMASS=45.0,   # tune as needed
    MMOLC_PER_G_MAB=132.0,      # tune as needed
    # ensemble
    ENSEMBLE_K=5, ENSEMBLE_BOOTSTRAP_FRAC=0.85,
)


# ------------------------ Utilities ------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_split(df, val_frac=0.2):
    runs = sorted(df["run_id"].unique())
    n_val = max(1, int(len(runs) * val_frac))
    val_runs = set(runs[-n_val:])
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, list(val_runs)


# --------------------------- Train One --------------------------
def train_one(df_train, df_val, feat_cols, obs_mu_t, obs_sd_t, cfg):
    raw_cols = ["run_id", "time_h", "dt", "X_gL", "Ab_gL"] + OBS_POOLS + DRIVING
    ds_tr = Seq2SeqDataset(apply_scaler(df_train, feat_cols, cfg._mu_all, cfg._sd_all),
                           df_train[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                           cfg.T_IN, cfg.T_OUT, stride=1)
    ds_va = Seq2SeqDataset(apply_scaler(df_val, feat_cols, cfg._mu_all, cfg._sd_all),
                           df_val[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                           cfg.T_IN, cfg.T_OUT, stride=1)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_TR, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.BATCH_EVAL, shuffle=False)

    in_dim = len(feat_cols)
    n_drv_aux = len(DRIVING) + len(cfg._aux_cols)
    model = StrictCSeq2Seq(in_dim, out_pools=len(OBS_POOLS), n_drv_aux=n_drv_aux,
                           hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
                           het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
                           logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)
    mse = nn.MSELoss()

    mu_xab = torch.tensor(df_train[["X_gL", "Ab_gL"]].mean().values, device=DEVICE, dtype=torch.float32)
    sd_xab = torch.tensor(df_train[["X_gL", "Ab_gL"]].std().replace(0, 1.0).values, device=DEVICE, dtype=torch.float32)

    def inv_scale_pools(y_sc): return obs_mu_t + obs_sd_t * y_sc

    best_val = float("inf"); patience = 0
    mb_warm_ep = max(1, int(0.4 * cfg.EPOCHS))
    nonneg_w = float(getattr(cfg, "NONNEG_W", 1e-3))

    for ep in range(1, cfg.EPOCHS + 1):
        tf_ratio = cfg.TF_END + (cfg.TF_START - cfg.TF_END) * max(0.0, 1.0 - (ep - 1) / (cfg.EPOCHS * 0.4))
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
                nll = 0.5 * (lv_sc + (xab_sc - mu_sc) ** 2 / var_sc)
                loss_xab = nll.mean()
                mu_raw = mu_sc * sd_xab + mu_xab
                X_pred_gL = mu_raw[..., 0]
                Ab_pred_gL = mu_raw[..., 1]
            else:
                pools_sc, cres_t, xab_hat = out
                p_next_raw = inv_scale_pools(pools_sc)
                eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                loss_state = mse(pools_sc, y_sc)
                loss_xab = mse(xab_hat, xab_next)
                X_pred_gL, Ab_pred_gL = xab_hat[..., 0], xab_hat[..., 1]

            BioC_pred = p_next_raw[..., 3]
            ProdC_pred = p_next_raw[..., 4]
            cons_bio = mse(BioC_pred, X_pred_gL * cfg.MMOLC_PER_G_BIOMASS)
            cons_ab = mse(ProdC_pred, Ab_pred_gL * cfg.MMOLC_PER_G_MAB)

            loss_mb = eps.abs().mean()
            loss_res = (cres_t ** 2).mean()
            loss_nonneg = (p_next_raw.clamp(max=0.0) ** 2).mean() * nonneg_w

            loss = (loss_state
                    + lam_mb * loss_mb
                    + cfg.GAMMA_RES * loss_res
                    + cfg.LAMBDA_XAB * loss_xab
                    + cfg.LAMBDA_CONS * (cons_bio + cons_ab)
                    + loss_nonneg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
            opt.step()

        # early stop on val state loss
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                enc_sc, dec_sc, y_sc, *_ = batch
                enc_sc = enc_sc.to(DEVICE); dec_sc = dec_sc.to(DEVICE); y_sc = y_sc.to(DEVICE)
                pools_sc, *_ = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                tot += nn.functional.mse_loss(pools_sc, y_sc).item(); n += 1
        v = tot / max(1, n)
        if v < best_val - 1e-6:
            best_val = v; patience = 0
            best_state = {k: v.cpu().clone() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                break

    if "best_state" in locals():
        model.load_state_dict(best_state)

    # Save X/Ab scalers for het head
    if cfg.HET_XAB:
        scal = {"mu_xab": mu_xab.detach().cpu().numpy().tolist(),
                "sd_xab": sd_xab.detach().cpu().numpy().tolist()}
        with open(os.path.join(cfg.TRAIN_OUTDIR, "scaler_xab.json"), "w") as f:
            json.dump(scal, f)

    dl_va_eval = DataLoader(ds_va, batch_size=cfg.BATCH_EVAL, shuffle=False)
    pools_tf, xab_tf = eval_seq_metrics(
        dl_va_eval, model, inv_scale_pools=inv_scale_pools,
        obs_names=OBS_POOLS, xab_names=["X_gL", "Ab_gL"],
        device=DEVICE, het_xab=cfg.HET_XAB, mu_xab=mu_xab, sd_xab=sd_xab, tf_ratio=1.0
    )
    return model, pools_tf, xab_tf, ds_tr, ds_va, mu_xab, sd_xab, (obs_mu_t, obs_sd_t)


# --------------------------- Orchestrator --------------------------
def run_train(cfg):
    os.makedirs(cfg.TRAIN_OUTDIR, exist_ok=True)
    set_seed(cfg.SEED)

    print(">>> Strict-C Seq2Seq — drivers-in-decoder + het X/Ab")
    df = pd.read_csv(cfg.TRAIN_INPUT_CSV)
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"] = df["batch_name"].astype(str)
    df = carbonize_df(df, cfg)

    # keep only long-enough runs
    need_len = cfg.T_IN + cfg.T_OUT
    sizes = df.groupby("run_id").size()
    keep = sizes.index[sizes >= need_len]
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)

    aux_cols = get_aux_cols_present(df)
    feat_cols = OBS_POOLS + DRIVING + aux_cols
    print(f"Aux present: {aux_cols}")

    df_tr, df_va, val_runs = train_val_split(df, val_frac=cfg.VAL_FRAC)
    mu_all, sd_all = build_scalers(df_tr, feat_cols)
    mu_obs, sd_obs = mu_all[OBS_POOLS], sd_all[OBS_POOLS]
    cfg._mu_all, cfg._sd_all, cfg._aux_cols = mu_all, sd_all, aux_cols

    # cast floats
    cols_to_cast = list({*feat_cols, "X_gL", "Ab_gL", *OBS_POOLS, *DRIVING} & set(df.columns))
    df[cols_to_cast] = df[cols_to_cast].astype(np.float32)

    obs_mu_t = torch.tensor(mu_obs.values, dtype=torch.float32, device=DEVICE)
    obs_sd_t = torch.tensor(sd_obs.values, dtype=torch.float32, device=DEVICE)

    model, pools_tf, xab_tf, ds_tr, ds_va, mu_xab, sd_xab, (obs_mu_t, obs_sd_t) = \
        train_one(df_tr, df_va, feat_cols, obs_mu_t, obs_sd_t, cfg)

    pd.DataFrame(pools_tf).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_pools_TF.csv"), index=False)
    pd.DataFrame(xab_tf).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_xab_TF.csv"), index=False)

    # open-loop
    raw_cols = ["run_id", "time_h", "dt", "X_gL", "Ab_gL"] + OBS_POOLS + DRIVING
    ds_va_open = Seq2SeqDataset(apply_scaler(df_va, feat_cols, mu_all, sd_all),
                                df_va[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                                cfg.T_IN, cfg.T_OUT, stride=1)
    dl_va_open = DataLoader(ds_va_open, batch_size=cfg.BATCH_EVAL, shuffle=False)

    def inv_scale(y_sc): return obs_mu_t + obs_sd_t * y_sc

    pools_ol, xab_ol = eval_seq_metrics(
        dl_va_open, model, inv_scale, OBS_POOLS, ["X_gL", "Ab_gL"],
        DEVICE, cfg.HET_XAB, mu_xab=mu_xab, sd_xab=sd_xab, tf_ratio=0.0
    )
    pd.DataFrame(pools_ol).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_pools_OL.csv"), index=False)
    pd.DataFrame(xab_ol).to_csv(os.path.join(cfg.TRAIN_OUTDIR, "VAL_xab_OL.csv"), index=False)

    # save weights + scalers
    torch.save(model.state_dict(), os.path.join(cfg.TRAIN_OUTDIR, "strictC_seq2seq_50L_XAb.pt"))
    with open(os.path.join(cfg.TRAIN_OUTDIR, "scaler_features.json"), "w") as f:
        json.dump({"mu_all": mu_all.to_dict(), "sd_all": sd_all.to_dict(), "feat_cols": feat_cols}, f)

    print("Saved to:", cfg.TRAIN_OUTDIR)

    # ----------------- Ensemble UQ -----------------
    uq_dir = Path(cfg.TRAIN_OUTDIR) / "UQ"
    uq_dir.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_K = int(getattr(cfg, "ENSEMBLE_K", 5))
    BOOT_FRAC = float(getattr(cfg, "ENSEMBLE_BOOTSTRAP_FRAC", 0.85))

    ds_va2 = Seq2SeqDataset(apply_scaler(df_va, feat_cols, mu_all, sd_all),
                            df_va[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                            cfg.T_IN, cfg.T_OUT, stride=1)
    dl_va_eval = DataLoader(ds_va2, batch_size=cfg.BATCH_EVAL, shuffle=False)

    member_paths = []
    all_runs = df_tr["run_id"].unique()
    n_boot = max(1, int(len(all_runs) * BOOT_FRAC))

    def train_member_from_scratch(seed: int, df_tr_member: pd.DataFrame):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        obs_mu_t_local = torch.tensor(mu_all[OBS_POOLS].values, dtype=torch.float32, device=DEVICE)
        obs_sd_t_local = torch.tensor(sd_all[OBS_POOLS].values, dtype=torch.float32, device=DEVICE)
        cfg._mu_all, cfg._sd_all = mu_all, sd_all
        model_m, *_ = train_one(df_tr_member, df_va, feat_cols, obs_mu_t_local, obs_sd_t_local, cfg)
        return model_m

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
                n_drv_aux=len(DRIVING) + len(cfg._aux_cols),
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

            MU = np.stack(MU_list)         # (S,B,T,2)
            VAR = np.stack(VAR_list)       # (S,B,T,2)
            mu_ens = MU.mean(0) * sd_xab.cpu().numpy() + mu_xab.cpu().numpy()
            var_epi = MU.var(0, ddof=1) * (sd_xab.cpu().numpy() ** 2)
            var_alea = VAR.mean(0) * (sd_xab.cpu().numpy() ** 2)
            var_tot = var_epi + var_alea
            Y = xab_next.numpy()

            B, T_out = mu_ens.shape[0], mu_ens.shape[1]
            rids = [rid] if isinstance(rid, str) else list(rid)
            for b in range(B):
                rid_b = rids[b] if isinstance(rids, list) else rid
                for t in range(T_out):
                    rows.append({
                        "run_id": str(rid_b), "t_idx": int(t),
                        "true_X": float(Y[b, t, 0]), "true_Ab": float(Y[b, t, 1]),
                        "mu_X": float(mu_ens[b, t, 0]), "mu_Ab": float(mu_ens[b, t, 1]),
                        "var_ale_X": float(var_alea[b, t, 0]), "var_ale_Ab": float(var_alea[b, t, 1]),
                        "var_epi_X": float(var_epi[b, t, 0]),  "var_epi_Ab": float(var_epi[b, t, 1]),
                        "var_tot_X": float(var_tot[b, t, 0]),  "var_tot_Ab": float(var_tot[b, t, 1]),
                    })

        dfE = pd.DataFrame(rows)
        dfE.to_csv(out_csv, index=False)
        print(f"[ensemble] saved → {out_csv}")

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        for tag, tcol, mu_col in [("X_gL", "true_X", "mu_X"), ("Ab_gL", "true_Ab", "mu_Ab")]:
            mae = mean_absolute_error(dfE[tcol], dfE[mu_col])
            rmse = mean_squared_error(dfE[tcol], dfE[mu_col], squared=False)
            r2 = r2_score(dfE[tcol], dfE[mu_col])
            print(f"{tag}: MAE {mae:.3f} | RMSE {rmse:.3f} | R² {r2:.3f}")

    out_ens = (Path(cfg.TRAIN_OUTDIR) / "UQ" / "VAL_UQ_ensemble.csv").as_posix()
    ensemble_predict_to_csv(member_paths, dl_va_eval, out_ens)

    # ----------------- Learning curve -----------------
    print("\n>>> Learning curve (validation TF & OL)")
    ds_va2 = Seq2SeqDataset(apply_scaler(df_va, feat_cols, mu_all, sd_all),
                            df_va[raw_cols].copy(), OBS_POOLS, DRIVING, cfg._aux_cols,
                            cfg.T_IN, cfg.T_OUT, stride=1)
    dl_va_lc = DataLoader(ds_va2, batch_size=cfg.BATCH_EVAL, shuffle=False)

    def inv_scale_lc(y_sc: torch.Tensor) -> torch.Tensor:
        return obs_mu_t + obs_sd_t * y_sc

    def tiny_train(model, dl, mu_xab=None, sd_xab=None, epochs=cfg.LC_EPOCHS, lr=3e-3, tf_start=1.0, tf_end=0.6):
        mse = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        model.train()
        for ep in range(1, epochs + 1):
            tf_ratio = tf_end + (tf_start - tf_end) * max(0.0, 1.0 - (ep - 1) / max(1, epochs * 0.5))
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
                    nll = 0.5 * (lv_sc + (xab_sc - mu_sc) ** 2 / torch.exp(lv_sc))
                    loss_xab = nll.mean()
                    mu_raw = mu_sc * sd_xab + mu_xab
                    X_pred_gL, Ab_pred_gL = mu_raw[..., 0], mu_raw[..., 1]
                else:
                    pools_sc, cres_t, xab_hat = out
                    p_next_raw = inv_scale_lc(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                    loss_state = mse(pools_sc, y_sc)
                    loss_xab = mse(xab_hat, xab_next)
                    X_pred_gL, Ab_pred_gL = xab_hat[..., 0], xab_hat[..., 1]

                cons_bio = mse(p_next_raw[..., 3], X_pred_gL * cfg.MMOLC_PER_G_BIOMASS)
                cons_ab = mse(p_next_raw[..., 4], Ab_pred_gL * cfg.MMOLC_PER_G_MAB)
                loss = (loss_state
                        + 0.5 * cfg.LAMBDA_MB * eps.abs().mean()
                        + 0.5 * cfg.GAMMA_RES * (cres_t ** 2).mean()
                        + 0.2 * cfg.LAMBDA_XAB * loss_xab
                        + 0.2 * cfg.LAMBDA_CONS * (cons_bio + cons_ab))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
                opt.step()

    all_train_runs = sorted(df_tr["run_id"].unique())
    rows_lc = []

    init_model = StrictCSeq2Seq(
        in_dim=len(feat_cols), out_pools=len(OBS_POOLS),
        n_drv_aux=len(DRIVING) + len(cfg._aux_cols),
        hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
        het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
        logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR
    ).to(DEVICE)
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)
    init_state = deepcopy(init_model.state_dict())

    for k in range(1, len(all_train_runs) + 1):
        use_runs = set(all_train_runs[:k])
        sub_tr = df_tr[df_tr["run_id"].isin(use_runs)].reset_index(drop=True)
        sub_tr_s = apply_scaler(sub_tr, feat_cols, mu_all, sd_all)
        sub_tr_r = sub_tr[["run_id", "time_h", "dt", "X_gL", "Ab_gL"] + OBS_POOLS + DRIVING].copy()

        ds_k = Seq2SeqDataset(sub_tr_s, sub_tr_r, OBS_POOLS, DRIVING, cfg._aux_cols, cfg.T_IN, cfg.T_OUT, stride=1)
        dl_k = DataLoader(ds_k, batch_size=cfg.BATCH_TR, shuffle=True, drop_last=True)

        mu_xab_k = torch.tensor(sub_tr[["X_gL", "Ab_gL"]].mean().values, dtype=torch.float32, device=DEVICE)
        sd_xab_k = torch.tensor(sub_tr[["X_gL", "Ab_gL"]].std().replace(0, 1.0).values, dtype=torch.float32, device=DEVICE)

        model_k = StrictCSeq2Seq(
            in_dim=len(feat_cols), out_pools=len(OBS_POOLS),
            n_drv_aux=len(DRIVING) + len(cfg._aux_cols),
            hidden=cfg.HIDDEN, layers=cfg.LAYERS, dropout=cfg.DROPOUT,
            het_xab=cfg.HET_XAB, logv_min=cfg.LOGV_MIN,
            logv_max=cfg.LOGV_MAX, softplus_var=cfg.SOFTPLUS_VAR
        ).to(DEVICE)
        model_k.load_state_dict(init_state)

        tiny_train(model_k, dl_k, mu_xab=(mu_xab_k if cfg.HET_XAB else None),
                   sd_xab=(sd_xab_k if cfg.HET_XAB else None))

        pools_tf_k, xab_tf_k = eval_seq_metrics(
            dl_va_lc, model_k, inv_scale_lc, OBS_POOLS, ["X_gL", "Ab_gL"],
            DEVICE, cfg.HET_XAB, mu_xab=mu_xab_k, sd_xab=sd_xab_k, tf_ratio=1.0
        )
        pools_ol_k, xab_ol_k = eval_seq_metrics(
            dl_va_lc, model_k, inv_scale_lc, OBS_POOLS, ["X_gL", "Ab_gL"],
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
class C: ...
def make_cfg_from_args(args):
    c = C()
    for k, v in DEFAULTS.items():
        setattr(c, k, v)
    for k, v in vars(args).items():
        if v is None:
            continue
        setattr(c, k, v)
    return c


def main():
    p = argparse.ArgumentParser(description="Strict-C 2L pipeline (train + UQ + LC)")
    p.add_argument("--TRAIN_INPUT_CSV", type=str)
    p.add_argument("--TRAIN_OUTDIR", type=str)
    # hparams
    for k in [
        "SEED", "EPOCHS", "BATCH_TR", "BATCH_EVAL", "T_IN", "T_OUT", "TF_START", "TF_END", "CLIP", "PATIENCE",
        "LAMBDA_MB", "GAMMA_RES", "LAMBDA_XAB", "LAMBDA_CONS", "HIDDEN", "LAYERS", "DROPOUT", "LC_EPOCHS", "VAL_FRAC"
    ]:
        p.add_argument(f"--{k}", type=type(DEFAULTS[k]))
    # uncertainty flags
    p.add_argument("--HET_XAB", type=bool)
    p.add_argument("--LOGV_MIN", type=float)
    p.add_argument("--LOGV_MAX", type=float)
    p.add_argument("--SOFTPLUS_VAR", type=bool)
    # ensemble knobs
    p.add_argument("--ENSEMBLE_K", type=int, default=DEFAULTS["ENSEMBLE_K"])
    p.add_argument("--ENSEMBLE_BOOTSTRAP_FRAC", type=float, default=DEFAULTS["ENSEMBLE_BOOTSTRAP_FRAC"])

    args = p.parse_args()
    cfg = make_cfg_from_args(args)
    run_train(cfg)


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        sys.argv = [""]  # allow running in notebooks
    main()

# cho_lstm_uq/models/strictc50/train_lstm.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import random
from sklearn.metrics import r2_score



from .data import (
    carbonize_df, build_scalers, apply_scaler, Seq2SeqDataset,
    OBS_POOLS, DRIVING, AUX_FEATS, train_val_split, get_aux_cols_present
)
from .model import StrictCSeq2Seq
from .physics import carbon_closure_eps_seq
from .metrics import eval_tf_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_argparser():
    ap = argparse.ArgumentParser(description="StrictC50 — Train LSTM only (no UQ).")
    # I/O
    ap.add_argument("--TRAIN_INPUT_CSV", type=str, required=True)
    ap.add_argument("--TWO_L_WEIGHTS",   type=str, required=False, default="")
    ap.add_argument("--OUTDIR_TL",       type=str, required=True)
    # Core HParams
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
    ap.add_argument("--VAL_FRAC", type=float, default=0.2)

    # Data Split
    ap.add_argument("--SPLIT_STRATEGY",
                type=str, default="auto",
                choices=["auto","basic","pairs","rep_index","doe_holdout"])
    ap.add_argument("--REP_VAL_IDX", type=int, default=3,
                help="When using rep_index (or auto with ≥3 reps), which _repN to place in VAL.")
    # Loss weights / physics
    ap.add_argument("--LAMBDA_MB",   type=float, default=0.05)
    ap.add_argument("--GAMMA_RES",   type=float, default=2e-5)
    ap.add_argument("--LAMBDA_XAB",  type=float, default=1.0)
    ap.add_argument("--LAMBDA_CONS", type=float, default=0.2)

    # TL stages
    ap.add_argument("--FREEZE_EPOCHS_HEAD", type=int, default=3)
    ap.add_argument("--FREEZE_EPOCHS_TOP",  type=int, default=4)
    ap.add_argument("--POLISH_EPOCHS", type=int, default=2)
    ap.add_argument("--LR_STAGE_A", type=float, default=3e-4)
    ap.add_argument("--LR_STAGE_B", type=float, default=3e-4)
    ap.add_argument("--LR_STAGE_C", type=float, default=5e-5)
    ap.add_argument("--L2SP_ALPHA", type=float, default=1e-4)

    # Uncertainty head (kept in model; not doing ensemble here)
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
    return ap

@torch.no_grad()
def _log_uq_ensemble(
    args,
    out_dir: Path,
    base_ckpt_path: Path,
    ds_tr,
    ds_va,
    feat_cols,
    n_drv_aux: int,
    mu_xab: torch.Tensor | None,
    sd_xab: torch.Tensor | None,
    device: str = "cpu",
):
    """
    Writes:
      - out_dir / "UQ" / "ensemble_val_predictions_with_uncert.csv"
      - out_dir / "UQ" / "ensemble_val_picp_by_h.csv"
    Schema matches your previous table:
      run_id,t_idx,true_X,true_Ab,mu_X,mu_Ab,var_ale_X,var_ale_Ab,var_epi_X,var_epi_Ab,var_tot_X,var_tot_Ab
    """
    uq_dir = out_dir / "UQ"
    uq_dir.mkdir(parents=True, exist_ok=True)

    # Build a model constructor that loads the saved TL/LSTM checkpoint
    def _fresh_model():
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
        ).to(device)
        sd = torch.load(base_ckpt_path, map_location=device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        return m

    # tiny ensemble for epistemic variance
    SEEDS = [0, 1, 2]   # adjust if you want
    models = []
    for s in SEEDS:
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        models.append(_fresh_model())

    # Sanity: we need X/Ab scalers if we’ll unscale UQ
    if args.HET_XAB:
        assert mu_xab is not None and sd_xab is not None, "HET_XAB=True but mu_xab/sd_xab not set."
        mu_xab_t = mu_xab.to(device)
        sd_xab_t = sd_xab.to(device)

    rows = []
    for batch in DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False):
        # Dataset signature: enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, meta
        if len(batch) == 6:
            enc_sc, dec_sc, y_sc, _, _, xab_next = batch
            meta = None
        else:
            enc_sc, dec_sc, y_sc, _, _, xab_next, meta = batch

        enc_sc = enc_sc.to(device)
        dec_sc = dec_sc.to(device)
        y_sc   = y_sc.to(device)
        xab_next = xab_next.to(device)  # raw g/L targets for X, Ab

        MU_raw_list = []
        VAR_ale_raw_list = []  # exp(lv) in raw units

        for m in models:
            if args.HET_XAB:
                _, _, (mu_sc, lv_sc) = m(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                # stabilize and map to RAW
                lv_sc = m._stabilize_logv(lv_sc)
                mu_raw  = mu_sc * sd_xab_t + mu_xab_t
                var_ale_raw = torch.exp(lv_sc) * (sd_xab_t ** 2)
            else:
                # If no heteroscedastic head, we still want an ensemble epistemic var.
                _, _, xab_hat = m(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                mu_raw = xab_hat  # already in raw (your head outputs raw for no-UQ)
                var_ale_raw = torch.zeros_like(mu_raw)

            MU_raw_list.append(mu_raw.detach().cpu().numpy())
            VAR_ale_raw_list.append(var_ale_raw.detach().cpu().numpy())

        MU  = np.stack(MU_raw_list)       # (S,B,T,2) raw g/L
        VAR = np.stack(VAR_ale_raw_list)  # (S,B,T,2) raw g/L^2

        mu_ens   = MU.mean(0)                    # (B,T,2)
        var_epi  = MU.var(0, ddof=1) if MU.shape[0] > 1 else np.zeros_like(mu_ens)
        var_ale  = VAR.mean(0)                   # aleatoric (raw)
        var_tot  = var_epi + var_ale
        Y        = xab_next.detach().cpu().numpy()  # (B,T,2)

        # meta: try to get run_id list; otherwise synthesize
        B, T = mu_ens.shape[0], mu_ens.shape[1]
        run_ids = None
        if isinstance(meta, dict) and "run_id" in meta:
            run_ids = meta["run_id"]
            # ensure list of strings length B
            if isinstance(run_ids, torch.Tensor):
                run_ids = [str(x) for x in run_ids]
        if run_ids is None:
            run_ids = [f"val_run_{i}" for i in range(B)]

        # write rows
        for b in range(B):
            rid = str(run_ids[b])
            for t in range(T):
                rows.append({
                    "run_id":     rid,
                    "t_idx":      int(t),
                    "true_X":     float(Y[b, t, 0]),
                    "true_Ab":    float(Y[b, t, 1]),
                    "mu_X":       float(mu_ens[b, t, 0]),
                    "mu_Ab":      float(mu_ens[b, t, 1]),
                    "var_ale_X":  float(var_ale[b, t, 0]),
                    "var_ale_Ab": float(var_ale[b, t, 1]),
                    "var_epi_X":  float(var_epi[b, t, 0]),
                    "var_epi_Ab": float(var_epi[b, t, 1]),
                    "var_tot_X":  float(var_tot[b, t, 0]),
                    "var_tot_Ab": float(var_tot[b, t, 1]),
                })

    df_unc = pd.DataFrame(rows)
    out_csv = uq_dir / "ensemble_val_predictions_with_uncert.csv"
    df_unc.to_csv(out_csv, index=False)
    print(f"[UQ] saved → {out_csv}")

    # PICP@95 + mean band width (per horizon)
    # NOTE: assume Normal approx; 95% interval via 1.96*sqrt(var_tot).
    # If you prefer aleatoric-only, swap var_tot for var_ale.
    if not df_unc.empty:
        # compute bounds
        df_unc["std_X"]  = np.sqrt(np.clip(df_unc["var_tot_X"].values, 0.0, None))
        df_unc["std_Ab"] = np.sqrt(np.clip(df_unc["var_tot_Ab"].values, 0.0, None))
        df_unc["lo95_X"]  = df_unc["mu_X"]  - 1.96 * df_unc["std_X"]
        df_unc["hi95_X"]  = df_unc["mu_X"]  + 1.96 * df_unc["std_X"]
        df_unc["lo95_Ab"] = df_unc["mu_Ab"] - 1.96 * df_unc["std_Ab"]
        df_unc["hi95_Ab"] = df_unc["mu_Ab"] + 1.96 * df_unc["std_Ab"]

        df_unc["hit95_X"]  = ((df_unc["true_X"]  >= df_unc["lo95_X"])  & (df_unc["true_X"]  <= df_unc["hi95_X"])).astype(float)
        df_unc["hit95_Ab"] = ((df_unc["true_Ab"] >= df_unc["lo95_Ab"]) & (df_unc["true_Ab"] <= df_unc["hi95_Ab"])).astype(float)

        by_h = df_unc.groupby("t_idx").agg(
            PICP95_X=("hit95_X", "mean"),
            PICP95_Ab=("hit95_Ab", "mean"),
            mean_band_X=("hi95_X", "mean"),
            mean_band_Ab=("hi95_Ab", "mean"),
        ).reset_index()

        # correct band widths to be hi - lo
        bw_X  = (df_unc.groupby("t_idx")["hi95_X"].mean()  - df_unc.groupby("t_idx")["lo95_X"].mean()).values
        bw_Ab = (df_unc.groupby("t_idx")["hi95_Ab"].mean() - df_unc.groupby("t_idx")["lo95_Ab"].mean()).values
        by_h["mean_band_X"]  = bw_X
        by_h["mean_band_Ab"] = bw_Ab

        out_picp = uq_dir / "ensemble_val_picp_by_h.csv"
        by_h.to_csv(out_picp, index=False)
        print(f"[UQ] saved → {out_picp}")


# --- add these helpers anywhere above run_train (NEW) ---
@torch.no_grad()
def _eval_r2_summary(
    model,
    dl_va: DataLoader,
    inv_scale_pools,           # maps scaled pools -> raw pools (tensor op)
    het_xab: bool,
    mu_xab: torch.Tensor | None,
    sd_xab: torch.Tensor | None,
    device: str = "cpu",
    tf_ratio: float = 1.0,     # 1.0 = TF, 0.0 = OL
):
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
        # enc_sc, dec_sc, y_sc (scaled pools), yprev_raw, flows_raw, xab_next (raw), ...
        if len(batch) >= 6:
            enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next = batch[:6]
        else:
            raise RuntimeError("VAL dataloader did not return the expected 6-tuple.")

        enc_sc = enc_sc.to(device)
        dec_sc = dec_sc.to(device)
        y_sc   = y_sc.to(device)
        xab_next = xab_next.to(device)     # raw g/L targets

        # forward; TF vs OL controlled by tf_ratio
        out = model(enc_sc, dec_sc,
                    y_tf_sc=(y_sc if tf_ratio > 0 else None),
                    tf_ratio=tf_ratio)

        if het_xab:
            pools_sc, cres_t, (mu_sc, lv_sc) = out
            # X/Ab predictions (raw) via unscaling mu
            mu_raw = mu_sc * sd_xab + mu_xab
            xab_hat_raw = mu_raw
        else:
            pools_sc, cres_t, xab_hat_raw = out  # xab head already raw in your non-UQ path

        # pools to raw
        p_next_raw = inv_scale_pools(pools_sc)

        # collect
        y_pools_all.append(y_sc.detach().cpu())                # still scaled here
        yhat_pools_all.append(pools_sc.detach().cpu())         # scaled preds; we’ll unscale next
        # but for R² we want raw; convert both:
        # (do conversion on the fly to avoid double memory)
        y_pools_all[-1]    = (inv_scale_pools(y_pools_all[-1].to(device))).cpu()
        yhat_pools_all[-1] = (inv_scale_pools(yhat_pools_all[-1].to(device))).cpu()

        yX_all.append(xab_next[..., 0].detach().cpu())
        yAb_all.append(xab_next[..., 1].detach().cpu())
        yXhat_all.append(xab_hat_raw[..., 0].detach().cpu())
        yAbhat_all.append(xab_hat_raw[..., 1].detach().cpu())

    # stack
    Yp  = torch.cat(y_pools_all, dim=0).numpy()      # [N, T_out, 5]
    Yph = torch.cat(yhat_pools_all, dim=0).numpy()
    yX  = torch.cat(yX_all, dim=0).numpy()           # [N, T_out]
    yXh = torch.cat(yXhat_all, dim=0).numpy()
    yAb = torch.cat(yAb_all, dim=0).numpy()
    yAbh= torch.cat(yAbhat_all, dim=0).numpy()

    # flatten across batch & horizon
    def flat(a): return a.reshape(-1)

    # pools: mean R² across the 5 observed pools in OBS_POOLS order
    r2_pools = []
    for k in range(Yp.shape[-1]):
        r2_pools.append(r2_score(flat(Yp[..., k]), flat(Yph[..., k])))
    R2_avg_pools = float(np.mean(r2_pools)) if len(r2_pools) else float("nan")

    R2_X_gL  = float(r2_score(flat(yX),  flat(yXh)))
    R2_Ab_gL = float(r2_score(flat(yAb), flat(yAbh)))

    return dict(R2_avg_pools=R2_avg_pools, R2_X_gL=R2_X_gL, R2_Ab_gL=R2_Ab_gL)


def set_seed(seed:int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def run_train(args):
    set_seed(args.SEED)
    out = Path(args.OUTDIR_TL)
    out.mkdir(parents=True, exist_ok=True)

    # ------------ Load + carbonize ------------
    df = pd.read_csv(args.TRAIN_INPUT_CSV)
    if "run_id" not in df.columns and "batch_name" in df.columns:
        df["run_id"] = df["batch_name"].astype(str)
    df = carbonize_df(df, args)

    # ------------ Feature selection (AUX optional) ------------
    aux_cols = get_aux_cols_present(df)  # subset of ["pCO2","V_L","vvd_per_day"]
    FEATURE_ORDER = [
        # pools (required, 5)
        "GlcC","LacC","DIC_mmolC_L","BioC","ProdC",
        # drivers (required, 3)
        "Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h",
        # AUX (optional, 0–3)
        "pCO2","V_L","vvd_per_day",
    ]
    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]

    required = {
        "GlcC","LacC","DIC_mmolC_L","BioC","ProdC",
        "Fin_over_V_1ph","CinC_mmolC_L","CTR_mmolC_L_h",
    }
    missing_req = sorted(required - set(feat_cols))
    if missing_req:
        raise RuntimeError(f"Missing required features: {missing_req}. Found: {feat_cols}")

    # ------------ Length filter ------------
    need_len = args.T_IN + args.T_OUT
    sizes = df.groupby("run_id").size()
    keep_runs = sizes.index[sizes >= need_len]
    df = df[df["run_id"].isin(keep_runs)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No runs meet T_IN+T_OUT.")

    # ---------- choose split strategy ----------
    from cho_lstm_uq.utils.data_split import (
        train_val_split_basic,
        train_val_split_pairs,
        train_val_split_reps_by_index,
        train_val_split_doe_holdout,
        train_val_split_auto,
    )

    def do_split(df, args):
        if args.SPLIT_STRATEGY == "basic":
            return train_val_split_basic(df, getattr(args, "VAL_FRAC", 0.2))
        elif args.SPLIT_STRATEGY == "pairs":
            return train_val_split_pairs(df, val_frac=0.5, random_state=args.SEED)
        elif args.SPLIT_STRATEGY == "rep_index":
            rep_idx = getattr(args, "REP_VAL_IDX", 3)
            return train_val_split_reps_by_index(df, rep_val_index=rep_idx)
        elif args.SPLIT_STRATEGY == "doe_holdout":
            return train_val_split_doe_holdout(df, val_frac_doe=0.25, random_state=args.SEED)
        else:  # "auto"
            return train_val_split_auto(
                df,
                default_pairs_val_frac=0.5,
                rep_val_index=getattr(args, "REP_VAL_IDX", 3),
                random_state=args.SEED,
            )

    # ---------- split (with diagnostics + safety fallback) ----------
    df_tr, df_va, val_runs = do_split(df, args)

    if len(val_runs) == 0 or df_va.empty:
        # Diagnostics
        runs = df["run_id"].astype(str)
        print("[split][diag] unique runs:", runs.nunique())
        print("[split][diag] counts per run_id (top 10):")
        print(runs.value_counts().head(10).to_string())
        # Safety fallback: force last run into VAL
        unique_runs = sorted(runs.unique())
        if len(unique_runs) < 2:
            raise RuntimeError(
                "Validation split is empty and there is fewer than 2 runs after length filtering. "
                "Add more data or reduce T_IN/T_OUT."
            )
        forced = [unique_runs[-1]]
        df_tr = df[~df["run_id"].isin(forced)].copy()
        df_va = df[df["run_id"].isin(forced)].copy()
        val_runs = forced
        print("[split] fallback applied: forced last run into VAL →", val_runs)

    print(f"[split] strategy={args.SPLIT_STRATEGY} | VAL runs: {val_runs}")

    # ------------ Scale ------------
    mu_all, sd_all = build_scalers(df_tr, feat_cols)
    mu_obs, sd_obs = mu_all[OBS_POOLS], sd_all[OBS_POOLS]
    obs_mu_t = torch.tensor(mu_obs.values, dtype=torch.float32, device=DEVICE)
    obs_sd_t = torch.tensor(sd_obs.values, dtype=torch.float32, device=DEVICE)
    inv_scale_pools = lambda y_sc: obs_mu_t + obs_sd_t * y_sc

    df_tr_s = apply_scaler(df_tr, feat_cols, mu_all, sd_all)
    df_va_s = apply_scaler(df_va, feat_cols, mu_all, sd_all)

    # ------------ Datasets / loaders ------------
    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING
    ds_tr = Seq2SeqDataset(df_tr_s, df_tr[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
    ds_va = Seq2SeqDataset(df_va_s, df_va[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
    dl_tr = DataLoader(ds_tr, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False)

    # --- which runs are being evaluated (match old "k/runs_used_json" idea)
    # val_runs comes from: df_tr, df_va, val_runs = do_split(df, args)
    val_runs_list = [str(r) for r in val_runs]   # ensure plain list[str]
    k_runs = len(val_runs_list)
    runs_used_json = json.dumps(val_runs_list)

    # ------------ Model ------------
    in_dim = len(feat_cols)
    n_drv_aux = len(DRIVING) + len(aux_cols)
    model = StrictCSeq2Seq(
        in_dim, out_pools=len(OBS_POOLS), n_drv_aux=n_drv_aux,
        hidden=args.HIDDEN, layers=args.LAYERS, dropout=args.DROPOUT,
        het_xab=args.HET_XAB, softplus_var=args.SOFTPLUS_VAR,
        logv_min=args.LOGV_MIN, logv_max=args.LOGV_MAX
    ).to(DEVICE)

    # Optional warm-start (same-scale)
    if args.TWO_L_WEIGHTS and Path(args.TWO_L_WEIGHTS).exists():
        sd = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)

    mse = nn.MSELoss()
    if args.HET_XAB:
        mu_xab = torch.tensor(df_tr[["X_gL","Ab_gL"]].mean().values, device=DEVICE, dtype=torch.float32)
        sd_xab = torch.tensor(df_tr[["X_gL","Ab_gL"]].std().replace(0,1.0).values, device=DEVICE, dtype=torch.float32)

    def l2sp_term(theta0):
        if not args.L2SP_ALPHA or not theta0:
            return 0.0
        pen = 0.0
        for (n, p) in model.named_parameters():
            if (not p.requires_grad) or (n not in theta0) or n.endswith(".bias"):
                continue
            pen = pen + torch.sum((p - theta0[n].to(p.device))**2)
        return args.L2SP_ALPHA * pen

    def train_stage(epochs, lr, lamb_mb, lamb_cons, theta0=None):
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-6)
        best = float("inf"); pat = 0
        for ep in range(1, epochs + 1):
            model.train()
            tf_ratio = 1.0
            for enc_sc, dec_sc, y_sc, yprev_raw, flows_raw, xab_next, _ in dl_tr:
                enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)
                yprev_raw, flows_raw, xab_next = yprev_raw.to(DEVICE), flows_raw.to(DEVICE), xab_next.to(DEVICE)

                out = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=tf_ratio)
                if args.HET_XAB:
                    pools_sc, cres_t, (mu_sc, lv_sc) = out
                    lv_sc = model._stabilize_logv(lv_sc)
                    p_next_raw = inv_scale_pools(pools_sc)
                    eps = carbon_closure_eps_seq(p_next_raw, yprev_raw, flows_raw, cres_t)
                    loss_state = mse(pools_sc, y_sc)
                    z = (xab_next - mu_xab) / sd_xab
                    nll = 0.5 * (lv_sc + (z - mu_sc)**2 / torch.exp(lv_sc))
                    loss_xab = nll.mean()
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
                ramp = 0.5 + 0.5 * min(1.0, ep / max(1, epochs))
                loss_mb  = eps.abs().mean()
                loss_res = (cres_t**2).mean()
                loss = (loss_state
                        + args.LAMBDA_XAB * loss_xab
                        + lamb_mb * loss_mb
                        + args.GAMMA_RES * loss_res
                        + (lamb_cons * ramp) * (cons_bio + cons_ab)
                        + l2sp_term(theta0))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.CLIP)
                opt.step()

            # TF validation
            model.eval(); tot = 0.0; n = 0
            with torch.no_grad():
                for enc_sc, dec_sc, y_sc, *_ in dl_va:
                    enc_sc, dec_sc, y_sc = enc_sc.to(DEVICE), dec_sc.to(DEVICE), y_sc.to(DEVICE)
                    pools_sc, *_ = model(enc_sc, dec_sc, y_tf_sc=y_sc, tf_ratio=1.0)
                    tot += mse(pools_sc, y_sc).item(); n += 1
            val = tot / max(1, n)
            if val < best - 1e-6:
                best = val; pat = 0
                best_state = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                              for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= args.PATIENCE:
                    break
        if 'best_state' in locals():
            model.load_state_dict(best_state)

    # ------------ Stage A/B/C ------------
    for p in model.parameters(): p.requires_grad = False
    for p in model.head_pools.parameters(): p.requires_grad = True
    for p in model.head_cres.parameters():  p.requires_grad = True
    if args.HET_XAB:
        for p in model.head_xab_mu.parameters(): p.requires_grad = True
        for p in model.head_xab_lv.parameters(): p.requires_grad = True
    else:
        for p in model.head_xab.parameters():    p.requires_grad = True
    L = model.dec.num_layers - 1
    for name, p in model.dec.named_parameters():
        if f"_l{L}" in name: p.requires_grad = True
    train_stage(args.FREEZE_EPOCHS_HEAD, args.LR_STAGE_A, args.LAMBDA_MB * 0.5, args.LAMBDA_CONS * 0.5)

    for name, p in model.enc.named_parameters():
        if f"_l{L}" in name: p.requires_grad = True
    train_stage(args.FREEZE_EPOCHS_TOP, args.LR_STAGE_B, args.LAMBDA_MB, args.LAMBDA_CONS)

    for p in model.parameters(): p.requires_grad = True
    train_stage(args.POLISH_EPOCHS, args.LR_STAGE_C, args.LAMBDA_MB, args.LAMBDA_CONS)

    # ------------ Final VAL metrics + save ------------
    if args.HET_XAB:
        pf, xf, pbh, xbh, pr = eval_tf_metrics(
            DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False),
            model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"], DEVICE,
            het_xab=True, mu_xab=mu_xab, sd_xab=sd_xab
        )
    else:
        pf, xf, pbh, xbh, pr = eval_tf_metrics(
            DataLoader(ds_va, batch_size=args.BATCH_EVAL, shuffle=False),
            model, inv_scale_pools, OBS_POOLS, ["X_gL","Ab_gL"], DEVICE
        )

    pd.DataFrame(pf).to_csv(out / "VAL_pools_flat.csv", index=False)
    pd.DataFrame(xf).to_csv(out / "VAL_xab_flat.csv", index=False)
    pd.DataFrame(pbh).to_csv(out / "VAL_pools_by_h.csv", index=False)
    pd.DataFrame(xbh).to_csv(out / "VAL_xab_by_h.csv", index=False)
    pd.DataFrame(pr).to_csv(out / "VAL_per_run_flat.csv", index=False)

    # save checkpoint + scalers
    torch.save(model.state_dict(), out / "strictC_seq2seq_50L_XAb.pt")
    with open(out / "scaler_features.json", "w") as f:
        json.dump({"mu_all": mu_all.to_dict(),
                   "sd_all": sd_all.to_dict(),
                   "feat_cols": feat_cols}, f)
    if args.HET_XAB:
        with open(out / "scaler_xab.json", "w") as f:
            json.dump({
                "mu_xab": mu_xab.detach().cpu().numpy().tolist(),
                "sd_xab": sd_xab.detach().cpu().numpy().tolist()
            }, f)
    
        # --- UQ logging (ensemble epistemic + het-head aleatoric) ---
    _log_uq_ensemble(
        args=args,
        out_dir=out,
        base_ckpt_path=out / "strictC_seq2seq_50L_XAb.pt",
        ds_tr=ds_tr,
        ds_va=ds_va,
        feat_cols=feat_cols,
        n_drv_aux=len(DRIVING) + len(aux_cols),
        mu_xab=(mu_xab if args.HET_XAB else None),
        sd_xab=(sd_xab if args.HET_XAB else None),
        device=DEVICE,
    )

    # --- OLD-SCHEMA one-row summary (TF vs OL R²) ---
    # IMPORTANT: always pass y_tf_sc=y_sc and control with tf_ratio to avoid None-path
    tf_scores = _eval_r2_summary(
        model=model,
        dl_va=dl_va,
        inv_scale_pools=inv_scale_pools,
        het_xab=bool(args.HET_XAB),
        mu_xab=(mu_xab if args.HET_XAB else None),
        sd_xab=(sd_xab if args.HET_XAB else None),
        device=DEVICE,
        tf_ratio=1.0,   # teacher forcing
    )

    ol_scores = _eval_r2_summary(
        model=model,
        dl_va=dl_va,
        inv_scale_pools=inv_scale_pools,
        het_xab=bool(args.HET_XAB),
        mu_xab=(mu_xab if args.HET_XAB else None),
        sd_xab=(sd_xab if args.HET_XAB else None),
        device=DEVICE,
        tf_ratio=1e-6,  # open-loop (near-zero TF), but still pass y_tf_sc=y_sc
    )

    row = {
        "k_runs": k_runs,
        "runs_used_json": runs_used_json,  # <-- this now lists VAL runs
        "TF_R2_avg_pools": tf_scores["R2_avg_pools"],
        "TF_R2_X_gL":      tf_scores["R2_X_gL"],
        "TF_R2_Ab_gL":     tf_scores["R2_Ab_gL"],
        "OL_R2_avg_pools": ol_scores["R2_avg_pools"],
        "OL_R2_X_gL":      ol_scores["R2_X_gL"],
        "OL_R2_Ab_gL":     ol_scores["R2_Ab_gL"],
    }

    # 1) Keep the original single-row summary
    out_summary = out / "VAL_k_summary.csv"
    pd.DataFrame([row]).to_csv(out_summary, index=False)
    print(f"[summary] wrote old-schema row → {out_summary}")

    # 2) Also append an extended row into LC_added_runs_metrics_TF_OL.csv (one row per run)
    row_ext = dict(row)
    row_ext.update({
        "split_strategy": args.SPLIT_STRATEGY,
        "seed": args.SEED,
        "T_IN": args.T_IN,
        "T_OUT": args.T_OUT,
        "timestamp": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "outdir": str(out),
    })
    lc_path = out / "LC_snapshot_metrics_TF_OL.csv"
    df_row = pd.DataFrame([row_ext])
    if lc_path.exists():
        df_row.to_csv(lc_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(lc_path, index=False)
    print(f"[summary] appended → {lc_path}")

    # ----------------- Learning Curve (production-faithful) -----------------
    from copy import deepcopy

    print("\n>>> Learning curve by added 50L runs (production-equivalent recipe)")

    # We'll reuse your train_stage() function for Stage A/B/C,
    # not the lightweight tiny_train.

    all_train_runs = sorted(df_tr["run_id"].astype(str).unique().tolist())
    rows_lc = []

    # Save base initialization to ensure identical random weights for each k
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
        logv_max=args.LOGV_MAX
    ).to(DEVICE)
    init_state = deepcopy(base_init.state_dict())

    # Fixed VAL raw data for all LC evaluations
    raw_cols = ["run_id","time_h","dt","X_gL","Ab_gL"] + OBS_POOLS + DRIVING

    for k in range(1, len(all_train_runs) + 1):
        use_runs = set(all_train_runs[:k])
        sub_tr = df_tr[df_tr["run_id"].astype(str).isin(use_runs)].reset_index(drop=True)

        # ----- Rebuild scalers on this k-subset -----
        mu_all_k, sd_all_k = build_scalers(sub_tr, feat_cols)
        mu_obs_k, sd_obs_k = mu_all_k[OBS_POOLS], sd_all_k[OBS_POOLS]
        obs_mu_t_k = torch.tensor(mu_obs_k.values, dtype=torch.float32, device=DEVICE)
        obs_sd_t_k = torch.tensor(sd_obs_k.values, dtype=torch.float32, device=DEVICE)
        inv_scale_k = lambda y_sc: obs_mu_t_k + obs_sd_t_k * y_sc

        # Scale train and validation separately with k-scalers
        sub_tr_s = apply_scaler(sub_tr, feat_cols, mu_all_k, sd_all_k)
        df_va_s_k = apply_scaler(df_va, feat_cols, mu_all_k, sd_all_k)

        ds_tr_k = Seq2SeqDataset(sub_tr_s, sub_tr[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
        ds_va_k = Seq2SeqDataset(df_va_s_k, df_va[raw_cols].copy(), OBS_POOLS, DRIVING, aux_cols, args.T_IN, args.T_OUT)
        dl_tr_k = DataLoader(ds_tr_k, batch_size=args.BATCH_TR, shuffle=True, drop_last=True)
        dl_va_k = DataLoader(ds_va_k, batch_size=args.BATCH_EVAL, shuffle=False)

        # X/Ab normalization for heteroscedastic loss
        if args.HET_XAB:
            mu_xab_k = torch.tensor(sub_tr[["X_gL","Ab_gL"]].mean().values, dtype=torch.float32, device=DEVICE)
            sd_xab_k = torch.tensor(sub_tr[["X_gL","Ab_gL"]].std().replace(0, 1.0).values, dtype=torch.float32, device=DEVICE)
        else:
            mu_xab_k = sd_xab_k = None

        # ----- Fresh model init -----
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
            logv_max=args.LOGV_MAX
        ).to(DEVICE)
        model_k.load_state_dict(init_state)

        # Optional warm-start from TWO_L_WEIGHTS (same as prod)
        if args.TWO_L_WEIGHTS and Path(args.TWO_L_WEIGHTS).exists():
            sd_tl = torch.load(args.TWO_L_WEIGHTS, map_location=DEVICE)
            model_k.load_state_dict(sd_tl, strict=False)

        # ----- Run full production recipe -----
        # Freeze/unfreeze identical to Stage A/B/C in run_train
        for p in model_k.parameters(): p.requires_grad = False
        for p in model_k.head_pools.parameters(): p.requires_grad = True
        for p in model_k.head_cres.parameters():  p.requires_grad = True
        if args.HET_XAB:
            for p in model_k.head_xab_mu.parameters(): p.requires_grad = True
            for p in model_k.head_xab_lv.parameters(): p.requires_grad = True
        else:
            for p in model_k.head_xab.parameters(): p.requires_grad = True
        L = model_k.dec.num_layers - 1
        for name, p in model_k.dec.named_parameters():
            if f"_l{L}" in name: p.requires_grad = True
        train_stage(args.FREEZE_EPOCHS_HEAD, args.LR_STAGE_A, args.LAMBDA_MB * 0.5, args.LAMBDA_CONS * 0.5)

        for name, p in model_k.enc.named_parameters():
            if f"_l{L}" in name: p.requires_grad = True
        train_stage(args.FREEZE_EPOCHS_TOP, args.LR_STAGE_B, args.LAMBDA_MB, args.LAMBDA_CONS)

        for p in model_k.parameters(): p.requires_grad = True
        train_stage(args.POLISH_EPOCHS, args.LR_STAGE_C, args.LAMBDA_MB, args.LAMBDA_CONS)

        # ----- Evaluate (TF & OL) -----
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
            "TF_R2_X_gL": tf_scores_k["R2_X_gL"],
            "TF_R2_Ab_gL": tf_scores_k["R2_Ab_gL"],
            "OL_R2_avg_pools": ol_scores_k["R2_avg_pools"],
            "OL_R2_X_gL": ol_scores_k["R2_X_gL"],
            "OL_R2_Ab_gL": ol_scores_k["R2_Ab_gL"],
        })

        print(f"  k={k:02d} | TF R²[X]={tf_scores_k['R2_X_gL']:.3f}  TF R²[Ab]={tf_scores_k['R2_Ab_gL']:.3f} "
            f"| OL R²[X]={ol_scores_k['R2_X_gL']:.3f}  OL R²[Ab]={ol_scores_k['R2_Ab_gL']:.3f}")

    # save LC table
    lc_csv = out / "LC_added_runs_metrics_TF_OL.csv"
    pd.DataFrame(rows_lc).to_csv(lc_csv, index=False)
    print(f"Saved learning-curve table → {lc_csv}")

    print(f"[train_lstm] Saved artifacts to {out}")
    return 0


def main():
    ap = build_argparser(); args = ap.parse_args()
    # resolve paths
    args.TRAIN_INPUT_CSV = str(Path(args.TRAIN_INPUT_CSV).resolve())
    if args.TWO_L_WEIGHTS: args.TWO_L_WEIGHTS = str(Path(args.TWO_L_WEIGHTS).resolve())
    args.OUTDIR_TL = str(Path(args.OUTDIR_TL).resolve())
    ec = run_train(args)
    raise SystemExit(ec)

if __name__ == "__main__":
    main()

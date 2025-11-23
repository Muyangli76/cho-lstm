# src/cho_lstm_uq/optuna_hpo/objective_tl.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Callable, List, Optional
import sys, math, time, subprocess, json, os
import optuna

# ---- Optional MLflow logging ----------------------------------------------
try:
    import mlflow  # type: ignore
    _HAS_MLFLOW = True
except Exception:
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

# ---- Local imports ---------------------------------------------------------
from .metrics import read_val_metrics


# ---- Helpers ---------------------------------------------------------------
def _make_trial_outdir(base_outdir: Path, number: int) -> Path:
    p = Path(base_outdir) / f"trial_{number:04d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_src_dir(anchor: Path) -> Path:
    """
    Walk up from `anchor` until we hit a folder literally named 'src'.
    Fallback to anchor.parent if not found.
    """
    p = Path(anchor).resolve()
    while p.name.lower() != "src" and p.parent != p:
        p = p.parent
    return p if p.name.lower() == "src" else Path(anchor).resolve().parent


def _module_from_train_script(train_script: Path, src_dir: Path) -> str:
    """
    Convert '.../src/cho_lstm_uq/models/strictc50_tl_train.py'
    -> 'cho_lstm_uq.models.strictc50_tl_train'
    """
    ts = Path(train_script).resolve()
    sd = Path(src_dir).resolve()
    rel = ts.with_suffix("").relative_to(sd)
    return ".".join(rel.parts)


# Keys that are passed as "--KEY value" (trainer argparse must accept them)
_ALLOWED_CLI_KEYS = {
    # core IO / training knobs
    "SEED", "T_IN", "T_OUT", "HIDDEN", "LAYERS", "DROPOUT",
    "CLIP", "BATCH_TR", "BATCH_EVAL", "PATIENCE",
    # loss weights
    "LAMBDA_MB", "GAMMA_RES", "LAMBDA_XAB", "LAMBDA_CONS",
    # TL schedule / L2SP / LRs
    "FREEZE_EPOCHS_HEAD", "FREEZE_EPOCHS_TOP",
    "LR_STAGE_A", "LR_STAGE_B", "LR_STAGE_C",
    "L2SP_ALPHA",
    # polish / learning-curve (kept for compatibility)
    "POLISH_EPOCHS", "LC_ORDERS", "LC_EPOCHS",
    # log-variance clamps
    "LOGV_MIN", "LOGV_MAX",
    # validation split
    "VAL_FRAC",
    # allow HPO to explicitly set split behaviour
    "SPLIT_STRATEGY", "REP_VAL_IDX",
}

# Boolean flags that trainer defines as `action="store_true"`
_FLAG_KEYS = {
    "HET_XAB",
    "SOFTPLUS_VAR",
}


def _build_args_for_tl_train(
    *,
    input_csv: Path,
    two_l_weights: Path,
    outdir_tl: Path | str,
    params: Dict[str, Any],
) -> List[str]:
    """
    Build CLI *arguments only* for your TL trainer.
    Caller is responsible for prepending 'python -m <module>'.
    Booleans: include flag only if True. Others: --KEY value.
    """
    args: List[str] = []

    # Required paths
    args += ["--TRAIN_INPUT_CSV", str(Path(input_csv))]
    args += ["--TWO_L_WEIGHTS",   str(Path(two_l_weights))]
    args += ["--OUTDIR_TL",       str(outdir_tl)]

    # Boolean flags first (when True only)
    for fk in _FLAG_KEYS:
        if params.get(fk, False):
            args += [f"--{fk}"]

    # Numeric / string args
    for k, v in params.items():
        if k in _FLAG_KEYS or k in {"OUTDIR_TL", "TWO_L_WEIGHTS"}:
            continue
        if k not in _ALLOWED_CLI_KEYS:
            continue
        args += [f"--{k}", str(v)]

    return args


def _read_optional_uq_penalty(metrics_dir: Path) -> Optional[float]:
    """
    Looks for OUTDIR_TL/UQ/ensemble_val_predictions_with_uncert.csv and
    returns mean(var_tot_X + var_tot_Ab)/2 if present; otherwise None.
    """
    f = Path(metrics_dir) / "UQ" / "ensemble_val_predictions_with_uncert.csv"
    if not f.exists():
        return None
    try:
        import pandas as pd  # local import to avoid hard dependency if not needed
        df = pd.read_csv(f)
        if {"var_tot_X", "var_tot_Ab"}.issubset(df.columns):
            vt = (df["var_tot_X"].astype(float) + df["var_tot_Ab"].astype(float)) / 2.0
            return float(vt.mean())
    except Exception:
        return None
    return None


# ---- Factory ---------------------------------------------------------------
def objective_factory(
    *,
    train_script: Path,
    input_csv: Path,
    base_outdir: Path,
    seed: int,
    lambda_mae: float,
    default_args: Dict[str, Any],
    mlflow_tracking_uri: str | None = None,
    use_mlflow: bool = False,
) -> Callable[[optuna.Trial], float]:
    """
    TL objective (2L → 50L). Requires `TWO_L_WEIGHTS` in `default_args` (or overrides).
    Optional: `LAMBDA_UQ` in `default_args` to penalize ensemble variance if UQ CSV is present.
    """

    # Resolve & validate static paths early (fail fast across all trials)
    train_script = Path(train_script).resolve()
    input_csv = Path(input_csv).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"[TL objective] train_script not found: {train_script}")
    if not input_csv.exists():
        raise FileNotFoundError(f"[TL objective] input_csv not found: {input_csv}")

    # Configure MLflow (optional)
    if use_mlflow and _HAS_MLFLOW:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)  # type: ignore
        exp_name = str(default_args.get("MLFLOW_EXPERIMENT", "strictC_TL_experiments"))
        mlflow.set_experiment(exp_name)  # type: ignore

    def objective(trial: optuna.Trial) -> float:
        """
        TL HPO objective (2L → 50L).

        - Keeps paths / chemistry / most constants from `default_args`
        - Lets Optuna tune:
            * window sizes, model size, dropout
            * batch sizes, grad clip
            * physics weights (MB / CONS / RES)
            * TL schedule (FREEZE_EPOCHS_*, POLISH_EPOCHS)
            * stage learning rates
            * L2SP_ALPHA
        - Uses doe_holdout split for stability on 50 L.
        """

        # --- Start from defaults, then override with trial-suggested values ---
        params: Dict[str, Any] = dict(default_args)

        # Seed (fixed per study unless you want to randomize)
        params["SEED"] = int(default_args.get("SEED", seed))

        # -----------------------------
        # 1) Common hyperparameters
        # -----------------------------
        # Time windows (keep near the regimes that worked for same-scale)
        T_IN  = trial.suggest_int("T_IN", 60, 96, step=12)   # {60,72,84,96}
        T_OUT = trial.suggest_int("T_OUT", 18, 24, step=6)   # {18,24}

        # Model size: must match 2L checkpoint for TL warm-start
        HIDDEN  = int(default_args.get("HIDDEN", 128))
        LAYERS  = int(default_args.get("LAYERS", 2))

        # You can still tune dropout independently
        DROPOUT = trial.suggest_float("DROPOUT", 0.05, 0.25)

        # Batching / clipping
        BATCH_TR   = trial.suggest_categorical("BATCH_TR",   [32, 48, 64, 80, 96])
        BATCH_EVAL = trial.suggest_categorical("BATCH_EVAL", [128, 192, 256])
        CLIP       = trial.suggest_float("CLIP", 0.5, 2.0)

        # Physics weights (same spirit as single-scale objective)
        LAMBDA_MB   = trial.suggest_float("LAMBDA_MB",   0.03, 0.08)
        LAMBDA_CONS = trial.suggest_float("LAMBDA_CONS", 0.10, 0.30)
        GAMMA_RES   = trial.suggest_float("GAMMA_RES", 1e-5, 5e-5, log=True)

        # -----------------------------
        # 2) TL-specific knobs
        # -----------------------------
        # Stage lengths
        FREEZE_EPOCHS_HEAD = trial.suggest_int("FREEZE_EPOCHS_HEAD", 1, 6)
        FREEZE_EPOCHS_TOP  = trial.suggest_int("FREEZE_EPOCHS_TOP",  2, 8)
        POLISH_EPOCHS      = trial.suggest_int("POLISH_EPOCHS",      1, 4)

        # Stage learning rates
        LR_STAGE_A = trial.suggest_float("LR_STAGE_A", 1e-5, 5e-4, log=True)
        LR_STAGE_B = trial.suggest_float("LR_STAGE_B", 1e-5, 5e-4, log=True)
        LR_STAGE_C = trial.suggest_float("LR_STAGE_C", 5e-6, 2e-4, log=True)

        # L2SP strength (penalty to 2 L reference)
        L2SP_ALPHA = trial.suggest_float("L2SP_ALPHA", 1e-5, 5e-3, log=True)

        # -----------------------------
        # 3) Commit trial choices into params
        # -----------------------------
        params.update({
            # split: for TL to 50 L, doe_holdout is usually what you described
            "SPLIT_STRATEGY": "doe_holdout",
            "REP_VAL_IDX": int(default_args.get("REP_VAL_IDX", 3)),   # harmless for doe_holdout
            "VAL_FRAC": float(default_args.get("VAL_FRAC", 0.2)),     # unused in doe_holdout

            # windows / model / batch / clip
            "T_IN": T_IN,
            "T_OUT": T_OUT,
            "HIDDEN": HIDDEN,
            "LAYERS": LAYERS,
            "DROPOUT": DROPOUT,
            "BATCH_TR": BATCH_TR,
            "BATCH_EVAL": BATCH_EVAL,
            "CLIP": CLIP,

            # physics weights
            "LAMBDA_MB": LAMBDA_MB,
            "LAMBDA_CONS": LAMBDA_CONS,
            "GAMMA_RES": GAMMA_RES,

            # TL schedule
            "FREEZE_EPOCHS_HEAD": FREEZE_EPOCHS_HEAD,
            "FREEZE_EPOCHS_TOP": FREEZE_EPOCHS_TOP,
            "POLISH_EPOCHS": POLISH_EPOCHS,
            "LR_STAGE_A": LR_STAGE_A,
            "LR_STAGE_B": LR_STAGE_B,
            "LR_STAGE_C": LR_STAGE_C,
            "L2SP_ALPHA": L2SP_ALPHA,
        })
        # NOTE:
        # - We intentionally *do not* touch HET_XAB / SOFTPLUS_VAR / LOGV_* /
        #   LAMBDA_XAB / LAMBDA_UQ here — they come from default_args.

        # -----------------------------
        # 4) Mandatory TL inputs
        # -----------------------------
        two_l_weights = params.get("TWO_L_WEIGHTS", None)
        if not two_l_weights:
            raise RuntimeError("TWO_L_WEIGHTS is required for TL but was not provided in default_args/overrides.")
        two_l_weights = Path(two_l_weights).resolve()
        if not two_l_weights.exists():
            raise FileNotFoundError(f"TWO_L_WEIGHTS not found: {two_l_weights}")

        # Trial directories (force isolated OUTDIR_TL per trial)
        trial_outdir = _make_trial_outdir(base_outdir, trial.number)
        outdir_tl = trial_outdir

        # Discover src & module, set PYTHONPATH and CWD
        src_dir = _find_src_dir(train_script)
        module = _module_from_train_script(train_script, src_dir)

        # Build CLI
        cli_args = _build_args_for_tl_train(
            input_csv=input_csv,
            two_l_weights=two_l_weights,
            outdir_tl=outdir_tl,
            params=params,
        )
        cmd = [sys.executable, "-m", module, *cli_args]

        # Persist spawn command
        (trial_outdir / "_spawn_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")

        # Environment with PYTHONPATH including src
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

        # -----------------------------
        # 5) Run trainer
        # -----------------------------
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(src_dir.resolve()),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as e:
            (trial_outdir / "spawn_error.txt").write_text(str(e), encoding="utf-8")
            return -1e9
        wall = time.time() - t0

        # Save logs
        (trial_outdir / "train_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (trial_outdir / "train_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            return -1e9

        # -----------------------------
        # 6) Read metrics & compute score
        # -----------------------------
        metrics_dir = outdir_tl
        try:
            xab_metrics, pool_metrics = read_val_metrics(metrics_dir)
        except Exception as e:
            (trial_outdir / "metrics_error.txt").write_text(str(e), encoding="utf-8")
            return -1e9

        R2_X      = float(xab_metrics.get("R2_X_gL",          float("nan")))
        R2_Ab     = float(xab_metrics.get("R2_Ab_gL",         float("nan")))
        macro_MAE = float(pool_metrics.get("macro_MAE_pools", float("nan")))

        if any(((z != z) or math.isinf(z)) for z in (R2_X, R2_Ab, macro_MAE)):
            return -1e9

        # Optional UQ penalty
        lambda_uq = float(params.get("LAMBDA_UQ", 0.0))
        uq_pen = 0.0
        if lambda_uq > 0.0:
            mean_var = _read_optional_uq_penalty(metrics_dir)
            if mean_var is not None and math.isfinite(float(mean_var)):
                uq_pen = lambda_uq * float(mean_var)

        score = 0.5 * (R2_X + R2_Ab) - (lambda_mae * macro_MAE) - uq_pen

        # -----------------------------
        # 7) MLflow logging (optional)
        # -----------------------------
        if use_mlflow and _HAS_MLFLOW:
            with mlflow.start_run(run_name=f"TL_trial_{trial.number:04d}"):  # type: ignore
                loggable_params = {
                    k: (str(v) if isinstance(v, (list, dict)) else v)
                    for k, v in params.items()
                }
                mlflow.log_params(loggable_params)  # type: ignore
                mlflow.log_metrics({                # type: ignore
                    "R2_X": R2_X,
                    "R2_Ab": R2_Ab,
                    "macro_MAE_pools": macro_MAE,
                    "objective": score,
                    "wall_time_s": wall,
                    "uq_penalty": uq_pen,
                })
                for name in (
                    "VAL_xab_flat.csv", "VAL_pools_flat.csv",
                    "train_stdout.txt", "train_stderr.txt", "_spawn_cmd.txt",
                ):
                    pth = (metrics_dir if name.startswith("VAL_") else trial_outdir) / name
                    if pth.exists():
                        try:
                            mlflow.log_artifact(str(pth))  # type: ignore
                        except Exception:
                            pass

        # Attach attributes for quick triage
        trial.set_user_attr("wall_time_s", wall)
        trial.set_user_attr("outdir", str(trial_outdir))
        trial.set_user_attr("metrics_dir", str(metrics_dir))
        trial.set_user_attr("R2_X", R2_X)
        trial.set_user_attr("R2_Ab", R2_Ab)
        trial.set_user_attr("macro_MAE_pools", macro_MAE)
        trial.set_user_attr("uq_penalty", uq_pen)

        return float(score)

    return objective


__all__ = ["objective_factory"]

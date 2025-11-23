# src/cho_lstm_uq/optuna_hpo/objective_single.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Callable, List
import os, sys, math, time, subprocess
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

_ALLOWED_CLI_KEYS = {
    "BATCH_TR", "BATCH_EVAL", "T_IN", "T_OUT", "CLIP", "PATIENCE",
    "LAMBDA_MB", "GAMMA_RES", "LAMBDA_XAB", "LAMBDA_CONS",
    "HIDDEN", "LAYERS", "DROPOUT",
    "LOGV_MIN", "LOGV_MAX",
    "SPLIT_STRATEGY", "REP_VAL_IDX", "VAL_FRAC",
}

def _build_args_for_train_script(
    train_script: Path,
    input_csv: Path,
    trial_outdir: Path,
    seed: int,
    params: Dict[str, Any],
) -> List[str]:
    """
    Build CLI args for subprocess call:
        python -m cho_lstm_uq.models.strictc50.train_lstm --TRAIN_INPUT_CSV ... --OUTDIR_TL ...
    We explicitly call the module (not the file) so that relative imports in train_lstm.py resolve.
    """
    cmd: List[str] = [sys.executable, "-m", "cho_lstm_uq.models.strictc50.train_lstm"]

    # required args
    cmd += ["--TRAIN_INPUT_CSV", str(Path(input_csv).resolve())]
    outdir_tl = params.get("OUTDIR_TL", str(Path(trial_outdir).resolve()))
    cmd += ["--OUTDIR_TL", outdir_tl]
    cmd += ["--SEED", str(int(seed))]

    # optional warm start
    if params.get("TWO_L_WEIGHTS"):
        cmd += ["--TWO_L_WEIGHTS", str(Path(params["TWO_L_WEIGHTS"]).resolve())]

    # booleans as flags
    if params.get("HET_XAB", False):
        cmd += ["--HET_XAB"]
    if params.get("SOFTPLUS_VAR", False):
        cmd += ["--SOFTPLUS_VAR"]

    # numeric/string args
    for k, v in params.items():
        if k in ("HET_XAB", "SOFTPLUS_VAR", "TWO_L_WEIGHTS", "OUTDIR_TL"):
            continue
        if k not in _ALLOWED_CLI_KEYS:
            continue
        cmd += [f"--{k}", str(v)]

    (Path(trial_outdir) / "_spawn_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
    return cmd

def _find_src_dir(anchor: Path) -> Path:
    """
    Walk upward from `anchor` until we find a folder named 'src'.
    Fallback to anchor.parent if not found.
    """
    p = Path(anchor).resolve()
    while p.name.lower() != "src" and p.parent != p:
        p = p.parent
    return p

# ---- Factory ---------------------------------------------------------------
def objective_factory(
    *,
    train_script: Path,
    input_csv: Path,
    base_outdir: Path,
    epochs: int,            # kept for API compatibility (your train script uses stage epochs)
    seed: int,
    lambda_mae: float,
    use_mlflow: bool,
    default_args: Dict[str, Any],
    mlflow_tracking_uri: str | None = None,
) -> Callable[[optuna.Trial], float]:

    # ---- Optional MLflow wiring ----
    if use_mlflow and _HAS_MLFLOW:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)  # type: ignore
        exp_name = str(default_args.get("MLFLOW_EXPERIMENT", "strictC_single_experiments"))
        mlflow.set_experiment(exp_name)  # type: ignore

    def objective(trial: optuna.Trial) -> float:
        # ===============================
        # 1) Search space (50L same-scale)
        # ===============================
        # Time windows (stick close to your good regimes)
        T_IN  = trial.suggest_int("T_IN", 60, 96, step=12)     # {60,72,84,96}
        T_OUT = trial.suggest_int("T_OUT", 18, 24, step=6)     # {18,24}

        # Model size + regularization
        HIDDEN   = trial.suggest_int("HIDDEN", 96, 224, step=32)   # {96,128,160,192,224}
        LAYERS   = trial.suggest_int("LAYERS", 1, 3)               # 1–3
        DROPOUT  = trial.suggest_float("DROPOUT", 0.15, 0.35)      # a bit stronger than 0.1

        # Batching / clipping
        BATCH_TR   = trial.suggest_categorical("BATCH_TR", [32, 48, 64, 80, 96])
        BATCH_EVAL = trial.suggest_categorical("BATCH_EVAL", [128, 192, 256])
        CLIP       = trial.suggest_float("CLIP", 0.5, 2.0)

        # Physics weights (narrower, but still exploratory)
        LAMBDA_MB   = trial.suggest_float("LAMBDA_MB", 0.03, 0.08)
        LAMBDA_CONS = trial.suggest_float("LAMBDA_CONS", 0.10, 0.30)
        GAMMA_RES   = trial.suggest_float("GAMMA_RES", 1e-5, 5e-5, log=True)

        # ===============================
        # 2) Assemble params for trainer
        # ===============================
        p: Dict[str, Any] = {
            # --- split: fixed to DoE holdout for now ---
            "SPLIT_STRATEGY": "doe_holdout",
            "REP_VAL_IDX": 3,      # harmless for doe_holdout
            "VAL_FRAC": 0.2,       # unused for doe_holdout, also harmless

            # --- core / io ---
            "T_IN": T_IN,
            "T_OUT": T_OUT,
            "HIDDEN": HIDDEN,
            "LAYERS": LAYERS,
            "DROPOUT": DROPOUT,
            "BATCH_TR": BATCH_TR,
            "BATCH_EVAL": BATCH_EVAL,
            "CLIP": CLIP,

            # --- losses / physics ---
            "LAMBDA_MB": LAMBDA_MB,
            "LAMBDA_CONS": LAMBDA_CONS,
            "GAMMA_RES": GAMMA_RES,

            # --- UQ head toggles (OFF for HPO) ---
            "HET_XAB": False,
            "SOFTPLUS_VAR": False,
            "LOGV_MIN": -10.0,
            "LOGV_MAX": 3.0,
        }

        # Keep your stage schedule & chemistry constants from defaults unless overridden
        merged_params = dict(default_args)
        merged_params.update({k: v for k, v in p.items() if v is not None})

        # ===============================
        # 3) Spawn a trial run
        # ===============================
        trial_outdir = _make_trial_outdir(base_outdir, trial.number)
        cmd = _build_args_for_train_script(
            train_script=train_script,
            input_csv=input_csv,
            trial_outdir=trial_outdir,
            seed=seed,
            params=merged_params,
        )

        # Ensure module importability
        src_dir = _find_src_dir(Path(train_script))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(src_dir.resolve()),
                env=env,
            )
        except Exception as e:
            (trial_outdir / "spawn_error.txt").write_text(str(e), encoding="utf-8")
            return -1e9
        wall = time.time() - t0

        (trial_outdir / "train_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (trial_outdir / "train_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            return -1e9

        # ===============================
        # 4) Read metrics & compute score
        # ===============================
        try:
            xab_metrics, pool_metrics = read_val_metrics(trial_outdir)
        except Exception as e:
            (trial_outdir / "metrics_error.txt").write_text(str(e), encoding="utf-8")
            return -1e9

        R2_X = float(xab_metrics.get("R2_X_gL", float("nan")))
        R2_Ab = float(xab_metrics.get("R2_Ab_gL", float("nan")))
        macro_MAE = float(pool_metrics.get("macro_MAE_pools", float("nan")))

        if any(((z != z) or math.isinf(z)) for z in (R2_X, R2_Ab, macro_MAE)):
            return -1e9

        score = 0.5 * R2_X + 0.5 * R2_Ab - (lambda_mae * macro_MAE)

        # ===============================
        # 5) Optional MLflow logging
        # ===============================
        if use_mlflow and _HAS_MLFLOW:
            with mlflow.start_run(run_name=f"trial_{trial.number:04d}"):  # type: ignore
                safe_params = {k: (str(v) if isinstance(v, (list, dict)) else v)
                               for k, v in merged_params.items()}
                mlflow.log_params(safe_params)  # type: ignore
                mlflow.log_metrics({            # type: ignore
                    "R2_X": R2_X,
                    "R2_Ab": R2_Ab,
                    "macro_MAE_pools": macro_MAE,
                    "objective": score,
                    "wall_time_s": wall,
                })
                for name in ("VAL_xab_flat.csv", "VAL_pools_flat.csv",
                             "train_stdout.txt", "train_stderr.txt", "_spawn_cmd.txt"):
                    pth = trial_outdir / name
                    if pth.exists():
                        mlflow.log_artifact(str(pth))  # type: ignore
                ckpt = trial_outdir / "strictC_seq2seq_50L_XAb.pt"
                if ckpt.exists():
                    mlflow.log_artifact(str(ckpt))  # type: ignore

        trial.set_user_attr("wall_time_s", wall)
        trial.set_user_attr("outdir", str(trial_outdir))
        trial.set_user_attr("R2_X", R2_X)
        trial.set_user_attr("R2_Ab", R2_Ab)
        trial.set_user_attr("macro_MAE_pools", macro_MAE)

        return score

    return objective

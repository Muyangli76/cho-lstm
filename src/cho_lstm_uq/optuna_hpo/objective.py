from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Callable, List
import sys, math, time, subprocess
import optuna

# ---- Optional MLflow logging ----------------------------------------------
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

# ---- Local imports ---------------------------------------------------------
from .metrics import read_val_metrics


# ---- Helpers ---------------------------------------------------------------
def _make_trial_outdir(base_outdir: Path, number: int) -> Path:
    """Create a per-trial directory such as base/trial_0003."""
    p = Path(base_outdir) / f"trial_{number:04d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


_ALLOWED_CLI_KEYS = {
    "BATCH_TR", "BATCH_EVAL", "T_IN", "T_OUT", "CLIP", "PATIENCE",
    "LAMBDA_MB", "GAMMA_RES", "LAMBDA_XAB", "LAMBDA_CONS",
    "HIDDEN", "LAYERS", "DROPOUT",
    "LOGV_MIN", "LOGV_MAX",
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
    """
    cmd: List[str] = [sys.executable, "-m", "cho_lstm_uq.models.strictc50.train_lstm"]

    # required args
    cmd += ["--TRAIN_INPUT_CSV", str(Path(input_csv))]
    outdir_tl = params.get("OUTDIR_TL", str(Path(trial_outdir)))
    cmd += ["--OUTDIR_TL", outdir_tl]
    cmd += ["--SEED", str(int(seed))]

    # optional warm start
    if params.get("TWO_L_WEIGHTS"):
        cmd += ["--TWO_L_WEIGHTS", str(params["TWO_L_WEIGHTS"])]

    # booleans as flags
    if params.get("HET_XAB", False):
        cmd += ["--HET_XAB"]
    if params.get("SOFTPLUS_VAR", False):
        cmd += ["--SOFTPLUS_VAR"]

    # numeric/string args
    for k, v in params.items():
        if k in ("HET_XAB", "SOFTPLUS_VAR"):
            continue
        if k not in _ALLOWED_CLI_KEYS:
            continue
        cmd += [f"--{k}", str(v)]

    # for debugging
    (Path(trial_outdir) / "_spawn_cmd.txt").write_text(" ".join(cmd))
    return cmd


# ---- Factory ---------------------------------------------------------------
def objective_factory(
    *,
    train_script: Path,
    input_csv: Path,
    base_outdir: Path,
    epochs: int,                 # kept for API compatibility
    seed: int,
    lambda_mae: float,
    use_mlflow: bool,
    default_args: Dict[str, Any],
    mlflow_tracking_uri: str | None = None,
) -> Callable[[optuna.Trial], float]:

    # Set MLflow tracking configuration
    if use_mlflow and _HAS_MLFLOW:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("strictC50L_experiments")

    def objective(trial: optuna.Trial) -> float:
        # ---- Sample hyperparameters ----
        p: Dict[str, Any] = {
            "T_IN": trial.suggest_int("T_IN", 48, 96, step=12),
            "T_OUT": trial.suggest_int("T_OUT", 12, 24, step=6),

            "HIDDEN": trial.suggest_int("HIDDEN", 96, 256, step=32),
            "LAYERS": trial.suggest_int("LAYERS", 1, 3),
            "DROPOUT": trial.suggest_float("DROPOUT", 0.0, 0.3),

            "BATCH_TR": trial.suggest_categorical("BATCH_TR", [32, 48, 64, 80, 96, 112, 128]),
            "BATCH_EVAL": 128,

            "LAMBDA_MB": trial.suggest_float("LAMBDA_MB", 0.02, 0.10),
            "LAMBDA_CONS": trial.suggest_float("LAMBDA_CONS", 0.10, 0.30),
            "GAMMA_RES": trial.suggest_float("GAMMA_RES", 1e-5, 5e-5, log=True),

            "LOGV_MIN": trial.suggest_float("LOGV_MIN", -12.0, -8.0),
            "LOGV_MAX": trial.suggest_float("LOGV_MAX", 1.0, 3.0),
        }

        merged_params = dict(default_args)
        merged_params.update({k: v for k, v in p.items() if v is not None})

        trial_outdir = _make_trial_outdir(base_outdir, trial.number)
        cmd = _build_args_for_train_script(
            train_script=train_script,
            input_csv=input_csv,
            trial_outdir=trial_outdir,
            seed=seed,
            params=merged_params,
        )

        # ---- Run training subprocess ----
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as e:
            (trial_outdir / "spawn_error.txt").write_text(str(e))
            return -1e9
        wall = time.time() - t0

        (trial_outdir / "train_stdout.txt").write_text(proc.stdout or "")
        (trial_outdir / "train_stderr.txt").write_text(proc.stderr or "")
        if proc.returncode != 0:
            return -1e9

        # ---- Read metrics ----
        try:
            xab_metrics, pool_metrics = read_val_metrics(trial_outdir)
        except Exception as e:
            (trial_outdir / "metrics_error.txt").write_text(str(e))
            return -1e9

        R2_X = float(xab_metrics.get("R2_X_gL", float("nan")))
        R2_Ab = float(xab_metrics.get("R2_Ab_gL", float("nan")))
        macro_MAE = float(pool_metrics.get("macro_MAE_pools", float("nan")))

        if any(((z != z) or math.isinf(z)) for z in (R2_X, R2_Ab, macro_MAE)):
            return -1e9

        score = 0.5 * R2_X + 0.5 * R2_Ab - (lambda_mae * macro_MAE)

        # ---- MLflow logging ----
        if use_mlflow and _HAS_MLFLOW:
            with mlflow.start_run(run_name=f"trial_{trial.number:04d}"):
                safe_params = {k: (str(v) if isinstance(v, (list, dict)) else v)
                               for k, v in merged_params.items()}
                mlflow.log_params(safe_params)
                mlflow.log_metrics({
                    "R2_X": R2_X,
                    "R2_Ab": R2_Ab,
                    "macro_MAE_pools": macro_MAE,
                    "objective": score,
                    "wall_time_s": wall,
                })

                # artifacts
                for name in (
                    "VAL_xab_flat.csv",
                    "VAL_pools_flat.csv",
                    "train_stdout.txt",
                    "train_stderr.txt",
                    "_spawn_cmd.txt",
                ):
                    pth = trial_outdir / name
                    if pth.exists():
                        mlflow.log_artifact(str(pth))

                # optional checkpoint
                ckpt = trial_outdir / "strictC_seq2seq_50L_XAb.pt"
                if ckpt.exists():
                    mlflow.log_artifact(str(ckpt))

        # ---- attach Optuna user attrs ----
        trial.set_user_attr("wall_time_s", wall)
        trial.set_user_attr("outdir", str(trial_outdir))
        trial.set_user_attr("R2_X", R2_X)
        trial.set_user_attr("R2_Ab", R2_Ab)
        trial.set_user_attr("macro_MAE_pools", macro_MAE)

        return score

    return objective


__all__ = ["objective_factory"]

# src/cho_lstm_uq/optuna_hpo/api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import optuna

# import both objective factories
from .objective_single import objective_factory as single_objective_factory
from .objective_tl import objective_factory as tl_objective_factory


# ---- Tiny YAML loader (no external import required) ------------------------
def _load_defaults_yaml(path: Optional[Path]) -> Dict[str, Any]:
    """Lightweight YAML / key:value parser for config files."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        # fallback: basic key:value per line
        data: Dict[str, Any] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = _coerce_scalar(v.strip())
        return data

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return {str(k): v for k, v in (data.items() if isinstance(data, dict) else [])}


def _coerce_scalar(s: str):
    sl = s.lower()
    if sl in ("true", "yes", "on"):
        return True
    if sl in ("false", "no", "off"):
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


# ----------------------------------------------------------------------------
def run_study(
    *,
    train_script: Path,
    input_csv: Path,
    base_outdir: Path,
    epochs: int,
    seed: int,
    n_trials: int,
    n_jobs: int,
    study_name: str,
    storage: Optional[str],
    sampler: str = "tpe",
    pruner: Optional[str] = None,
    lambda_mae: float = 0.1,
    trainer_kind: str = "single",  # NEW: "single" or "tl"
    mlflow_tracking_uri: Optional[str] = None,
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    code_defaults: Optional[Dict[str, Any]] = None,
) -> optuna.Study:
    """
    Unified HPO launcher.
    - trainer_kind="single" → same-scale objective (LSTM or UQ)
    - trainer_kind="tl" → transfer-learning objective (requires TWO_L_WEIGHTS)
    """

    # ---------------- Sampler ----------------
    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler(seed=seed)
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # ---------------- Pruner ----------------
    if pruner is None or pruner == "none":
        pruner_obj = optuna.pruners.NopPruner()
    elif pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner()
    elif pruner == "successive_halving":
        pruner_obj = optuna.pruners.SuccessiveHalvingPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner}")

    # ---------------- Study ----------------
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler_obj,
        pruner=pruner_obj,
        direction="maximize",
        load_if_exists=True,
    )

    # ---------------- Merge defaults ----------------
    merged: Dict[str, Any] = {}
    if code_defaults:
        merged.update(code_defaults)
    merged.update(_load_defaults_yaml(config_path))
    if overrides:
        merged.update(overrides)

    # ---------------- Objective factory ----------------
    kind = trainer_kind.lower().strip()
    if kind not in {"single", "tl"}:
        raise ValueError(f"Invalid trainer_kind='{trainer_kind}', must be 'single' or 'tl'")

    if kind == "tl":
        obj = tl_objective_factory(
            train_script=train_script.resolve(),
            input_csv=input_csv.resolve(),
            base_outdir=base_outdir.resolve(),
            seed=seed,
            lambda_mae=lambda_mae,
            default_args=merged,
            mlflow_tracking_uri=mlflow_tracking_uri,
            use_mlflow=bool(mlflow_tracking_uri),
        )
    else:
        obj = single_objective_factory(
            train_script=train_script.resolve(),
            input_csv=input_csv.resolve(),
            base_outdir=base_outdir.resolve(),
            epochs=epochs,
            seed=seed,
            lambda_mae=lambda_mae,
            use_mlflow=bool(mlflow_tracking_uri),
            default_args=merged,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    # ---------------- Run optimization ----------------
    study.optimize(
        obj,
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    return study

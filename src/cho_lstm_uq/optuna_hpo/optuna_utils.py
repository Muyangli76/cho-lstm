from __future__ import annotations
import optuna
from typing import Optional


def build_sampler(name: str):
    """Return an Optuna sampler based on name."""
    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler()
    return optuna.samplers.TPESampler()


def build_pruner(name: str):
    """Return an Optuna pruner based on name."""
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    if name == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    return optuna.pruners.NopPruner()


def create_study(
    *, study_name: str, storage: Optional[str], sampler, pruner
) -> optuna.Study:
    """Create or load an Optuna study."""
    kwargs = dict(direction="maximize", sampler=sampler, pruner=pruner)
    if storage:
        return optuna.create_study(
            study_name=study_name, storage=storage, load_if_exists=True, **kwargs
        )
    return optuna.create_study(study_name=study_name, **kwargs)

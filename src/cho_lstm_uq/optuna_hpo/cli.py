from __future__ import annotations
import argparse
from pathlib import Path
from .api import run_study


def _parse_args():
    ap = argparse.ArgumentParser(description="strictc_hpo: Optuna runner wrapper")
    ap.add_argument("--train_script", type=str, required=True)
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--base_outdir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--study_name", type=str, default="strictC50L_phase1")
    ap.add_argument("--storage", type=str, default=None)
    ap.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "cmaes"])
    ap.add_argument("--pruner", type=str, default="median", choices=["none", "median", "asha"])
    ap.add_argument("--lambda_mae", type=float, default=0.1)
    ap.add_argument("--mlflow_tracking_uri", type=str, default=None)
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON defaults file")
    return ap.parse_args()


def main():
    args = _parse_args()
    overrides = None  # could be extended to parse --override key=val pairs

    study = run_study(
        train_script=Path(args.train_script),
        input_csv=Path(args.input_csv),
        base_outdir=Path(args.base_outdir),
        epochs=args.epochs,
        seed=args.seed,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        storage=args.storage,
        sampler=args.sampler,
        pruner=(args.pruner if args.pruner != "none" else "none"),
        lambda_mae=args.lambda_mae,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        config_path=(Path(args.config) if args.config else None),
        overrides=overrides,
    )

    bt = study.best_trial
    print("\nBest trial:")
    print(f"  Value: {bt.value:.6f}")
    for k, v in bt.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

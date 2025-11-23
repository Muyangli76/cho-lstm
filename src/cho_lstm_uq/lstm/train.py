"""
Vanilla training entrypoint (no TL/UQ extras).
Expose: run_train(cfg) so notebooks can call directly.
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from ..core import (
    OBS_POOLS, DRIVING, AUX_FEATS,
    carbonize_df, build_scalers, apply_scaler, train_val_split, Seq2SeqDataset,
    StrictCSeq2Seq, carbon_closure_eps_seq, set_seed
)

def run_train(cfg):
    """
    Minimal shell. Port your full pipeline from the ‘vanilla’ script here.
    Keep the same cfg keys so notebook usage stays stable.
    """
    # TODO: paste your working training loop here
    raise NotImplementedError("Implement run_train(cfg) using your vanilla script.")

# optional: tiny helper the notebook can import for metrics signature parity
def eval_seq_metrics(*args, **kwargs):
    """Placeholder to keep import paths stable; port your function later."""
    raise NotImplementedError

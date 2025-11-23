"""
Transfer learning entrypoint with:
 - Stage A/B/C schedule
 - Scale adapter
 - Optional heteroscedastic X/Ab head & variance blending/alignment
Expose: run_transfer(cfg)
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from ..core import (
    OBS_POOLS, DRIVING, AUX_FEATS,
    carbonize_df, build_scalers, apply_scaler, train_val_split, Seq2SeqDataset,
    StrictCSeq2Seq, ScaleInputAdapter, carbon_closure_eps_seq, set_seed
)

def run_transfer(cfg):
    """
    Minimal shell. Port your TL v3 logic here (stages, snapshots, alignment).
    Expect the same cfg fields you listed in your TL script.
    """
    # TODO: wire up: load data → build adapters/model → Stage A/B/C → save snapshots
    raise NotImplementedError("Implement run_transfer(cfg) using your TL script.")

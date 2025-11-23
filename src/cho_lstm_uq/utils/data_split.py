# src/cho_lstm_uq/utils/data_split.py
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass

# -------------------------
# Helpers
# -------------------------

_REP_RE = re.compile(r"^(?P<core>.*?)(_rep(?P<idx>\d+))$")

def _ensure_run_id(df: pd.DataFrame) -> pd.DataFrame:
    if "run_id" not in df.columns:
        if "batch_name" in df.columns:
            df = df.copy()
            df["run_id"] = df["batch_name"].astype(str)
        else:
            raise ValueError("data_split: require 'run_id' or 'batch_name' column.")
    return df

@dataclass
class RepInfo:
    doe: str
    rep_idx: Optional[int]  # None if no explicit _repN

def _parse_doe_and_rep(run_or_batch: str) -> RepInfo:
    """
    Extracts DOE core id and replicate index from a string like '50L_1_50L_rep2'.
    If no _repN suffix, doe = full string, rep_idx=None.
    """
    m = _REP_RE.match(run_or_batch)
    if m:
        return RepInfo(doe=m.group("core"), rep_idx=int(m.group("idx")))
    return RepInfo(doe=run_or_batch, rep_idx=None)

def _rep_index_of(rid: str) -> Optional[int]:
    return _parse_doe_and_rep(rid).rep_idx

def _group_by_doe(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Map: DOE core -> list of UNIQUE run_ids (deduped).
    A 'DOE core' is the part before the '_repN' suffix.
    """
    # DEDUP here so each run_id appears once
    runs = sorted(df["run_id"].astype(str).unique())
    out: Dict[str, List[str]] = defaultdict(list)
    for rid in runs:
        info = _parse_doe_and_rep(rid)
        out[info.doe].append(rid)
    return dict(out)

# -------------------------
# 2-rep DoE deterministic split (one rep per DoE to VAL)
# -------------------------

def train_val_split_pairs(
    df: pd.DataFrame,
    val_frac: float = 0.5,          # kept for API compatibility
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    For DoEs that have exactly TWO explicit replicates (rep1/rep2):
    select one replicate per DoE for validation (≈50% val by default).
    Deterministic with random_state if you want to flip choices.
    """
    df = _ensure_run_id(df)
    rng = np.random.RandomState(random_state)
    by_doe = _group_by_doe(df)

    val_runs: List[str] = []
    for doe, run_ids in by_doe.items():
        reps = sorted(run_ids)
        # Expect exactly two runs for a pair DoE; if >2, skip to be safe
        rep_indices = [_rep_index_of(r) for r in reps]
        if len(reps) == 2 and all(idx in (1, 2) for idx in rep_indices):
            # choose one of the two for val; default 0.5 probability each
            choose_second = rng.rand() < val_frac
            chosen = reps[1] if choose_second else reps[0]
            val_runs.append(chosen)
        else:
            # Non-conforming DOE; keep all in train
            pass

    val_runs = sorted(set(val_runs))
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, val_runs

# -------------------------
# Fixed DoE holdout (LODO-style, but for a chosen DoE)
# -------------------------

def train_val_split_doe_fixed(
    df: pd.DataFrame,
    doe_to_holdout: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Deterministic DoE-holdout split:
      - ALL replicates of `doe_to_holdout` go into VAL
      - ALL others go into TRAIN
    """
    df = _ensure_run_id(df)
    by_doe = _group_by_doe(df)

    if doe_to_holdout not in by_doe:
        raise ValueError(f"DoE '{doe_to_holdout}' not found. Available: {list(by_doe.keys())}")

    val_runs = sorted(by_doe[doe_to_holdout])
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()

    return tr, va, val_runs

# -------------------------
# Basic split (last runs)
# -------------------------

def train_val_split_basic(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Simple last-runs split. Use only as a fallback (can be biased)."""
    df = _ensure_run_id(df)
    runs = sorted(df["run_id"].unique())
    n_val = max(1, int(len(runs) * val_frac))
    val_runs = set(runs[-n_val:])
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, sorted(list(val_runs))

# -------------------------
# k-rep DoE split by rep index (best for 2L with 3 reps)
# -------------------------

def train_val_split_reps_by_index(
    df: pd.DataFrame,
    rep_val_index: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Pick the SAME rep index as validation across ALL DoEs (e.g., all _rep3 are VAL).
    Great for 3× reps per DoE at 2L. If a given DoE lacks that rep index, we keep its runs in TRAIN.
    """
    df = _ensure_run_id(df)
    by_doe = _group_by_doe(df)

    val_runs: List[str] = []
    for doe, run_ids in by_doe.items():
        for rid in run_ids:
            if _rep_index_of(rid) == rep_val_index:
                val_runs.append(rid)

    val_runs = sorted(set(val_runs))
    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, val_runs

# -------------------------
# DoE holdout (unseen DoEs, random subset)
# -------------------------

def train_val_split_doe_holdout(
    df: pd.DataFrame,
    val_frac_doe: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Hold out entire DoEs for validation (all their reps go to VAL).
    Tests generalization to unseen experimental designs.
    """
    df = _ensure_run_id(df)
    rng = np.random.RandomState(random_state)
    by_doe = _group_by_doe(df)
    does = sorted(by_doe.keys())
    n_val = max(1, int(len(does) * val_frac_doe))
    val_does = set(rng.choice(does, size=n_val, replace=False))
    val_runs = sorted([rid for doe in val_does for rid in by_doe[doe]])

    tr = df[~df["run_id"].isin(val_runs)].copy()
    va = df[df["run_id"].isin(val_runs)].copy()
    return tr, va, val_runs

# -------------------------
# Auto strategy
# -------------------------

def train_val_split_auto(
    df: pd.DataFrame,
    *,
    default_pairs_val_frac: float = 0.5,
    rep_val_index: int = 1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Detects layout and picks:
      - all DoEs have exactly 2 reps with _rep1/_rep2  → pairs
      - all DoEs have ≥3 reps with consistent _repN    → reps_by_index(rep_val_index)
      - otherwise                                     → basic (fallback)
    """
    df = _ensure_run_id(df)
    by_doe = _group_by_doe(df)

    rep_counts = []
    only_two = True
    only_three_or_more = True
    has_any_rep_suffix = False

    for doe, run_ids in by_doe.items():
        idxs = [_rep_index_of(r) for r in run_ids]
        if any(i is not None for i in idxs):
            has_any_rep_suffix = True
        unique_idxs = sorted(set([i for i in idxs if i is not None]))
        rep_counts.append(len(unique_idxs))

    if not has_any_rep_suffix:
        # No explicit _repN, cannot be rep-aware → basic
        return train_val_split_basic(df, val_frac=0.2)

    for c in rep_counts:
        if c != 2:
            only_two = False
        if c < 3:
            only_three_or_more = False

    if only_two:
        # All DoEs have exactly two reps → use pairs split
        return train_val_split_pairs(df, val_frac=default_pairs_val_frac, random_state=random_state)
    if only_three_or_more:
        # All DoEs have 3+ reps → choose a fixed rep index as VAL
        return train_val_split_reps_by_index(df, rep_val_index=rep_val_index)

    # Mixed/irregular → safer to use DoE holdout, else basic
    try:
        return train_val_split_doe_holdout(df, val_frac_doe=0.25, random_state=random_state)
    except Exception:
        return train_val_split_basic(df, val_frac=0.2)

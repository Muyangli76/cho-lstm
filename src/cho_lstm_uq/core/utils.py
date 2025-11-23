import random, numpy as np, torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = 42):
    """Set python/numpy/torch RNGs."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class DotDict(dict):
    """Dict with attribute access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

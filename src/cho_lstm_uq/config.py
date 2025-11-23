"""
Run/config defaults for cho_lstm_uq.
Replace with your real dataclass or argparse loader.
"""
DEFAULTS = {
    "SEED": 1337, "EPOCHS": 30, "BATCH_TR": 32, "BATCH_EVAL": 64,
    "T_IN": 12, "T_OUT": 12, "TF_START": 1.0, "TF_END": 0.6,
    "CLIP": 1.0, "PATIENCE": 6, "VAL_FRAC": 0.2,
    "LAMBDA_MB": 1.0, "GAMMA_RES": 0.1, "LAMBDA_XAB": 1.0, "LAMBDA_CONS": 0.2,
    "HIDDEN": 128, "LAYERS": 2, "DROPOUT": 0.1,
    "HET_XAB": True, "LOGV_MIN": -10.0, "LOGV_MAX": 3.0, "SOFTPLUS_VAR": True,
    "LC_EPOCHS": 6
}

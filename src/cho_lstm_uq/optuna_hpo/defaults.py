# Hard defaults (lowest precedence)
# Keep these in sync with your training script arguments.

DEFAULTS = {
    "BATCH_TR": 64,
    "BATCH_EVAL": 128,
    "T_IN": 72,
    "T_OUT": 24,
    "TF_START": 1.0,
    "TF_END": 0.4,
    "CLIP": 1.0,
    "PATIENCE": 8,
    "LAMBDA_MB": 0.05,
    "GAMMA_RES": 2e-5,
    "LAMBDA_XAB": 1.0,
    "LAMBDA_CONS": 0.1,
    "HIDDEN": 128,
    "LAYERS": 2,
    "DROPOUT": 0.1,
    "HET_XAB": True,
    "LOGV_MIN": -10.0,
    "LOGV_MAX": 3.0,
    "SOFTPLUS_VAR": False,
}

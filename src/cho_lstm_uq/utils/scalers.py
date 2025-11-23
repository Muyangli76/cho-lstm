"""Feature scaling helpers (stubs)."""
import pandas as pd
def build_scalers(df: pd.DataFrame, cols): return df[cols].mean(), df[cols].std().replace(0, 1.0)
def apply_scaler(df: pd.DataFrame, cols, mu, sd):
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

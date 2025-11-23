"""IO helpers (stub)."""
import pandas as pd, os
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def read_csv(p): return pd.read_csv(p)
def write_csv(df, p): df.to_csv(p, index=False)

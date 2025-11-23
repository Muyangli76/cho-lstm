"""CLI: train seq2seq strict-C (stub)."""
import argparse, json
from cho_lstm_uq.config import DEFAULTS
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML/JSON config path", default=None)
    ap.add_argument("--TRAIN_INPUT_CSV")
    ap.add_argument("--TRAIN_OUTDIR")
    args = ap.parse_args()
    print("[stub] training would start with:", {k:v for k,v in vars(args).items() if v is not None})
if __name__ == "__main__": main()

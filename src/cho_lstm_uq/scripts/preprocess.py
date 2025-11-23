"""CLI: preprocess raw CSV -> cleaned CSV (stub)."""
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    # TODO: call resample_clean.resample_run per batch_name
    print("[stub] preprocess would read", args.in_csv, "and write", args.out_csv)
if __name__ == "__main__": main()

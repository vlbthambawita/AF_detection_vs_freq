"""
ecg_data_prepare.py

What this script does:
- Load raw ECG datasets such as PTB-XL
- Assign binary labels: AFIB vs NORMAL
- Perform dataset-specific filtering
- Provide CLI interface
- Forward records to preprocessing pipeline
- Each loader assigns stable patient_id for leakage safety
- No splitting is done here; splitting is handled by the preprocessor
- PTB-XL official strat_fold is preserved for fold-based training

To add a new dataset:
    1. Write a new load_<dataset>() function
    2. Return a list of Record objects
    3. Add a condition in main() to route to the loader
"""

import os
import argparse
import ast
from collections import Counter

import wfdb
import pandas as pd
import numpy as np

from ecg_data_preprocessor import Record, prepare_dataset, setup_logger


# ================= Utility =================

def normalize(r: str) -> str:
    """
    Normalize annotation strings by:
        - Converting to uppercase
        - Stripping whitespace

    Used for rhythm annotations.
    """

    return r.upper().strip()


# ================= PTB-XL Loader =================

def load_ptb_xl(dataset_path: str):
    """
    Load PTB-XL dataset.

    Rules:
        - Use ptbxl_database.csv metadata
        - Keep only AFIB vs NORMAL subset
        - Use official strat_fold for fold-based training
        - patient_id and record_id come from metadata

    Returns:
        List[Record]
    """

    meta_path = os.path.join(dataset_path, "ptbxl_database.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at: {meta_path}")

    df = pd.read_csv(meta_path)

    fs_set = set()
    lead_set = set()
    label_counter = Counter()
    label_set = set()

    for _, row in df.iterrows():
        scp_codes = ast.literal_eval(row["scp_codes"])

        fs_set.add(500)
        lead_set.add(12)

        if not scp_codes:
            label_counter["UNKNOWN"] += 1
            label_set.add("UNKNOWN")
        else:
            for c in scp_codes.keys():
                label_counter[c] += 1
                label_set.add(c)

    print("\nFULL DATASET OVERVIEW")
    print(f"  Total records : {len(df)}")
    print(f"  Sampling rates: {sorted(fs_set)}")
    print(f"  Leads         : {sorted(lead_set)}")
    print(f"  Unique labels : {len(label_set)}")
    print("  Labels found  :")

    items = [f"{k}({v})" for k, v in label_counter.items()]

    line = "    "
    max_width = 100

    for item in items:
        if len(line) + len(item) + 2 > max_width:
            print(line.rstrip(", "))
            line = "    "

        line += item + ", "

    if line.strip():
        print(line.rstrip(", "))

    print()

    records = []

    skipped_read_error = 0
    skipped_empty_signal = 0
    skipped_non_binary = 0

    for _, row in df.iterrows():
        scp_codes = ast.literal_eval(row["scp_codes"])

        # Binary labeling rules
        if "AFIB" in scp_codes:
            label = 1
        elif "NORM" in scp_codes and "AFIB" not in scp_codes:
            label = 0
        else:
            skipped_non_binary += 1
            continue

        record_name = row["filename_hr"].replace(".hea", "")
        record_path = os.path.join(dataset_path, record_name)

        try:
            rec = wfdb.rdrecord(record_path)
        except Exception:
            skipped_read_error += 1
            continue

        if rec.p_signal is None:
            skipped_empty_signal += 1
            continue

        records.append(
            Record(
                signal=rec.p_signal.astype(np.float32),
                fs=int(rec.fs),
                label=int(label),
                patient_id=str(row["patient_id"]),
                record_id=str(row["ecg_id"]),
                fold=int(row["strat_fold"]),
            )
        )

    label_counts = Counter([r.label for r in records])

    print("BINARY DATASET AFTER FILTERING")
    print(f"  Records kept       : {len(records)}")
    print(f"  NORMAL records     : {label_counts.get(0, 0)}")
    print(f"  AFIB records       : {label_counts.get(1, 0)}")
    print(f"  Skipped non-binary : {skipped_non_binary}")
    print(f"  Skipped read error : {skipped_read_error}")
    print(f"  Skipped empty      : {skipped_empty_signal}")
    print()

    return records


# ================= CLI =================

def main():
    """
    Command-line interface for loading datasets and forwarding them
    to the preprocessing pipeline.
    """

    ap = argparse.ArgumentParser()

    # Required arguments
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--name", required=True)

    # Optional preprocessing parameters
    ap.add_argument(
        "--fs",
        type=int,
        nargs="+",
        default=[500, 250, 100, 62],
        help="Target sampling frequencies, e.g. --fs 500 250 100 62",
    )

    ap.add_argument(
        "--out_root",
        type=str,
        default="prepared_data",
        help="Root output directory for prepared data.",
    )

    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for experiments. Use None for full dataset.",
    )

    ap.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Train/validation/test split ratio when not using folds.",
    )

    ap.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help=(
            "Patient-safe hold-out test split ratio, e.g. 0.2. "
            "If set, test is created first and CV is applied only on remaining data."
        ),
    )

    ap.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Enable patient-safe stratified K-fold metadata.",
    )

    ap.add_argument(
        "--balance_mode",
        choices=["none", "train", "global", "fold"],
        default="train",
        help=(
            "Balancing strategy. "
            "none=no balancing; "
            "train=save natural folds and balance training later; "
            "global=global downsampling; "
            "fold=balance each fold."
        ),
    )

    ap.add_argument(
        "--flatline_seconds",
        type=float,
        default=3.0,
        help="Minimum continuous flatline duration in seconds before zeroing a lead.",
    )

    args = ap.parse_args()

    setup_logger(os.path.join("logs", f"{args.name}.log"))

    name = args.name.upper()

    if "PTB" in name:
        records = load_ptb_xl(args.dataset_path)
    else:
        raise ValueError("Only PTB-XL is supported now.")

    prepare_dataset(
        dataset_name=args.name,
        records=records,
        out_root=args.out_root,
        target_fs=tuple(args.fs),
        max_samples=args.max_samples,
        split_ratio=tuple(args.split_ratio),
        balance_mode=args.balance_mode,
        folds=args.folds,
        test_ratio=args.test_ratio,
        flatline_seconds=args.flatline_seconds,
    )


if __name__ == "__main__":
    main()
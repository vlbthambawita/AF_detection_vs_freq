"""
ecg_data_loader.py

What this script do:
- Load raw ECG datasets; MIT-BIH, PTB-XL, ECG-Arrhythmia, or any other can be added.
- Assign binary labels (AFIB vs NORMAL)
- Perform dataset-specific filtering
- Provide CLI interface
- Forward records to preprocessing pipeline
- Each loader assigns a stable patient_id(Leakage safety)
- No splitting is done here; splitting is handled by the preprocessor
- PTB-XL official strat_fold is preserved for fold-based training

To add a new dataset:
    1. Write a new load_<dataset>() function
    2. Return a list of Record objects
    3. Add a condition in main() to route to the loader
"""

import os
import argparse
import wfdb
import pandas as pd
import numpy as np
import ast
from collections import Counter
from typing import List

from ecg_data_preprocessor import Record, prepare_dataset, setup_logger


# ================= Utility =================

def normalize(r: str) -> str:
    """
    Normalize annotation strings by:
        - Converting to uppercase
        - Stripping whitespace

    Used for MIT‑BIH rhythm annotations.
    """
    return r.upper().strip()


# ================= 3. PTB-XL Loader (with start fold) ================= 

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
        raise FileNotFoundError("ptbxl_database.csv not found")

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
    max_width = 100  # adjust if you want wider/narrower lines

    for item in items:
        if len(line) + len(item) + 2 > max_width:
            print(line.rstrip(", "))
            line = "    "
        line += item + ", "

    if line.strip():
        print(line.rstrip(", "))

    print()


    records = []

    

    for _, row in df.iterrows():
        scp_codes = ast.literal_eval(row["scp_codes"])

        # Labeling rules
        if "AFIB" in scp_codes:
            label = 1
        elif "NORM" in scp_codes and "AFIB" not in scp_codes:
            label = 0
        else:
            continue

        record_name = row["filename_hr"].replace(".hea", "")
        record_path = os.path.join(dataset_path, record_name)

        try:
            rec = wfdb.rdrecord(record_path)
        except Exception:
            continue

        if rec.p_signal is None:
            continue

        records.append(
            Record(
                signal=rec.p_signal.astype(np.float32),
                fs=int(rec.fs),
                label=label,
                patient_id=row["patient_id"],
                record_id=row["ecg_id"],
                fold=int(row["strat_fold"]),
            )
        )


    return records


# ================= CLI ================= 

def main():
    """
    Command-line interface for loading datasets and forwarding them to the
    preprocessing pipeline.

    Steps:
        1. Parse CLI arguments
        2. Load dataset based on --name
        3. Forward records to prepare_dataset()
    """
    ap = argparse.ArgumentParser()

    # Required arguments
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--name", required=True)

    # Optional preprocessing parameters
    ap.add_argument("--fs", type=int, nargs="+", default=[500, 250, 100])
    ap.add_argument("--out_root", type=str, default="prepared_data")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1])

    # The new feature test split before CV
    ap.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help="Patient-safe hold-out test split ratio (e.g. 0.3). "
            "If set, test is created first and CV is applied only on remaining data.",
    )


    ap.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Enable patient-safe stratified K-fold metadata",
    )

    ap.add_argument(
        "--balance_mode",
        choices=["train", "global", "none"],
        default="global",
    )

    args = ap.parse_args()

    # Create logger
    logger = setup_logger(os.path.join("logs", f"{args.name}.log"))

    name = args.name.upper()

    # Route to the correct loader
    if "PTB" in name:
        records = load_ptb_xl(args.dataset_path)
    else:
        raise ValueError("Only PTB-XL is supported now.")


    # Forward to preprocessing pipeline
    prepare_dataset(
        dataset_name=args.name,
        records=records,
        out_root=args.out_root,
        target_fs=tuple(args.fs),
        max_samples=args.max_samples,
        split_ratio=tuple(args.split_ratio),
        balance_mode=args.balance_mode,
        folds=args.folds,
        test_ratio=args.test_ratio,  # new

    )


if __name__ == "__main__":
    main()



import os
from pyparsing import Path
import wfdb
import pandas as pd
import numpy as np
import ast
from collections import Counter

# ================= Utility =================

# SNOMED codes used in ECG‑Arrhythmia dataset
AFIB = "164889003"
AF   = "164890007"
SR   = "426783006"
#Code used in PTB-XL dataset
norm_ids  = []
afib_ids  = []
aflt_ids  = []
other_ids = []


# ================= || Added Datasets || ================= 


# ================= 2. ECG-ARRHYTHMIA Loader ================= 

def load_ecg_arrhythmia(dataset_path: str, logger):
    """
    Load ECG-Arrhythmia dataset.

    Rules:
        - Labels derived from SNOMED codes in WFDB comments
        - Keep only AFIB or NORMAL
        - patient_id = record basename

    Returns:
        List[Record]
    """


    logger.info("Loading ECG-Arrhythmia raw records (this may take several minutes)...")

    total = 0
    fs_set = set()
    lead_set = set()
    label_counter = Counter()
    label_set = set()

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if not f.endswith(".hea"):
                continue
            total += 1
            base = os.path.join(root, f[:-4])

            try:
                rec = wfdb.rdrecord(base)
            except Exception:
                label_counter["UNREADABLE"] += 1
                label_set.add("UNREADABLE")
                continue

            fs_set.add(int(rec.fs))
            if rec.p_signal is not None:
                lead_set.add(rec.p_signal.shape[1])

            comments = [c.upper() for c in (rec.comments or [])]
            codes = {
                tok for c in comments
                for tok in c.replace(",", " ").split()
                if tok.isdigit()
            }

            if not codes:
                label_counter["UNKNOWN"] += 1
                label_set.add("UNKNOWN")
            else:
                for c in codes:
                    label_counter[c] += 1
                    label_set.add(c)

    print("\nFULL DATASET OVERVIEW")
    print(f"  Total records : {total}")
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

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if not f.endswith(".hea"):
                continue

            base = os.path.join(root, f[:-4])

            try:
                rec = wfdb.rdrecord(base)
            except Exception:
                continue

            if rec.p_signal is None:
                continue

            # Extract SNOMED codes from comments
            comments = [c.upper() for c in (rec.comments or [])]
            codes = {
                tok for c in comments
                for tok in c.replace(",", " ").split()
                if tok.isdigit()
            }

            # Labeling rules
            if AFIB in codes and SR not in codes and AF not in codes:
                label = 1
            elif SR in codes and AFIB not in codes and AF not in codes:
                label = 0
            else:
                continue

            records.append(
                Record(
                    signal=rec.p_signal.astype(np.float32),
                    fs=int(rec.fs),
                    label=label,
                    patient_id=os.path.basename(base),
                    record_id=os.path.basename(base),
                    fold=None,
                )
            )


    return records


# ================= 3. PTB-XL Loader (with start fold) ================= 
def get_label(scp_codes):
    if "AFIB" in scp_codes:
        return 1
    if "NORM" in scp_codes:
        return 0
    if "AFLT" in scp_codes:
        return 2
    
    if "NDT" in scp_codes:
        return 4
    if "NST_" in scp_codes:
        return 5
    if "SVARR" in scp_codes:
        return 6
    if "SVTAC" in scp_codes:
        return 7
    if "PAC" in scp_codes:
        return 8
    return None

def load_ptb_xl_ids(dataset_path: str):
    
    df = pd.read_csv(os.path.join(dataset_path, "ptbxl_database.csv"))
    
    for _, row in df.iterrows():
        scp_codes = ast.literal_eval(row["scp_codes"])
        label = get_label(scp_codes)
        ecg_id = row["ecg_id"]
        
        if label is None:
            continue
        
        if label == 0:
            norm_ids.append(ecg_id)
        if label == 1:
            afib_ids.append(ecg_id)
        if label == 2:
            aflt_ids.append(ecg_id)
        if label >= 4:
            other_ids.append(ecg_id)
    print(f"AFIB samples: {len(afib_ids)}")
    print(f"AFLT samples: {len(aflt_ids)}")
    print(f"Normal samples: {len(norm_ids)}")
    print(f"Other samples: {len(other_ids)}")

def build_ecg_index(root="../data/records500", suffix="_hr"):
    index = {}

    for p in Path(root).rglob(f"*{suffix}.hea"):
        ecg_id = int(p.stem.replace(suffix, ""))
        index[ecg_id] = str(p.with_suffix(""))
    return index



def load_ecg_fast(ecg_id):
    ECG_INDEX = build_ecg_index()
    record = wfdb.rdrecord(ECG_INDEX[ecg_id])
    return record.p_signal, int(record.fs)


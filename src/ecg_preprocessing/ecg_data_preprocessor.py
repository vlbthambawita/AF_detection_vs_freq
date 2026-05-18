"""
ecg_data_preprocessor.py

Memory-safer ECG preprocessing pipeline.

Balancing modes:
- none   : no balancing; folds/splits keep natural distribution
- fold   : balance each fold independently; validation fold is also balanced
- global : globally downsample majority class before saving folds/splits
- train  : keep folds/splits natural here; training balancing happens in train.py/loader.py

Recommended scientific setup:
- Preprocessing: --balance_mode train
- Training: balance only training subset per fold
- Validation: natural/unbalanced
- Test: natural/unbalanced + optional balanced secondary test
"""

import os
import csv
import gc
import logging
from dataclasses import dataclass
from typing import List
from collections import defaultdict, Counter

import numpy as np
import torch
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm


# ================= Data Structure =================

@dataclass
class Record:
    signal: np.ndarray
    fs: int
    label: int
    patient_id: str
    record_id: str
    fold: int | None = None


# ================= Logger Setup =================

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logger initialized successfully")
    return logger


# ================= Fold Utilities =================

def assert_no_patient_leakage(folds_dict):
    seen = set()

    for fold, pids in folds_dict.items():
        overlap = seen & pids

        if overlap:
            raise RuntimeError(f"Patient leakage detected in fold {fold}: {overlap}")

        seen |= pids


def build_patient_folds(records: List[Record], k: int, seed=42):
    if records and records[0].fold is not None and k == 10:
        folds = defaultdict(set)

        for r in records:
            folds[r.fold].add(r.patient_id)

        assert_no_patient_leakage(folds)
        return dict(folds)

    patient_labels = defaultdict(list)

    for r in records:
        patient_labels[r.patient_id].append(r.label)

    patients = list(patient_labels.keys())
    labels = [int(np.round(np.mean(patient_labels[p]))) for p in patients]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds = {}

    for i, (_, idx) in enumerate(skf.split(patients, labels), start=1):
        folds[i] = {patients[j] for j in idx}

    assert_no_patient_leakage(folds)
    return folds


# ================= Signal Processing =================

def clean_signal(x):
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)


def resample_signal(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x.astype(np.float32)

    n = int(round(len(x) * fs_out / fs_in))
    return resample(x, n).astype(np.float32)


def zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def make_segments(x, seg_len):
    return [
        x[i:i + seg_len]
        for i in range(0, len(x) - seg_len + 1, seg_len)
    ]


def zero_flatline_leads(x, fs, eps=1e-6, min_flat_seconds=3.0):
    """
    Zero leads that contain one continuous flatline lasting at least min_flat_seconds.
    x shape: (C, T)
    """

    C, T = x.shape
    zeroed = 0

    min_flat_samples = int(min_flat_seconds * fs)

    for c in range(C):
        dx = np.abs(np.diff(x[c]))

        signal_scale = np.std(x[c]) + 1e-8
        threshold = eps * signal_scale

        flat_mask = dx < threshold

        max_run = 0
        current_run = 0

        for is_flat in flat_mask:
            if is_flat:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        if max_run >= min_flat_samples:
            x[c] = 0.0
            zeroed += 1

    return x, zeroed


def clip_extremes(x, clip_value=15.0):
    before = np.abs(x) > clip_value
    n_clipped = int(before.sum())
    x = np.clip(x, -clip_value, clip_value)

    return x, n_clipped


def fix_shape(X_list):
    """
    Convert list of ECG segments to tensor shape (N, C, T).
    """

    X_np = np.stack(X_list)

    if X_np.ndim == 3 and X_np.shape[1] > X_np.shape[2]:
        X_np = np.transpose(X_np, (0, 2, 1))

    return torch.tensor(X_np, dtype=torch.float32).contiguous()


# ================= Hold-out Test Split =================

def split_patients_test(records: List[Record], test_ratio: float, seed=42):
    pid_to_labels = defaultdict(list)

    for r in records:
        pid_to_labels[r.patient_id].append(r.label)

    patients = list(pid_to_labels.keys())
    labels = [int(np.round(np.mean(pid_to_labels[p]))) for p in patients]

    trainval_pids, test_pids = train_test_split(
        patients,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    trainval_pids = set(trainval_pids)
    test_pids = set(test_pids)

    trainval_records = [
        r for r in records
        if r.patient_id in trainval_pids
    ]

    test_records = [
        r for r in records
        if r.patient_id in test_pids
    ]

    return trainval_records, test_records


# ================= Helper Functions =================

def get_split_key(record, folds, patient_folds, train_p=None, val_p=None):
    if folds is None:
        if record.patient_id in train_p:
            return "train"
        elif record.patient_id in val_p:
            return "val"
        else:
            return "test"

    for fid, pset in patient_folds.items():
        if record.patient_id in pset:
            return fid

    raise RuntimeError(f"Patient {record.patient_id} not found in any fold")


def process_record_to_segments(record, fs, seg_len, flatline_seconds=3.0):
    sig = resample_signal(clean_signal(record.signal), record.fs, fs).T

    sig, n_clipped = clip_extremes(sig, clip_value=15.0)

    sig, n_zeroed = zero_flatline_leads(
        sig,
        fs=fs,
        min_flat_seconds=flatline_seconds
    )

    sig = zscore(sig)

    segments = make_segments(sig.T, seg_len)

    del sig

    return segments, n_clipped, n_zeroed


def class_distribution_from_splits_by_class(splits_by_class):
    total_afib = sum(
        len(splits_by_class[k][1])
        for k in splits_by_class.keys()
    )

    total_normal = sum(
        len(splits_by_class[k][0])
        for k in splits_by_class.keys()
    )

    return total_afib, total_normal


def select_balanced_items(normal_items, afib_items, seed):
    n = min(len(normal_items), len(afib_items))

    if n == 0:
        return [], 0

    rng = np.random.default_rng(seed)

    normal_idx = rng.choice(
        len(normal_items),
        n,
        replace=False
    )

    afib_idx = rng.choice(
        len(afib_items),
        n,
        replace=False
    )

    selected = (
        [normal_items[i] for i in normal_idx] +
        [afib_items[i] for i in afib_idx]
    )

    rng.shuffle(selected)

    return selected, n


def items_to_split_tuple(items):
    if not items:
        return [], [], [], []

    X, y, rids, pids = zip(*items)

    return list(X), list(y), list(rids), list(pids)


def write_metadata_csv(csv_path, fs, split_col, splits):
    segment_index = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [split_col, "patient_id", "record_id", "label", "fs", "segment_index"]
        )

        for split, (X, y, rids, pids) in splits.items():
            for i in range(len(y)):
                writer.writerow(
                    [
                        split,
                        str(pids[i]),
                        str(rids[i]),
                        int(y[i]),
                        fs,
                        segment_index,
                    ]
                )
                segment_index += 1


def build_splits_from_splits_by_class(
    splits_by_class,
    balance_mode,
    folds,
    seed,
    logger,
    fs,
):
    splits = defaultdict(lambda: ([], [], [], []))

    total_afib, total_normal = class_distribution_from_splits_by_class(
        splits_by_class
    )

    logger.info(f"[{fs}Hz] SEGMENT DISTRIBUTION BEFORE BALANCING")
    logger.info(f"  AFIB segments   : {total_afib}")
    logger.info(f"  NORMAL segments : {total_normal}")
    logger.info(f"  TOTAL segments  : {total_afib + total_normal}")

    # ---------- K-FOLD MODE ----------
    if folds is not None:

        if balance_mode in ("none", "train"):
            logger.info(f"[{fs}Hz] K-FOLD MODE WITH balance_mode={balance_mode}")

            if balance_mode == "train":
                logger.info(
                    f"[{fs}Hz] Folds are saved NATURAL/UNBALANCED. "
                    "Training balancing must be applied in train.py/loader.py."
                )
            else:
                logger.info(f"[{fs}Hz] No balancing applied.")

            for fid in sorted(splits_by_class.keys()):
                items = (
                    splits_by_class[fid][0] +
                    splits_by_class[fid][1]
                )

                splits[fid] = items_to_split_tuple(items)

                logger.info(
                    f"[{fs}Hz] fold {fid}: "
                    f"AFIB={len(splits_by_class[fid][1])}, "
                    f"NORMAL={len(splits_by_class[fid][0])}, "
                    f"TOTAL={len(items)}"
                )

            logger.info(f"[{fs}Hz] TOTAL KEPT: {total_afib + total_normal}")
            logger.info(f"[{fs}Hz] DROPPED   : 0")

            return splits

        if balance_mode == "fold":
            logger.info(f"[{fs}Hz] FOLD-LEVEL BALANCING")
            logger.info("  Rule: min(AFIB, NORMAL) per fold")

            total_kept = 0
            kept_afib = 0
            kept_normal = 0

            for fid in sorted(splits_by_class.keys()):
                normal_items = splits_by_class[fid][0]
                afib_items = splits_by_class[fid][1]

                selected, n = select_balanced_items(
                    normal_items,
                    afib_items,
                    seed + int(fid)
                )

                if n == 0:
                    logger.warning(
                        f"[{fs}Hz] fold {fid} has no balanced segments"
                    )
                    continue

                splits[fid] = items_to_split_tuple(selected)

                total_kept += 2 * n
                kept_afib += n
                kept_normal += n

                logger.info(
                    f"[{fs}Hz] fold {fid}: segments={2*n} "
                    f"(AFIB={n}, NORMAL={n})"
                )

                del selected

            logger.info(f"[{fs}Hz] BALANCED TOTAL")
            logger.info(f"  AFIB kept   : {kept_afib}")
            logger.info(f"  NORMAL kept : {kept_normal}")
            logger.info(f"  TOTAL kept  : {total_kept}")
            logger.info(
                f"  DROPPED     : {(total_afib + total_normal) - total_kept}"
            )

            return splits

        if balance_mode == "global":
            logger.info(f"[{fs}Hz] GLOBAL BALANCING BEFORE FOLD SAVE")
            logger.info("  Rule: keep min(total AFIB, total NORMAL) globally")

            all_normal = []
            all_afib = []

            item_to_fold = {}

            for fid in splits_by_class.keys():
                for item in splits_by_class[fid][0]:
                    all_normal.append(item)
                    item_to_fold[id(item)] = fid

                for item in splits_by_class[fid][1]:
                    all_afib.append(item)
                    item_to_fold[id(item)] = fid

            selected, n = select_balanced_items(
                all_normal,
                all_afib,
                seed
            )

            if n == 0:
                raise RuntimeError(
                    f"[{fs}Hz] Cannot globally balance: one class is empty."
                )

            temp_by_fold = defaultdict(list)

            for item in selected:
                fid = item_to_fold[id(item)]
                temp_by_fold[fid].append(item)

            for fid in sorted(temp_by_fold.keys()):
                splits[fid] = items_to_split_tuple(temp_by_fold[fid])

                y = [int(v[1]) for v in temp_by_fold[fid]]

                logger.info(
                    f"[{fs}Hz] fold {fid}: "
                    f"AFIB={sum(y)}, NORMAL={len(y)-sum(y)}, TOTAL={len(y)}"
                )

            logger.info(f"[{fs}Hz] GLOBAL BALANCED TOTAL")
            logger.info(f"  AFIB kept   : {n}")
            logger.info(f"  NORMAL kept : {n}")
            logger.info(f"  TOTAL kept  : {2*n}")
            logger.info(
                f"  DROPPED     : {(total_afib + total_normal) - (2*n)}"
            )

            del all_normal, all_afib, selected, temp_by_fold, item_to_fold

            return splits

        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    # ---------- NON-FOLD MODE ----------
    logger.info(f"[{fs}Hz] SPLIT MODE WITH balance_mode={balance_mode}")

    for split_key in sorted(splits_by_class.keys()):
        normal_items = splits_by_class[split_key][0]
        afib_items = splits_by_class[split_key][1]

        if balance_mode == "none":
            selected = normal_items + afib_items

        elif balance_mode == "train":
            if split_key == "train":
                selected, n = select_balanced_items(
                    normal_items,
                    afib_items,
                    seed
                )

                if n == 0:
                    logger.warning(
                        f"[{fs}Hz] Train balancing skipped: one class missing"
                    )
                    selected = normal_items + afib_items
                else:
                    logger.info(
                        f"[{fs}Hz] TRAIN-ONLY BALANCE APPLIED: "
                        f"AFIB={n}, NORMAL={n}"
                    )
            else:
                selected = normal_items + afib_items

        elif balance_mode in ("global", "fold"):
            selected, n = select_balanced_items(
                normal_items,
                afib_items,
                seed
            )

            if n == 0:
                logger.warning(
                    f"[{fs}Hz] Balancing skipped for {split_key}: one class missing"
                )
                selected = normal_items + afib_items
            else:
                logger.info(
                    f"[{fs}Hz] BALANCE APPLIED TO {split_key}: "
                    f"AFIB={n}, NORMAL={n}"
                )

        else:
            raise ValueError(f"Unknown balance_mode: {balance_mode}")

        splits[split_key] = items_to_split_tuple(selected)

        y = [int(v[1]) for v in selected]

        logger.info(
            f"[{fs}Hz] {split_key}: "
            f"AFIB={sum(y)}, NORMAL={len(y)-sum(y)}, TOTAL={len(y)}"
        )

    return splits


# ================= Main Pipeline =================

def prepare_dataset(
    dataset_name: str,
    records: List[Record],
    out_root="prepared_data",
    target_fs=(500, 250, 100),
    segment_seconds=10,
    max_samples=None,
    split_ratio=(0.7, 0.2, 0.1),
    balance_mode="global",
    folds=None,
    test_ratio=None,
    flatline_seconds=3.0,
    seed=42,
):
    logger = setup_logger(os.path.join("logs", f"{dataset_name}.log"))

    if balance_mode not in ("none", "train", "global", "fold"):
        raise ValueError(
            "balance_mode must be one of: none, train, global, fold"
        )

    labels = [r.label for r in records]
    patients = {r.patient_id for r in records}
    c = Counter(labels)

    logger.info("DATASET SUMMARY (RAW RECORDS)")
    logger.info(f"  Total records  : {len(records)}")
    logger.info(f"  Total patients : {len(patients)}")
    logger.info(f"  AFIB records   : {c.get(1, 0)}")
    logger.info(f"  NORMAL records : {c.get(0, 0)}")
    logger.info(f"  Balance mode   : {balance_mode}")
    logger.info(f"  Flatline rule  : continuous >= {flatline_seconds} seconds")

    # ---------- Optional hold-out test ----------
    test_records = None

    if test_ratio is not None:
        logger.info(f"HOLD-OUT TEST MODE ENABLED (test_ratio={test_ratio})")

        records, test_records = split_patients_test(
            records,
            test_ratio,
            seed
        )

        logger.info(
            f"Patients after split: "
            f"train+val={len({r.patient_id for r in records})}, "
            f"test={len({r.patient_id for r in test_records})}"
        )

    # ---------- Patient-safe split/folds ----------
    if folds is not None:
        logger.info(f"FOLD MODE ENABLED (K={folds})")
        patient_folds = build_patient_folds(records, folds, seed)

        train_p = None
        val_p = None

    else:
        pid_to_label = {r.patient_id: r.label for r in records}

        pids = np.array(list(pid_to_label.keys()))
        pid_labels = np.array([pid_to_label[p] for p in pids])

        train_p, temp, train_y, temp_y = train_test_split(
            pids,
            pid_labels,
            test_size=1 - split_ratio[0],
            stratify=pid_labels,
            random_state=seed,
        )

        val_p, test_p, val_y, test_y = train_test_split(
            temp,
            temp_y,
            test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
            stratify=temp_y,
            random_state=seed,
        )

        train_p = set(train_p)
        val_p = set(val_p)

        patient_folds = None

    # ================= Process each sampling rate =================

    for fs in target_fs:
        logger.info("=" * 70)
        logger.info(f"START PROCESSING {fs} Hz")
        logger.info("=" * 70)

        out_dir = os.path.join(out_root, dataset_name, f"{fs}hz")
        os.makedirs(out_dir, exist_ok=True)

        seg_len = fs * segment_seconds

        records_with_clipping = 0
        records_with_flatlines = 0

        splits_by_class = defaultdict(lambda: {0: [], 1: []})

        # ---------- Stream records into split/fold containers ----------
        for r in tqdm(records, desc=f"{fs}Hz"):
            split_key = get_split_key(
                r,
                folds=folds,
                patient_folds=patient_folds,
                train_p=train_p,
                val_p=val_p,
            )

            segs, n_clipped, n_zeroed = process_record_to_segments(
                r,
                fs,
                seg_len,
                flatline_seconds=flatline_seconds
            )

            if n_clipped > 0:
                records_with_clipping += 1

            if n_zeroed > 0:
                records_with_flatlines += 1

            for seg in segs:
                splits_by_class[split_key][int(r.label)].append(
                    (seg, int(r.label), r.record_id, r.patient_id)
                )

            del segs

        gc.collect()

        logger.info(
            f"[{fs}Hz] QC SUMMARY: "
            f"{records_with_clipping}/{len(records)} records had extreme-value clipping, "
            f"{records_with_flatlines}/{len(records)} records had at least 1 flatline lead"
        )

        # ---------- Build final splits ----------
        splits = build_splits_from_splits_by_class(
            splits_by_class=splits_by_class,
            balance_mode=balance_mode,
            folds=folds,
            seed=seed,
            logger=logger,
            fs=fs,
        )

        gc.collect()

        # ---------- Save metadata and tensors ----------
        csv_path = os.path.join(out_dir, f"samples_{fs}hz.csv")

        if folds is not None:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["fold", "patient_id", "record_id", "label", "fs", "segment_index"]
                )

                segment_index = 0

                for fid in sorted(splits.keys()):
                    X, y, rids, pids = splits[fid]

                    if len(X) == 0:
                        logger.warning(f"[{fs}Hz] fold_{fid}.pt is empty — skipping")
                        continue

                    for i in range(len(y)):
                        writer.writerow(
                            [
                                fid,
                                str(pids[i]),
                                str(rids[i]),
                                int(y[i]),
                                fs,
                                segment_index,
                            ]
                        )
                        segment_index += 1

                    X_tensor = fix_shape(X)
                    y_tensor = torch.tensor(y, dtype=torch.long)

                    torch.save(
                        {
                            "X": X_tensor,
                            "y": y_tensor,
                            "record_ids": [str(v) for v in rids],
                            "patient_ids": [str(v) for v in pids],
                        },
                        os.path.join(out_dir, f"fold_{fid}.pt"),
                    )

                    logger.info(
                        f"[{fs}Hz] saved fold_{fid}.pt "
                        f"(samples={len(y)}, AFIB={sum(y)}, NORMAL={len(y)-sum(y)})"
                    )

                    del X_tensor
                    del y_tensor
                    gc.collect()

        else:
            write_metadata_csv(
                csv_path=csv_path,
                fs=fs,
                split_col="split",
                splits=splits,
            )

            for split in ("train", "val", "test"):
                X, y, rids, pids = splits.get(split, ([], [], [], []))

                if len(X) == 0:
                    logger.warning(f"[{fs}Hz] {split}.pt is empty — skipping")
                    continue

                X_tensor = fix_shape(X)

                torch.save(
                    {
                        "X": X_tensor,
                        "y": torch.tensor(y, dtype=torch.long),
                        "record_ids": list(rids),
                        "patient_ids": list(pids),
                    },
                    os.path.join(out_dir, f"{split}.pt"),
                )

                logger.info(
                    f"[{fs}Hz] saved {split}.pt "
                    f"(samples={len(y)}, AFIB={sum(y)}, NORMAL={len(y)-sum(y)})"
                )

                del X_tensor
                gc.collect()

        gc.collect()

        # ---------- Save hold-out test ----------
        if test_records is not None:
            test_dir = os.path.join(out_root, dataset_name, f"{fs}hz", "test")
            os.makedirs(test_dir, exist_ok=True)

            test_X = []
            test_y = []
            test_rids = []
            test_pids = []

            logger.info(f"[{fs}Hz] Processing hold-out test set")

            for r in tqdm(test_records, desc=f"{fs}Hz test"):
                segs, _, _ = process_record_to_segments(
                    r,
                    fs,
                    seg_len,
                    flatline_seconds=flatline_seconds
                )

                for seg in segs:
                    test_X.append(seg)
                    test_y.append(int(r.label))
                    test_rids.append(str(r.record_id))
                    test_pids.append(str(r.patient_id))

                del segs

            if test_X:
                X_tensor = fix_shape(test_X)

                torch.save(
                    {
                        "X": X_tensor,
                        "y": torch.tensor(test_y, dtype=torch.long),
                        "record_ids": test_rids,
                        "patient_ids": test_pids,
                    },
                    os.path.join(test_dir, "test.pt"),
                )

                logger.info(
                    f"[{fs}Hz] saved HOLD-OUT test.pt "
                    f"(samples={len(test_y)}, AFIB={sum(test_y)}, NORMAL={len(test_y)-sum(test_y)})"
                )

                del X_tensor

            del test_X
            del test_y
            del test_rids
            del test_pids

            gc.collect()

        # ---------- Full cleanup after this frequency ----------
        del splits
        del splits_by_class

        gc.collect()

        logger.info(f"[{fs}Hz] Memory cleaned after saving")

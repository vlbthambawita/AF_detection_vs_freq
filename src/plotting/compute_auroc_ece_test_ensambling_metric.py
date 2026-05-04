#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv

import numpy as np
from sklearn.metrics import roc_auc_score, recall_score


def compute_ece_posclass(y_true, y_prob, n_bins=10):
    """
    Positive-class Expected Calibration Error.

    Compares predicted AF probability with the observed AF prevalence
    inside fixed probability bins.
    """

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    if len(y_true) == 0:
        raise ValueError("y_true is empty.")

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} values, "
            f"but y_prob has {len(y_prob)} values."
        )

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = binids == b

        if np.any(mask):
            frac_pos = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(frac_pos - conf)

    return float(ece)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute final test AUROC, sensitivity, and positive-class ECE "
            "from unbalanced and balanced test NPZ files."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/ptbl-xl"),
        help="Root checkpoint directory. Default: checkpoints/ptbl-xl",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV file or folder. Default: <root>/test_metrics_summary.csv",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for ECE. Default: 10",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for sensitivity. Default: 0.5",
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimals printed in terminal. Default: 4",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if no test NPZ files are found.",
    )

    return parser.parse_args()


def infer_freq_model(npz_path, root):
    """
    Expected path:
        root / <freq> / <model> / roc_test.npz
        root / <freq> / <model> / roc_test_balanced.npz

    Example:
        checkpoints/ptbl-xl/250hz/cnn_lstm/roc_test_balanced.npz
    """

    rel = npz_path.relative_to(root)
    parts = rel.parts

    if len(parts) >= 3:
        freq = parts[0]
        model = parts[1]
    else:
        freq = "unknown"
        model = npz_path.parent.name

    return freq, model


def infer_test_type(npz_path):
    if npz_path.name == "roc_test_balanced.npz":
        return "Bal"
    if npz_path.name == "roc_test.npz":
        return "Unbal"
    return "Unknown"


def freq_as_int(freq):
    try:
        return int(str(freq).replace("hz", "").replace("Hz", ""))
    except ValueError:
        return 999999


def test_type_order(test_type):
    if test_type == "Unbal":
        return 0
    if test_type == "Bal":
        return 1
    return 2


def main():
    args = parse_args()

    root = args.root

    if args.out is None:
        out_csv = root / "test_metrics_summary.csv"
    else:
        out_csv = args.out

    # Allow --out to be either a folder or a CSV file.
    if out_csv.suffix.lower() != ".csv":
        out_csv = out_csv / "test_metrics_summary.csv"

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    rows = []

    npz_files = []
    npz_files.extend(root.rglob("roc_test.npz"))
    npz_files.extend(root.rglob("roc_test_balanced.npz"))

    for npz_path in sorted(npz_files):
        freq, model = infer_freq_model(npz_path, root)
        test_type = infer_test_type(npz_path)

        data = np.load(npz_path)

        if "y_true" not in data or "y_score" not in data:
            raise KeyError(
                f"{npz_path} must contain arrays named 'y_true' and 'y_score'. "
                f"Found: {list(data.keys())}"
            )

        y_true = data["y_true"].astype(int).ravel()
        y_prob = data["y_score"].astype(float).ravel()

        y_pred = (y_prob >= args.threshold).astype(int)

        auroc = roc_auc_score(y_true, y_prob)
        sensitivity = recall_score(y_true, y_pred)

        ece = compute_ece_posclass(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=args.bins,
        )

        mean_prob = float(y_prob.mean())
        prevalence = float(y_true.mean())

        rows.append([
            freq,
            model,
            test_type,
            auroc,
            sensitivity,
            ece,
            mean_prob,
            prevalence,
            str(npz_path),
        ])

    if not rows:
        message = f"No roc_test.npz or roc_test_balanced.npz files found under: {root}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)

    rows.sort(key=lambda x: (x[1], freq_as_int(x[0]), test_type_order(x[2])))

    print("\nFINAL TEST METRICS")
    print("=" * 130)
    print(
        f"{'Model':<12}"
        f"{'Freq':<8}"
        f"{'Test':<8}"
        f"{'AUROC':<12}"
        f"{'Sensitivity':<14}"
        f"{'ECE':<12}"
        f"{'MeanP(AF)':<14}"
        f"{'Prevalence':<12}"
    )
    print("-" * 130)

    d = args.decimals

    for r in rows:
        print(
            f"{r[1]:<12}"
            f"{r[0]:<8}"
            f"{r[2]:<8}"
            f"{r[3]:<12.{d}f}"
            f"{r[4]:<14.{d}f}"
            f"{r[5]:<12.{d}f}"
            f"{r[6]:<14.{d}f}"
            f"{r[7]:<12.{d}f}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frequency",
            "Model",
            "TestType",
            "AUROC",
            "Sensitivity",
            "ECE",
            "MeanPredictedAF",
            "TruePrevalence",
            "SourceFile",
        ])
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
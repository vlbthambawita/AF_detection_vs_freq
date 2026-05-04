#!/usr/bin/env python3
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score


def compute_ece_posclass(y_true, y_prob, n_bins=10):
    """
    Positive-class Expected Calibration Error.

    Compares predicted AF probability with true AF prevalence
    inside probability bins.
    """

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

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
        description="Compute validation AUROC, Sensitivity, and ECE from roc_val.npz files."
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
        help="Output CSV file or folder. Default: <root>/validation_missing_metrics_summary.csv",
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
        help="Decision threshold for sensitivity. Default: 0.5",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of validation folds. Default: 5",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = args.root

    if args.out is None:
        out_csv = root / "validation_missing_metrics_summary.csv"
    else:
        out_csv = args.out

    if out_csv.suffix.lower() != ".csv":
        out_csv = out_csv / "validation_missing_metrics_summary.csv"

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    freq_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.endswith("hz")],
        key=lambda p: int(p.name.replace("hz", ""))
    )

    rows = []

    for freq_dir in freq_dirs:
        for model_dir in sorted([p for p in freq_dir.iterdir() if p.is_dir()]):
            model = model_dir.name
            freq = freq_dir.name

            fold_aurocs = []
            fold_sensitivities = []
            fold_eces = []

            for fold in range(1, args.folds + 1):
                npz_path = model_dir / f"fold_{fold}" / "roc_val.npz"

                if not npz_path.exists():
                    print(f"Skipping missing file: {npz_path}")
                    continue

                data = np.load(npz_path)

                y_true = data["y_true"].astype(int).ravel()
                y_prob = data["y_score"].astype(float).ravel()

                auroc = roc_auc_score(y_true, y_prob)

                y_pred = (y_prob >= args.threshold).astype(int)
                sensitivity = recall_score(y_true, y_pred)

                ece = compute_ece_posclass(
                    y_true=y_true,
                    y_prob=y_prob,
                    n_bins=args.bins,
                )

                fold_aurocs.append(auroc)
                fold_sensitivities.append(sensitivity)
                fold_eces.append(ece)

            if len(fold_aurocs) == 0:
                continue

            rows.append({
                "Frequency": freq,
                "Model": model,
                "AUROC_mean": np.mean(fold_aurocs),
                "AUROC_std": np.std(fold_aurocs),
                "Sensitivity_mean": np.mean(fold_sensitivities),
                "Sensitivity_std": np.std(fold_sensitivities),
                "ECE_mean": np.mean(fold_eces),
                "ECE_std": np.std(fold_eces),
                "FoldsUsed": len(fold_aurocs),
            })

    df = pd.DataFrame(rows)

    print("\nVALIDATION MISSING METRICS")
    print("=" * 100)

    if df.empty:
        print("No validation roc_val.npz files found.")
    else:
        print(df.round(4).to_string(index=False))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
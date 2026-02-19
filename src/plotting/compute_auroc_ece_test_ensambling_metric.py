#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import csv

ROOT = Path("checkpoints/ptbl-xl")


# ================= ECE (positive-class) =================
def compute_ece_posclass(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = binids == b
        if np.any(m):
            frac_pos = y_true[m].mean()
            conf = y_prob[m].mean()
            ece += (m.sum() / N) * abs(frac_pos - conf)

    return float(ece)


# ================= MAIN =================
def main():

    rows = []

    # find all roc_test.npz files from checkpoints/ptbl-xl and compute metrics
    # and store in rows as: [freq, model, auroc, ece, mean_prob, prevalence]
    for npz_path in sorted(ROOT.rglob("roc_test.npz")):

        model_dir = npz_path.parent
        model = model_dir.name              # cnn1d / cnn_lstm
        freq = model_dir.parent.name        # 62hz / 100hz ...

        data = np.load(npz_path)
        y_true = data["y_true"]
        y_prob = data["y_score"]

        auroc = roc_auc_score(y_true, y_prob)
        ece = compute_ece_posclass(y_true, y_prob)
        mean_prob = float(y_prob.mean())
        prevalence = float(y_true.mean())

        rows.append([
            freq,
            model,
            auroc,
            ece,
            mean_prob,
            prevalence
        ])

    # sort nicely
    rows.sort(key=lambda x: (x[1], int(x[0].replace("hz",""))))

    # -------- PRINT TABLE --------
    print("\nFINAL TEST METRICS")
    print("="*80)
    print(f"{'Model':<10}{'Freq':<8}{'AUROC':<10}{'ECE':<10}"
          f"{'MeanP(AF)':<14}{'Prevalence':<12}")
    print("-"*80)

    for r in rows:
        print(f"{r[1]:<10}{r[0]:<8}{r[2]:<10.4f}{r[3]:<10.4f}"
              f"{r[4]:<14.4f}{r[5]:<12.4f}")

    # -------- SAVE CSV --------
    out_csv = ROOT / "test_metrics_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frequency",
            "Model",
            "AUROC",
            "ECE",
            "MeanPredictedAF",
            "TruePrevalence"
        ])
        writer.writerows(rows)

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()

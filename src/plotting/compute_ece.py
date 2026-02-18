import numpy as np
from pathlib import Path
import csv

# ================= CONFIG =================
ROOT = Path("../checkpoints/ptbl-xl")

FREQS = ["62hz", "100hz", "250hz", "500hz"]
MODELS = ["cnn1d", "cnn_lstm"]
FOLDS = 5

N_BINS = 15          # standard choice in medical ML
EPS = 1e-8

OUTFILE = ROOT / "ece_validation_summary.csv"


# ================= ECE FUNCTION =================
def compute_ece(y_true, y_prob, n_bins=15):
    """
    Expected Calibration Error (ECE)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])

        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / N

    return ece


# ================= LOAD NPZ =================
def load_fold_npz(base: Path, fold: int):
    p = base / f"fold_{fold}" / "roc_val.npz"
    if not p.exists():
        raise FileNotFoundError(p)

    data = np.load(p)

    y_true = np.asarray(data["y_true"]).astype(int).ravel()
    y_score = np.asarray(data["y_score"]).astype(float).ravel()

    y_score = np.clip(y_score, EPS, 1 - EPS)

    return y_true, y_score


# ================= MAIN =================
rows = []
print("\nComputing ECE from validation folds...\n")

for freq in FREQS:
    for model in MODELS:

        base = ROOT / freq / model
        eces = []

        for fold in range(1, FOLDS + 1):
            y, s = load_fold_npz(base, fold)
            ece = compute_ece(y, s, N_BINS)
            eces.append(ece)

            print(f"{freq} | {model} | fold {fold} → ECE = {ece:.5f}")

        mean_ece = float(np.mean(eces))
        std_ece = float(np.std(eces))

        print(f"==> {freq} | {model} | "
              f"ECE = {mean_ece:.5f} ± {std_ece:.5f}\n")

        rows.append([freq, model, mean_ece, std_ece])


# ================= SAVE CSV =================
with open(OUTFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frequency", "Model", "ECE_mean", "ECE_std"])
    writer.writerows(rows)

print("Saved:", OUTFILE)


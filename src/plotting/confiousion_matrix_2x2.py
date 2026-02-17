import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

# ================= CONFIG =================
ROOT = Path("checkpoints/ptbl-xl")
FREQS = ["62hz", "100hz", "250hz", "500hz"]
FOLDS = 5


MODELS = ["cnn1d", "cnn_lstm"]   

THRESH = 0.5
EPS = 1e-6

# normalize: None | "true" | "pred" | "all"
NORMALIZE = None

OUT_FILE = ROOT / "confusion_matrix_grid_4x2_cnn1d_vs_cnnlstm.png"


def load_pooled_from_folds(base: Path, folds: int):
    """Pool y_true and y_score across folds from roc_val.npz."""
    y_all = []
    p_all = []

    for fold in range(1, folds + 1):
        roc_path = base / f"fold_{fold}" / "roc_val.npz"
        if not roc_path.exists():
            raise FileNotFoundError(f"Missing: {roc_path}")

        data = np.load(roc_path)
        y = np.asarray(data["y_true"]).astype(int).ravel()
        p = np.asarray(data["y_score"]).astype(float).ravel()

        if y.size != p.size:
            raise ValueError(f"{roc_path}: y_true and y_score have different lengths")

        p = np.clip(p, EPS, 1 - EPS)
        y_all.append(y)
        p_all.append(p)

    return np.concatenate(y_all), np.concatenate(p_all)


# ================= PLOT =================
# ================= PLOT =================

for model in MODELS:

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        constrained_layout=True
    )

    axes = axes.flatten() 

    for i, freq in enumerate(FREQS):
        base = ROOT / freq / model

        y_true, y_prob = load_pooled_from_folds(base, FOLDS)
        y_pred = (y_prob >= THRESH).astype(int)

        ax = axes[i]

        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            ax=ax,
            normalize=NORMALIZE,
            values_format=".2f" if NORMALIZE else "d",
        )

        ax.set_title(freq.upper(), fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        f"{model.upper()} Confusion Matrices (pooled {FOLDS}-fold CV, thr={THRESH:.2f})",
        fontsize=14
    )

    out_file = ROOT / f"confusion_matrix_2x2_{model}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

ROOT = Path("checkpoints/ptbl-xl")
MODEL = "cnn_lstm"
FOLDS = 5
FREQS = ["62hz", "100hz", "250hz", "500hz"]

for freq in FREQS:
    base = ROOT / freq / MODEL

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    fig, ax = plt.subplots(figsize=(7, 7))


    for fold in range(1, FOLDS + 1):
        data = np.load(base / f"fold_{fold}" / "roc_val.npz")
        y_true, y_score = data["y_true"], data["y_score"]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.20,
            color="tab:blue",
        )

    # ---- Mean ROC ----
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="tab:blue",
        lw=3,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )

    # ---- ±1 std band ----
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="tab:blue",
        alpha=0.15,
        label="±1 std. dev.",
    )

    # ---- Chance level ----
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1.5,
        color="gray",
        alpha=0.7,
        label="Chance",
    )

    # ---- Axis formatting ----
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curve (5-Fold Cross-Validation) – {freq.replace('hz','')} Hz",
        fontsize=14,
        pad=12,
    )

    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()

    out_path = base / f"roc_cv_{freq}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

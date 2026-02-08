import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path("checkpoints/ptbl-xl")
MODEL = "cnn_lstm"
FOLDS = 5
FREQS = ["62hz", "100hz", "250hz", "500hz"]

colors = plt.cm.tab10.colors  

for freq in FREQS:
    base = ROOT / freq / MODEL

    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # ================= Fold ROC curves =================
    for i, fold in enumerate(range(1, FOLDS + 1)):
        data = np.load(base / f"fold_{fold}" / "roc_val.npz")
        y_true = data["y_true"]
        y_score = data["y_score"]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(
            fpr,
            tpr,
            lw=1.2,
            linestyle="--",
            color=colors[i],
            alpha=0.8,
            drawstyle="steps-post",
            label=f"ROC fold {i} (AUC = {roc_auc:.3f})",
        )

    # ================= Chance =================
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance level (AUC = 0.50)")

    # ================= Mean ROC =================
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        lw=3.0,
        drawstyle="steps-post",
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )

    # ================= Std band =================
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="grey",
        alpha=0.25,
        label="± 1 std. dev.",
    )

    # ================= Inset =================
    axins = inset_axes(ax, width="38%", height="38%", loc="lower left", borderpad=2)

    for i, tpr in enumerate(tprs):
        axins.plot(
            mean_fpr,
            tpr,
            linestyle="--",
            lw=1.0,
            color=colors[i],
            drawstyle="steps-post",
        )

    axins.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        lw=2.5,
        drawstyle="steps-post",
    )

    axins.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="grey",
        alpha=0.25,
    )

    # zoom region
    axins.set_xlim(0.0, 0.08)
    axins.set_ylim(0.85, 1.01)
    axins.set_xticks([0.0, 0.04, 0.08])
    axins.set_yticks([0.85, 0.90, 0.95, 1.00])
    axins.grid(True, linestyle=":", alpha=0.5)

    # ================= Formatting =================
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(
        f"Mean ROC curve with variability\n(5-fold CV, {freq.replace('hz','')} Hz)",
        fontsize=15,
    )
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)

    out_path = base / f"roc_cv_sklearn_style_{freq}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
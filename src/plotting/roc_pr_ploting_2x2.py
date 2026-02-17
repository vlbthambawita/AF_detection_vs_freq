import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
)

# ================= CONFIG =================
ROOT = Path("checkpoints/ptbl-xl")
FREQS = ["62hz", "100hz", "250hz", "500hz"]
FOLDS = 5

MODELS = ["cnn1d", "cnn_lstm"]
MODEL_TITLES = {"cnn1d": "CNN1D", "cnn_lstm": "CNN-LSTM"}

# requested: nicer colors
MODEL_COLORS = {"cnn1d": "tab:blue", "cnn_lstm": "tab:orange"}

N_POINTS = 300
EPS = 1e-6
PLOT_STD_BAND = True

ROC_ZOOM_XLIM = (0.0, 0.08)
ROC_ZOOM_YLIM = (0.85, 1.01)

PR_ZOOM_XLIM = (0.70, 1.00)
PR_ZOOM_YLIM = (0.70, 1.01)

OUT_ROC = ROOT / "roc_mean_grid_4x2_cnn1d_cnnlstm.png"
OUT_PR  = ROOT / "pr_mean_grid_4x2_cnn1d_cnnlstm.png"

def pooled_y_true(base: Path, folds: int) -> np.ndarray:
    ys = []
    for fold in range(1, folds + 1):
        y, _ = load_fold_npz(base, fold)
        ys.append(y)
    return np.concatenate(ys)

def load_fold_npz(base: Path, fold: int):
    p = base / f"fold_{fold}" / "roc_val.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    data = np.load(p)
    y = np.asarray(data["y_true"]).astype(int).ravel()
    s = np.asarray(data["y_score"]).astype(float).ravel()
    if y.size != s.size:
        raise ValueError(f"{p}: y_true and y_score must have same length")

    s = np.clip(s, EPS, 1 - EPS)
    return y, s


def mean_roc_over_folds(base: Path, folds: int, n_points: int):
    mean_fpr = np.linspace(0.0, 1.0, n_points)
    tprs, aucs = [], []

    for fold in range(1, folds + 1):
        y, s = load_fold_npz(base, fold)
        fpr, tpr, _ = roc_curve(y, s)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)

    tprs = np.vstack(tprs)
    return mean_fpr, tprs.mean(0), tprs.std(0), float(np.mean(aucs)), float(np.std(aucs))


def _interp_on_sorted_x(x_new, x, y):
    """
    Safe interpolation:
    - sort by x ascending
    - remove duplicate x by keeping max y for same x (stable for PR)
    """
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # merge duplicates in x by taking max y at each x
    # (PR curves can repeat recall values)
    x_unique = []
    y_unique = []
    i = 0
    while i < len(x):
        j = i
        y_max = y[i]
        while j + 1 < len(x) and x[j + 1] == x[i]:
            j += 1
            y_max = max(y_max, y[j])
        x_unique.append(x[i])
        y_unique.append(y_max)
        i = j + 1

    x_unique = np.asarray(x_unique, dtype=float)
    y_unique = np.asarray(y_unique, dtype=float)

    return np.interp(x_new, x_unique, y_unique)


def mean_pr_over_folds(base: Path, folds: int, n_points: int):
    mean_recall = np.linspace(0.0, 1.0, n_points)
    precs, aps = [], []

    for fold in range(1, folds + 1):
        y, s = load_fold_npz(base, fold)

        precision, recall, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        aps.append(ap)

        # FIX: recall may not be sorted for interpolation; sort + dedup first
        interp_prec = _interp_on_sorted_x(mean_recall, recall, precision)
        precs.append(interp_prec)

    precs = np.vstack(precs)
    return mean_recall, precs.mean(0), precs.std(0), float(np.mean(aps)), float(np.std(aps))


def add_inset(ax, x, y, y_std, xlim, ylim, xticks, yticks, color, baseline_y=None):
    axins = inset_axes(ax, width="38%", height="38%", loc="lower left", borderpad=1.6)
    if baseline_y is not None:
         axins.hlines(baseline_y, xlim[0], xlim[1], linestyles="--", linewidth=0.9, color="0.5", alpha=0.7)
    axins.plot(x, y, lw=2.0, drawstyle="steps-post", color=color)
    if y_std is not None and PLOT_STD_BAND:
        axins.fill_between(
            x,
            np.maximum(y - y_std, 0),
            np.minimum(y + y_std, 1),
            alpha=0.18,
            color=color,
        )

    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xticks(xticks)
    axins.set_yticks(yticks)
    axins.grid(True, linestyle=":", alpha=0.55)


# ================= ROC (2x2 per model) =================
for model in MODELS:

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, freq in enumerate(FREQS):

        base = ROOT / freq / model
        fpr, mean_tpr, std_tpr, mean_auc, std_auc = mean_roc_over_folds(
            base, FOLDS, N_POINTS
        )

        ax = axes[i]
        color = MODEL_COLORS[model]

        ax.plot([0, 1], [0, 1], "--", lw=1.0, color="0.3")
        ax.plot(fpr, mean_tpr, lw=2.7, drawstyle="steps-post", color=color)

        if PLOT_STD_BAND:
            ax.fill_between(
                fpr,
                np.maximum(mean_tpr - std_tpr, 0),
                np.minimum(mean_tpr + std_tpr, 1),
                alpha=0.18,
                color=color,
            )

        add_inset(
            ax, fpr, mean_tpr, std_tpr,
            ROC_ZOOM_XLIM, ROC_ZOOM_YLIM,
            [0.0, 0.04, 0.08],
            [0.85, 0.90, 0.95, 1.00],
            color=color,
        )

        ax.set_title(freq.upper())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.55)

        ax.text(0.55, 0.07,
                f"AUC={mean_auc:.3f}±{std_auc:.3f}",
                transform=ax.transAxes,
                fontsize=10)

    fig.suptitle(f"{MODEL_TITLES[model]} — Mean ROC ({FOLDS}-fold CV)", fontsize=14)

    out = ROOT / f"roc_2x2_{model}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out)
# ================= PR (2x2 per model) =================
for model in MODELS:

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, freq in enumerate(FREQS):

        base = ROOT / freq / model
        recall, mean_prec, std_prec, mean_ap, std_ap = mean_pr_over_folds(
            base, FOLDS, N_POINTS
        )

        ax = axes[i]
        color = MODEL_COLORS[model]

        y_pool = pooled_y_true(base, FOLDS)
        prevalence = float(y_pool.mean())

        ax.hlines(prevalence, 0, 1, linestyle="--", lw=1.0, color="0.5")

        ax.plot(recall, mean_prec, lw=2.7,
                drawstyle="steps-post", color=color)

        if PLOT_STD_BAND:
            ax.fill_between(
                recall,
                np.maximum(mean_prec - std_prec, 0),
                np.minimum(mean_prec + std_prec, 1),
                alpha=0.18,
                color=color,
            )

        add_inset(
            ax, recall, mean_prec, std_prec,
            PR_ZOOM_XLIM, PR_ZOOM_YLIM,
            [0.70, 0.85, 1.00],
            [0.70, 0.80, 0.90, 1.00],
            color=color,
            baseline_y=prevalence,
        )

        ax.set_title(freq.upper())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.55)

        ax.text(0.55, 0.07,
                f"AP={mean_ap:.3f}±{std_ap:.3f}",
                transform=ax.transAxes,
                fontsize=10)

    fig.suptitle(f"{MODEL_TITLES[model]} — Mean PR ({FOLDS}-fold CV)", fontsize=14)

    out = ROOT / f"pr_2x2_{model}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out)
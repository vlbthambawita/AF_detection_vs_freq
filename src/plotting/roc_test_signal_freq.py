import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import roc_curve, auc


# ================= DEFAULT CONFIG =================
EPS = 1e-6
N_POINTS = 300
PLOT_STD_BAND = True

MODEL = "cnn_lstm"
MODEL_TITLE = "CNN-LSTM"
MODEL_COLOR = "tab:orange"

ROC_ZOOM_XLIM = (0.0, 0.08)
ROC_ZOOM_YLIM = (0.85, 1.01)


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
    tprs = []
    aucs = []

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

    return (
        mean_fpr,
        tprs.mean(axis=0),
        tprs.std(axis=0),
        float(np.mean(aucs)),
        float(np.std(aucs)),
    )


def add_inset(ax, x, y, y_std, color):
    axins = inset_axes(
        ax,
        width="38%",
        height="38%",
        loc="lower left",
        borderpad=1.6,
    )

    axins.plot(x, y, lw=2.0, drawstyle="steps-post", color=color)

    if y_std is not None and PLOT_STD_BAND:
        axins.fill_between(
            x,
            np.maximum(y - y_std, 0),
            np.minimum(y + y_std, 1),
            alpha=0.18,
            color=color,
        )

    axins.set_xlim(*ROC_ZOOM_XLIM)
    axins.set_ylim(*ROC_ZOOM_YLIM)
    axins.set_xticks([0.0, 0.04, 0.08])
    axins.set_yticks([0.85, 0.90, 0.95, 1.00])
    axins.grid(True, linestyle=":", alpha=0.55)


def plot_roc(root: Path, freq: str, folds: int, out: Path | None):
    base = root / freq / MODEL

    fpr, mean_tpr, std_tpr, mean_auc, std_auc = mean_roc_over_folds(
        base=base,
        folds=folds,
        n_points=N_POINTS,
    )

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    ax.plot([0, 1], [0, 1], "--", lw=1.0, color="0.3")
    ax.plot(
        fpr,
        mean_tpr,
        lw=2.7,
        drawstyle="steps-post",
        color=MODEL_COLOR,
        label=f"AUC={mean_auc:.3f}±{std_auc:.3f}",
    )

    if PLOT_STD_BAND:
        ax.fill_between(
            fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.18,
            color=MODEL_COLOR,
        )

    add_inset(ax, fpr, mean_tpr, std_tpr, MODEL_COLOR)

    ax.set_title(f"{MODEL_TITLE} — Mean ROC ({folds}-fold CV, {freq.upper()})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower right")

    if out is None:
        out = root / f"roc_{freq}_{MODEL}.png"

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean ROC curve for CNN-LSTM at one sampling frequency."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/"),
        help="Root checkpoint directory. Default: checkpoints/ptbl-xl",
    )

    parser.add_argument(
        "--freq",
        type=str,
        default="250hz",
        choices=["62hz", "100hz", "250hz", "500hz"],
        help="Sampling frequency to plot. Default: 250hz",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds. Default: 5",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: root/roc_<freq>_cnn_lstm.png",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_roc(
        root=args.root,
        freq=args.freq,
        folds=args.folds,
        out=args.out,
    )
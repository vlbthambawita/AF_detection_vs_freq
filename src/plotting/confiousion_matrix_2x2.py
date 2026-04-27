import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay


# ================= DEFAULT CONFIG =================

EPS = 1e-6

DEFAULT_FREQS = ["62hz", "100hz", "250hz", "500hz"]
DEFAULT_MODELS = ["cnn1d", "cnn_lstm"]

MODEL_TITLES = {
    "cnn1d": "CNN1D",
    "cnn_lstm": "CNN-LSTM",
}


def load_pooled_from_folds(base: Path, folds: int):
    """
    Pool y_true and y_score across folds from roc_val.npz.
    """
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
            raise ValueError(
                f"{roc_path}: y_true and y_score have different lengths"
            )

        p = np.clip(p, EPS, 1 - EPS)

        y_all.append(y)
        p_all.append(p)

    return np.concatenate(y_all), np.concatenate(p_all)


def get_grid_shape(n_items: int):
    """
    Return dynamic subplot grid shape.
    1 item  -> 1x1
    2 items -> 1x2
    3-4     -> 2x2
    More    -> automatic compact grid
    """
    if n_items == 1:
        return 1, 1

    if n_items == 2:
        return 1, 2

    if n_items <= 4:
        return 2, 2

    ncols = 2
    nrows = math.ceil(n_items / ncols)

    return nrows, ncols


def plot_confusion_matrices(
    root: Path,
    freqs: list[str],
    models: list[str],
    folds: int,
    threshold: float,
    normalize: str | None,
    out_dir: Path | None,
):
    if out_dir is None:
        out_dir = root

    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        n_freqs = len(freqs)
        nrows, ncols = get_grid_shape(n_freqs)

        fig_width = 5 * ncols
        fig_height = 4.5 * nrows

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
        )

        if n_freqs == 1:
            axes = np.array([axes])
        else:
            axes = np.asarray(axes).flatten()

        for i, freq in enumerate(freqs):
            base = root / freq / model

            y_true, y_prob = load_pooled_from_folds(base, folds)
            y_pred = (y_prob >= threshold).astype(int)

            ax = axes[i]

            ConfusionMatrixDisplay.from_predictions(
                y_true=y_true,
                y_pred=y_pred,
                ax=ax,
                normalize=normalize,
                values_format=".2f" if normalize else "d",
                display_labels=["Normal", "AFIB"],
            )

            ax.set_title(freq.upper(), fontsize=12)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")

        # Hide unused subplot positions if grid has extra axes
        for j in range(n_freqs, len(axes)):
            axes[j].axis("off")

        model_title = MODEL_TITLES.get(model, model.upper())

        if n_freqs == 1:
            title = (
                f"{model_title} Confusion Matrix "
                f"({freqs[0].upper()}, pooled {folds}-fold CV, threshold={threshold:.2f})"
            )
            out_file = out_dir / f"confusion_matrix_{freqs[0]}_{model}.png"
        else:
            title = (
                f"{model_title} Confusion Matrices "
                f"(pooled {folds}-fold CV, threshold={threshold:.2f})"
            )
            freq_name = "_".join(freqs)
            out_file = out_dir / f"confusion_matrix_{freq_name}_{model}.png"

        fig.suptitle(title, fontsize=14)

        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot pooled confusion matrices from fold-level roc_val.npz files."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/ptbl-xl"),
        help="Root checkpoint directory. Default: checkpoints/ptbl-xl",
    )

    parser.add_argument(
        "--freqs",
        nargs="+",
        default=DEFAULT_FREQS,
        choices=DEFAULT_FREQS,
        help="Sampling frequencies to plot. Example: --freqs 250hz",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to plot. Example: --models cnn_lstm",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds. Default: 5",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probability to class label. Default: 0.5",
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "true", "pred", "all"],
        help="Normalize confusion matrix: none, true, pred, or all. Default: none",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same as root.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    normalize = None if args.normalize == "none" else args.normalize

    plot_confusion_matrices(
        root=args.root,
        freqs=args.freqs,
        models=args.models,
        folds=args.folds,
        threshold=args.threshold,
        normalize=normalize,
        out_dir=args.out_dir,
    )
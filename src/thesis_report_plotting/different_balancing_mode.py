import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


LABEL_NAMES = {
    0: "NORMAL",
    1: "AFIB",
}

COLORS = {
    0: "#2ca02c",  # NORMAL
    1: "#d62728",  # AFIB
}


# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------

def load_y_from_pt(pt_path: Path):
    if not pt_path.exists():
        return None

    data = torch.load(pt_path, map_location="cpu")
    y = data["y"]

    if torch.is_tensor(y):
        y = y.cpu().numpy()

    return np.asarray(y).astype(int)


def count_y(y):
    if y is None:
        return {0: 0, 1: 0}

    return {
        0: int((y == 0).sum()),
        1: int((y == 1).sum()),
    }


def count_pt(pt_path: Path):
    y = load_y_from_pt(pt_path)
    return count_y(y)


def add_count_dicts(dicts):
    return {
        0: sum(d[0] for d in dicts),
        1: sum(d[1] for d in dicts),
    }


def mean_count_dicts(dicts):
    if not dicts:
        return {0: 0, 1: 0}

    return {
        0: int(round(np.mean([d[0] for d in dicts]))),
        1: int(round(np.mean([d[1] for d in dicts]))),
    }


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------

def plot_stacked_bars(ax, labels, counts, title, rotate=0):
    normal = [c[0] for c in counts]
    afib = [c[1] for c in counts]
    x = np.arange(len(labels))

    ax.bar(x, normal, color=COLORS[0], label="NORMAL")
    ax.bar(x, afib, bottom=normal, color=COLORS[1], label="AFIB")

    max_total = max([n + a for n, a in zip(normal, afib)] + [1])

    for i, (n, a) in enumerate(zip(normal, afib)):
        total = n + a

        if n > 0:
            ax.text(
                i,
                n / 2,
                str(n),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        if a > 0:
            ax.text(
                i,
                n + a / 2,
                str(a),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.text(
            i,
            total + max_total * 0.02,
            str(total),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of ECG segments")
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=rotate,
        ha="right" if rotate else "center",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")


# ------------------------------------------------------------
# Manual split plotting
# ------------------------------------------------------------

def plot_manual_mode(mode_dir: Path, mode_name: str, out_file: Path):
    train_pt = mode_dir / "train.pt"
    val_pt = mode_dir / "val.pt"
    test_pt = mode_dir / "test.pt"

    train_counts = count_pt(train_pt)
    val_counts = count_pt(val_pt)
    test_counts = count_pt(test_pt)

    total_counts = add_count_dicts([train_counts, val_counts, test_counts])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_stacked_bars(
        axes[0],
        ["Prepared dataset"],
        [total_counts],
        title=f"{mode_name}: total prepared data",
    )

    plot_stacked_bars(
        axes[1],
        ["Train", "Validation", "Test"],
        [train_counts, val_counts, test_counts],
        title=f"{mode_name}: train / validation / test",
    )

    fig.suptitle(
        f"PTB-XL balancing mode: {mode_name}",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Fold plotting with held-out test
# ------------------------------------------------------------

def get_fold_files(mode_dir: Path):
    return sorted(
        mode_dir.glob("fold_*.pt"),
        key=lambda p: int(p.stem.replace("fold_", "")),
    )


def plot_fold_mode(mode_dir: Path, out_file: Path):
    fold_files = get_fold_files(mode_dir)

    if not fold_files:
        raise FileNotFoundError(f"No fold_*.pt files found in: {mode_dir}")

    fold_counts = []

    for p in fold_files:
        fold_counts.append(count_pt(p))

    test_pt = mode_dir / "test" / "test.pt"
    test_counts = count_pt(test_pt)

    has_test = test_pt.exists()

    # In K-fold training:
    # validation = one fold
    # training = all other folds
    cv_train_counts = []
    cv_val_counts = []

    for i in range(len(fold_counts)):
        val_c = fold_counts[i]
        train_c = add_count_dicts(
            [fold_counts[j] for j in range(len(fold_counts)) if j != i]
        )

        cv_train_counts.append(train_c)
        cv_val_counts.append(val_c)

    mean_train = mean_count_dicts(cv_train_counts)
    mean_val = mean_count_dicts(cv_val_counts)

    total_fold_data = add_count_dicts(fold_counts)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel 1: total fold data and test
    left_labels = ["K-fold data"]
    left_counts = [total_fold_data]

    if has_test:
        left_labels.append("Held-out test")
        left_counts.append(test_counts)

    plot_stacked_bars(
        axes[0],
        left_labels,
        left_counts,
        title="Fold mode dataset structure",
    )

    # Panel 2: average CV train/validation + test
    middle_labels = ["CV train\n(mean)", "CV validation\n(mean)"]
    middle_counts = [mean_train, mean_val]

    if has_test:
        middle_labels.append("Held-out\ntest")
        middle_counts.append(test_counts)

    plot_stacked_bars(
        axes[1],
        middle_labels,
        middle_counts,
        title="Cross-validation view with test",
    )

    # Panel 3: individual fold files + test
    fold_labels = [p.stem.replace("_", " ").title() for p in fold_files]
    fold_panel_counts = fold_counts.copy()

    if has_test:
        fold_labels.append("Held-out\nTest")
        fold_panel_counts.append(test_counts)

    plot_stacked_bars(
        axes[2],
        fold_labels,
        fold_panel_counts,
        title="Saved fold files and held-out test",
        rotate=30,
    )

    if not has_test:
        axes[2].text(
            0.5,
            -0.25,
            "No held-out test was found. Run preprocessing with --test_ratio to create test/test.pt.",
            transform=axes[2].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.suptitle(
        "PTB-XL balancing mode: Fold",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Combined overview
# ------------------------------------------------------------

def plot_combined_overview(base_root: Path, dataset_name: str, fs: int, out_file: Path):
    modes = ["none", "train", "global", "fold"]

    labels = []
    counts = []

    for mode in modes:
        mode_dir = base_root / mode / dataset_name / f"{fs}hz"

        if mode == "fold":
            fold_files = get_fold_files(mode_dir)

            if not fold_files:
                continue

            fold_total = add_count_dicts([count_pt(p) for p in fold_files])
            test_pt = mode_dir / "test" / "test.pt"

            labels.append("Fold\nCV data")
            counts.append(fold_total)

            if test_pt.exists():
                labels.append("Fold\nTest")
                counts.append(count_pt(test_pt))

        else:
            train_pt = mode_dir / "train.pt"
            val_pt = mode_dir / "val.pt"
            test_pt = mode_dir / "test.pt"

            if not train_pt.exists():
                continue

            total = add_count_dicts([
                count_pt(train_pt),
                count_pt(val_pt),
                count_pt(test_pt),
            ])

            labels.append(mode.capitalize())
            counts.append(total)

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_stacked_bars(
        ax,
        labels,
        counts,
        title="Prepared PTB-XL data across balancing modes",
        rotate=20,
    )

    fig.suptitle(
        f"PTB-XL balancing overview at {fs} Hz",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot prepared PTB-XL balancing modes from .pt files."
    )

    parser.add_argument(
        "--base_root",
        type=str,
        default="prepared_balance",
        help="Root folder containing none/train/global/fold outputs.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ptbl-xl",
        help="Dataset folder name used by ecg_data_prepare.py.",
    )

    parser.add_argument(
        "--fs",
        type=int,
        default=250,
        help="Sampling frequency folder to plot, e.g. 250.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures/balancing_modes",
        help="Output directory for plots.",
    )

    args = parser.parse_args()

    base_root = Path(args.base_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_dir_name = f"{args.fs}hz"

    # B1: none
    none_dir = base_root / "none" / args.dataset_name / fs_dir_name
    if none_dir.exists():
        plot_manual_mode(
            none_dir,
            "B1 — None",
            out_dir / f"b1_none_{args.fs}hz.png",
        )

    # B2: train
    train_dir = base_root / "train" / args.dataset_name / fs_dir_name
    if train_dir.exists():
        plot_manual_mode(
            train_dir,
            "B2 — Train",
            out_dir / f"b2_train_{args.fs}hz.png",
        )

    # B4: global
    global_dir = base_root / "global" / args.dataset_name / fs_dir_name
    if global_dir.exists():
        plot_manual_mode(
            global_dir,
            "B4 — Global",
            out_dir / f"b4_global_{args.fs}hz.png",
        )

    # B3: fold
    fold_dir = base_root / "fold" / args.dataset_name / fs_dir_name
    if fold_dir.exists():
        plot_fold_mode(
            fold_dir,
            out_dir / f"b3_fold_{args.fs}hz.png",
        )

    plot_combined_overview(
        base_root=base_root,
        dataset_name=args.dataset_name,
        fs=args.fs,
        out_file=out_dir / f"combined_balancing_modes_{args.fs}hz.png",
    )

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
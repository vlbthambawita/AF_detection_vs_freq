from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


sample_index = 0
lead_index = 0
duration_sec = 2.0


def label_to_name(label_value: int) -> str:
    """Convert the binary class label to a readable class name."""
    return "AF" if int(label_value) == 1 else "NORMAL"


def extract_signal(obj, sample_index: int = 0, lead_index: int = 0) -> np.ndarray:
    """
    Extract a single ECG signal from a loaded .pt object.

    Supported tensor layouts:
    - [N, C, L]
    - [N, L, C]
    - [N, L]
    - [L]
    """
    if isinstance(obj, dict):
        for key in ["X", "data", "signals", "samples"]:
            if key in obj:
                obj = obj[key]
                break

    if not torch.is_tensor(obj):
        raise ValueError(f"Unsupported object type: {type(obj)}")

    x = obj

    if x.ndim == 3:
        if x.shape[1] <= 16 and x.shape[2] > x.shape[1]:
            return x[sample_index, lead_index, :].cpu().numpy()

        if x.shape[2] <= 16 and x.shape[1] > x.shape[2]:
            return x[sample_index, :, lead_index].cpu().numpy()

        raise ValueError(f"Ambiguous 3D tensor shape: {x.shape}")

    if x.ndim == 2:
        return x[sample_index, :].cpu().numpy()

    if x.ndim == 1:
        return x.cpu().numpy()

    raise ValueError(f"Unsupported tensor shape: {x.shape}")


def main() -> None:
    base_dir = Path("prepared_data") / "ptbl-xl"
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = {
        500: base_dir / "500hz" / "data.pt",
        250: base_dir / "250hz" / "data.pt",
        100: base_dir / "100hz" / "data.pt",
        62: base_dir / "62hz" / "data.pt",
    }

    csv_path = base_dir / "500hz" / "samples_500hz.csv"
    meta_df = pd.read_csv(csv_path)

    if sample_index < 0 or sample_index >= len(meta_df):
        raise IndexError(
            f"sample_index={sample_index} is out of range "
            f"(number of rows: {len(meta_df)})"
        )

    row = meta_df.iloc[sample_index]
    label_name = label_to_name(int(row["label"]))
    patient_id = row["patient_id"]
    record_id = row["record_id"]
    segment_index = row["segment_index"]

    signals = {}
    for fs, path in data_paths.items():
        obj = torch.load(path, map_location="cpu")
        signals[fs] = extract_signal(obj, sample_index=sample_index, lead_index=lead_index)

    segments = {}
    times = {}

    for fs, sig in signals.items():
        n = int(round(duration_sec * fs))
        seg = sig[:n]
        t = np.arange(len(seg)) / fs
        segments[fs] = seg
        times[fs] = t

    all_values = np.concatenate([segments[fs] for fs in [500, 250, 100, 62]])
    ymin, ymax = all_values.min(), all_values.max()
    margin = 0.05 * (ymax - ymin + 1e-12)
    ymin -= margin
    ymax += margin

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    order = [500, 250, 100, 62]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, fs, lab in zip(axes, order, labels):
        ax.plot(times[fs], segments[fs], linewidth=1)
        ax.plot(times[fs], segments[fs], linestyle="None", marker="o", markersize=2.2)
        ax.set_title(f"{lab} {fs} Hz ({len(segments[fs])} samples in 2 s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "ECG resampling comparison over 2 seconds "
        f"({label_name} record, patient_id={patient_id}, "
        f"record_id={record_id}, segment_index={segment_index})",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / "ecg_resampling_comparison_2s_2x2.png"
    pdf_path = output_dir / "ecg_resampling_comparison_2s_2x2.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
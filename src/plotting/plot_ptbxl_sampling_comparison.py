import argparse
import ast
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import wfdb


# ============================================================
# Text shown inside each subplot
# ============================================================

DESCRIPTIONS = {
    62: "Less waveform detail; QRS and P-wave morphology may be compressed",
    100: "Moderate detail, but still limited for fine morphology",
    250: "Good balance between detail and efficiency",
    500: "More detail, but higher input size and more computational cost",
}


# ============================================================
# Utility functions
# ============================================================

def parse_scp_codes(value):
    """
    Convert PTB-XL scp_codes column from string to dictionary.

    Example input:
        "{'NORM': 100.0}"
    """
    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    try:
        return ast.literal_eval(value)
    except Exception:
        return {}


def resample_signal(x, fs_in, fs_out):
    """
    Resample a 1D ECG segment from fs_in to fs_out.
    """
    x = np.asarray(x, dtype=np.float32)

    if fs_in == fs_out:
        return x

    n_out = int(round(len(x) * fs_out / fs_in))
    return resample(x, n_out).astype(np.float32)


def choose_record(df, ecg_id=None, label="NORM"):
    """
    Choose a PTB-XL record.

    If ecg_id is given, select that record.
    Otherwise, select the first record with the requested label.
    """
    if ecg_id is not None:
        if ecg_id not in df.index:
            raise ValueError(f"ecg_id={ecg_id} was not found in ptbxl_database.csv")
        return df.loc[ecg_id]

    label = label.upper()

    if label not in {"NORM", "AFIB"}:
        raise ValueError("label must be either NORM or AFIB")

    mask = df["scp_codes"].apply(lambda codes: label in codes)
    subset = df[mask]

    if subset.empty:
        raise ValueError(f"No PTB-XL record found with label={label}")

    return subset.iloc[0]


def resolve_wfdb_record_path(data_root, filename_hr):
    """
    Resolve PTB-XL WFDB base path.

    PTB-XL filename_hr usually looks like:
        records500/00000/00001_hr

    WFDB expects the base path without extension:
        data/records500/00000/00001_hr

    The actual files are:
        data/records500/00000/00001_hr.hea
        data/records500/00000/00001_hr.dat
    """
    record_base = data_root / filename_hr

    hea_path = record_base.with_suffix(".hea")
    dat_path = record_base.with_suffix(".dat")

    if not hea_path.exists() or not dat_path.exists():
        raise FileNotFoundError(
            "\nWFDB record files were not found.\n\n"
            f"Expected header file:\n  {hea_path}\n\n"
            f"Expected data file:\n  {dat_path}\n\n"
            "Check that --data_root points to the folder containing:\n"
            "  ptbxl_database.csv\n"
            "  records500/\n"
        )

    return record_base


def get_lead_index(sig_names, lead_name):
    """
    Return index of selected ECG lead.
    """
    sig_names = list(sig_names)

    if lead_name not in sig_names:
        raise ValueError(
            f"Lead '{lead_name}' was not found.\n"
            f"Available leads are: {sig_names}"
        )

    return sig_names.index(lead_name)


def infer_label_from_scp_codes(scp_codes):
    """
    Infer simple display label from PTB-XL scp_codes.
    """
    if "AFIB" in scp_codes:
        return "AFIB"
    if "NORM" in scp_codes:
        return "NORM"
    return "UNKNOWN"


def style_axis(ax):
    """
    Clean academic plot style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9)


# ============================================================
# Plotting function
# ============================================================

def make_sampling_plot(
    segment_500,
    fs_original,
    lead_name,
    ecg_id,
    label,
    start_sec,
    duration_sec,
    out_path,
):
    """
    Create 2x2 sampling-frequency comparison plot.
    """
    target_freqs = [62, 100, 250, 500]

    signals = {
        fs: resample_signal(segment_500, fs_original, fs)
        for fs in target_freqs
    }

    # Same y-axis range for fair visual comparison
    y_min = min(np.min(sig) for sig in signals.values())
    y_max = max(np.max(sig) for sig in signals.values())
    y_padding = 0.08 * (y_max - y_min + 1e-8)
    y_limits = (y_min - y_padding, y_max + y_padding)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharey=True,
    )

    axes = axes.ravel()

    for ax, fs in zip(axes, target_freqs):
        sig = signals[fs]
        time_axis = np.arange(len(sig)) / fs

        # Main ECG line
        ax.plot(
            time_axis,
            sig,
            linewidth=1.15,
            color="black",
        )

        # Sampling points make low-frequency detail loss visible
        ax.scatter(
            time_axis,
            sig,
            s=7,
            color="black",
            alpha=0.65,
        )

        ax.set_xlim(0, duration_sec)
        ax.set_ylim(*y_limits)

        ax.set_title(
            f"{fs} Hz",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )

        description = fill(DESCRIPTIONS[fs], width=38)
        sample_count = f"Samples in {duration_sec:.1f}s: {len(sig)}"

        ax.text(
            0.02,
            0.98,
            f"{description}\n{sample_count}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.92,
            ),
        )

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Amplitude (mV)", fontsize=10)
        style_axis(ax)

    fig.suptitle(
        (
            f"PTB-XL sampling-frequency comparison "
            f"(Lead {lead_name}, ECG ID {ecg_id}, {label})"
        ),
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.012,
        (
            f"The same {duration_sec:.1f}-second ECG segment "
            f"from {start_sec:.1f}s to {start_sec + duration_sec:.1f}s "
            "was used for all sampling frequencies."
        ),
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.035, 1, 0.945])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved PDF: {out_path}")
    print(f"Saved PNG: {png_path}")

    plt.show()


# ============================================================
# Main program
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create a clean 2x2 PTB-XL ECG sampling-frequency comparison figure."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Folder containing ptbxl_database.csv and records500/",
    )

    parser.add_argument(
        "--ecg_id",
        type=int,
        default=None,
        help="Specific PTB-XL ecg_id to plot. If not given, label is used.",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="NORM",
        choices=["NORM", "AFIB", "norm", "afib"],
        help="Record label to choose automatically if ecg_id is not given.",
    )

    parser.add_argument(
        "--lead",
        type=str,
        default="II",
        help="ECG lead to plot, e.g. I, II, III, aVR, aVL, aVF, V1, V2, ..., V6.",
    )

    parser.add_argument(
        "--start_sec",
        type=float,
        default=2.0,
        help="Start time of the displayed ECG segment in seconds.",
    )

    parser.add_argument(
        "--duration_sec",
        type=float,
        default=2.0,
        help="Duration of the displayed ECG segment in seconds.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="plotting_bachelor_thesis/resampling",
        help="Output directory for PDF and PNG files.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata_path = data_root / "ptbxl_database.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"\nCould not find metadata file:\n  {metadata_path}\n\n"
            "Make sure --data_root points to the PTB-XL data folder."
        )

    # ------------------------------------------------------------
    # Load PTB-XL metadata
    # ------------------------------------------------------------
    df = pd.read_csv(metadata_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)

    # ------------------------------------------------------------
    # Select record
    # ------------------------------------------------------------
    row = choose_record(
        df=df,
        ecg_id=args.ecg_id,
        label=args.label,
    )

    ecg_id = row.name
    label = infer_label_from_scp_codes(row["scp_codes"])

    # ------------------------------------------------------------
    # Resolve and load WFDB record
    # ------------------------------------------------------------
    record_base_path = resolve_wfdb_record_path(
        data_root=data_root,
        filename_hr=row["filename_hr"],
    )

    signal, meta = wfdb.rdsamp(str(record_base_path))

    fs_original = int(meta["fs"])
    sig_names = meta["sig_name"]

    if fs_original != 500:
        raise ValueError(
            f"This script expects PTB-XL high-resolution records at 500 Hz, "
            f"but this record has fs={fs_original}."
        )

    # ------------------------------------------------------------
    # Select lead
    # ------------------------------------------------------------
    lead_index = get_lead_index(sig_names, args.lead)
    lead_signal = signal[:, lead_index].astype(np.float32)

    # ------------------------------------------------------------
    # Extract same time segment from original 500 Hz signal
    # ------------------------------------------------------------
    start_index = int(round(args.start_sec * fs_original))
    end_index = int(round((args.start_sec + args.duration_sec) * fs_original))

    if start_index < 0 or end_index > len(lead_signal):
        signal_duration = len(lead_signal) / fs_original
        raise ValueError(
            f"\nRequested segment is outside the ECG signal length.\n\n"
            f"Requested: {args.start_sec:.2f}s to "
            f"{args.start_sec + args.duration_sec:.2f}s\n"
            f"Available signal duration: {signal_duration:.2f}s\n"
        )

    segment_500 = lead_signal[start_index:end_index]

    # ------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------
    safe_lead_name = args.lead.replace("/", "_")
    out_file = (
        f"ptbxl_sampling_comparison_"
        f"ecg{ecg_id}_lead_{safe_lead_name}_"
        f"{args.start_sec:.1f}s_{args.duration_sec:.1f}s.pdf"
    )

    out_path = Path(args.out_dir) / out_file

    # ------------------------------------------------------------
    # Create plot
    # ------------------------------------------------------------
    make_sampling_plot(
        segment_500=segment_500,
        fs_original=fs_original,
        lead_name=args.lead,
        ecg_id=ecg_id,
        label=label,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
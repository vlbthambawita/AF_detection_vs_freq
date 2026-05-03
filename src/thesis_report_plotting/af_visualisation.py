import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks


def parse_scp_codes(value):
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def load_af_record_from_metadata(csv_path, records_root, ecg_id):
    df = pd.read_csv(csv_path)
    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)
    row = df[df["ecg_id"] == ecg_id].iloc[0]

    if "AFIB" not in row["scp_codes"]:
        raise ValueError(f"Record {ecg_id} is not labeled as AFIB in PTB-XL metadata.")

    record_path = records_root / row["filename_hr"]
    rec = wfdb.rdrecord(str(record_path))
    signal = rec.p_signal
    fs = rec.fs

    return signal, fs, row


def detect_r_peaks(x, fs):
    distance = int(0.25 * fs)
    prominence = max(0.05, 0.2 * np.std(x))
    peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    return peaks


def compute_rr_intervals(r_peaks, fs):
    rr = np.diff(r_peaks) / fs
    return rr


def extract_pre_qrs_windows(x, r_peaks, fs, pre=0.20, post=0.05):
    segments = []
    n_pre = int(pre * fs)
    n_post = int(post * fs)

    for r in r_peaks:
        start = r - n_pre
        end = r + n_post
        if start >= 0 and end < len(x):
            segments.append(x[start:end])

    return segments


def estimate_baseline_activity(x, r_peaks, fs, window=0.6):
    segments = []
    half = int((window * fs) / 2)

    for i in range(len(r_peaks) - 1):
        mid = (r_peaks[i] + r_peaks[i + 1]) // 2
        start = max(0, mid - half)
        end = min(len(x), mid + half)
        seg = x[start:end]
        seg = seg - np.mean(seg)
        segments.append(seg)

    return segments

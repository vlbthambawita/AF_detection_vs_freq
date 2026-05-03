from pathlib import Path
import numpy as np
import wfdb


def find_ptbxl_records(records_root: Path):
    """
    Locate all WFDB records by finding .hea files
    and returning record base paths without extension.
    """
    return sorted(hea.with_suffix("") for hea in records_root.rglob("*.hea"))


def read_record(record_base: Path):
    """
    Load one WFDB record and return the record object
    together with its physical signal matrix.
    """
    rec = wfdb.rdrecord(str(record_base))
    sig = rec.p_signal

    if sig is None:
        raise ValueError(f"No p_signal found in record: {record_base}")

    return rec, sig


def scan_record(record_base: Path):
    """
    Count NaN, +Inf, -Inf, and total non-finite values
    in one ECG record.
    """
    rec, sig = read_record(record_base)

    nan_mask = np.isnan(sig)
    posinf_mask = np.isposinf(sig)
    neginf_mask = np.isneginf(sig)
    nonfinite_mask = ~np.isfinite(sig)

    return {
        "record_id": record_base.name,
        "fs": int(rec.fs) if rec.fs is not None else None,
        "n_samples": int(sig.shape[0]),
        "n_leads": int(sig.shape[1]),
        "total_values": int(sig.size),
        "nan_count": int(nan_mask.sum()),
        "posinf_count": int(posinf_mask.sum()),
        "neginf_count": int(neginf_mask.sum()),
        "nonfinite_count": int(nonfinite_mask.sum()),
    }


def scan_dataset(records_root: Path):
    """
    Scan all PTB-XL records and accumulate dataset-level counts.
    """
    record_bases = find_ptbxl_records(records_root)

    total_nan = 0
    total_posinf = 0
    total_neginf = 0
    total_nonfinite = 0
    total_values = 0

    for record_base in record_bases:
        stats = scan_record(record_base)
        total_nan += stats["nan_count"]
        total_posinf += stats["posinf_count"]
        total_neginf += stats["neginf_count"]
        total_nonfinite += stats["nonfinite_count"]
        total_values += stats["total_values"]

    return {
        "total_records": len(record_bases),
        "total_values": total_values,
        "total_nan": total_nan,
        "total_posinf": total_posinf,
        "total_neginf": total_neginf,
        "total_nonfinite": total_nonfinite,
    }
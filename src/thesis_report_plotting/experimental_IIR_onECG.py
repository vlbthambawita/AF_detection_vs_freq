from pathlib import Path
import numpy as np
import wfdb


def load_record(record_base: Path):
    """
    Load one WFDB ECG record and return the signal matrix.
    """
    rec = wfdb.rdrecord(str(record_base))
    signal = np.asarray(rec.p_signal, dtype=np.float64)

    if signal is None:
        raise ValueError(f"No p_signal found for record: {record_base}")

    return signal, int(rec.fs)


def iir_filter_first_order(signal: np.ndarray, alpha: float = 0.2, beta: float = 0.8) -> np.ndarray:
    """
    First-order recursive IIR filter:
        y[n] = alpha * x[n] + beta * y[n-1]
    """
    y = np.empty_like(signal, dtype=np.float64)
    y[0, :] = alpha * signal[0, :]

    with np.errstate(invalid="ignore", over="ignore"):
        for n in range(1, signal.shape[0]):
            y[n, :] = alpha * signal[n, :] + beta * y[n - 1, :]

    return y


def inject_nonfinite_values(signal: np.ndarray, total_injections: int, rng: np.random.Generator) -> np.ndarray:
    """
    Inject NaN, +Inf, and -Inf into random unique positions.
    """
    corrupted = np.array(signal, copy=True, dtype=np.float64)
    flat_indices = rng.choice(corrupted.size, size=total_injections, replace=False)

    kinds = ["nan", "posinf", "neginf"]
    for i, flat_idx in enumerate(flat_indices):
        r, c = np.unravel_index(flat_idx, corrupted.shape)
        kind = kinds[i % len(kinds)]

        if kind == "nan":
            corrupted[r, c] = np.nan
        elif kind == "posinf":
            corrupted[r, c] = np.inf
        else:
            corrupted[r, c] = -np.inf

    return corrupted


def count_nonfinite(signal: np.ndarray):
    """
    Count NaN, +Inf, -Inf, and total non-finite values.
    """
    return {
        "nan_count": int(np.isnan(signal).sum()),
        "posinf_count": int(np.isposinf(signal).sum()),
        "neginf_count": int(np.isneginf(signal).sum()),
        "nonfinite_count": int((~np.isfinite(signal)).sum()),
    }


def compute_rmse_and_corr(reference: np.ndarray, test: np.ndarray):
    """
    Compare two signals using only finite overlapping values.
    """
    mask = np.isfinite(reference) & np.isfinite(test)

    ref = reference[mask]
    tst = test[mask]

    mse = np.mean((tst - ref) ** 2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(ref, tst)[0, 1] if ref.size > 1 else np.nan

    return {"mse": float(mse), "rmse": float(rmse), "corr": float(corr)}


def run_nonfinite_iir_experiment(record_base: Path, injections: int = 30, alpha: float = 0.2, beta: float = 0.8):
    """
    Controlled experiment on one real ECG record:
    1. load clean signal
    2. inject NaN / +Inf / -Inf
    3. replace the same corrupted positions with zero
    4. apply recursive IIR to all versions
    5. quantify propagated corruption and repair quality
    """
    rng = np.random.default_rng(42)

    clean_signal, fs = load_record(record_base)
    corrupted_signal = inject_nonfinite_values(clean_signal, injections, rng)
    zero_replaced_signal = np.nan_to_num(corrupted_signal, nan=0.0, posinf=0.0, neginf=0.0)

    clean_iir = iir_filter_first_order(clean_signal, alpha=alpha, beta=beta)
    corrupted_iir = iir_filter_first_order(corrupted_signal, alpha=alpha, beta=beta)
    zero_iir = iir_filter_first_order(zero_replaced_signal, alpha=alpha, beta=beta)

    injected_counts = count_nonfinite(corrupted_signal)
    propagated_counts = count_nonfinite(corrupted_iir)
    repaired_metrics = compute_rmse_and_corr(clean_iir, zero_iir)

    propagation_factor = (
        propagated_counts["nonfinite_count"] / injected_counts["nonfinite_count"]
        if injected_counts["nonfinite_count"] > 0 else np.nan
    )

    return {
        "fs": fs,
        "injected_nonfinite": injected_counts["nonfinite_count"],
        "propagated_nonfinite_after_iir": propagated_counts["nonfinite_count"],
        "propagation_factor": float(propagation_factor),
        "zero_vs_clean_mse": repaired_metrics["mse"],
        "zero_vs_clean_rmse": repaired_metrics["rmse"],
        "zero_vs_clean_corr": repaired_metrics["corr"],
    }
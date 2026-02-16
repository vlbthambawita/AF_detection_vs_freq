import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

try:
    from scipy.optimize import minimize_scalar
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ================= CONFIG =================
ROOT = Path("checkpoints/ptbl-xl")
MODEL = "cnn_lstm"
FOLDS = 5
FREQS = ["62hz", "100hz", "250hz", "500hz"]

N_BINS = 10
CALIB_RATIO = 0.5          # spliting fold val into calib/eval
SEED = 42
EPS = 1e-6                 # clipping probabilities


# ============== helpers ==============
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_scaling(s_calib: np.ndarray, y_calib: np.ndarray) -> float:
    """
    Fit temperature T > 0 by minimizing NLL on calibration split.
    Uses logits s_calib (logit(p)).

    If scipy is not available, returns T=1.0.
    """
    if not _HAS_SCIPY:
        return 1.0

    def nll(T):
        # constrain T positive
        if T <= 0:
            return 1e9
        pT = sigmoid(s_calib / T)
        pT = np.clip(pT, EPS, 1 - EPS)
        # binary NLL
        return -np.mean(y_calib * np.log(pT) + (1 - y_calib) * np.log(1 - pT))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x) if res.success else 1.0


# ================= MAIN =================
for freq in FREQS:
    base = ROOT / freq / MODEL


    y_eval_all = []

    probs_raw_all = []
    probs_platt_all = []
    probs_iso_all = []
    probs_temp_all = []

    # ---------- read folds + calibrate ----------
    for fold in range(1, FOLDS + 1):
        roc_path = base / f"fold_{fold}" / "roc_val.npz"
        if not roc_path.exists():
            raise FileNotFoundError(f"Missing: {roc_path}")

        data = np.load(roc_path)
        y_true = data["y_true"].astype(int).ravel()
        y_score = data["y_score"].astype(float).ravel()

        # clip and create "logit score" for calibration models
        y_score = np.clip(y_score, EPS, 1 - EPS)
        s = logit(y_score)  # "score" space for Platt and temp scaling (logits)

        # split fold val into calib/eval
        idx = np.arange(len(y_true))
        calib_idx, eval_idx = train_test_split(
            idx,
            test_size=(1 - CALIB_RATIO),
            random_state=SEED + fold,
            stratify=y_true,
        )

        y_calib, y_eval = y_true[calib_idx], y_true[eval_idx]
        p_calib, p_eval = y_score[calib_idx], y_score[eval_idx]
        s_calib, s_eval = s[calib_idx], s[eval_idx]

        # ---- 1) Raw (uncalibrated) ----
        p_raw = p_eval

        # ---- 2) Platt scaling (sigmoid) via logistic regression on logits ----
        #    p' = sigmoid(a*s + b)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(s_calib.reshape(-1, 1), y_calib)
        p_platt = lr.predict_proba(s_eval.reshape(-1, 1))[:, 1]
        p_platt = np.clip(p_platt, EPS, 1 - EPS)

        # ---- 3) Isotonic regression on probabilities ----
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_calib, y_calib)
        p_iso = iso.predict(p_eval)
        p_iso = np.clip(p_iso, EPS, 1 - EPS)

        # ---- 4) Temperature scaling on logits ----
        #    p' = sigmoid(s / T)
        T = fit_temperature_scaling(s_calib, y_calib)
        p_temp = sigmoid(s_eval / T)
        p_temp = np.clip(p_temp, EPS, 1 - EPS)

        # collect pooled eval
        y_eval_all.append(y_eval)
        probs_raw_all.append(p_raw)
        probs_platt_all.append(p_platt)
        probs_iso_all.append(p_iso)
        probs_temp_all.append(p_temp)

    # concat pooled eval across folds
    y_eval_all = np.concatenate(y_eval_all)

    p_raw_all = np.concatenate(probs_raw_all)
    p_platt_all = np.concatenate(probs_platt_all)
    p_iso_all = np.concatenate(probs_iso_all)
    p_temp_all = np.concatenate(probs_temp_all)

    # ==========================
    # Plot similar to sklearn's CalibrationDisplay but with multiple methods + histograms of predicted probabilities.
    # ==========================
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)
    colors = plt.get_cmap("Dark2")

    ax_cal = fig.add_subplot(gs[:2, :2])
    ax_cal.grid(True)
    hz = freq.replace("hz", "")
    ax_cal.set_title(f"Calibration plots (5-fold CV, {hz} Hz)")

    # perfect line for reference
    ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # take Brier score on pooled eval for each method to show in legend
    methods = [
        (p_raw_all,   "Uncalibrated (raw)"),
        (p_platt_all, "Platt scaling (sigmoid)"),
        (p_iso_all,   "Isotonic"),
        (p_temp_all,  "Temperature scaling"),
    ]

    calibration_displays = {}
    for i, (p, name) in enumerate(methods):
        brier = brier_score_loss(y_eval_all, p)
        display = CalibrationDisplay.from_predictions(
            y_true=y_eval_all,
            y_prob=p,
            n_bins=N_BINS,
            name=f"{name} (Brier={brier:.3f})",
            ax=ax_cal,
            color=colors(i),
        )
        calibration_displays[name] = display

    # ---- histograms of predicted probabilities ----
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (p, name) in enumerate(methods):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            p,
            range=(0, 1),
            bins=N_BINS,
            label=name,
            color=colors(i),
            alpha=0.9,
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
        ax.grid(True, alpha=0.2)

    out_path = base / f"calibration_curve_2_{freq}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

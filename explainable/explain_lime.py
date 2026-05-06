"""LIME-style post-hoc explanations for ECG AFIB vs NORMAL classifiers.

This module complements Grad-CAM rather than replacing it:

- Grad-CAM highlights gradient-based temporal relevance inside the model.
- LIME perturbs interpretable input-space windows and fits a local surrogate.

For multilead ECG, the interpretable units are lead-time windows. The
explanation is therefore local and approximate: it highlights windows whose
presence supports or opposes a selected class around one specific sample, but
it is not proof of exact clinical reasoning. Physiological perturbations are
also imperfect, so masking artifacts should be interpreted cautiously.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.cnn1d import CNN1D
from models.cnn_lstm import CNN_LSTM_ECG


LabelName = Literal["NORMAL", "AFIB"]
MaskStrategy = Literal["zero", "lead_mean", "local_mean"]
ExplainClassMode = Literal["pred", "true"]
RepresentativeCase = Literal[
    "correct_afib",
    "correct_normal",
    "false_positive",
    "false_negative",
    "uncertain",
]

LABEL_MAP: dict[int, LabelName] = {
    0: "NORMAL",
    1: "AFIB",
}

STANDARD_12_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


@dataclass(frozen=True)
class FeatureWindow:
    """One interpretable feature corresponding to a lead-time ECG region."""

    feature_index: int
    lead_index: int
    lead_name: str
    window_index: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    display_name: str


@dataclass
class PredictionResult:
    """Container for ensemble outputs on one sample."""

    logits: np.ndarray
    probabilities: np.ndarray
    predicted_class: int


@dataclass
class LimeExplanation:
    """Results from the local surrogate fitted around one ECG sample."""

    feature_weights: np.ndarray
    intercept: float
    locality_weights: np.ndarray
    masks: np.ndarray
    target_scores: np.ndarray
    surrogate_predictions: np.ndarray
    weighted_r2: float
    weighted_mse: float
    explained_class: int
    original_target_score: float
    original_surrogate_score: float
    perturbation_seed: int
    mask_strategy: MaskStrategy
    num_perturbations: int
    kernel_width: float


def set_deterministic_seed(seed: int) -> None:
    """Seed NumPy, Python, and Torch for reproducible perturbations."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ECG LIME explainer."""

    parser = argparse.ArgumentParser(
        description=(
            "LIME-style post-hoc explanation for ensemble ECG classifiers "
            "(AFIB vs NORMAL)."
        )
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help=(
            "Path to a prepared data frequency folder such as "
            "prepared_data/ptb-xl/100hz."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn1d", "cnn_lstm"],
        help="Trained architecture to explain.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Sample index inside the selected split. Required unless representative selection is used.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "all"],
        default="test",
        help="'test' loads <data_path>/test/test.pt, 'all' loads <data_path>/data.pt.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of best-fold checkpoints to ensemble.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example 'cpu' or 'cuda'. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional custom output directory. Defaults to project_root/lime_results/...",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=0.5,
        help="Interpretable ECG window size in seconds.",
    )
    parser.add_argument(
        "--num_perturbations",
        type=int,
        default=1024,
        help="Number of perturbed neighborhood samples for the local surrogate.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of strongest positive and negative windows to highlight.",
    )
    parser.add_argument(
        "--explain_class",
        choices=["pred", "true"],
        default="pred",
        help="Explain the predicted class or the ground-truth class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic perturbation generation.",
    )
    parser.add_argument(
        "--mask_strategy",
        choices=["zero", "lead_mean", "local_mean"],
        default="local_mean",
        help="How masked lead-time windows are replaced during perturbation.",
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for the weighted local surrogate.",
    )
    parser.add_argument(
        "--kernel_width",
        type=float,
        default=0.25,
        help="Kernel width for the locality weighting function on normalized Hamming distance.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for running perturbed samples through the ensemble.",
    )
    parser.add_argument(
        "--representative_case",
        choices=[
            "correct_afib",
            "correct_normal",
            "false_positive",
            "false_negative",
            "uncertain",
        ],
        default=None,
        help="Automatically choose a representative sample from the selected split.",
    )
    return parser.parse_args()


def resolve_existing_path(raw_path: str | Path) -> Path:
    """Resolve a path against likely project roots and ensure it exists."""

    path = Path(raw_path).expanduser()
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                PROJECT_ROOT / path,
                SRC_ROOT / path,
            ]
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    searched = "\n".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve path '{raw_path}'. Tried:\n{searched}"
    )


def resolve_checkpoints_root() -> Path:
    """Find the checkpoints root without assuming one fixed repository layout."""

    candidates = [
        PROJECT_ROOT / "checkpoints",
        SRC_ROOT / "checkpoints",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "checkpoints"


def resolve_output_path(raw_path: str | Path) -> Path:
    """Resolve an output path relative to the project root when needed."""

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_sampling_rate(fs_dir_name: str) -> int:
    """Extract an integer sampling rate from a directory name such as '100hz'."""

    digits = "".join(character for character in fs_dir_name if character.isdigit())
    if not digits:
        raise ValueError(
            f"Could not parse sampling rate from directory name '{fs_dir_name}'."
        )
    return int(digits)


def load_split_tensor(data_dir: Path, split: Literal["test", "all"]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a prepared ECG tensor split from disk."""

    if split == "test":
        tensor_path = data_dir / "test" / "test.pt"
    else:
        tensor_path = data_dir / "data.pt"

    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing prepared tensor file: {tensor_path}")

    data = torch.load(tensor_path, map_location="cpu")
    if "X" not in data or "y" not in data:
        raise KeyError(f"Expected 'X' and 'y' keys in {tensor_path}")
    return data["X"], data["y"]


class ModelFactory:
    """Construct supported ECG model architectures without changing training logic."""

    @staticmethod
    def build(model_name: str, in_channels: int, num_classes: int) -> torch.nn.Module:
        """Instantiate one supported architecture."""

        if model_name == "cnn1d":
            return CNN1D(in_channels=in_channels, num_classes=num_classes)
        if model_name == "cnn_lstm":
            return CNN_LSTM_ECG(in_channels=in_channels, num_classes=num_classes)
        raise ValueError(f"Unsupported model '{model_name}'.")
    

def resolve_torch_device(device: str | None) -> torch.device:
    """Resolve GUI/CLI device values into a valid torch.device."""

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] CUDA was selected, but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{device}'. Use one of: auto, cuda, cpu."
    )

class EnsemblePredictor:
    """Load best-fold checkpoints and reproduce ensemble test-time prediction."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        fs_dir_name: str,
        in_channels: int,
        num_classes: int,
        folds: int,
        device: str | None = None,
        checkpoint_root: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.fs_dir_name = fs_dir_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.folds = folds
        self.device = resolve_torch_device(device)
        self.checkpoint_root = checkpoint_root or resolve_checkpoints_root()
        self.models = self._load_models()

    def _fold_checkpoint_path(self, fold: int) -> Path:
        """Return the expected checkpoint path for one fold."""

        return (
            self.checkpoint_root
            / self.dataset_name
            / self.fs_dir_name
            / self.model_name
            / f"fold_{fold}"
            / "best.pt"
        )

    def _load_models(self) -> list[torch.nn.Module]:
        """Load all requested fold checkpoints for ensemble inference."""

        models: list[torch.nn.Module] = []
        for fold in range(1, self.folds + 1):
            checkpoint_path = self._fold_checkpoint_path(fold)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

            model = ModelFactory.build(
                model_name=self.model_name,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            )
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            models.append(model)
        return models

    @torch.inference_mode()
    def predict_logits(self, batch: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Predict averaged ensemble logits for a batch of ECG samples."""

        if batch.ndim != 3:
            raise ValueError(
                f"Expected batch with shape [N, C, T], got {tuple(batch.shape)}"
            )

        outputs: list[torch.Tensor] = []
        for start in range(0, batch.shape[0], batch_size):
            end = start + batch_size
            batch_chunk = batch[start:end].to(self.device).float()

            logits_sum: torch.Tensor | None = None
            for model in self.models:
                logits = model(batch_chunk)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            assert logits_sum is not None
            outputs.append((logits_sum / len(self.models)).cpu())

        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def predict_probabilities(self, batch: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Predict ensemble probabilities after averaging logits across folds."""

        logits = self.predict_logits(batch=batch, batch_size=batch_size)
        return torch.softmax(logits, dim=1)

    def predict_single(self, sample: torch.Tensor) -> PredictionResult:
        """Predict logits and probabilities for one ECG sample."""

        logits = self.predict_logits(sample, batch_size=1).squeeze(0).numpy()
        probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        predicted_class = int(np.argmax(probabilities))
        return PredictionResult(
            logits=logits,
            probabilities=probabilities,
            predicted_class=predicted_class,
        )


class RepresentativeSampleSelector:
    """Select practical example cases from one split for later explanation."""

    def __init__(self, predictor: EnsemblePredictor, batch_size: int = 64) -> None:
        self.predictor = predictor
        self.batch_size = batch_size

    def select(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        case: RepresentativeCase,
    ) -> tuple[int, dict[str, Any]]:
        """Select one representative sample index for a requested test-set case."""

        probabilities = self.predictor.predict_probabilities(X.float(), batch_size=self.batch_size)
        probabilities_np = probabilities.numpy()
        pred = probabilities_np.argmax(axis=1)
        p_afib = probabilities_np[:, 1]
        y_np = y.numpy().astype(int)

        if case == "correct_afib":
            candidates = np.where((y_np == 1) & (pred == 1))[0]
            ranking = np.argsort(-p_afib[candidates])
        elif case == "correct_normal":
            candidates = np.where((y_np == 0) & (pred == 0))[0]
            ranking = np.argsort(p_afib[candidates])
        elif case == "false_positive":
            candidates = np.where((y_np == 0) & (pred == 1))[0]
            ranking = np.argsort(-p_afib[candidates])
        elif case == "false_negative":
            candidates = np.where((y_np == 1) & (pred == 0))[0]
            ranking = np.argsort(p_afib[candidates])
        elif case == "uncertain":
            candidates = np.arange(len(y_np))
            ranking = np.argsort(np.abs(p_afib[candidates] - 0.5))
        else:
            raise ValueError(f"Unsupported representative case '{case}'.")

        if len(candidates) == 0:
            raise ValueError(f"No samples found for representative case '{case}'.")

        selected_idx = int(candidates[ranking[0]])
        return selected_idx, {
            "representative_case": case,
            "selected_probability_afib": float(p_afib[selected_idx]),
            "selected_true_label": LABEL_MAP[int(y_np[selected_idx])],
            "selected_predicted_label": LABEL_MAP[int(pred[selected_idx])],
        }


class ECGWindowInterpreter:
    """Map between ECG tensors and interpretable lead-time windows."""

    def __init__(
        self,
        num_channels: int,
        time_length: int,
        sampling_rate_hz: int,
        window_sec: float,
        lead_names: Sequence[str] | None = None,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive.")

        self.num_channels = num_channels
        self.time_length = time_length
        self.sampling_rate_hz = sampling_rate_hz
        self.window_sec = window_sec
        self.window_size_samples = max(1, int(round(window_sec * sampling_rate_hz)))
        self.num_time_windows = math.ceil(time_length / self.window_size_samples)
        self.lead_names = list(lead_names or self._default_lead_names())
        self.features = self._build_features()

    def _default_lead_names(self) -> list[str]:
        """Return clinically familiar lead names when the signal has 12 channels."""

        if self.num_channels == 12:
            return STANDARD_12_LEAD_NAMES.copy()
        return [f"Lead_{index}" for index in range(self.num_channels)]

    def _build_features(self) -> list[FeatureWindow]:
        """Create the full interpretable feature list."""

        features: list[FeatureWindow] = []
        feature_index = 0
        for lead_index in range(self.num_channels):
            lead_name = self.lead_names[lead_index]
            lead_label = lead_name if lead_name.startswith("Lead_") else f"Lead {lead_name}"
            for window_index in range(self.num_time_windows):
                start_sample = window_index * self.window_size_samples
                end_sample = min(self.time_length, start_sample + self.window_size_samples)
                start_sec = start_sample / self.sampling_rate_hz
                end_sec = end_sample / self.sampling_rate_hz
                features.append(
                    FeatureWindow(
                        feature_index=feature_index,
                        lead_index=lead_index,
                        lead_name=lead_name,
                        window_index=window_index,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        display_name=f"{lead_label} | {start_sec:.1f}s-{end_sec:.1f}s",
                    )
                )
                feature_index += 1
        return features

    @property
    def num_features(self) -> int:
        """Number of interpretable binary features."""

        return len(self.features)

    def build_feature_rows(self, feature_weights: np.ndarray) -> list[dict[str, Any]]:
        """Combine feature metadata with surrogate coefficients."""

        rows: list[dict[str, Any]] = []
        for feature in self.features:
            weight = float(feature_weights[feature.feature_index])
            rows.append(
                {
                    "feature_index": feature.feature_index,
                    "lead_index": feature.lead_index,
                    "lead_name": feature.lead_name,
                    "window_index": feature.window_index,
                    "start_sec": feature.start_sec,
                    "end_sec": feature.end_sec,
                    "display_name": feature.display_name,
                    "surrogate_weight": weight,
                    "absolute_weight": abs(weight),
                }
            )
        return rows

    def feature_weight_matrix(self, feature_weights: np.ndarray) -> np.ndarray:
        """Reshape 1D feature weights into a lead x time-window matrix."""

        matrix = np.zeros((self.num_channels, self.num_time_windows), dtype=np.float64)
        for feature in self.features:
            matrix[feature.lead_index, feature.window_index] = feature_weights[feature.feature_index]
        return matrix

    def _moving_average(self, signal: np.ndarray, kernel_size: int) -> np.ndarray:
        """Compute a simple edge-padded moving average baseline."""

        kernel_size = max(1, int(kernel_size))
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        left_pad = kernel_size // 2
        right_pad = kernel_size - 1 - left_pad
        padded = np.pad(signal, (left_pad, right_pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _baseline_template(self, sample: np.ndarray, strategy: MaskStrategy) -> np.ndarray:
        """Create the replacement baseline used for masked ECG windows."""

        if strategy == "zero":
            return np.zeros_like(sample, dtype=np.float32)

        if strategy == "lead_mean":
            lead_mean = sample.mean(axis=1, keepdims=True)
            return np.repeat(lead_mean, repeats=sample.shape[1], axis=1).astype(np.float32)

        if strategy == "local_mean":
            baseline = np.zeros_like(sample, dtype=np.float32)
            kernel_size = max(3, self.window_size_samples)
            for lead_index in range(sample.shape[0]):
                baseline[lead_index] = self._moving_average(
                    sample[lead_index],
                    kernel_size=kernel_size,
                ).astype(np.float32)
            return baseline

        raise ValueError(f"Unsupported mask strategy '{strategy}'.")

    def apply_masks(
        self,
        sample: np.ndarray,
        masks: np.ndarray,
        strategy: MaskStrategy,
    ) -> np.ndarray:
        """Create perturbed ECG tensors from interpretable binary masks.

        A mask value of 1 keeps the original window. A mask value of 0 replaces
        that lead-time region with the selected baseline strategy.
        """

        if sample.ndim != 2:
            raise ValueError(f"Expected sample shape [C, T], got {sample.shape}")
        if masks.ndim != 2 or masks.shape[1] != self.num_features:
            raise ValueError(
                f"Expected masks shape [N, {self.num_features}], got {masks.shape}"
            )

        baseline = self._baseline_template(sample=sample, strategy=strategy)
        perturbed = np.repeat(sample[None, :, :], repeats=masks.shape[0], axis=0).astype(np.float32)

        for feature in self.features:
            off_rows = np.where(masks[:, feature.feature_index] == 0)[0]
            if off_rows.size == 0:
                continue
            perturbed[
                off_rows,
                feature.lead_index,
                feature.start_sample:feature.end_sample,
            ] = baseline[
                feature.lead_index,
                feature.start_sample:feature.end_sample,
            ]

        return perturbed


class LimeExplainerECG:
    """Generate perturbations and fit a weighted local surrogate in ECG input space."""

    def __init__(
        self,
        predictor: EnsemblePredictor,
        interpreter: ECGWindowInterpreter,
        num_perturbations: int,
        mask_strategy: MaskStrategy,
        ridge_alpha: float,
        seed: int,
        kernel_width: float = 0.25,
        batch_size: int = 64,
    ) -> None:
        if num_perturbations < 2:
            raise ValueError("num_perturbations must be at least 2.")
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative.")
        if kernel_width <= 0:
            raise ValueError("kernel_width must be positive.")

        self.predictor = predictor
        self.interpreter = interpreter
        self.num_perturbations = num_perturbations
        self.mask_strategy = mask_strategy
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.kernel_width = kernel_width
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def _sample_masks(self) -> np.ndarray:
        """Sample a local neighborhood of interpretable binary masks."""

        num_features = self.interpreter.num_features
        masks = np.ones((self.num_perturbations, num_features), dtype=np.float32)

        for row_index in range(1, self.num_perturbations):
            drop_fraction = float(self.rng.uniform(0.05, 0.45))
            num_off = max(1, int(round(drop_fraction * num_features)))
            off_indices = self.rng.choice(num_features, size=num_off, replace=False)
            masks[row_index, off_indices] = 0.0

        return masks

    def _locality_weights(self, masks: np.ndarray) -> np.ndarray:
        """Compute LIME-style locality weights from the all-on reference mask."""

        off_fraction = 1.0 - masks.mean(axis=1)
        distances = np.sqrt(off_fraction)
        weights = np.exp(-np.square(distances) / (self.kernel_width**2 + 1e-12))
        weights[0] = 1.0
        return weights.astype(np.float64)

    def _fit_surrogate(
        self,
        masks: np.ndarray,
        target_scores: np.ndarray,
        locality_weights: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Fit a weighted ridge surrogate on interpretable features."""

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        model.fit(masks, target_scores, sample_weight=locality_weights)

        surrogate_predictions = model.predict(masks)
        intercept = float(model.intercept_)
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        return intercept, coefficients, surrogate_predictions

    def _weighted_metrics(
        self,
        target_scores: np.ndarray,
        surrogate_predictions: np.ndarray,
        locality_weights: np.ndarray,
    ) -> tuple[float, float]:
        """Compute weighted surrogate diagnostics."""

        mse = float(np.average(np.square(target_scores - surrogate_predictions), weights=locality_weights))
        weighted_mean = float(np.average(target_scores, weights=locality_weights))
        ss_res = float(np.sum(locality_weights * np.square(target_scores - surrogate_predictions)))
        ss_tot = float(np.sum(locality_weights * np.square(target_scores - weighted_mean)))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        return r2, mse

    def explain(
        self,
        sample: torch.Tensor,
        true_class: int,
        explain_class: ExplainClassMode,
    ) -> tuple[LimeExplanation, PredictionResult]:
        """Explain one ECG sample using ensemble predictions and window perturbations."""

        if sample.ndim != 3 or sample.shape[0] != 1:
            raise ValueError(
                f"Expected sample shape [1, C, T], got {tuple(sample.shape)}"
            )

        original_prediction = self.predictor.predict_single(sample.float())
        explained_class = (
            original_prediction.predicted_class
            if explain_class == "pred"
            else int(true_class)
        )

        masks = self._sample_masks()
        locality_weights = self._locality_weights(masks)
        sample_np = sample.squeeze(0).detach().cpu().numpy().astype(np.float32)

        target_scores: list[np.ndarray] = []
        for start in range(0, self.num_perturbations, self.batch_size):
            end = min(self.num_perturbations, start + self.batch_size)
            mask_chunk = masks[start:end]
            perturbed_chunk = self.interpreter.apply_masks(
                sample=sample_np,
                masks=mask_chunk,
                strategy=self.mask_strategy,
            )
            perturbed_tensor = torch.from_numpy(perturbed_chunk).float()
            probs_chunk = self.predictor.predict_probabilities(
                perturbed_tensor,
                batch_size=self.batch_size,
            ).numpy()
            target_scores.append(probs_chunk[:, explained_class])

        target_scores_np = np.concatenate(target_scores, axis=0).astype(np.float64)
        intercept, feature_weights, surrogate_predictions = self._fit_surrogate(
            masks=masks,
            target_scores=target_scores_np,
            locality_weights=locality_weights,
        )
        weighted_r2, weighted_mse = self._weighted_metrics(
            target_scores=target_scores_np,
            surrogate_predictions=surrogate_predictions,
            locality_weights=locality_weights,
        )

        explanation = LimeExplanation(
            feature_weights=feature_weights,
            intercept=intercept,
            locality_weights=locality_weights,
            masks=masks,
            target_scores=target_scores_np,
            surrogate_predictions=surrogate_predictions,
            weighted_r2=weighted_r2,
            weighted_mse=weighted_mse,
            explained_class=explained_class,
            original_target_score=float(original_prediction.probabilities[explained_class]),
            original_surrogate_score=float(intercept + feature_weights.sum()),
            perturbation_seed=self.seed,
            mask_strategy=self.mask_strategy,
            num_perturbations=self.num_perturbations,
            kernel_width=self.kernel_width,
        )
        return explanation, original_prediction


class LimeResultWriter:
    """Save LIME summaries, tables, and thesis-friendly plots."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _top_positive_rows(
        self,
        feature_rows: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the strongest windows that support the explained class."""

        positive = [row for row in feature_rows if row["surrogate_weight"] > 0]
        positive.sort(key=lambda row: row["surrogate_weight"], reverse=True)
        return positive[:top_k]

    def _top_negative_rows(
        self,
        feature_rows: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the strongest windows that oppose the explained class."""

        negative = [row for row in feature_rows if row["surrogate_weight"] < 0]
        negative.sort(key=lambda row: row["surrogate_weight"])
        return negative[:top_k]

    def write_feature_weights_csv(self, feature_rows: Sequence[dict[str, Any]]) -> Path:
        """Write the full interpretable feature table to CSV."""

        csv_path = self.output_dir / "feature_weights.csv"
        fieldnames = [
            "feature_index",
            "lead_index",
            "lead_name",
            "window_index",
            "start_sec",
            "end_sec",
            "display_name",
            "surrogate_weight",
            "absolute_weight",
        ]

        with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in feature_rows:
                writer.writerow(row)

        return csv_path

    def write_summary_json(self, summary: dict[str, Any]) -> Path:
        """Write the main summary payload to JSON."""

        json_path = self.output_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2)
        return json_path

    def _figure_context(self):
        """Import plotting dependencies only when figures are actually needed."""

        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        return plt, Patch

    def save_leadwise_plot(
        self,
        sample: np.ndarray,
        interpreter: ECGWindowInterpreter,
        top_positive_rows: Sequence[dict[str, Any]],
        top_negative_rows: Sequence[dict[str, Any]],
        title: str,
    ) -> Path:
        """Plot all leads with the most influential windows shaded."""

        plt, Patch = self._figure_context()
        time_axis = np.arange(sample.shape[1], dtype=np.float32) / interpreter.sampling_rate_hz

        num_channels = sample.shape[0]
        ncols = 3 if num_channels >= 6 else (2 if num_channels > 1 else 1)
        nrows = math.ceil(num_channels / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.0 * ncols, 2.8 * nrows),
            constrained_layout=True,
            sharex=False,
        )

        axes_array = np.atleast_1d(np.array(axes, dtype=object)).reshape(nrows, ncols)
        selected_rows = list(top_positive_rows) + list(top_negative_rows)
        max_abs_weight = max(
            (abs(row["surrogate_weight"]) for row in selected_rows),
            default=1.0,
        )

        for lead_index in range(num_channels):
            ax = axes_array.flat[lead_index]
            lead_name = interpreter.lead_names[lead_index]
            signal = sample[lead_index]
            ax.plot(time_axis, signal, color="black", linewidth=1.0)

            y0, y1 = np.percentile(signal, 1), np.percentile(signal, 99)
            pad = 0.08 * (y1 - y0 + 1e-8)
            ax.set_ylim(y0 - pad, y1 + pad)

            for row in top_positive_rows:
                if row["lead_index"] != lead_index:
                    continue
                alpha = 0.15 + 0.35 * (abs(row["surrogate_weight"]) / max_abs_weight)
                ax.axvspan(row["start_sec"], row["end_sec"], color="#c0392b", alpha=alpha)

            for row in top_negative_rows:
                if row["lead_index"] != lead_index:
                    continue
                alpha = 0.15 + 0.35 * (abs(row["surrogate_weight"]) / max_abs_weight)
                ax.axvspan(row["start_sec"], row["end_sec"], color="#2c7fb8", alpha=alpha)

            ax.set_title(f"Lead {lead_name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.20)

        for spare_index in range(num_channels, nrows * ncols):
            axes_array.flat[spare_index].axis("off")

        legend_handles = [
            Patch(facecolor="#c0392b", alpha=0.35, label="Supports explained class"),
            Patch(facecolor="#2c7fb8", alpha=0.35, label="Opposes explained class"),
        ]
        fig.legend(handles=legend_handles, loc="upper right")
        fig.suptitle(title, fontsize=14)

        output_path = self.output_dir / "leadwise_windows.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def save_heatmap(
        self,
        weight_matrix: np.ndarray,
        interpreter: ECGWindowInterpreter,
        title: str,
    ) -> Path:
        """Save a heatmap over lead by time-window importance."""

        plt, _ = self._figure_context()
        fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)

        max_abs = float(np.max(np.abs(weight_matrix))) if weight_matrix.size else 1.0
        max_abs = max(max_abs, 1e-8)
        image = ax.imshow(
            weight_matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
            origin="upper",
        )

        xticks = np.arange(interpreter.num_time_windows)
        xticklabels = [
            f"{(index * interpreter.window_size_samples) / interpreter.sampling_rate_hz:.1f}"
            for index in xticks
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        ax.set_yticks(np.arange(interpreter.num_channels))
        ax.set_yticklabels(interpreter.lead_names)
        ax.set_xlabel("Window Start Time (s)")
        ax.set_ylabel("Lead")
        ax.set_title(title)

        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Surrogate Weight")

        output_path = self.output_dir / "importance_heatmap.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def save_top_bar_plot(
        self,
        top_positive_rows: Sequence[dict[str, Any]],
        top_negative_rows: Sequence[dict[str, Any]],
        title: str,
    ) -> Path:
        """Save a bar plot of the most positive and negative windows."""

        plt, _ = self._figure_context()
        rows = list(top_positive_rows) + list(reversed(top_negative_rows))
        fig_height = max(4.0, 0.42 * max(len(rows), 1))
        fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)

        if rows:
            labels = [row["display_name"] for row in rows]
            values = [row["surrogate_weight"] for row in rows]
            colors = ["#c0392b" if value >= 0 else "#2c7fb8" for value in values]

            positions = np.arange(len(rows))
            ax.barh(positions, values, color=colors)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_xlabel("Surrogate Weight")
            ax.set_title(title)
            ax.grid(True, axis="x", alpha=0.20)
        else:
            ax.text(
                0.5,
                0.5,
                "No non-zero surrogate weights were found for this explanation.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            ax.axis("off")

        output_path = self.output_dir / "top_windows_barplot.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def write_all(
        self,
        sample: np.ndarray,
        interpreter: ECGWindowInterpreter,
        explanation: LimeExplanation,
        prediction: PredictionResult,
        true_class: int,
        summary_context: dict[str, Any],
        top_k: int,
    ) -> dict[str, Path]:
        """Write all requested tables, figures, and JSON summaries."""

        feature_rows = interpreter.build_feature_rows(explanation.feature_weights)
        csv_path = self.write_feature_weights_csv(feature_rows)

        top_positive_rows = self._top_positive_rows(feature_rows, top_k=top_k)
        top_negative_rows = self._top_negative_rows(feature_rows, top_k=top_k)
        weight_matrix = interpreter.feature_weight_matrix(explanation.feature_weights)

        explanation_title = (
            f"{summary_context['dataset']} {summary_context['sampling_rate_hz']} Hz | "
            f"true={LABEL_MAP[true_class]} | pred={LABEL_MAP[prediction.predicted_class]} | "
            f"P(AFIB)={prediction.probabilities[1]:.4f} | "
            f"explained={LABEL_MAP[explanation.explained_class]}"
        )

        leadwise_path = self.save_leadwise_plot(
            sample=sample,
            interpreter=interpreter,
            top_positive_rows=top_positive_rows,
            top_negative_rows=top_negative_rows,
            title=explanation_title,
        )
        heatmap_path = self.save_heatmap(
            weight_matrix=weight_matrix,
            interpreter=interpreter,
            title=f"LIME Heatmap | {explanation_title}",
        )
        barplot_path = self.save_top_bar_plot(
            top_positive_rows=top_positive_rows,
            top_negative_rows=top_negative_rows,
            title=f"Top Supporting/Opposing Windows | {explanation_title}",
        )

        summary_payload = {
            "dataset": summary_context["dataset"],
            "sampling_rate_hz": summary_context["sampling_rate_hz"],
            "model": summary_context["model"],
            "split": summary_context["split"],
            "sample_index": summary_context["sample_index"],
            "true_label": LABEL_MAP[true_class],
            "predicted_label": LABEL_MAP[prediction.predicted_class],
            "explained_class": LABEL_MAP[explanation.explained_class],
            "class_probabilities": {
                LABEL_MAP[class_index]: float(probability)
                for class_index, probability in enumerate(prediction.probabilities.tolist())
            },
            "window_size_sec": interpreter.window_sec,
            "window_size_samples": interpreter.window_size_samples,
            "num_time_windows": interpreter.num_time_windows,
            "num_interpretable_features": interpreter.num_features,
            "num_perturbations": explanation.num_perturbations,
            "mask_strategy": explanation.mask_strategy,
            "kernel_width": explanation.kernel_width,
            "ridge_alpha": summary_context["ridge_alpha"],
            "seed": explanation.perturbation_seed,
            "surrogate_diagnostics": {
                "intercept": explanation.intercept,
                "weighted_r2": explanation.weighted_r2,
                "weighted_mse": explanation.weighted_mse,
                "original_target_score": explanation.original_target_score,
                "original_surrogate_score": explanation.original_surrogate_score,
            },
            "top_positive_windows": top_positive_rows,
            "top_negative_windows": top_negative_rows,
            "selection_context": summary_context.get("selection_context"),
            "method_notes": [
                "LIME is a local approximation around one ECG sample, not a proof of exact model reasoning.",
                "The interpretable units are lead-time windows in input space, so this complements gradient-based Grad-CAM.",
                "Masking ECG windows can create unrealistic physiological patterns, especially with aggressive perturbations.",
                "Deep models may rely on distributed or abstract features rather than only textbook-visible ECG markers.",
                "Disagreement between LIME and Grad-CAM is possible because they answer different explanation questions.",
            ],
            "files": {
                "feature_weights_csv": str(csv_path),
                "leadwise_windows_plot": str(leadwise_path),
                "importance_heatmap": str(heatmap_path),
                "top_windows_barplot": str(barplot_path),
            },
        }
        json_path = self.write_summary_json(summary_payload)
        return {
            "summary_json": json_path,
            "feature_weights_csv": csv_path,
            "leadwise_plot": leadwise_path,
            "heatmap_plot": heatmap_path,
            "barplot": barplot_path,
        }


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations after parsing."""

    if args.sample_idx is None and args.representative_case is None:
        raise ValueError("Provide --sample_idx or use --representative_case.")
    if args.window_sec <= 0:
        raise ValueError("--window_sec must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")
    if args.num_perturbations < 2:
        raise ValueError("--num_perturbations must be at least 2.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")


def default_output_dir(
    dataset_name: str,
    fs_dir_name: str,
    model_name: str,
    split: str,
    sample_index: int,
) -> Path:
    """Create the default output location for one explanation run."""

    return (
        PROJECT_ROOT
        / "lime_results"
        / dataset_name
        / fs_dir_name
        / model_name
        / f"{split}_sample_{sample_index}"
    )


def run() -> None:
    """CLI entrypoint for LIME-based ECG explanations."""

    args = parse_args()
    validate_args(args)
    set_deterministic_seed(args.seed)

    data_dir = resolve_existing_path(args.data_path)
    dataset_name = data_dir.parent.name
    fs_dir_name = data_dir.name
    sampling_rate_hz = parse_sampling_rate(fs_dir_name)

    X, y = load_split_tensor(data_dir=data_dir, split=args.split)
    if len(y) == 0:
        raise ValueError(f"No samples were found in split '{args.split}' at {data_dir}")

    in_channels = int(X[0].shape[0])
    num_classes = int(torch.max(y).item() + 1)
    predictor = EnsemblePredictor(
        model_name=args.model,
        dataset_name=dataset_name,
        fs_dir_name=fs_dir_name,
        in_channels=in_channels,
        num_classes=num_classes,
        folds=args.folds,
        device=args.device,
        checkpoint_root=resolve_checkpoints_root(),
    )

    selection_context: dict[str, Any] | None = None
    if args.representative_case is not None:
        selector = RepresentativeSampleSelector(predictor=predictor, batch_size=args.batch_size)
        sample_index, selection_context = selector.select(
            X=X,
            y=y,
            case=args.representative_case,
        )
    else:
        assert args.sample_idx is not None
        sample_index = args.sample_idx

    if sample_index < 0 or sample_index >= len(y):
        raise IndexError(
            f"sample index {sample_index} is out of range for split size {len(y)}"
        )

    sample = X[sample_index].unsqueeze(0).float()
    true_class = int(y[sample_index].item())
    interpreter = ECGWindowInterpreter(
        num_channels=sample.shape[1],
        time_length=sample.shape[2],
        sampling_rate_hz=sampling_rate_hz,
        window_sec=args.window_sec,
    )

    explainer = LimeExplainerECG(
        predictor=predictor,
        interpreter=interpreter,
        num_perturbations=args.num_perturbations,
        mask_strategy=args.mask_strategy,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        kernel_width=args.kernel_width,
        batch_size=args.batch_size,
    )
    explanation, prediction = explainer.explain(
        sample=sample,
        true_class=true_class,
        explain_class=args.explain_class,
    )

    output_dir = (
        resolve_output_path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(
            dataset_name=dataset_name,
            fs_dir_name=fs_dir_name,
            model_name=args.model,
            split=args.split,
            sample_index=sample_index,
        )
    )
    writer = LimeResultWriter(output_dir=output_dir)

    saved_files = writer.write_all(
        sample=sample.squeeze(0).numpy(),
        interpreter=interpreter,
        explanation=explanation,
        prediction=prediction,
        true_class=true_class,
        summary_context={
            "dataset": dataset_name,
            "sampling_rate_hz": sampling_rate_hz,
            "model": args.model,
            "split": args.split,
            "sample_index": sample_index,
            "ridge_alpha": args.ridge_alpha,
            "selection_context": selection_context,
        },
        top_k=args.top_k,
    )

    print(f"Dataset           : {dataset_name}")
    print(f"Sampling rate     : {sampling_rate_hz} Hz")
    print(f"Model             : {args.model}")
    print(f"Split             : {args.split}")
    print(f"Sample index      : {sample_index}")
    print(f"True label        : {LABEL_MAP[true_class]}")
    print(f"Predicted label   : {LABEL_MAP[prediction.predicted_class]}")
    print(f"P(NORMAL)         : {prediction.probabilities[0]:.4f}")
    print(f"P(AFIB)           : {prediction.probabilities[1]:.4f}")
    print(f"Explained class   : {LABEL_MAP[explanation.explained_class]}")
    print(f"Mask strategy     : {explanation.mask_strategy}")
    print(f"Perturbations     : {explanation.num_perturbations}")
    print(f"Weighted surrogate R^2 : {explanation.weighted_r2:.4f}")
    print(f"Output directory  : {writer.output_dir}")
    print("Saved files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    run()

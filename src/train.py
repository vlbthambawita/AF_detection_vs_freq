import argparse
import gc
from pathlib import Path
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from loader import load_kfold, ECGDataset
from models.cnn1d import CNN1D
from models.cnn_lstm import CNN_LSTM_ECG
from sklearn.metrics import roc_auc_score


# ================= REPRODUCIBILITY =================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= DEFAULT SETTINGS =================

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_KFOLDS = 5
DEFAULT_EARLY_STOPPING_PATIENCE = 10


# ================= DATASET HELPERS =================

def print_dataset_stats(name, dataset):
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    total = len(labels)

    if total == 0:
        print(f"{name:<18}: EMPTY")
        return

    afib = sum(1 for y in labels if y == 1)
    normal = total - afib

    print(
        f"{name:<18}: total={total} | "
        f"Normal={normal} ({100 * normal / total:.2f}%) | "
        f"AFIB={afib} ({100 * afib / total:.2f}%)"
    )


def make_balanced_subset_binary(dataset, seed=42):
    """
    Downsample majority class to match minority class.

    Used for:
    - training subset balancing
    - secondary balanced test evaluation

    Validation is not balanced by this function unless the user already
    prepared a balanced validation split during preprocessing.
    """

    rng = np.random.RandomState(seed)

    labels = np.array(
        [int(dataset[i][1]) for i in range(len(dataset))],
        dtype=int,
    )

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return dataset

    n = min(len(idx0), len(idx1))

    idx0 = rng.choice(idx0, size=n, replace=False)
    idx1 = rng.choice(idx1, size=n, replace=False)

    keep = np.concatenate([idx0, idx1])
    rng.shuffle(keep)

    return Subset(dataset, keep.tolist())


def build_model(model_name, in_ch, num_classes):
    if model_name == "cnn1d":
        return CNN1D(in_ch, num_classes)

    if model_name == "cnn_lstm":
        return CNN_LSTM_ECG(
            in_channels=in_ch,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model type: {model_name}")


def make_loader(dataset, batch_size, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )


def tensor_file_to_dataset(pt_path: Path):
    """
    Load a .pt split file containing X and y into the same lightweight
    ECGDataset object style used by the existing code.
    """

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing dataset split file: {pt_path}")

    data = torch.load(pt_path, map_location="cpu")

    if "X" not in data or "y" not in data:
        raise KeyError(f"{pt_path} must contain keys 'X' and 'y'.")

    ds = ECGDataset.__new__(ECGDataset)
    ds.X = data["X"]
    ds.y = data["y"]

    # Preserve optional metadata if present.
    if "record_ids" in data:
        ds.record_ids = data["record_ids"]
    if "patient_ids" in data:
        ds.patient_ids = data["patient_ids"]

    return ds


def dataset_labels(dataset):
    """
    Return labels from either a full ECGDataset-like object or a torch Subset.

    make_balanced_subset_binary() returns torch.utils.data.Subset.
    Subset does not expose .y directly, so label access must also work
    through __getitem__.
    """

    if hasattr(dataset, "y"):
        y = dataset.y

        if isinstance(y, torch.Tensor):
            return y.detach().cpu().long()

        return torch.tensor(y, dtype=torch.long)

    labels = [int(dataset[i][1]) for i in range(len(dataset))]

    if not labels:
        raise ValueError("Cannot infer labels from an empty dataset.")

    return torch.tensor(labels, dtype=torch.long)


def infer_num_classes(*datasets):
    """
    Infer the number of classes from one or more datasets.
    Works for both ECGDataset-like objects and torch Subset objects.
    """

    max_label = None

    for dataset in datasets:
        labels = dataset_labels(dataset)

        if labels.numel() == 0:
            continue

        current_max = int(torch.max(labels).item())
        max_label = current_max if max_label is None else max(max_label, current_max)

    if max_label is None:
        raise ValueError("Cannot infer number of classes from empty dataset(s).")

    return max_label + 1


def infer_in_channels(dataset):
    if len(dataset) == 0:
        raise ValueError("Cannot infer input channels from an empty dataset.")

    return int(dataset[0][0].shape[0])


def find_test_file(data_dir: Path) -> Path | None:
    """
    Support both:
    - k-fold + hold-out structure: data_dir/test/test.pt
    - manual split structure:      data_dir/test.pt
    """

    candidates = [
        data_dir / "test" / "test.pt",
        data_dir / "test.pt",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def detect_split_mode(data_dir: Path, requested_mode: str, kfolds: int) -> str:
    """
    Decide whether training should use:
    - kfold:  fold_1.pt ... fold_k.pt
    - manual: train.pt and val.pt

    requested_mode can be:
    - auto
    - kfold
    - manual
    """

    fold_files = [data_dir / f"fold_{i}.pt" for i in range(1, kfolds + 1)]
    has_all_folds = all(p.exists() for p in fold_files)

    has_manual = (data_dir / "train.pt").exists() and (data_dir / "val.pt").exists()

    if requested_mode == "kfold":
        if not has_all_folds:
            missing = [str(p) for p in fold_files if not p.exists()]
            raise FileNotFoundError(
                "split_mode=kfold was requested, but not all fold files exist.\n"
                "Missing:\n" + "\n".join(missing)
            )
        return "kfold"

    if requested_mode == "manual":
        if not has_manual:
            raise FileNotFoundError(
                "split_mode=manual was requested, but train.pt and/or val.pt is missing.\n"
                f"Expected:\n{data_dir / 'train.pt'}\n{data_dir / 'val.pt'}"
            )
        return "manual"

    # Auto mode: prefer k-fold if all fold files exist, otherwise manual.
    if has_all_folds:
        return "kfold"

    if has_manual:
        return "manual"

    raise FileNotFoundError(
        "Could not detect split structure.\n\n"
        "Expected either k-fold files:\n"
        + "\n".join(str(p) for p in fold_files)
        + "\n\nor manual split files:\n"
        + f"{data_dir / 'train.pt'}\n{data_dir / 'val.pt'}\n"
    )


# ================= METRICS =================

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)


def compute_metrics_from_cm(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1e-12
    mcc = mcc_num / mcc_den

    return recall, specificity, precision, mcc


def compute_ece_posclass(y_true, y_prob_pos, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(y_prob_pos, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = binids == b

        if np.any(m):
            frac_pos = y_true[m].mean()
            conf_b = y_prob_pos[m].mean()
            ece += (m.sum() / N) * abs(frac_pos - conf_b)

    return ece


# ================= EVALUATION =================

def evaluate(model, loader, device, return_scores=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total = 0
    correct = 0

    tp = tn = fp = fn = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            loss = loss_fn(logits, y)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (preds == y).sum().item()

            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

            if return_scores:
                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

    acc = correct / total
    f1 = compute_f1(tp, fp, fn)
    cm = [[tn, fp], [fn, tp]]
    avg_loss = total_loss / total

    if return_scores:
        return (
            acc,
            f1,
            cm,
            avg_loss,
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
        )

    return acc, f1, cm, avg_loss


def evaluate_ensemble(models, loader, device):
    for m in models:
        m.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total = 0
    correct = 0

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            logits_sum = None

            for m in models:
                logits = m(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            logits_avg = logits_sum / len(models)
            loss = loss_fn(logits_avg, y)

            preds = logits_avg.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (preds == y).sum().item()

            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    acc = correct / total
    f1 = compute_f1(tp, fp, fn)
    cm = [[tn, fp], [fn, tp]]

    return acc, f1, cm, total_loss / total


def evaluate_ensemble_with_probs(models, loader, device):
    for m in models:
        m.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            logits_sum = None

            for m in models:
                logits = m(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits

            logits_avg = logits_sum / len(models)
            probs = torch.softmax(logits_avg, dim=1)[:, 1]

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    return y_true, y_prob


# ================= TRAIN ONE SPLIT =================

def train_one_split(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    out_dir,
    epochs,
    early_stopping_patience,
    split_label="fold",
):
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_f1 = -1
    best_acc = None
    best_cm = None
    best_epoch = None
    best_val_loss = None
    best_roc = None

    bad_epochs = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    split_start = time.time()

    print()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            total += y.size(0)

        train_loss /= total

        acc, f1, cm, val_loss, y_score, y_true = evaluate(
            model,
            val_loader,
            device,
            return_scores=True,
        )

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"ACC {acc * 100:.2f}% | "
            f"F1 {f1:.4f} | "
            f"Time {epoch_time:.2f}s"
        )

        torch.save(model.state_dict(), out_dir / "last.pt")

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_cm = cm
            best_epoch = epoch
            best_val_loss = val_loss
            best_roc = {
                "y_true": y_true,
                "y_score": y_score,
            }

            bad_epochs = 0

            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1

        if best_roc is not None:
            np.savez(
                out_dir / "roc_val.npz",
                y_true=best_roc["y_true"],
                y_score=best_roc["y_score"],
            )

        if bad_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    split_time = time.time() - split_start

    recall, specificity, precision, mcc = compute_metrics_from_cm(best_cm)

    tn, fp = best_cm[0]
    fn, tp = best_cm[1]

    print("\n" + "=" * 70)
    print(f"{split_label} Results – Training time: {split_time / 60:.2f} minutes")
    print("Confusion Matrix (Validation)")
    print(f"[[{tn:4d} {fp:3d}]")
    print(f" [{fn:4d} {tp:3d}]]\n")

    print("Best F1    Accuracy   Recall(Sens)  Specificity  Precision")
    print("-" * 58)
    print(
        f"{best_f1:<10.4f}"
        f"{best_acc:<11.4f}"
        f"{recall:<15.4f}"
        f"{specificity:<13.4f}"
        f"{precision:<10.4f}"
    )
    print("=" * 70)

    metrics_path = out_dir / "metrics.txt"

    with open(metrics_path, "w") as f:
        f.write("BEST VALIDATION METRICS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Split              : {out_dir.name}\n")
        f.write(f"Best epoch         : {best_epoch}\n")
        f.write(f"Validation loss    : {best_val_loss:.4f}\n")
        f.write(f"Validation F1      : {best_f1:.4f}\n")
        f.write(f"Validation Accuracy: {best_acc:.4f}\n\n")

        f.write("CONFUSION MATRIX (VAL)\n")
        f.write(f"[[{tn} {fp}]\n")
        f.write(f" [{fn} {tp}]]\n\n")

        f.write("DERIVED METRICS\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity         : {specificity:.4f}\n")
        f.write(f"Precision           : {precision:.4f}\n")
        f.write(f"MCC                 : {mcc:.4f}\n\n")

        f.write("RUNTIME\n")
        f.write(f"Training time (sec): {split_time:.2f}\n")
        f.write(f"Training time (min): {split_time / 60:.2f}\n\n")

        f.write("RUN INFO\n")
        f.write("-" * 60 + "\n")
        f.write("Random seed        : 42\n")

    return {
        "fold": out_dir.name,
        "best_epoch": int(best_epoch),
        "val_loss": float(best_val_loss),
        "accuracy": float(best_acc),
        "f1": float(best_f1),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "time_min": float(split_time / 60.0),
    }


# Backward-compatible function name if other scripts import it.
train_one_fold = train_one_split


# ================= TRAINING MODES =================

def train_kfold(args, data_dir, fs, dataset_name, device):
    training_start = time.time()
    fold_results = []

    best_fold = None
    best_f1_overall = -1

    for fold in range(1, args.kfolds + 1):
        print(f"\n=== Fold {fold}/{args.kfolds} ===")

        train_ds, val_ds, num_classes = load_kfold(
            data_dir,
            fs,
            fold,
            args.kfolds,
        )

        print_dataset_stats("Train raw", train_ds)
        print_dataset_stats("Validation", val_ds)

        if args.train_balance == "downsample":
            train_ds = make_balanced_subset_binary(
                train_ds,
                seed=42 + fold,
            )
            print_dataset_stats("Train balanced", train_ds)
        else:
            print("Train balancing: disabled")

        train_loader = make_loader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            device=device,
        )

        val_loader = make_loader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            device=device,
        )

        in_ch = infer_in_channels(train_ds)

        model = build_model(
            args.model,
            in_ch,
            num_classes,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        metrics = train_one_split(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=Path("checkpoints") / dataset_name / f"{fs}hz" / args.model / f"fold_{fold}",
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            split_label=f"Fold {fold}",
        )

        fold_results.append(metrics)

        if metrics["f1"] > best_f1_overall:
            best_f1_overall = metrics["f1"]
            best_fold = fold

        del model
        del optimizer
        del train_loader
        del val_loader
        del train_ds
        del val_ds

        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    training_total_time = time.time() - training_start

    print(f"\nBest fold           : {best_fold}")
    print(f"Best validation F1  : {best_f1_overall:.4f}")
    print(f"Total training time : {training_total_time / 60:.2f} minutes")

    write_validation_table(
        dataset_name=dataset_name,
        fs=fs,
        model_name=args.model,
        split_mode="kfold",
        fold_results=fold_results,
    )

    return training_total_time, fold_results


def train_manual_split(args, data_dir, fs, dataset_name, device):
    training_start = time.time()

    print("\n=== Manual split training ===")

    train_ds = tensor_file_to_dataset(data_dir / "train.pt")
    val_ds = tensor_file_to_dataset(data_dir / "val.pt")

    print_dataset_stats("Train raw", train_ds)
    print_dataset_stats("Validation", val_ds)

    if args.train_balance == "downsample":
        train_ds = make_balanced_subset_binary(
            train_ds,
            seed=42,
        )
        print_dataset_stats("Train balanced", train_ds)
    else:
        print("Train balancing: disabled")

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )

    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )

    in_ch = infer_in_channels(train_ds)
    num_classes = infer_num_classes(train_ds, val_ds)

    model = build_model(
        args.model,
        in_ch,
        num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    metrics = train_one_split(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=Path("checkpoints") / dataset_name / f"{fs}hz" / args.model / "manual_split",
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        split_label="Manual split",
    )

    training_total_time = time.time() - training_start

    print(f"\nBest validation F1  : {metrics['f1']:.4f}")
    print(f"Total training time : {training_total_time / 60:.2f} minutes")

    write_validation_table(
        dataset_name=dataset_name,
        fs=fs,
        model_name=args.model,
        split_mode="manual",
        fold_results=[metrics],
    )

    del model
    del optimizer
    del train_loader
    del val_loader
    del train_ds
    del val_ds

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return training_total_time, [metrics]


# ================= RESULT WRITING =================

def write_validation_table(dataset_name, fs, model_name, split_mode, fold_results):
    table_path = (
        Path("checkpoints")
        / dataset_name
        / "validation_table.txt"
    )

    table_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not table_path.exists()

    freq_str = f"{fs} Hz"
    model_str = model_name

    acc_avg = float(np.mean([r["accuracy"] for r in fold_results]))
    f1_avg = float(np.mean([r["f1"] for r in fold_results]))
    prec_avg = float(np.mean([r["precision"] for r in fold_results]))
    spec_avg = float(np.mean([r["specificity"] for r in fold_results]))
    mcc_avg = float(np.mean([r["mcc"] for r in fold_results]))
    time_avg = float(np.mean([r["time_min"] for r in fold_results]))

    with open(table_path, "a") as f:
        if write_header:
            f.write("VALIDATION PERFORMANCE ACROSS SAMPLING FREQUENCIES\n")
            f.write("=" * 150 + "\n\n")
            f.write(
                f"{'Model':<10}"
                f"{'Freq':<10}"
                f"{'Mode':<12}"
                f"{'Split':<14}"
                f"{'Acc':<10}"
                f"{'F1':<10}"
                f"{'Prec':<10}"
                f"{'Spec':<10}"
                f"{'MCC':<10}"
                f"{'Time':<10}\n"
            )
            f.write("-" * 110 + "\n")

        mid_row = (len(fold_results) // 2) + 1

        for i, r in enumerate(fold_results, 1):
            model_cell = model_str if i == mid_row else ""
            freq_cell = freq_str if i == mid_row else ""
            mode_cell = split_mode if i == mid_row else ""

            if split_mode == "manual":
                split_cell = "Manual"
            else:
                split_cell = f"Fold {i}"

            f.write(
                f"{model_cell:<10}"
                f"{freq_cell:<10}"
                f"{mode_cell:<12}"
                f"{split_cell:<14}"
                f"{r['accuracy']:<10.4f}"
                f"{r['f1']:<10.4f}"
                f"{r['precision']:<10.4f}"
                f"{r['specificity']:<10.4f}"
                f"{r['mcc']:<10.4f}"
                f"{r['time_min']:<10.2f}\n"
            )

        if len(fold_results) > 1:
            f.write(
                f"{'':<10}{'':<10}{'':<12}{'Avg':<14}"
                f"{acc_avg:<10.4f}"
                f"{f1_avg:<10.4f}"
                f"{prec_avg:<10.4f}"
                f"{spec_avg:<10.4f}"
                f"{mcc_avg:<10.4f}"
                f"{time_avg:<10.2f}\n"
            )

        f.write("-" * 110 + "\n")

    print(f"\nValidation table updated: {table_path}")


def load_models_for_final_evaluation(
    split_mode,
    data_dir,
    dataset_name,
    fs,
    model_name,
    kfolds,
    in_ch,
    num_classes,
    device,
):
    models = []

    if split_mode == "kfold":
        for fold in range(1, kfolds + 1):
            m = build_model(
                model_name,
                in_ch,
                num_classes,
            ).to(device)

            best_model_path = (
                Path("checkpoints")
                / dataset_name
                / f"{fs}hz"
                / model_name
                / f"fold_{fold}"
                / "best.pt"
            )

            if not best_model_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {best_model_path}")

            m.load_state_dict(torch.load(best_model_path, map_location=device))
            m.to(device)
            models.append(m)

        return models

    if split_mode == "manual":
        m = build_model(
            model_name,
            in_ch,
            num_classes,
        ).to(device)

        best_model_path = (
            Path("checkpoints")
            / dataset_name
            / f"{fs}hz"
            / model_name
            / "manual_split"
            / "best.pt"
        )

        if not best_model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {best_model_path}")

        m.load_state_dict(torch.load(best_model_path, map_location=device))
        m.to(device)
        models.append(m)

        return models

    raise ValueError(f"Unknown split_mode: {split_mode}")


# ================= FINAL TEST =================

def run_final_test(
    data_dir,
    dataset_name,
    fs,
    model_name,
    batch_size,
    kfolds,
    split_mode,
    device,
    training_total_time=None,
):
    test_file = find_test_file(data_dir)

    if test_file is None:
        print("\nTest file not found — skipping test evaluation.")
        print("Expected one of:")
        print(f"  {data_dir / 'test' / 'test.pt'}")
        print(f"  {data_dir / 'test.pt'}")
        return

    print("\n" + "=" * 70)
    if split_mode == "kfold":
        print("FINAL TEST EVALUATION (Ensemble of all folds' best epochs)")
    else:
        print("FINAL TEST EVALUATION (Best manual-split checkpoint)")
    print("=" * 70)

    test_ds = tensor_file_to_dataset(test_file)

    print_dataset_stats("Test unbalanced", test_ds)

    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    test_balanced_ds = make_balanced_subset_binary(
        test_ds,
        seed=42,
    )

    print_dataset_stats("Test balanced", test_balanced_ds)

    test_balanced_loader = make_loader(
        test_balanced_ds,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    in_ch = infer_in_channels(test_ds)
    num_classes = infer_num_classes(test_ds)

    models = load_models_for_final_evaluation(
        split_mode=split_mode,
        data_dir=data_dir,
        dataset_name=dataset_name,
        fs=fs,
        model_name=model_name,
        kfolds=kfolds,
        in_ch=in_ch,
        num_classes=num_classes,
        device=device,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    # ---------- Unbalanced Test ----------
    start = time.time()

    acc_u, f1_u, cm_u, _ = evaluate_ensemble(
        models,
        test_loader,
        device,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_u = time.time() - start

    recall_u, specificity_u, precision_u, mcc_u = compute_metrics_from_cm(cm_u)

    # ---------- Balanced Test ----------
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    acc_b, f1_b, cm_b, _ = evaluate_ensemble(
        models,
        test_balanced_loader,
        device,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_b = time.time() - start

    recall_b, specificity_b, precision_b, mcc_b = compute_metrics_from_cm(cm_b)

    # ---------- ROC / AUROC / ECE ----------
    y_true_u, y_prob_u = evaluate_ensemble_with_probs(
        models,
        test_loader,
        device,
    )

    auroc_u = roc_auc_score(y_true_u, y_prob_u)
    ece_u = compute_ece_posclass(y_true_u, y_prob_u)

    results_dir = (
        Path("checkpoints")
        / dataset_name
        / f"{fs}hz"
        / model_name
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "roc_test.npz",
        y_true=y_true_u,
        y_score=y_prob_u,
    )

    # ---------- ROC / AUROC / ECE: Balanced ----------
    y_true_b, y_prob_b = evaluate_ensemble_with_probs(
        models,
        test_balanced_loader,
        device,
    )

    auroc_b = roc_auc_score(y_true_b, y_prob_b)
    ece_b = compute_ece_posclass(y_true_b, y_prob_b)

    np.savez(
        results_dir / "roc_test_balanced.npz",
        y_true=y_true_b,
        y_score=y_prob_b,
    )

    # ================= SAVE TEST TABLE =================

    test_table_path = (
        Path("checkpoints")
        / dataset_name
        / "test_table.txt"
    )

    write_header = not test_table_path.exists()

    thpt_u = len(test_ds) / (elapsed_u + 1e-12)
    thpt_b = len(test_balanced_ds) / (elapsed_b + 1e-12)

    model_str = model_name
    freq_str = f"{fs} Hz"

    with open(test_table_path, "a") as f:
        if write_header:
            f.write("FINAL TEST PERFORMANCE\n")
            f.write("=" * 150 + "\n\n")
            f.write(
                f"{'Model':<10}"
                f"{'Freq':<10}"
                f"{'Mode':<12}"
                f"{'Test':<10}"
                f"{'Acc':<10}"
                f"{'F1':<10}"
                f"{'Prec':<10}"
                f"{'Spec':<10}"
                f"{'MCC':<10}"
                f"{'AUROC':<10}"
                f"{'ECE':<10}"
                f"{'Time(s)':<10}"
                f"{'Thpt':<10}\n"
            )
            f.write("-" * 140 + "\n")

        f.write(
            f"{model_str:<10}"
            f"{freq_str:<10}"
            f"{split_mode:<12}"
            f"{'Unbal':<10}"
            f"{acc_u:<10.4f}"
            f"{f1_u:<10.4f}"
            f"{precision_u:<10.4f}"
            f"{specificity_u:<10.4f}"
            f"{mcc_u:<10.4f}"
            f"{auroc_u:<10.4f}"
            f"{ece_u:<10.4f}"
            f"{elapsed_u:<10.2f}"
            f"{thpt_u:<10.2f}\n"
        )

        f.write(
            f"{'':<10}"
            f"{'':<10}"
            f"{'':<12}"
            f"{'Bal':<10}"
            f"{acc_b:<10.4f}"
            f"{f1_b:<10.4f}"
            f"{precision_b:<10.4f}"
            f"{specificity_b:<10.4f}"
            f"{mcc_b:<10.4f}"
            f"{auroc_b:<10.4f}"
            f"{ece_b:<10.4f}"
            f"{elapsed_b:<10.2f}"
            f"{thpt_b:<10.2f}\n"
        )

        f.write("-" * 140 + "\n")

    print(f"\nMaster test table updated: {test_table_path}")

    # ================= SAVE FINAL TEST RESULTS =================

    results_path = results_dir / "test_results.txt"

    tn_u, fp_u = cm_u[0]
    fn_u, tp_u = cm_u[1]

    tn_b, fp_b = cm_b[0]
    fn_b, tp_b = cm_b[1]

    with open(results_path, "w") as f:
        f.write("FINAL TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("RUN TYPE\n")
        f.write("-" * 60 + "\n")
        f.write(f"Split mode           : {split_mode}\n")
        if split_mode == "kfold":
            f.write(f"Folds ensembled      : {kfolds}\n")
        else:
            f.write("Folds ensembled      : 1 manual-split model\n")
        f.write("\n")

        f.write("UNBALANCED TEST\n")
        f.write("-" * 60 + "\n")
        f.write("CONFUSION MATRIX (TEST)\n")
        f.write(f"[[{tn_u}  {fp_u}]\n")
        f.write(f" [{fn_u}   {tp_u}]]\n\n")

        f.write("METRICS\n")
        f.write(f"Accuracy            : {acc_u:.4f}\n")
        f.write(f"F1-score            : {f1_u:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall_u:.4f}\n")
        f.write(f"Specificity         : {specificity_u:.4f}\n")
        f.write(f"Precision           : {precision_u:.4f}\n")
        f.write(f"MCC                 : {mcc_u:.4f}\n")
        f.write(f"AUROC               : {auroc_u:.4f}\n")
        f.write(f"ECE                 : {ece_u:.4f}\n\n")

        f.write("PERFORMANCE\n")
        if training_total_time is not None:
            f.write(f"Training time (total) : {training_total_time / 60:.2f} minutes\n")
        f.write(f"Inference time        : {elapsed_u:.2f} seconds\n")
        f.write(f"Throughput            : {thpt_u:.2f} samples/sec\n\n\n")

        f.write("BALANCED TEST (DOWNSAMPLED MAJORITY)\n")
        f.write("-" * 60 + "\n")
        f.write("CONFUSION MATRIX (TEST)\n")
        f.write(f"[[{tn_b}  {fp_b}]\n")
        f.write(f" [{fn_b}   {tp_b}]]\n\n")

        f.write("METRICS\n")
        f.write(f"Accuracy            : {acc_b:.4f}\n")
        f.write(f"F1-score            : {f1_b:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall_b:.4f}\n")
        f.write(f"Specificity         : {specificity_b:.4f}\n")
        f.write(f"Precision           : {precision_b:.4f}\n")
        f.write(f"MCC                 : {mcc_b:.4f}\n")
        f.write(f"AUROC               : {auroc_b:.4f}\n")
        f.write(f"ECE                 : {ece_b:.4f}\n\n")

        f.write("PERFORMANCE\n")
        f.write(f"Inference time        : {elapsed_b:.2f} seconds\n")
        f.write(f"Throughput            : {thpt_b:.2f} samples/sec\n\n\n")

        f.write("RUN INFORMATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset              : {dataset_name}\n")
        f.write(f"Sampling rate        : {fs} Hz\n")
        f.write(f"Model                : {model_name}\n")
        f.write(f"Device               : {device}\n")
        f.write(f"Split mode           : {split_mode}\n\n")

        f.write("DATASET SIZES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test samples (unbalanced) : {len(test_ds)}\n")
        f.write(f"Test samples (balanced)   : {len(test_balanced_ds)}\n")

    # ---------- Save confusion matrices ----------
    cm_csv_u = results_dir / "confusion_matrix_test_unbalanced.csv"

    with open(cm_csv_u, "w") as f:
        f.write("TN,FP\n")
        f.write(f"{tn_u},{fp_u}\n")
        f.write("FN,TP\n")
        f.write(f"{fn_u},{tp_u}\n")

    cm_csv_b = results_dir / "confusion_matrix_test_balanced.csv"

    with open(cm_csv_b, "w") as f:
        f.write("TN,FP\n")
        f.write(f"{tn_b},{fp_b}\n")
        f.write("FN,TP\n")
        f.write(f"{fn_b},{tp_b}\n")

    # ---------- Print final results ----------
    print("\nUNBALANCED TEST")
    print("Confusion Matrix (Test)")
    print(f"[[{tn_u:4d} {fp_u:3d}]")
    print(f" [{fn_u:4d} {tp_u:3d}]]\n")
    print("F1        Accuracy   Recall(Sens)  Specificity  Precision MCC      AUROC    ECE")
    print("-" * 90)
    print(
        f"{f1_u:<9.4f}"
        f"{acc_u:<11.4f}"
        f"{recall_u:<15.4f}"
        f"{specificity_u:<13.4f}"
        f"{precision_u:<10.4f}"
        f"{mcc_u:<9.4f}"
        f"{auroc_u:<9.4f}"
        f"{ece_u:<.4f}"
    )
    print(f"\nInference time : {elapsed_u:.2f} seconds")
    print(f"Throughput     : {thpt_u:.2f} samples/sec")
    print("=" * 70)

    print("\nBALANCED TEST")
    print("Confusion Matrix (Test)")
    print(f"[[{tn_b:4d} {fp_b:3d}]")
    print(f" [{fn_b:4d} {tp_b:3d}]]\n")
    print("F1        Accuracy   Recall(Sens)  Specificity  Precision MCC      AUROC    ECE")
    print("-" * 90)
    print(
        f"{f1_b:<9.4f}"
        f"{acc_b:<11.4f}"
        f"{recall_b:<15.4f}"
        f"{specificity_b:<13.4f}"
        f"{precision_b:<10.4f}"
        f"{mcc_b:<9.4f}"
        f"{auroc_b:<9.4f}"
        f"{ece_b:<.4f}"
    )
    print(f"\nInference time : {elapsed_b:.2f} seconds")
    print(f"Throughput     : {thpt_b:.2f} samples/sec")
    print("=" * 70)

    del models
    del test_loader
    del test_balanced_loader
    del test_ds
    del test_balanced_ds

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ================= TEST ONLY =================

def run_test_only(
    data_path,
    model_name,
    batch_size,
    kfolds,
    split_mode,
    device,
):
    data_dir = Path(data_path)
    fs = int(data_dir.name.replace("hz", ""))
    dataset_name = data_dir.parent.name

    print("\nTEST ONLY MODE")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Dataset       : {dataset_name}")
    print(f"Sampling rate : {fs} Hz")
    print(f"Model         : {model_name}")
    print(f"Split mode    : {split_mode}")
    print("=" * 60)

    run_final_test(
        data_dir=data_dir,
        dataset_name=dataset_name,
        fs=fs,
        model_name=model_name,
        batch_size=batch_size,
        kfolds=kfolds,
        split_mode=split_mode,
        device=device,
        training_total_time=None,
    )


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)

    parser.add_argument(
        "--model",
        choices=["cnn1d", "cnn_lstm"],
        required=True,
    )

    parser.add_argument(
        "--split_mode",
        choices=["auto", "kfold", "manual"],
        default="auto",
        help=(
            "Training split mode. "
            "auto detects fold_*.pt or train.pt/val.pt; "
            "kfold requires fold_1.pt ... fold_k.pt; "
            "manual requires train.pt and val.pt."
        ),
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Skip training and run final test evaluation only.",
    )

    parser.add_argument(
        "--train_balance",
        choices=["none", "downsample"],
        default="downsample",
        help=(
            "Runtime balancing for training data only. "
            "With kfold, it balances only the training folds. "
            "With manual split, it balances only train.pt."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--kfolds",
        type=int,
        default=DEFAULT_KFOLDS,
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    args = parser.parse_args()

    set_seed(42)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    data_dir = Path(args.data_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    fs = int(data_dir.name.replace("hz", ""))
    dataset_name = data_dir.parent.name

    split_mode = detect_split_mode(
        data_dir=data_dir,
        requested_mode=args.split_mode,
        kfolds=args.kfolds,
    )

    print("=" * 70)
    print("ECG TRAINING")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Dataset       : {dataset_name}")
    print(f"Sampling rate : {fs} Hz")
    print(f"Model         : {args.model}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"LR            : {args.lr}")
    print(f"Split mode    : {split_mode}")
    if split_mode == "kfold":
        print(f"K-folds       : {args.kfolds}")
    print(f"Train balance : {args.train_balance}")
    print("=" * 70)

    if args.test_only:
        run_test_only(
            data_path=args.data_path,
            model_name=args.model,
            batch_size=args.batch_size,
            kfolds=args.kfolds,
            split_mode=split_mode,
            device=device,
        )
        return

    if split_mode == "kfold":
        training_total_time, _ = train_kfold(
            args=args,
            data_dir=data_dir,
            fs=fs,
            dataset_name=dataset_name,
            device=device,
        )
    else:
        training_total_time, _ = train_manual_split(
            args=args,
            data_dir=data_dir,
            fs=fs,
            dataset_name=dataset_name,
            device=device,
        )

    run_final_test(
        data_dir=data_dir,
        dataset_name=dataset_name,
        fs=fs,
        model_name=args.model,
        batch_size=args.batch_size,
        kfolds=args.kfolds,
        split_mode=split_mode,
        device=device,
        training_total_time=training_total_time,
    )


if __name__ == "__main__":
    main()

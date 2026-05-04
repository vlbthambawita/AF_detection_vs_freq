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

    Validation is NOT balanced.
    """

    rng = np.random.RandomState(seed)

    labels = np.array(
        [int(dataset[i][1]) for i in range(len(dataset))],
        dtype=int
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
            num_classes=num_classes
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


# ================= TRAIN ONE FOLD =================

def train_one_fold(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    out_dir,
    epochs,
    early_stopping_patience,
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
    fold_start = time.time()

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

    fold_time = time.time() - fold_start

    recall, specificity, precision, mcc = compute_metrics_from_cm(best_cm)

    tn, fp = best_cm[0]
    fn, tp = best_cm[1]

    print("\n" + "=" * 70)
    print(f"Fold Results – Training time: {fold_time / 60:.2f} minutes")
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
        f.write("BEST VALIDATION METRICS (PER FOLD)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Fold               : {out_dir.name}\n")
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
        f.write(f"Fold training time (sec): {fold_time:.2f}\n")
        f.write(f"Fold training time (min): {fold_time / 60:.2f}\n\n")

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
        "time_min": float(fold_time / 60.0),
    }


# ================= TEST ONLY =================

def run_test_only(
    data_path,
    model_name,
    batch_size,
    kfolds,
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
    print("=" * 60)

    test_data = torch.load(
        data_dir / "test" / "test.pt",
        map_location="cpu"
    )

    test_ds = ECGDataset.__new__(ECGDataset)
    test_ds.X = test_data["X"]
    test_ds.y = test_data["y"]

    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    in_ch = test_ds.X[0].shape[0]
    num_classes = int(torch.max(test_ds.y).item() + 1)

    models = []

    for fold in range(1, kfolds + 1):
        m = build_model(model_name, in_ch, num_classes)

        ckpt = (
            Path("checkpoints")
            / dataset_name
            / f"{fs}hz"
            / model_name
            / f"fold_{fold}"
            / "best.pt"
        )

        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device)
        models.append(m)

    y_true, y_prob = evaluate_ensemble_with_probs(
        models,
        test_loader,
        device,
    )

    auroc = roc_auc_score(y_true, y_prob)
    ece = compute_ece_posclass(y_true, y_prob)

    print(f"\nAUROC : {auroc:.4f}")
    print(f"ECE   : {ece:.4f}")
    print(f"Prob min/mean/max: {y_prob.min():.4f} / {y_prob.mean():.4f} / {y_prob.max():.4f}")
    print(f"Positive rate    : {y_true.mean():.4f}")

    out = Path("checkpoints") / dataset_name / f"{fs}hz" / model_name
    out.mkdir(parents=True, exist_ok=True)

    np.savez(
        out / "roc_test.npz",
        y_true=y_true,
        y_score=y_prob,
    )

    print("Saved:", out / "roc_test.npz")


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
        "--test_only",
        action="store_true",
        help="Skip training and run ensemble test evaluation only.",
    )

    parser.add_argument(
        "--train_balance",
        choices=["none", "downsample"],
        default="downsample",
        help=(
            "Runtime balancing for training folds only. "
            "Use downsample with preprocessing --balance_mode train."
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
    fs = int(data_dir.name.replace("hz", ""))
    dataset_name = data_dir.parent.name

    print("=" * 70)
    print("ECG K-FOLD TRAINING")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Dataset       : {dataset_name}")
    print(f"Sampling rate : {fs} Hz")
    print(f"Model         : {args.model}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"LR            : {args.lr}")
    print(f"K-folds       : {args.kfolds}")
    print(f"Train balance : {args.train_balance}")
    print("=" * 70)

    if args.test_only:
        run_test_only(
            data_path=args.data_path,
            model_name=args.model,
            batch_size=args.batch_size,
            kfolds=args.kfolds,
            device=device,
        )
        return

    training_start = time.time()
    fold_results = []

    best_fold = None
    best_f1_overall = -1

    # ================= K-FOLD LOOP =================

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

        in_ch = train_ds[0][0].shape[0]

        model = build_model(
            args.model,
            in_ch,
            num_classes,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        metrics = train_one_fold(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=Path("checkpoints") / dataset_name / f"{fs}hz" / args.model / f"fold_{fold}",
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
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
    training_total_min = training_total_time / 60

    print(f"\nBest fold           : {best_fold}")
    print(f"Best validation F1  : {best_f1_overall:.4f}")
    print(f"Total training time : {training_total_min:.2f} minutes")

    # ================= VALIDATION TABLE =================

    table_path = (
        Path("checkpoints")
        / dataset_name
        / "validation_table.txt"
    )

    table_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not table_path.exists()

    freq_str = f"{fs} Hz"
    model_str = args.model

    acc_avg = float(np.mean([r["accuracy"] for r in fold_results]))
    f1_avg = float(np.mean([r["f1"] for r in fold_results]))
    prec_avg = float(np.mean([r["precision"] for r in fold_results]))
    spec_avg = float(np.mean([r["specificity"] for r in fold_results]))
    mcc_avg = float(np.mean([r["mcc"] for r in fold_results]))
    time_avg = float(np.mean([r["time_min"] for r in fold_results]))

    with open(table_path, "a") as f:
        if write_header:
            f.write("VALIDATION PERFORMANCE ACROSS SAMPLING FREQUENCIES (5-FOLD)\n")
            f.write("=" * 140 + "\n\n")
            f.write(
                f"{'Model':<10}"
                f"{'Freq':<10}"
                f"{'Fold':<10}"
                f"{'Acc':<10}"
                f"{'F1':<10}"
                f"{'Prec':<10}"
                f"{'Spec':<10}"
                f"{'MCC':<10}"
                f"{'Time':<10}\n"
            )
            f.write("-" * 90 + "\n")

        mid_row = (len(fold_results) // 2) + 1

        for i, r in enumerate(fold_results, 1):
            model_cell = model_str if i == mid_row else ""
            freq_cell = freq_str if i == mid_row else ""

            f.write(
                f"{model_cell:<10}"
                f"{freq_cell:<10}"
                f"{('Fold ' + str(i)):<10}"
                f"{r['accuracy']:<10.4f}"
                f"{r['f1']:<10.4f}"
                f"{r['precision']:<10.4f}"
                f"{r['specificity']:<10.4f}"
                f"{r['mcc']:<10.4f}"
                f"{r['time_min']:<10.2f}\n"
            )

        f.write(
            f"{'':<10}{'':<10}{'Avg':<10}"
            f"{acc_avg:<10.4f}"
            f"{f1_avg:<10.4f}"
            f"{prec_avg:<10.4f}"
            f"{spec_avg:<10.4f}"
            f"{mcc_avg:<10.4f}"
            f"{time_avg:<10.2f}\n"
        )

        f.write("-" * 90 + "\n")

    print(f"\nValidation table updated: {table_path}")

    # ================= FINAL TEST =================

    test_dir = data_dir / "test"

    if not test_dir.exists():
        print("\nTest folder not found — skipping test evaluation.")
        return

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (Ensemble of all folds' best epochs)")
    print("=" * 70)

    test_data = torch.load(
        test_dir / "test.pt",
        map_location="cpu",
    )

    test_ds = ECGDataset.__new__(ECGDataset)
    test_ds.X = test_data["X"]
    test_ds.y = test_data["y"]

    print_dataset_stats("Test unbalanced", test_ds)

    test_loader = make_loader(
        test_ds,
        batch_size=args.batch_size,
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
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )

    in_ch = test_ds.X[0].shape[0]
    num_classes = int(torch.max(test_ds.y).item() + 1)

    models = []

    for fold in range(1, args.kfolds + 1):
        m = build_model(
            args.model,
            in_ch,
            num_classes,
        ).to(device)

        best_model_path = (
            Path("checkpoints")
            / dataset_name
            / f"{fs}hz"
            / args.model
            / f"fold_{fold}"
            / "best.pt"
        )

        if not best_model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {best_model_path}")

        m.load_state_dict(torch.load(best_model_path, map_location=device))
        m.to(device)
        models.append(m)

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
        / args.model
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

    with open(test_table_path, "a") as f:
        if write_header:
            f.write("FINAL TEST PERFORMANCE (Ensemble of folds)\n")
            f.write("=" * 130 + "\n\n")
            f.write(
                f"{'Model':<10}"
                f"{'Freq':<10}"
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
            f.write("-" * 130 + "\n")

        f.write(
            f"{model_str:<10}"
            f"{freq_str:<10}"
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
            f"{'Bal':<10}"
            f"{acc_b:<10.4f}"
            f"{f1_b:<10.4f}"
            f"{precision_b:<10.4f}"
            f"{specificity_b:<10.4f}"
            f"{mcc_b:<10.4f}"
            f"{'':<10}"
            f"{'':<10}"
            f"{elapsed_b:<10.2f}"
            f"{thpt_b:<10.2f}\n"
        )

        f.write("-" * 130 + "\n")

    print(f"\nMaster test table updated: {test_table_path}")

    # ================= SAVE FINAL TEST RESULTS =================

    results_path = results_dir / "test_results.txt"

    tn_u, fp_u = cm_u[0]
    fn_u, tp_u = cm_u[1]

    tn_b, fp_b = cm_b[0]
    fn_b, tp_b = cm_b[1]

    with open(results_path, "w") as f:
        f.write("FINAL TEST RESULTS (Ensemble of all folds' best.pt)\n")
        f.write("=" * 60 + "\n\n")

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
        f.write(f"Training time (total) : {training_total_min:.2f} minutes\n")
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
        f.write(f"MCC                 : {mcc_b:.4f}\n\n")

        f.write("PERFORMANCE\n")
        f.write(f"Inference time        : {elapsed_b:.2f} seconds\n")
        f.write(f"Throughput            : {thpt_b:.2f} samples/sec\n\n\n")

        f.write("RUN INFORMATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset              : {dataset_name}\n")
        f.write(f"Sampling rate        : {fs} Hz\n")
        f.write(f"Model                : {args.model}\n")
        f.write(f"Folds ensembled      : {args.kfolds}\n")
        f.write(f"Device               : {device}\n")
        f.write(f"Training balance     : {args.train_balance}\n\n")

        f.write("DATASET SIZES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test samples (unbalanced) : {len(test_ds)}\n")
        f.write(f"Test samples (balanced)   : {len(test_balanced_ds)}\n\n")

        f.write("TRAINING SETTINGS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Epochs (max)         : {args.epochs}\n")
        f.write(f"Batch size           : {args.batch_size}\n")
        f.write(f"Learning rate        : {args.lr}\n")
        f.write(f"K-Folds              : {args.kfolds}\n")
        f.write(f"Early stop patience  : {args.early_stopping_patience}\n")

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
    print("F1        Accuracy   Recall(Sens)  Specificity  Precision MCC")
    print("-" * 70)
    print(
        f"{f1_b:<9.4f}"
        f"{acc_b:<11.4f}"
        f"{recall_b:<15.4f}"
        f"{specificity_b:<13.4f}"
        f"{precision_b:<10.4f}"
        f"{mcc_b:<.4f}"
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


if __name__ == "__main__":
    main()
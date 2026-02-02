import argparse
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




# ================= REPRODUCIBILITY =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= FIXED SETTINGS =================
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
KFOLDS = 5
EARLY_STOPPING_PATIENCE = 10
# ================================================


# ---------- Dataset statistics ----------
def print_dataset_stats(name, dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    total = len(labels)
    afib = sum(1 for y in labels if y == 1)
    normal = total - afib

    print(
        f"{name:<12}: total={total} | "
        f"Normal={normal} ({100*normal/total:.2f}%) | "
        f"AFIB={afib} ({100*afib/total:.2f}%)"
    )


# ---------- Metrics ----------
def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)


# ---------- Evaluation (VAL / TEST) ----------
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = total = correct = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()

            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(dim=1)

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


# ---------- Ensemble Evaluation (TEST) ----------
def evaluate_ensemble(models, loader, device):
    for m in models:
        m.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = total = correct = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()

            # average logits across folds
            logits_sum = None
            for m in models:
                logits = m(x)
                logits_sum = logits if logits_sum is None else (logits_sum + logits)
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


def make_balanced_subset_binary(dataset, seed=42):
    """
    Downsample the majority class to match the minority class (binary labels 0/1).
    Deterministic using `seed`.
    """
    rng = np.random.RandomState(seed)

    labels = np.array([dataset[i][1] for i in range(len(dataset))], dtype=int)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        # cannot balance if only one class exists
        return Subset(dataset, list(range(len(dataset))))

    n = min(len(idx0), len(idx1))

    if len(idx0) > len(idx1):
        idx0 = rng.choice(idx0, size=n, replace=False)
        keep = np.concatenate([idx0, idx1])
    else:
        idx1 = rng.choice(idx1, size=n, replace=False)
        keep = np.concatenate([idx0, idx1])

    rng.shuffle(keep)
    return Subset(dataset, keep.tolist())


# ---------- Train ONE fold ----------
def train_one_fold(model, optimizer, train_loader, val_loader, device, out_dir):
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_f1 = -1
    best_acc = None
    best_cm = None
    bad_epochs = 0

    best_epoch = None
    best_val_loss = None

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_start = time.time()

    print()  # space before epochs

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ---- Training ----
        model.train()
        train_loss = total = 0

        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            total += y.size(0)

        train_loss /= total

        # ---- Validation ----
        acc, f1, cm, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"ACC {acc*100:.2f}% | "
            f"F1 {f1:.4f} | "
            f"Time {epoch_time:.2f}s"
        )

        torch.save(model.state_dict(), out_dir / "last.pt")

        # ---- Early stopping ----
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_cm = cm
            best_epoch = epoch
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break

    fold_time = time.time() - fold_start

    tn, fp = best_cm[0]
    fn, tp = best_cm[1]

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    print("\n" + "=" * 70)
    print(f"Fold Results – Training time: {fold_time/60:.2f} minutes")
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

    tn, fp = best_cm[0]
    fn, tp = best_cm[1]

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5 + 1e-12
    mcc = mcc_num / mcc_den


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
        f.write(f"Specificity        : {specificity:.4f}\n")
        f.write(f"Precision          : {precision:.4f}\n")
        f.write(f"MCC                : {mcc:.4f}\n\n")

        f.write("RUNTIME\n")
        f.write(f"Fold training time (sec): {fold_time:.2f}\n")
        f.write(f"Fold training time (min): {fold_time/60:.2f}\n\n")

        f.write("RUN INFO\n")
        f.write("-" * 60 + "\n")
        f.write("Random seed        : 42\n")


    return {
        "fold": out_dir.name,                 # e.g., fold_1
        "best_epoch": best_epoch,
        "val_loss": float(best_val_loss),

        "accuracy": float(best_acc),
        "f1": float(best_f1),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "mcc": float(mcc),

        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),

        "time_min": float(fold_time / 60.0),
    }



# ============================ MAIN ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
    "--model",
    choices=["cnn1d", "cnn_lstm"],
    default="cnn1d"
)

    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print("=" * 70)

    best_fold = None
    best_f1_overall = -1

    training_start = time.time()
    fold_results = []


    # ---------- K-FOLD LOOP ----------
    for fold in range(1, KFOLDS + 1):
        print(f"\n=== Fold {fold}/{KFOLDS} ===")

        train_ds, val_ds, num_classes = load_kfold(data_dir, fs, fold, KFOLDS)

        print_dataset_stats("Train", train_ds)
        print_dataset_stats("Validation", val_ds)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, BATCH_SIZE)

        in_ch = train_ds[0][0].shape[0]

        if args.model == "cnn1d":
            model = CNN1D(in_ch, num_classes)
        
        elif args.model == "cnn_lstm":
            model = CNN_LSTM_ECG(in_channels=in_ch, num_classes=num_classes)
        
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        model = model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        metrics = train_one_fold(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            Path("checkpoints") / dataset_name / f"{fs}hz" / args.model / f"fold_{fold}",
        )

        fold_results.append(metrics)
        


        if metrics["f1"] > best_f1_overall:
            best_f1_overall = metrics["f1"]
            best_fold = fold


    training_total_time = time.time() - training_start
    training_total_min = training_total_time / 60
    print(f"Total training time : {training_total_min:.2f} minutes")

    


    table_path = (
        Path("checkpoints")
        / dataset_name
        / "validation_table.txt"  # one master table per dataset
    )

    write_header = not table_path.exists()

    # Frequency string exactly like screenshot: "500 Hz"
    freq_str = f"{fs} Hz"
    model_str = args.model  # e.g., "cnn1d", "cnn_lstm"

    # averages
    acc_avg  = float(np.mean([r["accuracy"] for r in fold_results]))
    f1_avg   = float(np.mean([r["f1"] for r in fold_results]))
    prec_avg = float(np.mean([r["precision"] for r in fold_results]))
    spec_avg = float(np.mean([r["specificity"] for r in fold_results]))
    mcc_avg  = float(np.mean([r["mcc"] for r in fold_results]))
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


        # ---- Write folds (Model/Frequency appear only once like screenshot) ----
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


        # ---- Avg row (Frequency blank like screenshot; Avg bolding is for LaTeX, here plain) ----
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

    test_data = torch.load(test_dir / "test.pt", map_location="cpu")
    test_ds = ECGDataset.__new__(ECGDataset)
    test_ds.X = test_data["X"]
    test_ds.y = test_data["y"]

    # Unbalanced (as-is)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    # Balanced (downsample majority)
    test_balanced_ds = make_balanced_subset_binary(test_ds, seed=42)
    test_balanced_loader = DataLoader(test_balanced_ds, BATCH_SIZE)

    # Build ensemble: one model per fold (load best.pt from each fold)
    # NOTE: uses same architecture + in_ch/num_classes inferred from test tensors
    in_ch = test_ds.X[0].shape[0]
    # num_classes from labels (binary assumed, but derived safely)
    num_classes = int(torch.max(test_ds.y).item() + 1)

    models = []
    for fold in range(1, KFOLDS + 1):
        if args.model == "cnn1d":
            m = CNN1D(in_ch, num_classes)
       
        elif args.model == "cnn_lstm":
            m = CNN_LSTM_ECG(in_channels=in_ch, num_classes=num_classes)
       

        else:
            raise ValueError(f"Unknown model type: {args.model}")

        m = m.to(device)


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

    # ---- Unbalanced test eval ----
    start = time.time()
    acc_u, f1_u, cm_u, _ = evaluate_ensemble(models, test_loader, device)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_u = time.time() - start

    tn_u, fp_u = cm_u[0]
    fn_u, tp_u = cm_u[1]
    recall_u = tp_u / (tp_u + fn_u + 1e-12)
    specificity_u = tn_u / (tn_u + fp_u + 1e-12)
    precision_u = tp_u / (tp_u + fp_u + 1e-12)
    mcc_num_u = (tp_u * tn_u) - (fp_u * fn_u)
    mcc_den_u = ((tp_u+fp_u)*(tp_u+fn_u)*(tn_u+fp_u)*(tn_u+fn_u)) ** 0.5 + 1e-12
    mcc_u = mcc_num_u / mcc_den_u


    # ---- Balanced test eval ----
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    acc_b, f1_b, cm_b, _ = evaluate_ensemble(models, test_balanced_loader, device)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_b = time.time() - start

    tn_b, fp_b = cm_b[0]
    fn_b, tp_b = cm_b[1]
    recall_b = tp_b / (tp_b + fn_b + 1e-12)
    specificity_b = tn_b / (tn_b + fp_b + 1e-12)
    precision_b = tp_b / (tp_b + fp_b + 1e-12)
    mcc_num_b = (tp_b * tn_b) - (fp_b * fn_b)
    mcc_den_b = ((tp_b+fp_b)*(tp_b+fn_b)*(tn_b+fp_b)*(tn_b+fn_b)) ** 0.5 + 1e-12
    mcc_b = mcc_num_b / mcc_den_b


    # ================= SAVE/APPEND MASTER TEST TABLE =================
    test_table_path = (
        Path("checkpoints")
        / dataset_name
        / "test_table.txt"   # one master test table per dataset
    )

    write_header = not test_table_path.exists()

    freq_str = f"{fs} Hz"
    model_str = args.model

    # throughput (samples/sec)
    thpt_u = len(test_ds) / (elapsed_u + 1e-12)
    thpt_b = len(test_balanced_ds) / (elapsed_b + 1e-12)

    with open(test_table_path, "a") as f:
        if write_header:
            f.write("FINAL TEST PERFORMANCE (Ensemble of folds)\n")
            f.write("=" * 110 + "\n\n")
            f.write(
                f"{'Model':<10}"
                f"{'Freq':<10}"
                f"{'Test':<10}"
                f"{'Acc':<10}"
                f"{'F1':<10}"
                f"{'Prec':<10}"
                f"{'Spec':<10}"
                f"{'MCC':<10}"
                f"{'Time(s)':<10}"
                f"{'Thpt':<10}\n"
            )
            f.write("-" * 110 + "\n")

        # Unbalanced row (print Model/Freq here)
        f.write(
            f"{model_str:<10}"
            f"{freq_str:<10}"
            f"{'Unbal':<10}"
            f"{acc_u:<10.4f}"
            f"{f1_u:<10.4f}"
            f"{precision_u:<10.4f}"
            f"{specificity_u:<10.4f}"
            f"{mcc_u:<10.4f}"
            f"{elapsed_u:<10.2f}"
            f"{thpt_u:<10.2f}\n"
        )

        # Balanced row (blank Model/Freq like your validation table)
        f.write(
            f"{'':<10}"
            f"{'':<10}"
            f"{'Bal':<10}"
            f"{acc_b:<10.4f}"
            f"{f1_b:<10.4f}"
            f"{precision_b:<10.4f}"
            f"{specificity_b:<10.4f}"
            f"{mcc_b:<10.4f}"
            f"{elapsed_b:<10.2f}"
            f"{thpt_b:<10.2f}\n"
        )

        f.write("-" * 110 + "\n")

    print(f"\nMaster test table updated: {test_table_path}")




    # ================= SAVE FINAL TEST RESULTS =================
    results_dir = (
        Path("checkpoints")
        / dataset_name
        / f"{fs}hz"
        / args.model
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "test_results.txt"

    with open(results_path, "w") as f:
        f.write("FINAL TEST RESULTS (Ensemble of all folds' best.pt)\n")
        f.write("=" * 60 + "\n\n")

        # ---------- UNBALANCED ----------
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
        f.write(f"MCC                : {mcc_u:.4f}\n")


        f.write("PERFORMANCE\n")
        f.write(f"Training time (total) : {training_total_min:.2f} minutes\n")
        f.write(f"Inference time      : {elapsed_u:.2f} seconds\n")
        f.write(f"Throughput          : {len(test_ds)/elapsed_u:.2f} samples/sec\n\n\n")

        # ---------- BALANCED ----------
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
        f.write(f"MCC                : {mcc_b:.4f}\n")


        f.write("PERFORMANCE\n")
        f.write(f"Inference time      : {elapsed_b:.2f} seconds\n")
        f.write(f"Throughput          : {len(test_balanced_ds)/elapsed_b:.2f} samples/sec\n\n\n")

        # ---------- RUN INFO ----------
        f.write("RUN INFORMATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset            : {dataset_name}\n")
        f.write(f"Sampling rate      : {fs} Hz\n")
        f.write(f"Model              : {args.model}\n")
        f.write(f"Folds ensembled    : {KFOLDS}\n")
        f.write(f"Device             : {device}\n\n")

        f.write("DATASET SIZES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test samples (unbalanced) : {len(test_ds)}\n")
        f.write(f"Test samples (balanced)   : {len(test_balanced_ds)}\n\n")

        f.write("TRAINING SETTINGS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Epochs (max)       : {EPOCHS}\n")
        f.write(f"Batch size         : {BATCH_SIZE}\n")
        f.write(f"Learning rate      : {LEARNING_RATE}\n")
        f.write(f"K-Folds            : {KFOLDS}\n")
        f.write(f"Early stop patience: {EARLY_STOPPING_PATIENCE}\n")

    # ---------- Save confusion matrices as CSV ----------
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

    # ---------- Print to console ----------
    print("\nUNBALANCED TEST")
    print("Confusion Matrix (Test)")
    print(f"[[{tn_u:4d} {fp_u:3d}]")
    print(f" [{fn_u:4d} {tp_u:3d}]]\n")
    print("F1        Accuracy   Recall(Sens)  Specificity  Precision MCC")
    print("-" * 70)
    print(
        f"{f1_u:<9.4f}"
        f"{acc_u:<11.4f}"
        f"{recall_u:<15.4f}"
        f"{specificity_u:<13.4f}"
        f"{precision_u:<11.4f}"
        f"{mcc_u:<.4f}"
    )
    print(f"\nInference time : {elapsed_u:.2f} seconds")
    print(f"Throughput     : {len(test_ds)/elapsed_u:.2f} samples/sec")
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
        f"{precision_b:<11.4f}"
        f"{mcc_b:<.4f}"
    )
    print(f"\nInference time : {elapsed_b:.2f} seconds")
    print(f"Throughput     : {len(test_balanced_ds)/elapsed_b:.2f} samples/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()

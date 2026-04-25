"""
main.py - ECG PTB-XL AFIB Detection Pipeline

Interactive runner for:

1. Preparing PTB-XL AFIB vs NORMAL data
2. Creating patient-safe K-fold data
3. Creating optional hold-out test data
4. Training CNN1D or CNN-LSTM
5. Evaluating with:
   - balanced training
   - natural/unbalanced validation
   - natural/unbalanced hold-out test
   - secondary balanced test

Recommended scientific setup:
- Data preparation: --balance_mode train
- Training: --train_balance downsample
- Validation: natural/unbalanced
- Test: natural/unbalanced + balanced secondary test
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule


# ================= Paths =================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

console = Console()


# ================= Rich UI Helper =================

def section(title, style="bold cyan"):
    console.print(
        Panel(
            Text(title, justify="center", style=style),
            border_style="cyan",
            padding=(0, 2),
        )
    )


# ================= Logger =================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("ECG-MAIN")


# ================= Input Helpers =================

def ask_choice(question, choices):
    console.print(f"\n[bold]{question}[/bold]")

    for i, c in enumerate(choices, 1):
        console.print(f"  [cyan]{i})[/cyan] {c}")

    while True:
        ans = input("\nSelect option number: ").strip()

        try:
            ans = int(ans)

            if 1 <= ans <= len(choices):
                return ans

        except ValueError:
            pass

        console.print("[red]Invalid input. Please enter a valid number.[/red]")


def ask_yes_no(question, default=None):
    if default is True:
        suffix = "(Y/n)"
    elif default is False:
        suffix = "(y/N)"
    else:
        suffix = "(y/n)"

    while True:
        ans = input(f"\n{question} {suffix}: ").strip().lower()

        if not ans and default is not None:
            return default

        if ans in ("y", "yes"):
            return True

        if ans in ("n", "no"):
            return False

        console.print("[red]Invalid input. Please enter 'y' or 'n'.[/red]")


def ask_text(question, default=None):
    if default is not None:
        ans = input(f"\n{question} (default: {default}): ").strip()
        return ans if ans else default

    ans = input(f"\n{question}: ").strip()
    return ans


def ask_int(question, default):
    while True:
        ans = input(f"\n{question} (default: {default}): ").strip()

        if not ans:
            return int(default)

        try:
            return int(ans)

        except ValueError:
            console.print("[red]Invalid input. Please enter an integer.[/red]")


def ask_float(question, default):
    while True:
        ans = input(f"\n{question} (default: {default}): ").strip()

        if not ans:
            return float(default)

        try:
            return float(ans)

        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


# ================= Sampling Rate Selection =================

def display_sampling_rates():
    console.print("\n" + "=" * 50)
    console.print("[bold]AVAILABLE SAMPLING RATES[/bold]")
    console.print("=" * 50)
    console.print("  [cyan]1)[/cyan] 62 Hz")
    console.print("  [cyan]2)[/cyan] 100 Hz")
    console.print("  [cyan]3)[/cyan] 250 Hz")
    console.print("  [cyan]4)[/cyan] 500 Hz")
    console.print("=" * 50)


def select_sampling_rates():
    available_rates = {
        1: 62,
        2: 100,
        3: 250,
        4: 500,
    }

    selected = []

    console.print("\nSelect one or more sampling rates.")
    console.print("Example: [bold]1 2 4[/bold] selects 62 Hz, 100 Hz, and 500 Hz.")
    console.print("Press Enter without input to use all rates.")

    display_sampling_rates()

    ans = input("\nEnter sampling rate numbers: ").strip()

    if not ans:
        return [62, 100, 250, 500]

    try:
        numbers = [int(x) for x in ans.split()]

        for num in numbers:
            if num in available_rates:
                rate = available_rates[num]

                if rate not in selected:
                    selected.append(rate)

    except ValueError:
        console.print("[yellow]Invalid input. Using default: 62, 100, 250, 500[/yellow]")
        return [62, 100, 250, 500]

    if not selected:
        console.print("[yellow]No valid rates selected. Using default: 62, 100, 250, 500[/yellow]")
        return [62, 100, 250, 500]

    return sorted(selected)


# ================= Hardware Warning =================

def display_hardware_warning(selected_rates, model_type):
    console.print("\n" + "=" * 70)
    console.print("[bold red]HARDWARE REQUIREMENT WARNING[/bold red]")
    console.print("=" * 70)

    console.print("\nTraining deep ECG models can be computationally intensive.")
    console.print("\nRecommended:")
    console.print("  • CUDA-enabled GPU")
    console.print("  • Batch size 8 for CNN1D")
    console.print("  • Batch size 4 or 8 for CNN-LSTM")
    console.print("  • Run 500 Hz separately on limited laptops")

    if 500 in selected_rates:
        console.print(
            "\n[yellow]500 Hz selected: this has the highest RAM and compute cost.[/yellow]"
        )

    if model_type == "cnn_lstm":
        console.print(
            "\n[yellow]CNN-LSTM selected: consider smaller batch size, e.g. 4.[/yellow]"
        )

    console.print("=" * 70)

    return ask_yes_no("Continue with training?", default=True)


# ================= Validation Helpers =================

def validate_ptbxl_path(raw_data_path):
    raw_data_path = Path(raw_data_path)

    if not raw_data_path.exists():
        console.print(f"[red]Directory does not exist:[/red] {raw_data_path}")
        return False

    db_path = raw_data_path / "ptbxl_database.csv"

    if not db_path.exists():
        console.print(f"[red]ptbxl_database.csv not found in:[/red] {raw_data_path}")
        return False

    console.print(f"[green]Found PTB-XL database:[/green] {db_path}")
    return True


def validate_prepared_rate(prepared_root, rate):
    prepared_root = Path(prepared_root)
    rate_path = prepared_root / f"{rate}hz"

    if not rate_path.exists():
        return False

    fold_files = [
        rate_path / f"fold_{i}.pt"
        for i in range(1, 6)
    ]

    old_data_file = rate_path / "data.pt"

    return all(p.exists() for p in fold_files) or old_data_file.exists()


# ================= Subprocess Runners =================

def run_data_preparation_for_rate(
    dataset_path,
    dataset_name,
    out_root,
    rate,
    folds,
    test_ratio,
    balance_mode,
    flatline_seconds,
):
    """
    Run data preparation for one sampling rate at a time.

    This is safer for laptops, especially for 500 Hz.
    """

    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]STARTING DATA PREPARATION: {rate} Hz[/bold cyan]")
    console.print("=" * 70)

    script_path = PROJECT_ROOT / "src" / "ecg_preprocessing" / "ecg_data_prepare.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_path", str(dataset_path),
        "--name", str(dataset_name),
        "--fs", str(rate),
        "--out_root", str(out_root),
        "--folds", str(folds),
        "--test_ratio", str(test_ratio),
        "--balance_mode", str(balance_mode),
        "--flatline_seconds", str(flatline_seconds),
    ]

    console.print("\n[bold]Command:[/bold]")
    console.print(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            console.print(f"\n[green]Data preparation completed for {rate} Hz[/green]")
            return True

        console.print(f"\n[red]Data preparation failed for {rate} Hz[/red]")
        return False

    except Exception as e:
        console.print(f"\n[red]Error running data preparation:[/red] {e}")
        return False


def run_training(
    prepared_root,
    sampling_rate,
    model_type,
    train_balance,
    batch_size,
    epochs,
    lr,
    kfolds,
    device,
):
    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]STARTING TRAINING: {sampling_rate} Hz | {model_type}[/bold cyan]")
    console.print("=" * 70)

    script_path = PROJECT_ROOT / "src" / "train.py"
    data_path = Path(prepared_root) / f"{sampling_rate}hz"

    cmd = [
        sys.executable,
        str(script_path),
        "--data_path", str(data_path),
        "--model", model_type,
        "--train_balance", train_balance,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--kfolds", str(kfolds),
        "--device", device,
    ]

    console.print("\n[bold]Command:[/bold]")
    console.print(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            console.print(f"\n[green]Training completed for {sampling_rate} Hz[/green]")
            return True

        console.print(f"\n[red]Training failed for {sampling_rate} Hz[/red]")
        return False

    except Exception as e:
        console.print(f"\n[red]Error running training:[/red] {e}")
        return False


# ================= Main =================

def main():
    console.print(
        Panel(
            Align.center(
                "[bold green]ECG PTB-XL AFIB DETECTION PIPELINE[/bold green]\n"
                "Data Preparation | K-Fold Training | Hold-Out Test Evaluation"
            ),
            border_style="green",
        )
    )

    # ================= Step 1 =================
    section("STEP 1: DATA SOURCE")

    data_source_choice = ask_choice(
        "Choose data source",
        [
            "Use existing prepared data",
            "Prepare new data from raw PTB-XL",
        ],
    )

    dataset_name = ask_text(
        "Dataset output name",
        default="ptbl-xl",
    )

    selected_rates = select_sampling_rates()

    console.print(f"\n[green]Selected sampling rates:[/green] {selected_rates}")

    # ================= Step 2 =================
    section("STEP 2: DATA PREPARATION SETTINGS")

    if data_source_choice == 1:
        prepared_root = ask_text(
            "Path to prepared dataset root",
            default=str(PROJECT_ROOT / "prepared_data" / dataset_name),
        )

        prepared_root = Path(prepared_root)

        if not prepared_root.exists():
            console.print(f"[red]Prepared data root not found:[/red] {prepared_root}")
            return

        missing_rates = [
            rate for rate in selected_rates
            if not validate_prepared_rate(prepared_root, rate)
        ]

        if missing_rates:
            console.print(
                f"[yellow]Missing or incomplete prepared data for rates:[/yellow] {missing_rates}"
            )
            console.print("[yellow]Prepare these rates first.[/yellow]")
            return

        console.print(f"[green]Using prepared data:[/green] {prepared_root}")

    else:
        raw_data_path = ask_text(
            "Path to raw PTB-XL dataset",
            default=str(PROJECT_ROOT / "data"),
        )

        raw_data_path = Path(raw_data_path)

        if not validate_ptbxl_path(raw_data_path):
            return

        out_root = ask_text(
            "Output root directory",
            default=str(PROJECT_ROOT / "prepared_data"),
        )

        out_root = Path(out_root)

        folds = ask_int(
            "Number of K-folds",
            default=5,
        )

        test_ratio = ask_float(
            "Hold-out test ratio",
            default=0.2,
        )

        balance_choice = ask_choice(
            "Choose preprocessing balance mode",
            [
                "train  - recommended: save natural folds; balance training later",
                "none   - no balancing",
                "fold   - balance each fold, validation also balanced",
                "global - globally downsample classes before saving",
            ],
        )

        balance_modes = {
            1: "train",
            2: "none",
            3: "fold",
            4: "global",
        }

        balance_mode = balance_modes[balance_choice]

        flatline_seconds = ask_float(
            "Flatline duration threshold in seconds",
            default=3.0,
        )

        console.print("\n[bold]Preparation summary[/bold]")
        console.print(f"  Raw data path     : {raw_data_path}")
        console.print(f"  Output root       : {out_root}")
        console.print(f"  Dataset name      : {dataset_name}")
        console.print(f"  Sampling rates    : {selected_rates}")
        console.print(f"  Folds             : {folds}")
        console.print(f"  Test ratio        : {test_ratio}")
        console.print(f"  Balance mode      : {balance_mode}")
        console.print(f"  Flatline seconds  : {flatline_seconds}")

        proceed = ask_yes_no("Start data preparation?", default=True)

        if not proceed:
            console.print("[yellow]Data preparation cancelled.[/yellow]")
            return

        prepared_root = out_root / dataset_name

        prepared_ok_rates = []

        for rate in selected_rates:
            ok = run_data_preparation_for_rate(
                dataset_path=raw_data_path,
                dataset_name=dataset_name,
                out_root=out_root,
                rate=rate,
                folds=folds,
                test_ratio=test_ratio,
                balance_mode=balance_mode,
                flatline_seconds=flatline_seconds,
            )

            if ok:
                prepared_ok_rates.append(rate)

        if not prepared_ok_rates:
            console.print("[red]No sampling rates were prepared successfully.[/red]")
            return

        selected_rates = prepared_ok_rates

        console.print(f"\n[green]Prepared rates:[/green] {selected_rates}")
        console.print(f"[green]Prepared data root:[/green] {prepared_root}")

    # ================= Step 3 =================
    section("STEP 3: MODEL SELECTION")

    model_choice = ask_choice(
        "Choose model",
        [
            "cnn1d    - 1D convolutional neural network",
            "cnn_lstm - CNN feature extraction + LSTM temporal modeling",
        ],
    )

    model_type = "cnn1d" if model_choice == 1 else "cnn_lstm"

    console.print(f"[green]Selected model:[/green] {model_type}")

    # ================= Step 4 =================
    section("STEP 4: TRAINING SETTINGS")

    train_balance_choice = ask_choice(
        "Choose runtime training balance",
        [
            "downsample - recommended with preprocessing --balance_mode train",
            "none       - train on natural/unbalanced folds",
        ],
    )

    train_balance = "downsample" if train_balance_choice == 1 else "none"

    default_batch = 4 if model_type == "cnn_lstm" else 8

    batch_size = ask_int(
        "Batch size",
        default=default_batch,
    )

    epochs = ask_int(
        "Maximum epochs",
        default=50,
    )

    lr = ask_float(
        "Learning rate",
        default=1e-3,
    )

    kfolds = ask_int(
        "Number of K-folds used in prepared data",
        default=5,
    )

    device_choice = ask_choice(
        "Choose device",
        [
            "auto - use CUDA if available",
            "cuda - force CUDA",
            "cpu  - force CPU",
        ],
    )

    device_map = {
        1: "auto",
        2: "cuda",
        3: "cpu",
    }

    device = device_map[device_choice]

    console.print("\n[bold]Training summary[/bold]")
    console.print(f"  Prepared root     : {prepared_root}")
    console.print(f"  Sampling rates    : {selected_rates}")
    console.print(f"  Model             : {model_type}")
    console.print(f"  Train balance     : {train_balance}")
    console.print(f"  Batch size        : {batch_size}")
    console.print(f"  Epochs            : {epochs}")
    console.print(f"  Learning rate     : {lr}")
    console.print(f"  K-folds           : {kfolds}")
    console.print(f"  Device            : {device}")

    if not display_hardware_warning(selected_rates, model_type):
        console.print("[yellow]Training cancelled.[/yellow]")
        return

    # ================= Step 5 =================
    section("STEP 5: TRAINING", style="bold green")

    training_results = []

    for rate in selected_rates:
        rate_path = Path(prepared_root) / f"{rate}hz"

        if not rate_path.exists():
            console.print(f"[yellow]No data found for {rate} Hz — skipping.[/yellow]")
            continue

        console.print(
            Rule(f"[bold cyan]TRAINING {rate} Hz | {model_type}")
        )

        ok = run_training(
            prepared_root=prepared_root,
            sampling_rate=rate,
            model_type=model_type,
            train_balance=train_balance,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            kfolds=kfolds,
            device=device,
        )

        if ok:
            training_results.append(rate)

            checkpoint_dir = (
                PROJECT_ROOT
                / "checkpoints"
                / dataset_name
                / f"{rate}hz"
                / model_type
            )

            if checkpoint_dir.exists():
                console.print(f"[green]Checkpoint directory:[/green] {checkpoint_dir}")

                fold_dirs = sorted(checkpoint_dir.glob("fold_*"))

                console.print(f"[green]Completed fold directories:[/green] {len(fold_dirs)}")

                for fold_dir in fold_dirs:
                    best_pt = fold_dir / "best.pt"

                    if best_pt.exists():
                        size_mb = best_pt.stat().st_size / (1024 * 1024)

                        console.print(
                            f"  • {fold_dir.name}/best.pt "
                            f"[dim]({size_mb:.2f} MB)[/dim]"
                        )

    # ================= Summary =================
    section("PIPELINE COMPLETE", style="bold green")

    if training_results:
        console.print(f"[green]Successfully trained rates:[/green] {training_results}")
        console.print(f"[green]Checkpoints root:[/green] {PROJECT_ROOT / 'checkpoints'}")
    else:
        console.print("[yellow]No training completed.[/yellow]")

    console.print(
        Align.center(
            "[bold cyan]ECG PTB-XL AFIB Detection Pipeline finished.[/bold cyan]"
        )
    )


if __name__ == "__main__":
    main()
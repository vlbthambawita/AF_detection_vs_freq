"""
main.py - ECG AFIB Detection Pipeline

Interactive runner for:
1. Preparing ECG AFIB vs NORMAL data
2. Creating either k-fold data or manual train/val/test data
3. Training CNN1D or CNN-LSTM with split_mode support
4. Running final test evaluation from the updated train.py

This version is aligned with train.py that supports:
- --split_mode auto
- --split_mode kfold
- --split_mode manual

Path design:
- The runner does not rely on a fixed dataset folder name such as ptbl-xl.
- The user provides or creates the prepared dataset root.
- Each rate folder is resolved dynamically as <prepared_root>/<rate>hz.
- The checkpoint dataset name is inferred by train.py from the parent folder of data_path.
"""

from __future__ import annotations

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.table import Table


SRC_DIR = Path(__file__).resolve().parent

if SRC_DIR.name == "src":
    PROJECT_ROOT = SRC_DIR.parent
else:
    PROJECT_ROOT = SRC_DIR

console = Console()


def section(title: str, style: str = "bold cyan") -> None:
    console.print(
        Panel(
            Text(title, justify="center", style=style),
            border_style="cyan",
            padding=(0, 2),
        )
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("ECG-MAIN")


def ask_choice(question: str, choices: list[str]) -> int:
    console.print(f"\n[bold]{question}[/bold]")

    for i, c in enumerate(choices, 1):
        console.print(f"  [cyan]{i})[/cyan] {c}")

    while True:
        ans = input("\nSelect option number: ").strip()

        try:
            ans_int = int(ans)
            if 1 <= ans_int <= len(choices):
                return ans_int
        except ValueError:
            pass

        console.print("[red]Invalid input. Please enter a valid number.[/red]")


def ask_yes_no(question: str, default: bool | None = None) -> bool:
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


def ask_text(question: str, default: str | None = None) -> str:
    if default is not None:
        ans = input(f"\n{question} (default: {default}): ").strip()
        return ans if ans else default

    return input(f"\n{question}: ").strip()


def ask_int(question: str, default: int) -> int:
    while True:
        ans = input(f"\n{question} (default: {default}): ").strip()

        if not ans:
            return int(default)

        try:
            return int(ans)
        except ValueError:
            console.print("[red]Invalid input. Please enter an integer.[/red]")


def ask_float(question: str, default: float) -> float:
    while True:
        ans = input(f"\n{question} (default: {default}): ").strip()

        if not ans:
            return float(default)

        try:
            return float(ans)
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


def norm_path(value: str | Path) -> Path:
    return Path(str(value).strip().strip('"')).expanduser()


def find_existing_file(candidates: Iterable[Path]) -> Path | None:
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p.resolve()
        except OSError:
            continue

    return None


def find_train_script() -> Path:
    script = find_existing_file(
        [
            PROJECT_ROOT / "src" / "train.py",
            PROJECT_ROOT / "train.py",
            PROJECT_ROOT / "old_AF" / "train.py",
            SRC_DIR / "train.py",
        ]
    )

    if script is None:
        raise FileNotFoundError(
            "Could not find train.py. Expected one of:\n"
            f"  {PROJECT_ROOT / 'src' / 'train.py'}\n"
            f"  {PROJECT_ROOT / 'train.py'}\n"
            f"  {PROJECT_ROOT / 'old_AF' / 'train.py'}\n"
            f"  {SRC_DIR / 'train.py'}"
        )

    return script


def find_prepare_script() -> Path:
    script = find_existing_file(
        [
            PROJECT_ROOT / "src" / "ecg_preprocessing" / "ecg_data_prepare.py",
            PROJECT_ROOT / "ecg_preprocessing" / "ecg_data_prepare.py",
            PROJECT_ROOT / "ecg_data_prepare.py",
            PROJECT_ROOT / "old_AF" / "ecg_data_prepare.py",
            SRC_DIR / "ecg_preprocessing" / "ecg_data_prepare.py",
            SRC_DIR / "ecg_data_prepare.py",
        ]
    )

    if script is None:
        raise FileNotFoundError(
            "Could not find ecg_data_prepare.py. Expected one of:\n"
            f"  {PROJECT_ROOT / 'src' / 'ecg_preprocessing' / 'ecg_data_prepare.py'}\n"
            f"  {PROJECT_ROOT / 'ecg_preprocessing' / 'ecg_data_prepare.py'}\n"
            f"  {PROJECT_ROOT / 'ecg_data_prepare.py'}\n"
            f"  {PROJECT_ROOT / 'old_AF' / 'ecg_data_prepare.py'}\n"
            f"  {SRC_DIR / 'ecg_preprocessing' / 'ecg_data_prepare.py'}\n"
            f"  {SRC_DIR / 'ecg_data_prepare.py'}"
        )

    return script


DEFAULT_RATES = [62, 100, 250, 500]

SUGGESTED_RATES = [
    50,
    62,
    75,
    100,
    125,
    150,
    200,
    250,
    300,
    400,
    500,
]


def display_sampling_rates(rates: list[int]) -> None:
    console.print("\n" + "=" * 58)
    console.print("[bold]AVAILABLE SAMPLING RATES[/bold]")
    console.print("=" * 58)

    for i, rate in enumerate(rates, 1):
        console.print(f"  [cyan]{i})[/cyan] {rate} Hz")

    console.print("=" * 58)


def parse_custom_frequencies(text: str) -> list[int]:
    """
    Parse custom frequency values written by the user.
    Only integer values are accepted because ecg_data_prepare.py expects --fs as int.
    Values above 500 Hz are rejected because the PTB-XL high-resolution source is 500 Hz.
    """

    cleaned = text.replace(",", " ").replace(";", " ")
    parts = [p.strip() for p in cleaned.split() if p.strip()]

    if not parts:
        return DEFAULT_RATES.copy()

    selected: list[int] = []

    for part in parts:
        if "." in part:
            raise ValueError(
                f"Float frequency '{part}' is not supported. "
                "Use an integer value such as 62 instead of 62.5."
            )

        try:
            fs = int(part)
        except ValueError:
            raise ValueError(
                f"Invalid frequency '{part}'. Frequencies must be integers."
            )

        if fs <= 0:
            raise ValueError(
                f"Invalid frequency '{fs}'. Frequency must be greater than 0 Hz."
            )

        if fs > 500:
            raise ValueError(
                f"Invalid frequency '{fs}'. Values above 500 Hz are not allowed "
                "because they would upsample instead of downsample."
            )

        if fs not in selected:
            selected.append(fs)

    return sorted(selected)


def select_from_numbered_list(rates: list[int]) -> list[int]:
    """
    Select sampling frequencies from a numbered list.
    """

    console.print("\nSelect one or more sampling rates from the list.")
    console.print("Example: [bold]1 3 4[/bold]")
    console.print("Press Enter without input to use all listed rates.")

    display_sampling_rates(rates)

    while True:
        ans = input("\nEnter option numbers: ").strip()

        if not ans:
            return rates.copy()

        try:
            numbers = [int(x) for x in ans.replace(",", " ").split()]
        except ValueError:
            console.print("[red]Invalid input. Enter numbers only, for example: 1 3 4[/red]")
            continue

        selected: list[int] = []

        for num in numbers:
            if 1 <= num <= len(rates):
                rate = rates[num - 1]
                if rate not in selected:
                    selected.append(rate)
            else:
                console.print(f"[yellow]Ignoring invalid option number: {num}[/yellow]")

        if selected:
            return sorted(selected)

        console.print("[red]No valid sampling rates selected.[/red]")


def write_custom_sampling_rates() -> list[int]:
    """
    Ask the user to write custom integer sampling frequencies.
    """

    console.print("\nWrite custom integer sampling frequencies.")
    console.print("Examples:")
    console.print("  [cyan]500 250 125 100 62[/cyan]")
    console.print("  [cyan]50, 75, 100, 150, 250, 500[/cyan]")
    console.print("\n[yellow]Float values such as 62.5 are not supported here.[/yellow]")
    console.print("[yellow]Use 62 instead of 62.5 unless the preprocessing script is changed to support floats.[/yellow]")

    while True:
        ans = input("\nEnter custom frequencies: ").strip()

        try:
            selected = parse_custom_frequencies(ans)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        if selected:
            return selected

        console.print("[red]No valid frequencies entered.[/red]")


def select_sampling_rates() -> list[int]:
    """
    Select sampling frequencies either from default values, a suggested list, or custom input.
    """

    console.print("\n[bold]Sampling frequency selection[/bold]")
    console.print("The data will be prepared as separate folders such as 62hz, 100hz, 125hz, and 250hz.")

    choice = ask_choice(
        "Choose sampling-rate input method",
        [
            "Use default thesis rates: 62, 100, 250, 500 Hz",
            "Select from suggested list",
            "Write custom integer frequencies",
        ],
    )

    if choice == 1:
        return DEFAULT_RATES.copy()

    if choice == 2:
        return select_from_numbered_list(SUGGESTED_RATES)

    return write_custom_sampling_rates()


def rate_dir_from_prepared_root(prepared_root: Path, rate: int) -> Path:
    """
    Resolve a rate directory without assuming a fixed dataset name.

    Supported inputs:
    - prepared_root = .../<dataset>             -> returns .../<dataset>/<rate>hz
    - prepared_root = .../<dataset>/<rate>hz    -> returns prepared_root directly
    """

    prepared_root = prepared_root.expanduser().resolve()
    rate_name = f"{rate}hz"

    if prepared_root.name.lower() == rate_name.lower():
        return prepared_root

    return prepared_root / rate_name


def detect_prepared_split_mode(rate_path: Path, requested_mode: str, kfolds: int) -> str | None:
    """
    Detect prepared data type for one frequency folder.

    Returns:
    - kfold if fold_1.pt ... fold_k.pt exist
    - manual if train.pt and val.pt exist
    - None if not valid for the requested mode
    """

    fold_files = [rate_path / f"fold_{i}.pt" for i in range(1, kfolds + 1)]
    has_all_folds = all(p.exists() for p in fold_files)
    has_manual = (rate_path / "train.pt").exists() and (rate_path / "val.pt").exists()

    if requested_mode == "kfold":
        return "kfold" if has_all_folds else None

    if requested_mode == "manual":
        return "manual" if has_manual else None

    if has_all_folds:
        return "kfold"

    if has_manual:
        return "manual"

    return None


def test_file_status(rate_path: Path) -> str:
    """
    Check whether a test file exists inside the prepared frequency folder.
    """

    if (rate_path / "test" / "test.pt").exists():
        return "test/test.pt"

    if (rate_path / "test.pt").exists():
        return "test.pt"

    return "not found"


def validate_prepared_rate(prepared_root: Path, rate: int, split_mode: str, kfolds: int) -> tuple[bool, str]:
    """
    Validate that one selected frequency has a usable prepared data structure.
    """

    rate_path = rate_dir_from_prepared_root(prepared_root, rate)

    if not rate_path.exists():
        return False, f"missing rate folder: {rate_path}"

    detected = detect_prepared_split_mode(rate_path, split_mode, kfolds)

    if detected is None:
        expected_folds = ", ".join(f"fold_{i}.pt" for i in range(1, kfolds + 1))
        return (
            False,
            "split structure not valid. Expected "
            f"{expected_folds} for kfold or train.pt + val.pt for manual."
        )

    test_status = test_file_status(rate_path)

    return True, f"{detected} split detected | test: {test_status} | {rate_path}"


def print_prepared_summary(prepared_root: Path, rates: list[int], split_mode: str, kfolds: int) -> None:
    """
    Print a table showing whether the selected prepared data folders are valid.
    """

    table = Table(title="Prepared data validation")
    table.add_column("Rate", style="cyan")
    table.add_column("Status")
    table.add_column("Detected")

    for rate in rates:
        ok, msg = validate_prepared_rate(prepared_root, rate, split_mode, kfolds)
        table.add_row(f"{rate} Hz", "OK" if ok else "MISSING", msg)

    console.print(table)


def display_hardware_warning(selected_rates: list[int], model_type: str, split_mode: str) -> bool:
    """
    Display hardware guidance before training.
    """

    console.print("\n" + "=" * 70)
    console.print("[bold red]HARDWARE REQUIREMENT WARNING[/bold red]")
    console.print("=" * 70)

    console.print("\nTraining deep ECG models can be computationally intensive.")
    console.print("\nRecommended:")
    console.print("  • CUDA-enabled GPU")
    console.print("  • Batch size 8 for CNN1D")
    console.print("  • Batch size 4 or 8 for CNN-LSTM")
    console.print("  • Run 500 Hz separately on limited laptops")

    if split_mode == "kfold":
        console.print("  • K-fold mode trains one model per fold")
    elif split_mode == "manual":
        console.print("  • Manual mode trains one model from train.pt and val.pt")
    else:
        console.print("  • Auto mode detects k-fold or manual split per frequency")

    if 500 in selected_rates:
        console.print("\n[yellow]500 Hz selected: this has the highest RAM and compute cost.[/yellow]")

    if model_type == "cnn_lstm":
        console.print("\n[yellow]CNN-LSTM selected: consider smaller batch size, e.g. 4.[/yellow]")

    console.print("=" * 70)

    return ask_yes_no("Continue with training?", default=True)


def validate_ptbxl_path(raw_data_path: Path) -> bool:
    """
    Validate that the given directory looks like a PTB-XL dataset folder.
    """

    raw_data_path = raw_data_path.expanduser().resolve()

    if not raw_data_path.exists():
        console.print(f"[red]Directory does not exist:[/red] {raw_data_path}")
        return False

    db_path = raw_data_path / "ptbxl_database.csv"

    if not db_path.exists():
        console.print(f"[red]ptbxl_database.csv not found in:[/red] {raw_data_path}")
        return False

    console.print(f"[green]Found PTB-XL database:[/green] {db_path}")
    return True


def run_command(cmd: list[str], title: str) -> bool:
    """
    Run a subprocess command and stream its output directly to the terminal.
    """

    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("=" * 70)

    console.print("\n[bold]Command:[/bold]")
    console.print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    console.print(f"[bold]Working directory:[/bold] {PROJECT_ROOT}")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
        )

        if result.returncode == 0:
            console.print(f"\n[green]{title} completed[/green]")
            return True

        console.print(f"\n[red]{title} failed with exit code {result.returncode}[/red]")
        return False

    except Exception as exc:
        console.print(f"\n[red]Error while running command:[/red] {exc}")
        return False


def run_data_preparation_for_rate(
    dataset_path: Path,
    dataset_name: str,
    out_root: Path,
    rate: int,
    preparation_split_mode: str,
    folds: int | None,
    split_ratio: tuple[float, float, float],
    test_ratio: float | None,
    balance_mode: str,
    flatline_seconds: float,
) -> bool:
    """
    Run data preparation for one sampling rate at a time.

    preparation_split_mode:
    - kfold: passes --folds and optionally --test_ratio
    - manual: passes --split_ratio and does not pass --folds
    """

    script_path = find_prepare_script()

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_path", str(dataset_path),
        "--name", str(dataset_name),
        "--fs", str(rate),
        "--out_root", str(out_root),
        "--balance_mode", str(balance_mode),
        "--flatline_seconds", str(flatline_seconds),
    ]

    if preparation_split_mode == "kfold":
        if folds is None:
            raise ValueError("folds must be provided for kfold preparation.")

        cmd += ["--folds", str(folds)]

        if test_ratio is not None:
            cmd += ["--test_ratio", str(test_ratio)]

    elif preparation_split_mode == "manual":
        cmd += [
            "--split_ratio",
            str(split_ratio[0]),
            str(split_ratio[1]),
            str(split_ratio[2]),
        ]

        if test_ratio is not None:
            cmd += ["--test_ratio", str(test_ratio)]

    else:
        raise ValueError(f"Unknown preparation_split_mode: {preparation_split_mode}")

    return run_command(cmd, f"DATA PREPARATION: {rate} Hz | {preparation_split_mode}")


def run_training(
    rate_path: Path,
    model_type: str,
    split_mode: str,
    train_balance: str,
    batch_size: int,
    epochs: int,
    lr: float,
    kfolds: int,
    early_stopping_patience: int,
    device: str,
    test_only: bool,
) -> bool:
    """
    Run train.py for one prepared frequency folder.
    """

    script_path = find_train_script()

    cmd = [
        sys.executable,
        str(script_path),
        "--data_path", str(rate_path),
        "--model", model_type,
        "--split_mode", split_mode,
        "--train_balance", train_balance,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--kfolds", str(kfolds),
        "--early_stopping_patience", str(early_stopping_patience),
        "--device", device,
    ]

    if test_only:
        cmd.append("--test_only")

    rate_name = rate_path.name.replace("hz", " Hz")
    mode_label = "TEST ONLY" if test_only else "TRAINING"

    return run_command(cmd, f"{mode_label}: {rate_name} | {model_type} | split_mode={split_mode}")


def print_checkpoint_summary(rate_path: Path, model_type: str, split_mode: str, kfolds: int) -> None:
    """
    Print checkpoint files generated by train.py.

    train.py saves checkpoints under:
    checkpoints/<dataset_name>/<rate>hz/<model>/...
    where dataset_name = rate_path.parent.name.
    """

    dataset_name = rate_path.parent.name
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / dataset_name / rate_path.name / model_type

    if not checkpoint_dir.exists():
        console.print(f"[yellow]Checkpoint directory not found:[/yellow] {checkpoint_dir}")
        return

    console.print(f"[green]Checkpoint directory:[/green] {checkpoint_dir}")

    detected = detect_prepared_split_mode(rate_path, split_mode, kfolds)

    if detected == "manual":
        best_pt = checkpoint_dir / "manual_split" / "best.pt"
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            console.print(f"  • manual_split/best.pt [dim]({size_mb:.2f} MB)[/dim]")
        else:
            console.print("  • [yellow]manual_split/best.pt not found[/yellow]")
        return

    fold_dirs = sorted(checkpoint_dir.glob("fold_*"))
    console.print(f"[green]Completed fold directories:[/green] {len(fold_dirs)}")

    for fold_dir in fold_dirs:
        best_pt = fold_dir / "best.pt"
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            console.print(f"  • {fold_dir.name}/best.pt [dim]({size_mb:.2f} MB)[/dim]")


def main() -> None:
    """
    Run the interactive ECG AFIB detection pipeline.
    """

    console.print(
        Panel(
            Align.center(
                "[bold green]ECG AFIB DETECTION PIPELINE[/bold green]\n"
                "Data Preparation | K-Fold or Manual Training | Final Test Evaluation"
            ),
            border_style="green",
        )
    )

    console.print(f"[dim]Project root detected as:[/dim] {PROJECT_ROOT}")

    section("STEP 1: DATA SOURCE")

    data_source_choice = ask_choice(
        "Choose data source",
        [
            "Use existing prepared data",
            "Prepare new data from raw PTB-XL",
        ],
    )

    selected_rates = select_sampling_rates()
    console.print(f"\n[green]Selected sampling rates:[/green] {selected_rates}")

    section("STEP 2: SPLIT MODE")

    split_choice = ask_choice(
        "Choose training split mode",
        [
            "auto   - detect k-fold or manual split from files",
            "kfold  - require fold_1.pt ... fold_k.pt",
            "manual - require train.pt and val.pt",
        ],
    )

    split_mode_map = {
        1: "auto",
        2: "kfold",
        3: "manual",
    }

    train_split_mode = split_mode_map[split_choice]

    if train_split_mode in ("auto", "kfold"):
        kfolds = ask_int(
            "Number of K-folds if k-fold data is used",
            default=5,
        )
    else:
        kfolds = 5
        console.print(
            "[dim]Manual split selected: K-fold count is not requested because "
            "training will use train.pt and val.pt.[/dim]"
        )

    section("STEP 3: DATA PREPARATION SETTINGS")

    if data_source_choice == 1:
        prepared_root = norm_path(
            ask_text(
                "Path to prepared dataset root. This should contain folders for the selected rates, e.g. 62hz/100hz/125hz/250hz",
                default=str(PROJECT_ROOT / "prepared_data"),
            )
        )

        if prepared_root.exists() and not any((prepared_root / f"{r}hz").exists() for r in selected_rates):
            candidates = sorted(
                p for p in prepared_root.iterdir()
                if p.is_dir() and any((p / f"{r}hz").exists() for r in selected_rates)
            )

            if len(candidates) == 1:
                prepared_root = candidates[0]
                console.print(
                    "\n[yellow]The path looks like a parent prepared_data folder.[/yellow]"
                )
                console.print(
                    f"[green]Auto-selected dataset folder:[/green] {prepared_root}"
                )

            elif len(candidates) > 1:
                console.print("\n[yellow]The path looks like a parent prepared_data folder.[/yellow]")
                console.print("Select the dataset folder to use:")

                for i, c in enumerate(candidates, 1):
                    console.print(f"  [cyan]{i})[/cyan] {c}")

                while True:
                    ans = input(
                        "\nSelect dataset folder number "
                        "or press Enter to use option 1: "
                    ).strip()

                    if not ans:
                        prepared_root = candidates[0]
                        console.print(
                            f"[green]Auto-selected dataset folder:[/green] {prepared_root}"
                        )
                        break

                    try:
                        idx = int(ans)
                        if 1 <= idx <= len(candidates):
                            prepared_root = candidates[idx - 1]
                            break
                    except ValueError:
                        pass

                    console.print("[red]Invalid input.[/red]")

        if not prepared_root.exists():
            console.print(f"[red]Prepared data root not found:[/red] {prepared_root}")
            return

        print_prepared_summary(prepared_root, selected_rates, train_split_mode, kfolds)

        missing_rates = []

        for rate in selected_rates:
            ok, _ = validate_prepared_rate(prepared_root, rate, train_split_mode, kfolds)
            if not ok:
                missing_rates.append(rate)

        if missing_rates:
            console.print(f"[red]Missing or incomplete prepared data for rates:[/red] {missing_rates}")
            console.print("[yellow]Fix the prepared data or choose another split mode.[/yellow]")
            return

        console.print(f"[green]Using prepared data root:[/green] {prepared_root}")

    else:
        dataset_name = ask_text(
            "Dataset output folder name. This can be any experiment name",
            default="ptbl-xl",
        )

        raw_data_path = norm_path(
            ask_text(
                "Path to raw PTB-XL dataset",
                default=str(PROJECT_ROOT / "data"),
            )
        )

        if not validate_ptbxl_path(raw_data_path):
            return

        out_root = norm_path(
            ask_text(
                "Output root directory",
                default=str(PROJECT_ROOT / "prepared_data"),
            )
        )

        prep_choice = ask_choice(
            "Choose prepared split structure to create",
            [
                "kfold  - save fold_1.pt ... fold_k.pt, optional hold-out test/test.pt",
                "manual - save train.pt, val.pt, test.pt using split_ratio",
            ],
        )

        preparation_split_mode = "kfold" if prep_choice == 1 else "manual"

        if train_split_mode == "auto":
            effective_train_split_mode = "auto"
        else:
            effective_train_split_mode = train_split_mode

        if preparation_split_mode == "kfold" and effective_train_split_mode == "manual":
            console.print("[red]You selected manual training but k-fold preparation.[/red]")
            return

        if preparation_split_mode == "manual" and effective_train_split_mode == "kfold":
            console.print("[red]You selected k-fold training but manual preparation.[/red]")
            return

        folds: int | None = None
        split_ratio = (0.7, 0.2, 0.1)
        test_ratio: float | None = None

        if preparation_split_mode == "kfold":
            folds = kfolds
            test_ratio = ask_float(
                "Hold-out test ratio for k-fold mode",
                default=0.2,
            )
        else:
            train_ratio = ask_float("Manual split train ratio", default=0.7)
            val_ratio = ask_float("Manual split validation ratio", default=0.2)
            test_ratio_part = ask_float("Manual split test ratio", default=0.1)
            split_ratio = (train_ratio, val_ratio, test_ratio_part)

            total_ratio = sum(split_ratio)
            if abs(total_ratio - 1.0) > 1e-6:
                console.print(
                    f"[yellow]Split ratios sum to {total_ratio:.4f}. They will be passed as entered; "
                    "recommended sum is 1.0.[/yellow]"
                )

            extra_holdout = ask_yes_no(
                "Create an additional patient-safe hold-out test/test.pt besides manual test.pt?",
                default=False,
            )

            if extra_holdout:
                test_ratio = ask_float("Additional hold-out test ratio", default=0.2)

        balance_choice = ask_choice(
            "Choose preprocessing balance mode",
            [
                "train  - recommended: keep validation/test natural; balance training later or train split only",
                "none   - no balancing",
                "fold   - balance each fold/split",
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
        console.print(f"  Raw data path       : {raw_data_path}")
        console.print(f"  Output root         : {out_root}")
        console.print(f"  Dataset folder name : {dataset_name}")
        console.print(f"  Prepared root       : {out_root / dataset_name}")
        console.print(f"  Sampling rates      : {selected_rates}")
        console.print(f"  Prepared split      : {preparation_split_mode}")
        console.print(f"  Training split mode : {train_split_mode}")

        if preparation_split_mode == "kfold":
            console.print(f"  Folds               : {folds}")
            console.print(f"  Hold-out test ratio : {test_ratio}")
        else:
            console.print(f"  Split ratio         : {split_ratio}")
            console.print(f"  Extra hold-out test : {test_ratio}")

        console.print(f"  Balance mode        : {balance_mode}")
        console.print(f"  Flatline seconds    : {flatline_seconds}")

        if not ask_yes_no("Start data preparation?", default=True):
            console.print("[yellow]Data preparation cancelled.[/yellow]")
            return

        prepared_root = out_root / dataset_name
        prepared_ok_rates: list[int] = []

        for rate in selected_rates:
            ok = run_data_preparation_for_rate(
                dataset_path=raw_data_path,
                dataset_name=dataset_name,
                out_root=out_root,
                rate=rate,
                preparation_split_mode=preparation_split_mode,
                folds=folds,
                split_ratio=split_ratio,
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

        print_prepared_summary(prepared_root, selected_rates, train_split_mode, kfolds)

    section("STEP 4: MODEL SELECTION")

    model_choice = ask_choice(
        "Choose model",
        [
            "cnn1d    - 1D convolutional neural network",
            "cnn_lstm - CNN feature extraction + LSTM temporal modeling",
        ],
    )

    model_type = "cnn1d" if model_choice == 1 else "cnn_lstm"
    console.print(f"[green]Selected model:[/green] {model_type}")

    section("STEP 5: TRAINING SETTINGS")

    action_choice = ask_choice(
        "Choose action",
        [
            "Train and run final test evaluation",
            "Test only using existing checkpoints",
        ],
    )

    test_only = action_choice == 2

    if test_only:
        train_balance = "downsample"
        console.print(
            "[dim]Test-only selected: training balance, epochs, learning rate, "
            "and early stopping are not requested.[/dim]"
        )
    else:
        train_balance_choice = ask_choice(
            "Choose runtime training balance",
            [
                "downsample - recommended with preprocessing --balance_mode train",
                "none       - train on natural/unbalanced training data",
            ],
        )

        train_balance = "downsample" if train_balance_choice == 1 else "none"

    default_batch = 4 if model_type == "cnn_lstm" else 8

    batch_size = ask_int(
        "Batch size",
        default=default_batch,
    )

    if test_only:
        epochs = 1
        lr = 1e-3
        early_stopping_patience = 10
    else:
        epochs = ask_int(
            "Maximum epochs",
            default=50,
        )

        lr = ask_float(
            "Learning rate",
            default=1e-3,
        )

        early_stopping_patience = ask_int(
            "Early stopping patience",
            default=10,
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
    console.print(f"  Prepared root        : {prepared_root}")
    console.print(f"  Sampling rates       : {selected_rates}")
    console.print(f"  Model                : {model_type}")
    console.print(f"  Action               : {'test_only' if test_only else 'train + final test'}")
    console.print(f"  Train split mode     : {train_split_mode}")
    console.print(f"  Train balance        : {train_balance}")
    console.print(f"  Batch size           : {batch_size}")
    console.print(f"  Epochs               : {epochs}")
    console.print(f"  Learning rate        : {lr}")
    console.print(f"  K-folds if needed    : {kfolds}")
    console.print(f"  Early stop patience  : {early_stopping_patience}")
    console.print(f"  Device               : {device}")

    if not test_only:
        if not display_hardware_warning(selected_rates, model_type, train_split_mode):
            console.print("[yellow]Training cancelled.[/yellow]")
            return

    section("STEP 6: TRAINING / TESTING", style="bold green")

    successful_rates: list[int] = []

    for rate in selected_rates:
        rate_path = rate_dir_from_prepared_root(prepared_root, rate)

        if not rate_path.exists():
            console.print(f"[yellow]No data found for {rate} Hz — skipping:[/yellow] {rate_path}")
            continue

        ok, msg = validate_prepared_rate(prepared_root, rate, train_split_mode, kfolds)

        if not ok:
            console.print(f"[yellow]Invalid data for {rate} Hz — skipping:[/yellow] {msg}")
            continue

        console.print(Rule(f"[bold cyan]{'TEST ONLY' if test_only else 'TRAINING'} {rate} Hz | {model_type}"))
        console.print(f"[dim]Detected structure:[/dim] {msg}")

        ok = run_training(
            rate_path=rate_path,
            model_type=model_type,
            split_mode=train_split_mode,
            train_balance=train_balance,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            kfolds=kfolds,
            early_stopping_patience=early_stopping_patience,
            device=device,
            test_only=test_only,
        )

        if ok:
            successful_rates.append(rate)
            print_checkpoint_summary(
                rate_path=rate_path,
                model_type=model_type,
                split_mode=train_split_mode,
                kfolds=kfolds,
            )

    section("PIPELINE COMPLETE", style="bold green")

    if successful_rates:
        console.print(f"[green]Successfully completed rates:[/green] {successful_rates}")
        console.print(f"[green]Checkpoints root:[/green] {PROJECT_ROOT / 'checkpoints'}")
    else:
        console.print("[yellow]No training/testing completed.[/yellow]")

    console.print(
        Align.center(
            "[bold cyan]ECG AFIB Detection Pipeline finished.[/bold cyan]"
        )
    )


if __name__ == "__main__":
    main()

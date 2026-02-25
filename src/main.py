"""
main.py - ECG Arrhythmia Detection Pipeline

This script provides an interactive interface for:
1. Loading PTB-XL dataset
2. Preprocessing ECG data
3. Training models at different sampling rates
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
from pathlib import Path
import os

console = Console()
#=================== Rich UI Helper =================
def section(title, style="bold cyan"):
    """Reusable section header."""
    console.print(
        Panel(
            Text(title, justify="center", style=style),
            border_style="cyan",
            padding=(0, 2),
        )
    )

# ================= Logger file=================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger("ECG-MAIN")


# ================= Helper Functions =================

def ask_choice(question, choices):
    """
    Ask user a numbered question.
    """
    print(question)
    for i, c in enumerate(choices, 1):
        print(f"  {i}) {c}")

    while True:
        try:
            ans = int(input("\nSelect option number: "))
            if 1 <= ans <= len(choices):
                return ans
        except ValueError:
            pass
        print("[bold red]Invalid input. Please enter a valid number.[/bold red]\n")


def ask_yes_no(question):
    """
    Ask user a yes/no question.
    """
    while True:
        ans = input(f"\n{question} (y/n): ").strip().lower()
        if ans in ['y', 'yes']:
            return True
        elif ans in ['n', 'no']:
            return False
        print("[bold red]Invalid input. Please enter 'y' or 'n'.[bold red]\n")


def display_hardware_warning():
    """
    Display hardware requirements warning before training.
    """
    print("\n" + "=" * 60)
    print("[red]⚠️  HARDWARE REQUIREMENT WARNING  ⚠️[red]")
    print("[yellow]=" * 60)
    print("\nThis training process is computationally intensive.")
    print("\nFor reasonable performance, you need ONE of the following:")
    print("  • GPU (CUDA-enabled) - RECOMMENDED")
    print("  • At least 12 CPU cores with multithreading enabled")
    print("\nWithout adequate hardware, training will be VERY SLOW.")
    print("Each fold may take significant time on CPU-only systems.")
    print("=" * 60 + "\n[/yellow]")

    # Ask user to confirm
    while True:
        ans = input("Do you want to continue? (y/n): ").strip().lower()
        if ans in ['y', 'yes']:
            return True
        elif ans in ['n', 'no']:
            return False
        print("Invalid input. Please enter 'y' or 'n'.\n")


def display_sampling_rates():
    """
    Display available sampling rates for selection.
    """
    print("\n" + "=" * 50)
    print(" AVAILABLE SAMPLING RATES")
    print("=" * 50)
    print("  1) 62 Hz")
    print("  2) 100 Hz")
    print("  3) 250 Hz")
    print("  4) 500 Hz")
    print("=" * 50)


def select_sampling_rates():
    """
    Allow user to select one or more sampling rates.
    Returns a list of selected sampling rates.
    """
    available_rates = [
        (1, 62),
        (2, 100),
        (3, 250),
        (4, 500)
    ]
    
    selected = []
    
    print("\nSelect sampling rates one by one (enter numbers, separated by space)")
    print("Example: 1 2 4 (this will select 62Hz, 100Hz, and 500Hz)")
    print("Press Enter without numbers to finish selection.")
    
    display_sampling_rates()
    
    while True:
        try:
            ans = input("\nEnter sampling rate numbers (or press Enter to finish): ").strip()
            if not ans:
                break
            
            numbers = [int(x) for x in ans.split()]
            for num in numbers:
                if 1 <= num <= 4:
                    rate = dict(available_rates)[num]
                    if rate not in selected:
                        selected.append(rate)
            
            if selected:
                print(f"\nCurrently selected: {selected}")
            else:
                print("\nNo valid sampling rates selected yet.")
                
        except ValueError:
            print("Invalid input. Please enter numbers separated by space.")
    
    if not selected:
        print("\nNo sampling rates selected. Using default: [62, 100, 250, 500]")
        return [62, 100, 250, 500]
    
    return sorted(selected)


def run_training(data_path, sampling_rate, model_type):
    """
    Run the training script for a specific sampling rate.
    """
    print("\n" + "=" * 60)
    print(f" STARTING TRAINING FOR {sampling_rate} Hz")
    print("=" * 60)
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        "train.py",
        "--data_path", str(data_path / f"{sampling_rate}hz"),
        "--model", model_type
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"\n✓ Training completed for {sampling_rate} Hz")
            print(f"✓ Checkpoints saved to: checkpoints/ptb-xl/{sampling_rate}hz/{model_type}/")
        else:
            print(f"\n✗ Training failed for {sampling_rate} Hz")
            
    except Exception as e:
        print(f"\n✗ Error running training: {e}")


def run_data_preparation(dataset_path, dataset_name, out_root):
    """
    Run the ECG data preparation script.
    """
    print("\n" + "=" * 60)
    print(" STARTING DATA PREPARATION")
    print("=" * 60)
    
    # Import the necessary functions from ecg_preprocessing
    sys.path.insert(0, str(Path(__file__).parent / "ecg_preprocessing"))
    from ecg_data_prepare import main as prepare_main
    
    # Set up arguments for data preparation
    sys.argv = [
        "ecg_data_prepare.py",
        "--dataset_path", dataset_path,
        "--name", dataset_name,
        "--fs", "62", "100", "250", "500",
        "--out_root", out_root,
        "--folds", "10"  # Use K-fold mode
    ]
    
    try:
        prepare_main()
        print("\n✓ Data preparation completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Data preparation failed: {e}")
        return False


# ================= Main =================

def main():

    # ================= HEADER =================
    console.print(
        Panel(
            Align.center(
                "[bold green]ECG PTB-XL DETECTION PIPELINE[/bold green]\n"
                "PTB-XL Dataset Processing & Training"
            ),
            border_style="green",
        )
    )

    # ================= Step 1 =================
    section("STEP 1: PTB-XL DATA LOCATION")

    is_ptbxl = ask_yes_no("Is the data PTB-XL dataset?")
    if not is_ptbxl:
        console.print(
            "[yellow]⚠ This pipeline is currently configured for PTB-XL only.[/yellow]"
        )
        console.print("[red]Exiting...[/red]")
        return

    default_ro_path = "/ro/data"
    data_ready = ask_yes_no(
        f"Is the PTB-XL data ready at '{default_ro_path}'?"
    )

    if data_ready:
        raw_data_path = default_ro_path
        console.print(f"[green]✓ Using PTB-XL data from:[/green] {raw_data_path}")
    else:
        raw_data_path = input(
            "\nEnter PTB-XL data directory path: "
        ).strip()
        if not raw_data_path:
            console.print("[red]No path provided. Exiting...[/red]")
            return

    if not os.path.exists(raw_data_path):
        console.print(
            f"[red]✗ Directory does not exist:[/red] {raw_data_path}"
        )
        return

    db_path = os.path.join(raw_data_path, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        console.print(
            f"[red]✗ ptbxl_database.csv not found in {raw_data_path}[/red]"
        )
        console.print(
            "[yellow]Please ensure the PTB-XL dataset is properly downloaded.[/yellow]"
        )
        return

    console.print(f"[green]✓ Found PTB-XL database:[/green] {db_path}")

    # ================= Step 2 =================
    section("STEP 2: OUTPUT DATA LOCATION")

    console.print("\n[bold]Options:[/bold]")
    console.print("  [cyan]1)[/cyan] Use existing prepared data")
    console.print("  [cyan]2)[/cyan] Prepare new data from raw PTB-XL")

    data_source_choice = ask_choice(
        "\nWhere should the data be loaded from?",
        [
            "Use existing prepared data",
            "Prepare new data from raw PTB-XL",
        ],
    )

    if data_source_choice == 1:
        default_data_path = "prepared_data/ptb-xl"
        data_path = input(
            f"\nEnter path to prepared data (default: {default_data_path}): "
        ).strip()

        if not data_path:
            data_path = default_data_path

        if not os.path.exists(data_path):
            console.print(
                f"[red]✗ Prepared data not found:[/red] {data_path}"
            )
            console.print(
                "[yellow]Prepare the data first (option 2).[/yellow]"
            )
            return

        console.print(f"[green]✓ Using prepared data:[/green] {data_path}")

    else:
        default_out_root = "prepared_data"
        out_root = input(
            f"\nEnter output directory (default: {default_out_root}): "
        ).strip()

        if not out_root:
            out_root = default_out_root

        console.print(f"[green]✓ Data will be saved to:[/green] {out_root}")

        success = run_data_preparation(raw_data_path, "ptb-xl", out_root)
        if not success:
            console.print("[red]✗ Data preparation failed.[/red]")
            return

        data_path = os.path.join(out_root, "ptb-xl")
        console.print(f"[green]✓ Prepared data available:[/green] {data_path}")

    # ================= Step 3 =================
    section("STEP 3: SAMPLING RATE SELECTION")

    selected_rates = select_sampling_rates()
    console.print(
        f"[green]✓ Selected sampling rates:[/green] {selected_rates}"
    )

    # ================= Step 4 =================
    section("STEP 4: MODEL SELECTION")

    model_choice = ask_choice(
        "\nWhich model would you like to train?",
        [
            "CNN1D (1D Convolutional Neural Network)",
            "CNN-LSTM (CNN with LSTM layers)",
        ],
    )

    model_type = "cnn1d" if model_choice == 1 else "cnn_lstm"
    console.print(f"[green]✓ Selected model:[/green] {model_type}")

    # ================= Step 5 =================
    section("STEP 5: TRAINING")

    if not display_hardware_warning():
        console.print("[yellow]⚠ Training cancelled by user.[/yellow]")
        return

    # ================= Step 6 =================
    section("STARTING TRAINING PROCESS", style="bold green")

    data_path_obj = Path(data_path)
    training_results = []

    for rate in selected_rates:
        rate_path = data_path_obj / f"{rate}hz"

        if not rate_path.exists():
            console.print(
                f"[yellow]⚠ No data found for {rate} Hz — skipping.[/yellow]"
            )
            continue

        console.print(
            Rule(f"[bold cyan]TRAINING: {rate} Hz with {model_type}")
        )

        run_training(data_path_obj, rate, model_type)

        checkpoint_dir = (
            Path("checkpoints") / "ptb-xl" / f"{rate}hz" / model_type
        )

        if checkpoint_dir.exists():
            console.print(
                f"[green]✓ Checkpoint saved:[/green] {checkpoint_dir}"
            )

            fold_dirs = list(checkpoint_dir.glob("fold_*"))
            console.print(
                f"[green]✓ Folds completed:[/green] {len(fold_dirs)}"
            )

            for fold_dir in sorted(fold_dirs):
                best_pt = fold_dir / "best.pt"
                if best_pt.exists():
                    size_mb = best_pt.stat().st_size / (1024 * 1024)
                    console.print(
                        f"   • {fold_dir.name}/best.pt "
                        f"[dim]({size_mb:.2f} MB)[/dim]"
                    )

        training_results.append(rate)

    # ================= SUMMARY =================
    section("TRAINING COMPLETE", style="bold green")

    if training_results:
        console.print(
            f"[green]✓ Successfully trained:[/green] {training_results}"
        )
        console.print(
            "[green]✓ Checkpoints saved in:[/green] checkpoints/ptb-xl/"
        )
    else:
        console.print("[yellow]⚠ No training completed.[/yellow]")

    console.print(
        Align.center(
            "[bold cyan]Thank you for using the ECG PTB-XL Detection Pipeline![/bold cyan]"
        )
    )
if __name__ == "__main__":
    main()


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
        print("Invalid input. Please enter a valid number.\n")


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
        print("Invalid input. Please enter 'y' or 'n'.\n")


def display_hardware_warning():
    """
    Display hardware requirements warning before training.
    """
    print("\n" + "=" * 60)
    print("⚠️  HARDWARE REQUIREMENT WARNING  ⚠️")
    print("=" * 60)
    print("\nThis training process is computationally intensive.")
    print("\nFor reasonable performance, you need ONE of the following:")
    print("  • GPU (CUDA-enabled) - RECOMMENDED")
    print("  • At least 12 CPU cores with multithreading enabled")
    print("\nWithout adequate hardware, training will be VERY SLOW.")
    print("Each fold may take significant time on CPU-only systems.")
    print("=" * 60 + "\n")

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

    print("\n" + "=" * 70)
    print(" ECG ARRHYTHMIA DETECTION PIPELINE")
    print(" PTB-XL Dataset Processing & Training")
    print("=" * 70 + "\n")

    # ================= Step 1: PTB-XL Data Location =================
    
    print("=" * 70)
    print(" STEP 1: PTB-XL DATA LOCATION")
    print("=" * 70)
    
    # Ask if data is PTBXL
    is_ptbxl = ask_yes_no("Is the data PTB-XL dataset?")
    
    if not is_ptbxl:
        print("\n⚠️  This pipeline is currently configured for PTB-XL only.")
        print("Exiting...")
        return
    
    # Ask if data is ready on the "ro" directory
    default_ro_path = "/ro/data"
    data_ready = ask_yes_no(f"Is the PTB-XL data ready at '{default_ro_path}'?")
    
    if data_ready:
        raw_data_path = default_ro_path
        print(f"\n✓ Using PTB-XL data from: {raw_data_path}")
    else:
        raw_data_path = input(f"\nEnter PTB-XL data directory path: ").strip()
        if not raw_data_path:
            print("No path provided. Exiting...")
            return
    
    # Verify the data path exists
    if not os.path.exists(raw_data_path):
        print(f"\n✗ Error: Directory does not exist: {raw_data_path}")
        return
    
    # Check for ptbxl_database.csv
    db_path = os.path.join(raw_data_path, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        print(f"\n✗ Error: ptbxl_database.csv not found in {raw_data_path}")
        print("Please ensure the PTB-XL dataset is properly downloaded.")
        return
    
    print(f"✓ Found PTB-XL database at: {db_path}")

    # ================= Step 2: Data Output Location =================
    
    print("\n" + "=" * 70)
    print(" STEP 2: OUTPUT DATA LOCATION")
    print("=" * 70)
    
    # Ask where to read data from (can reuse existing prepared data)
    print("\nOptions:")
    print("  1) Use existing prepared data (from previous run)")
    print("  2) Prepare new data from raw PTB-XL")
    
    data_source_choice = ask_choice(
        "\nWhere should the data be loaded from?",
        [
            "Use existing prepared data",
            "Prepare new data from raw PTB-XL"
        ]
    )
    
    if data_source_choice == 1:
        # Ask for existing data path
        default_data_path = "prepared_data/ptb-xl"
        data_path = input(f"\nEnter path to prepared data (default: {default_data_path}): ").strip()
        if not data_path:
            data_path = default_data_path
        
        if not os.path.exists(data_path):
            print(f"\n✗ Error: Prepared data not found at: {data_path}")
            print("Please prepare the data first (option 2).")
            return
        
        print(f"✓ Using prepared data from: {data_path}")
        
    else:
        # Prepare new data
        default_out_root = "prepared_data"
        out_root = input(f"\nEnter output directory for prepared data (default: {default_out_root}): ").strip()
        if not out_root:
            out_root = default_out_root
        
        print(f"\n✓ Data will be saved to: {out_root}")
        
        # Run data preparation
        success = run_data_preparation(raw_data_path, "ptb-xl", out_root)
        if not success:
            print("\n✗ Data preparation failed. Exiting...")
            return
        
        data_path = os.path.join(out_root, "ptb-xl")
        print(f"✓ Prepared data available at: {data_path}")

    # ================= Step 3: Sampling Rate Selection =================
    
    print("\n" + "=" * 70)
    print(" STEP 3: SAMPLING RATE SELECTION")
    print("=" * 70)
    
    selected_rates = select_sampling_rates()
    print(f"\n✓ Selected sampling rates: {selected_rates}")

    # ================= Step 4: Model Selection =================
    
    print("\n" + "=" * 70)
    print(" STEP 4: MODEL SELECTION")
    print("=" * 70)
    
    model_choice = ask_choice(
        "\nWhich model would you like to train?",
        [
            "CNN1D (1D Convolutional Neural Network)",
            "CNN-LSTM (CNN with LSTM layers)"
        ]
    )
    
    model_type = "cnn1d" if model_choice == 1 else "cnn_lstm"
    print(f"\n✓ Selected model: {model_type}")

    # ================= Step 5: Hardware Warning =================
    
    print("\n" + "=" * 70)
    print(" STEP 5: TRAINING")
    print("=" * 70)
    
    # Display hardware warning
    if not display_hardware_warning():
        print("\n⚠️  Training cancelled by user.")
        return

    # ================= Step 6: Training =================
    
    print("\n" + "=" * 70)
    print(" STARTING TRAINING PROCESS")
    print("=" * 70)
    
    # Convert data_path to Path object
    data_path_obj = Path(data_path)
    
    # Run training for each selected sampling rate
    training_results = []
    
    for rate in selected_rates:
        rate_path = data_path_obj / f"{rate}hz"
        
        # Check if data exists for this sampling rate
        if not rate_path.exists():
            print(f"\n⚠️  No data found for {rate} Hz at {rate_path}")
            print(f"   Skipping {rate} Hz...")
            continue
        
        # Run training
        print(f"\n{'='*60}")
        print(f" TRAINING: {rate} Hz with {model_type}")
        print(f"{'='*60}")
        
        run_training(data_path_obj, rate, model_type)
        
        # Display checkpoint information
        checkpoint_dir = Path("checkpoints") / "ptb-xl" / f"{rate}hz" / model_type
        if checkpoint_dir.exists():
            print(f"\n✓ Checkpoint saved at: {checkpoint_dir}")
            
            # Count saved checkpoints
            fold_dirs = list(checkpoint_dir.glob("fold_*"))
            print(f"✓ Folds completed: {len(fold_dirs)}")
            
            for fold_dir in sorted(fold_dirs):
                best_pt = fold_dir / "best.pt"
                if best_pt.exists():
                    size_mb = best_pt.stat().st_size / (1024 * 1024)
                    print(f"   - {fold_dir.name}/best.pt ({size_mb:.2f} MB)")
        
        training_results.append(rate)
    
    # ================= Summary =================
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    
    if training_results:
        print(f"\n✓ Successfully trained for sampling rates: {training_results}")
        print(f"\n✓ All checkpoints saved in: checkpoints/ptb-xl/")
    else:
        print("\n⚠️  No training completed.")
    
    print("\nThank you for using the ECG Arrhythmia Detection Pipeline!")


if __name__ == "__main__":
    main()


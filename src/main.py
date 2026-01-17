import logging
from data_loader.ecg_data_prepare import (
    load_ecg_arrhythmia,
    load_ptb_xl_ids
)

# ================= Logger =================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger("ECG-MAIN")


# ================= Helper =================

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


# ================= Main =================

def main():

    print("\n==========================================")
    print(" ECG ARRHYTHMIA DETECTION PIPELINE")
    print(" Step 1 — Dataset Loading")
    print("==========================================\n")

    print("Current pipeline status:")
    print(" • Loading ECG datasets")
    print(" • Label inspection")
    print(" • No preprocessing yet")
    print(" • No model training yet\n")

    print("📁 Important:")
    print("All datasets must be located under the project /data directory.\n")

    # ================= Dataset Question =================

    dataset_choice = ask_choice(
        "Which dataset would you like to load?",
        [
            "ECG-Arrhythmia dataset (WFDB with SNOMED codes)",
            "PTB-XL dataset (12-lead clinical ECG)"
        ]
    )

    # ================= Data Path =================

    data_dir = input("\nEnter dataset folder path (default: data): ").strip()
    if not data_dir:
        data_dir = "data"

    print("\n------------------------------------------")

    # ================= Dataset Logic =================

    if dataset_choice == 1:
        print("Dataset selected: ECG-Arrhythmia\n")

        print("Supported labels:")
        print(" • NORMAL (SR)")
        print(" • ATRIAL FIBRILLATION (AFIB)\n")

        records = load_ecg_arrhythmia(
            dataset_path=data_dir,
            logger=logger
        )

        print(f"\n✔ Successfully loaded {len(records)} ECG records")

    elif dataset_choice == 2:
        print("Dataset selected: PTB-XL\n")

        print("Supported labels:")
        print(" • NORM")
        print(" • AFIB")
        print(" • AFLT")
        print(" • OTHER arrhythmias\n")

        load_ptb_xl_ids(data_dir)

        print("\n✔ PTB-XL metadata successfully loaded")

    # ================= End Message =================

    print("\n==========================================")
    print(" PIPELINE STATUS")
    print("==========================================")
    print("✔ Dataset loaded")
    print("⏳ Preprocessing not started")
    print("⏳ Feature extraction not started")
    print("⏳ Model training not started")
    print("\nNext step: ECG preprocessing & segmentation\n")


if __name__ == "__main__":
    main()

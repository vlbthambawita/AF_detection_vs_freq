from pathlib import Path
import re
import csv

# ================= CONFIG =================
ROOT = Path(__file__).resolve().parents[1] / "checkpoints" / "ptbl-xl"

FREQS = ["62hz", "100hz", "250hz", "500hz"]
MODELS = ["cnn1d", "cnn_lstm"]

OUTCSV = ROOT / "test_summary_table.csv"


# ================= PARSER =================
def parse_metrics(file_path):
    text = file_path.read_text()

    def grab(name):
        m = re.search(rf"{name}\s*:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None

    return {
        "Accuracy": grab("Accuracy"),
        "F1": grab("F1-score"),
        "Sensitivity": grab("Recall \\(Sensitivity\\)"),
        "Specificity": grab("Specificity"),
        "Precision": grab("Precision"),
        "MCC": grab("MCC"),
    }


# ================= MAIN =================
rows = []

for freq in FREQS:
    for model in MODELS:

        path = ROOT / freq / model / "test_results.txt"

        if not path.exists():
            print("Missing:", path)
            continue

        metrics = parse_metrics(path)

        rows.append([
            freq,
            model,
            metrics["Accuracy"],
            metrics["F1"],
            metrics["Precision"],
            metrics["Specificity"],
            metrics["Sensitivity"],
            metrics["MCC"],
        ])

        print(f"Loaded {freq} | {model}")

# save csv
with open(OUTCSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Frequency","Model",
        "Accuracy","F1","Precision",
        "Specificity","Sensitivity","MCC"
    ])
    writer.writerows(rows)

print("\nSaved:", OUTCSV)


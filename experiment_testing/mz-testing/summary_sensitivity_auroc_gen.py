from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    recall_score
)

BASE = Path("checkpoints/ptbl-xl")
HZ_LIST = ["62hz", "100hz", "250hz", "500hz"]
MODELS = ["cnn1d", "cnn_lstm"]

THRESHOLD = 0.5  # Binary argmax decision is equivalent to prob >= 0.5 for class 1 preds = logits.argmax(dim=1)
#for more information READ threshol.md 
rows = []

for hz in HZ_LIST:
    for model in MODELS:
        model_dir = BASE / hz / model

        fold_aurocs = []
        fold_recalls = []

        for fold in range(1, 6):
            p = model_dir / f"fold_{fold}" / "roc_val.npz"
            data = np.load(p)

            y_true = data["y_true"].astype(int).ravel()
            y_score = data["y_score"].astype(float).ravel()

            # AUROC
            auc = roc_auc_score(y_true, y_score)
            fold_aurocs.append(auc)

            # Sensitivity (Recall)
            y_pred = (y_score >= THRESHOLD).astype(int)
            recall = recall_score(y_true, y_pred)
            fold_recalls.append(recall)

        rows.append({
            "Hz": hz,
            "Model": model,
            "AUROC_mean": np.mean(fold_aurocs),
            "AUROC_std": np.std(fold_aurocs),
            "Sensitivity_mean": np.mean(fold_recalls),
            "Sensitivity_std": np.std(fold_recalls),
        })

df = pd.DataFrame(rows)
print(df)
df.to_csv("summary_sensitivity_auroc.csv", index=False)

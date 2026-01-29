print("SCRIPT STARTED")

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# ------------------
# CONFIG
# ------------------
DATA_DIR = "ecg_dataset/ecg_arrhythmia_dataset_CSV/WFDBRecords"
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 16
WINDOW_SIZE = 500
STEP_SIZE = 250
PATIENCE = 10  # for early stopping
K_FOLDS = 5  # number of folds for cross-validation

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------
# LOAD FILES
# ------------------
def get_all_csv_files(data_dir):
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    return all_files

all_files = get_all_csv_files(DATA_DIR)
# Use subset for faster training
all_files = all_files[:1000]  # Use first 1000 files
print(f"Using {len(all_files)} CSV files (subset)")

# ------------------
# DATASET
# ------------------
class ECGDataset(Dataset):
    def __init__(self, file_list, window_size=500, step_size=250, scaler=None):
        self.samples = []
        self.scaler = scaler

        for file in file_list:
            df = pd.read_csv(file)
            df = df.loc[:, ~df.columns.duplicated()]
            data = df.values.T  # (12, N)

            if self.scaler is None:
                self.scaler = StandardScaler()
                data_flat = data.flatten().reshape(-1, 1)
                self.scaler.fit(data_flat)
            
            # Normalize
            data_flat = data.flatten().reshape(-1, 1)
            data_norm = self.scaler.transform(data_flat).reshape(data.shape)
            
            num_samples = data.shape[1]

            for start in range(0, num_samples - window_size + 1, step_size):
                window = data_norm[:, start:start + window_size]
                self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = x[0]  # reconstruct lead I
        return x, y

# ------------------
# MODEL
# ------------------
class ECG1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(256, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
           
            nn.Conv1d(64, 1, 7, padding=3)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ------------------
# K-FOLD CROSS-VALIDATION
# ------------------
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    
    print(f"\nFold {fold+1}/{K_FOLDS}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Fit scaler on training data
    train_ds = ECGDataset(train_files, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    scaler = train_ds.scaler
    
    val_ds = ECGDataset(val_files, window_size=WINDOW_SIZE, step_size=STEP_SIZE, scaler=scaler)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # ------------------
    # MODEL
    # ------------------
    model = ECG1DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # ------------------
    # TRAINING
    # ------------------
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            mae = torch.mean(torch.abs(pred - y))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            train_mae += mae.item()
    
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
    
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_mae += torch.mean(torch.abs(pred - y)).item()
    
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        scheduler.step(val_loss)
    
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}"
        )
        
        # Early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
            print("Saved best model for fold")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break
    
    print(f"Fold {fold+1} completed!")

print("All folds training completed!")

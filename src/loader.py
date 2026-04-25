# loader.py

from pathlib import Path
import torch
from torch.utils.data import Dataset, ConcatDataset


class ECGTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ECGDataset(Dataset):
    """
    Backward-compatible dataset loader.

    Used mainly for test.pt loading or old data.pt format.
    """

    def __init__(self, data_dir):
        data_dir = Path(data_dir)

        if (data_dir / "data.pt").exists():
            data = torch.load(data_dir / "data.pt", map_location="cpu")
        elif (data_dir / "test.pt").exists():
            data = torch.load(data_dir / "test.pt", map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No data.pt or test.pt found in {data_dir}"
            )

        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_fold_file(data_dir, fold):
    """
    Load one saved fold file.

    Expected:
        prepared_data/ptbl-xl/500hz/fold_1.pt
        prepared_data/ptbl-xl/500hz/fold_2.pt
        ...
    """

    data_dir = Path(data_dir)
    fold_path = data_dir / f"fold_{fold}.pt"

    if not fold_path.exists():
        raise FileNotFoundError(f"Missing fold file: {fold_path}")

    data = torch.load(fold_path, map_location="cpu")

    return ECGTensorDataset(
        X=data["X"],
        y=data["y"],
    )


def load_kfold(data_dir, fs, val_fold, n_folds=5):
    """
    Scientific K-fold loading.

    Training:
        all folds except val_fold

    Validation:
        only val_fold

    Important:
        Validation remains natural/unbalanced.
        Training balancing is handled later in train.py using --train_balance downsample.
    """

    data_dir = Path(data_dir)

    train_datasets = []

    for fold in range(1, n_folds + 1):
        fold_ds = load_fold_file(data_dir, fold)

        if fold == val_fold:
            val_ds = fold_ds
        else:
            train_datasets.append(fold_ds)

    train_ds = ConcatDataset(train_datasets)

    all_labels = []

    for fold in range(1, n_folds + 1):
        fold_ds = load_fold_file(data_dir, fold)
        all_labels.append(fold_ds.y)

    y_all = torch.cat(all_labels)
    num_classes = int(y_all.max().item() + 1)

    return train_ds, val_ds, num_classes
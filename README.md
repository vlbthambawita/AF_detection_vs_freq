

# SEARCH_AF_detection_OsloMet_BachelorGroup

Pipeline for **AFIB vs NORMAL ECG classification** using **PTB-XL**, with:

* Patient-safe splitting
* K-fold cross-validation
* CNN1D and CNN–LSTM models
* Optional hold-out test evaluation

---

## Pipeline Overview

![ECG preprocessing pipeline](src/ecg_preprocessing_pipeline.png)

---

# Installation

```bash
git clone <repo>
cd SEARCH_AF_detection_OsloMet_BachelorGroup
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

# Dataset

Download PTB-XL:

[https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)

---

# 1. Data Preparation

```bash
python src/ecg_preprocessing/ecg_data_loader.py ^
  --dataset_path C:\path\to\ptb-xl ^
  --name ptb-xl ^
  --fs 250 ^
  --out_root prepared_data ^
  --folds 5 ^
  --test_ratio 0.3 ^
  --balance_mode train
```

---

## Arguments (Data Preparation)

### Core

* `--dataset_path`
  Path to raw PTB-XL

* `--name`
  Dataset name (e.g., `ptb-xl`)

* `--fs`
  Sampling rates
  Example:

  ```
  --fs 62 100 250 500
  ```

---

### Splitting

* `--folds`
  Number of folds (e.g., 5)

* `--test_ratio`
  Hold-out test split (patient-safe)
  Example:

  ```
  0.3 → 30% patients reserved for final test
  ```

---

### Balancing (IMPORTANT)

* `--balance_mode train`  **(RECOMMENDED)**

  * Preprocessing → no balancing
  * Training → balanced
  * Validation → natural
  * Test → natural

* `--balance_mode none`
  No balancing anywhere

* `--balance_mode fold`
  Each fold balanced (NOT recommended for science)

* `--balance_mode global`
  Entire dataset balanced before splitting

---

## Output Structure

```text
prepared_data/ptb-xl/250hz/
├── fold_1.pt
├── fold_2.pt
├── fold_3.pt
├── fold_4.pt
├── fold_5.pt
├── samples_250hz.csv
└── test/
    └── test.pt
```

---

# 2. Training

```bash
cd src

python train.py ^
  --data_path ..\prepared_data\ptb-xl\250hz ^
  --model cnn1d ^
  --train_balance downsample ^
  --batch_size 8 ^
  --device cuda
```

---

## Arguments (Training)

### Required

* `--data_path`
  Path to prepared data

* `--model`

  ```
  cnn1d | cnn_lstm
  ```

---

### Training Behavior

* `--train_balance downsample`

  * ONLY training set balanced
  * Validation remains natural
  * Matches scientific setup

---

### Hardware

* `--device cuda` → GPU (recommended)
* `--device cpu` → slower

---

### Optional

* `--batch_size`
  Default: 32
  Reduce if memory issues

---

# 3. Testing

After training:

* Automatic test evaluation runs if `test.pt` exists
* Ensemble of all folds is used

---

## Test Evaluation Types

### Unbalanced (REALISTIC)

* Uses natural distribution
* Reflects real-world performance

### Balanced

* Downsampled test
* Fair comparison across classes

---

# Scientific Setup (IMPORTANT)

Recommended pipeline:

| Stage      | Distribution |
| ---------- | ------------ |
| Training   | Balanced     |
| Validation | Natural      |
| Test       | Natural      |
| Extra Test | Balanced     |

---

# Models

### CNN1D

* Learns local ECG morphology
* QRS, waveform patterns

### CNN-LSTM

* CNN → feature extraction
* LSTM → temporal dependencies

---

# Outputs

```text
src/checkpoints/ptb-xl/250hz/cnn1d/
├── fold_1/
│   ├── best.pt
│   ├── last.pt
│   ├── metrics.txt
│   └── roc_val.npz
```

---

# Notes

* **Patient-safe splitting is enforced**
* No leakage between folds or test
* Each segment inherits its patient split



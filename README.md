
## Project Goal

The goal of this project is to investigate how different ECG sampling frequencies affect the development, evaluation, and interpretation of deep learning models for atrial fibrillation detection.

The project compares model performance across sampling frequencies such as:

```text
62 Hz, 100 Hz, 250 Hz, 500 Hz
```

The main experimental focus is to evaluate whether lower or higher ECG sampling frequencies influence classification performance, training cost, and model reliability.

---

## Pipeline Overview

![ECG preprocessing pipeline](src/ecg_preprocessing_pipeline.png)

The general pipeline is:

```text
Raw PTB-XL ECG data
        ↓
AFIB/NORMAL label extraction
        ↓
Signal cleaning
        ↓
Resampling
        ↓
Segmentation
        ↓
Normalization
        ↓
Patient-safe splitting
        ↓
CNN1D / CNN-LSTM training
        ↓
Validation and test evaluation
```

---

# 1. Installation

Clone the repository:

```bash
git clone <repo>
cd SEARCH_AF_detection_OsloMet_BachelorGroup
```

Run the setup script:

```bash
python setup_env.py
```

The setup script will:

* Create a Python virtual environment called `venv`
* Upgrade `pip`
* Install all required packages from `requirements.txt`

---

## Activate the Virtual Environment

After running `setup_env.py`, activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

# 2. Dataset

This project uses the PTB-XL dataset.

Download PTB-XL from PhysioNet:

```text
https://physionet.org/content/ptb-xl/1.0.3/
```

After downloading the dataset, place it inside the project folder, for example:

```text
SEARCH_AF_detection_OsloMet_BachelorGroup/data/
```

The dataset folder should contain files and folders such as:

```text
data/
├── ptbxl_database.csv
├── records100/
└── records500/
```

The file `ptbxl_database.csv` is required because it contains the ECG metadata and diagnostic labels.

---

# 3. Recommended Way to Run the Project

The easiest way to run the full pipeline is the interactive runner:

```bash
python src/main.py
```

The interactive runner guides the user through:

1. Selecting whether to use existing prepared data or prepare new data
2. Selecting one or more sampling frequencies
3. Choosing data preparation settings
4. Choosing the model type
5. Choosing training settings
6. Running training for the selected frequencies

This is the recommended method for examiners, supervisors, and new users.

---

# 4. Interactive Pipeline Guide

When running:

```bash
python src/main.py
```

the program first asks for the data source.

## Step 1: Choose Data Source

```text
1) Use existing prepared data
2) Prepare new data from raw PTB-XL
```

Choose option `1` if the data has already been prepared.

Choose option `2` if starting from the raw PTB-XL dataset.

---

## Step 2: Choose Sampling Frequencies

The runner allows one or more sampling rates:

```text
1) 62 Hz
2) 100 Hz
3) 250 Hz
4) 500 Hz
```

Example:

```text
2 3 4
```

This selects:

```text
100 Hz, 250 Hz, 500 Hz
```

Pressing Enter without input selects all rates:

```text
62 Hz, 100 Hz, 250 Hz, 500 Hz
```

---

## Step 3: Data Preparation Settings

If preparing new data, the program asks for:

* Path to raw PTB-XL dataset
* Output directory
* Number of K-folds
* Hold-out test ratio
* Balance mode
* Flatline duration threshold

Recommended values:

```text
Number of folds: 5
Hold-out test ratio: 0.2
Balance mode: train
Flatline seconds: 3.0
```

---

# 5. Recommended Scientific Setup

The recommended setup is:

```text
Data preparation: balance_mode train
Training: train_balance downsample
Validation: natural/unbalanced
Test: natural/unbalanced
Extra test: balanced
```

This setup is used because the training set can be balanced while the validation and test sets remain closer to the natural PTB-XL distribution.

| Stage      | Distribution |
| ---------- | ------------ |
| Training   | Balanced     |
| Validation | Natural      |
| Test       | Natural      |
| Extra Test | Balanced     |

---

# 6. Balance Modes

The data preparation script supports different balance modes.

## Recommended

```text
train
```

This means:

* Preprocessing keeps the dataset natural
* Training can be balanced later
* Validation remains natural
* Test remains natural

This is the recommended scientific setup.

---

## Other Options

```text
none
```

No balancing is applied.

```text
fold
```

Each fold is balanced. This is not recommended for the main scientific experiment because validation also becomes artificially balanced.

```text
global
```

The entire dataset is globally balanced before splitting. This is also not recommended for the main scientific setup because it changes the natural dataset distribution before evaluation.

---

# 7. Model Selection

The runner supports two model types:

```text
cnn1d
cnn_lstm
```

---

## CNN1D

The CNN1D model learns local ECG morphology, such as:

* QRS-related patterns
* Waveform shapes
* Local amplitude changes
* Short-range temporal features

---

## CNN-LSTM

The CNN-LSTM model combines convolutional feature extraction with temporal sequence modeling.

The structure is:

```text
ECG signal → CNN feature extraction → LSTM temporal modeling → classification
```

This allows the model to learn both local ECG patterns and longer temporal dependencies.

---

# 8. Training Settings

During training, the runner asks for:

* Training balance mode
* Batch size
* Number of epochs
* Learning rate
* Number of K-folds
* Device

Recommended values for CNN1D:

```text
train_balance: downsample
batch_size: 8
epochs: 50
learning rate: 0.001
device: auto or cuda
```

Recommended values for CNN-LSTM:

```text
train_balance: downsample
batch_size: 4 or 8
epochs: 50
learning rate: 0.001
device: auto or cuda
```

---

# 9. Hardware Notes

Training ECG deep learning models can require significant compute resources.

Recommended hardware:

* CUDA-enabled GPU
* At least 8 GB RAM
* More memory for 500 Hz experiments
* Smaller batch size for CNN-LSTM

Suggested batch sizes:

| Model    | Suggested Batch Size |
| -------- | -------------------- |
| CNN1D    | 8                    |
| CNN-LSTM | 4 or 8               |

The 500 Hz frequency has the highest memory and compute cost. On limited laptops, it is recommended to run 500 Hz separately.

---

# 10. Manual Data Preparation

The interactive runner is recommended, but the data preparation script can also be run manually.

## Windows Example

```bash
python src/ecg_preprocessing/ecg_data_prepare.py ^
  --dataset_path data ^
  --name ptb-xl ^
  --fs 250 ^
  --out_root prepared_data ^
  --folds 5 ^
  --test_ratio 0.2 ^
  --balance_mode train ^
  --flatline_seconds 3.0
```

## macOS / Linux Example

```bash
python src/ecg_preprocessing/ecg_data_prepare.py \
  --dataset_path data \
  --name ptb-xl \
  --fs 250 \
  --out_root prepared_data \
  --folds 5 \
  --test_ratio 0.2 \
  --balance_mode train \
  --flatline_seconds 3.0
```

---

## Data Preparation Arguments

### `--dataset_path`

Path to the raw PTB-XL dataset.

Example:

```text
data
```

or:

```text
C:\path\to\ptb-xl
```

---

### `--name`

Name of the prepared dataset output folder.

Example:

```text
ptb-xl
```

---

### `--fs`

Sampling frequency.

Example:

```text
250
```

The interactive runner allows several frequencies to be selected one after another.

---

### `--out_root`

Output directory for prepared data.

Example:

```text
prepared_data
```

---

### `--folds`

Number of K-folds.

Example:

```text
5
```

---

### `--test_ratio`

Patient-safe hold-out test ratio.

Example:

```text
0.2
```

This means that 20% of patients are reserved for final testing.

---

### `--balance_mode`

Balance mode used during data preparation.

Recommended:

```text
train
```

---

### `--flatline_seconds`

Duration threshold used for flatline detection.

Example:

```text
3.0
```

This means that a lead with a continuous flatline of at least 3 seconds can be detected as problematic.

---

# 11. Prepared Data Output Structure

After data preparation, the output structure should look like this:

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

If several frequencies are prepared, the structure becomes:

```text
prepared_data/ptb-xl/
├── 62hz/
├── 100hz/
├── 250hz/
└── 500hz/
```

Each frequency folder contains its own K-fold files and optional hold-out test data.

---

# 12. Manual Training

The interactive runner is recommended, but training can also be run manually.

## Windows Example

```bash
python src/train.py ^
  --data_path prepared_data\ptb-xl\250hz ^
  --model cnn1d ^
  --train_balance downsample ^
  --batch_size 8 ^
  --epochs 50 ^
  --lr 0.001 ^
  --kfolds 5 ^
  --device cuda
```

## macOS / Linux Example

```bash
python src/train.py \
  --data_path prepared_data/ptb-xl/250hz \
  --model cnn1d \
  --train_balance downsample \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.001 \
  --kfolds 5 \
  --device cuda
```

---

## Training Arguments

### `--data_path`

Path to the prepared data folder for one sampling frequency.

Example:

```text
prepared_data/ptb-xl/250hz
```

---

### `--model`

Model type.

Options:

```text
cnn1d
cnn_lstm
```

---

### `--train_balance`

Runtime training balance.

Recommended:

```text
downsample
```

This balances only the training set. Validation and test sets remain natural.

---

### `--batch_size`

Batch size used during training.

Example:

```text
8
```

Use a smaller batch size if memory errors occur.

---

### `--epochs`

Maximum number of training epochs.

Example:

```text
50
```

---

### `--lr`

Learning rate.

Example:

```text
0.001
```

---

### `--kfolds`

Number of folds used in the prepared data.

Example:

```text
5
```

---

### `--device`

Device used for training.

Options:

```text
auto
cuda
cpu
```

Recommended:

```text
auto
```

or:

```text
cuda
```

if a CUDA-enabled GPU is available.

---

# 13. Checkpoint Output Structure

After training, model checkpoints are saved under:

```text
checkpoints/ptb-xl/250hz/cnn1d/
├── fold_1/
│   ├── best.pt
│   ├── last.pt
│   ├── metrics.txt
│   └── roc_val.npz
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/
```

For CNN-LSTM, the output folder is:

```text
checkpoints/ptb-xl/250hz/cnn_lstm/
```

Each fold stores:

* `best.pt` — best model checkpoint
* `last.pt` — final model checkpoint
* `metrics.txt` — training and validation metrics
* `roc_val.npz` — validation ROC data

---

# 14. Testing

If the prepared data contains:

```text
test/test.pt
```

the training script automatically performs final test evaluation after training.

The test evaluation uses an ensemble of trained folds.

---

## Test Evaluation Types

### Unbalanced Test

The unbalanced test uses the natural PTB-XL class distribution.

This is the most realistic evaluation because real clinical datasets are usually not perfectly balanced.

---

### Balanced Test

The balanced test downsamples the test set so that AFIB and NORMAL classes are more equal.

This is useful as a secondary comparison because it shows performance when both classes are equally represented.

---

# 15. Expected Workflow for Examiners

A simple workflow for checking the project is:

```bash
git clone <repo>
cd SEARCH_AF_detection_OsloMet_BachelorGroup
python setup_env.py
venv\Scripts\activate
python src\main.py
```

On macOS / Linux:

```bash
git clone <repo>
cd SEARCH_AF_detection_OsloMet_BachelorGroup
python setup_env.py
source venv/bin/activate
python src/main.py
```

Then choose:

```text
1) Use existing prepared data
```

if prepared data is already included or available.

Or choose:

```text
2) Prepare new data from raw PTB-XL
```

if running from the raw PTB-XL dataset.

---

# 16. Important Notes

* Patient-safe splitting is enforced.
* No patient should appear in more than one split.
* ECG segments inherit the split of their patient.
* Validation and test data should remain natural/unbalanced for the main experiment.
* Training can be balanced using downsampling.
* The 500 Hz setting requires more memory and compute time.
* The interactive `main.py` runner is the easiest way to run the project.
* Manual commands are provided for reproducibility and advanced use.

---

# 17. Summary

This repository provides a complete ECG classification pipeline for AFIB detection using PTB-XL.

The recommended command to start is:

```bash
python src/main.py
```

The recommended scientific setup is:

```text
Data preparation: balance_mode train
Training: train_balance downsample
Validation: natural
Test: natural
Extra test: balanced
```

This setup supports fair model development while preserving realistic evaluation conditions.

```
```

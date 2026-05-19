# ECG AFIB Detection Pipeline

## Project Goal

The goal of this project is to investigate how different ECG sampling frequencies affect the development, evaluation, and interpretation of deep learning models for atrial fibrillation detection.

The project compares ECG classification performance across several sampling frequencies. The default thesis frequencies are:

```text
62 Hz, 100 Hz, 250 Hz, 500 Hz
```

The updated interactive runner also allows additional integer sampling frequencies up to 500 Hz, such as:

```text
50 Hz, 75 Hz, 125 Hz, 150 Hz, 200 Hz, 300 Hz, 400 Hz
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
Validation and final test evaluation
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

The recommended way to run the project is the interactive runner:

```bash
python src/main.py
```

The runner guides the user through:

1. Selecting whether to use existing prepared data or prepare new data from raw PTB-XL
2. Selecting one or more sampling frequencies
3. Choosing the training split mode
4. Choosing data preparation settings if new data is created
5. Choosing the model type
6. Choosing whether to train or only test existing checkpoints
7. Choosing training settings
8. Running training and final test evaluation for the selected frequencies

This is the recommended method for examiners, supervisors, and new users.

---

# 4. Interactive Runner Overview

When running:

```bash
python src/main.py
```

the program starts the interactive ECG AFIB detection pipeline.

The current `main.py` supports:

```text
Data source:
1) Use existing prepared data
2) Prepare new data from raw PTB-XL

Sampling rates:
1) Use default thesis rates
2) Select from suggested list
3) Write custom integer frequencies

Training split mode:
1) auto
2) kfold
3) manual

Model:
1) cnn1d
2) cnn_lstm

Action:
1) Train and run final test evaluation
2) Test only using existing checkpoints
```

---

# 5. Step-by-Step Interactive Guide

## Step 1: Choose Data Source

```text
1) Use existing prepared data
2) Prepare new data from raw PTB-XL
```

Choose option `1` if the data has already been prepared.

Choose option `2` if starting from the raw PTB-XL dataset.

---

## Step 2: Choose Sampling Frequencies

The runner provides three ways to choose frequencies.

### Option 1: Default thesis rates

```text
62 Hz, 100 Hz, 250 Hz, 500 Hz
```

### Option 2: Suggested list

The suggested list contains:

```text
50 Hz, 62 Hz, 75 Hz, 100 Hz, 125 Hz, 150 Hz, 200 Hz, 250 Hz, 300 Hz, 400 Hz, 500 Hz
```

The user can select one or more option numbers.

Example:

```text
2 4 8
```

This selects the corresponding rates from the displayed list.

### Option 3: Custom integer frequencies

The user can write integer frequencies directly.

Examples:

```text
500 250 125 100 62
```

or:

```text
50, 75, 100, 150, 250, 500
```

Important restrictions:

* Float values such as `62.5` are not supported.
* Frequencies above `500 Hz` are rejected because PTB-XL high-resolution ECG data is 500 Hz.
* Duplicate frequencies are removed automatically.

---

## Step 3: Choose Split Mode

The runner supports three training split modes.

```text
auto   - detect k-fold or manual split from files
kfold  - require fold_1.pt ... fold_k.pt
manual - require train.pt and val.pt
```

### `auto`

The runner checks the prepared data folder and detects whether it contains:

```text
fold_1.pt ... fold_k.pt
```

or:

```text
train.pt
val.pt
```

This is useful when the prepared structure already exists.

### `kfold`

This mode requires k-fold files:

```text
fold_1.pt
fold_2.pt
...
fold_k.pt
```

The number of folds is requested by the runner.

### `manual`

This mode requires:

```text
train.pt
val.pt
```

Manual mode trains one model using the prepared train and validation split.

---

# 6. Existing Prepared Data

If using existing prepared data, the runner asks for the path to the prepared dataset root.

Example:

```text
prepared_data/ptb-xl
```

or:

```text
prepared_data/ptbl-xl
```

The runner expects each selected frequency to have its own folder:

```text
prepared_data/<dataset_name>/
├── 62hz/
├── 100hz/
├── 250hz/
└── 500hz/
```

The dataset folder name is not fixed. The runner resolves rate folders dynamically as:

```text
<prepared_root>/<rate>hz
```

The runner can also help if the user enters the parent folder:

```text
prepared_data
```

If exactly one matching dataset folder is found inside it, the runner can auto-select that dataset folder.

---

## Prepared Data Validation

Before training, the runner validates the selected frequencies and prints a table.

For each frequency, it checks:

* Whether the rate folder exists
* Whether the split structure is valid
* Whether test data exists

The runner detects test files in either of these forms:

```text
test/test.pt
```

or:

```text
test.pt
```

If a selected frequency is missing or incomplete, the runner stops and asks the user to fix the prepared data or choose another split mode.

---

# 7. Preparing New Data from Raw PTB-XL

If preparing new data, the runner asks for:

* Dataset output folder name
* Path to raw PTB-XL dataset
* Output root directory
* Prepared split structure
* Balance mode
* Flatline duration threshold

The dataset output folder name can be any experiment name. The current default in `main.py` is:

```text
ptbl-xl
```

The prepared root will then become:

```text
prepared_data/ptbl-xl
```

A different dataset name can be entered if required, for example:

```text
ptb-xl
```

---

## Prepared Split Structures

When preparing new data, the runner supports two prepared split structures.

### K-fold preparation

K-fold preparation saves:

```text
fold_1.pt
fold_2.pt
...
fold_k.pt
```

It can also create an optional patient-safe hold-out test set:

```text
test/test.pt
```

Recommended values:

```text
Number of folds: 5
Hold-out test ratio: 0.2
Balance mode: train
Flatline seconds: 3.0
```

### Manual preparation

Manual preparation saves:

```text
train.pt
val.pt
test.pt
```

Recommended split ratio:

```text
train: 0.7
validation: 0.2
test: 0.1
```

Manual mode can also create an additional patient-safe hold-out test file:

```text
test/test.pt
```

This extra hold-out test is optional.

---

# 8. Recommended Scientific Setup

The recommended scientific setup is:

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

# 9. Balance Modes

The data preparation script supports different balance modes.

## Recommended

```text
train
```

This means:

* Preprocessing keeps validation and test data natural
* Training can be balanced later during runtime
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

Each fold or split is balanced. This is not recommended for the main scientific experiment because validation may also become artificially balanced.

```text
global
```

The entire dataset is globally balanced before splitting. This is not recommended for the main scientific setup because it changes the natural dataset distribution before evaluation.

---

# 10. Model Selection

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

# 11. Training and Test Settings

After model selection, the runner asks whether to:

```text
1) Train and run final test evaluation
2) Test only using existing checkpoints
```

---

## Train and Final Test Evaluation

Training mode asks for:

* Runtime training balance
* Batch size
* Maximum epochs
* Learning rate
* Early stopping patience
* Device

Recommended values for CNN1D:

```text
train_balance: downsample
batch_size: 8
epochs: 50
learning rate: 0.001
early stopping patience: 10
device: auto or cuda
```

Recommended values for CNN-LSTM:

```text
train_balance: downsample
batch_size: 4 or 8
epochs: 50
learning rate: 0.001
early stopping patience: 10
device: auto or cuda
```

---

## Test-Only Mode

Test-only mode uses existing checkpoints and runs evaluation without training.

In test-only mode:

* Training balance is not requested interactively
* Epochs and learning rate are not requested interactively
* The runner passes `--test_only` to `train.py`
* Existing checkpoints must already be available under the expected checkpoint directory

This is useful when models have already been trained and the user only wants to repeat final evaluation.

---

## Runtime Training Balance

The runner supports:

```text
downsample
none
```

Recommended:

```text
downsample
```

This balances only the training data during runtime. Validation and test data remain natural.

---

## Device Selection

The runner supports:

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

# 12. Hardware Notes

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

In k-fold mode, one model is trained per fold. This increases training time compared with manual split mode.

---

# 13. Manual Data Preparation

The interactive runner is recommended, but the data preparation script can also be run manually.

## K-fold Preparation

### Windows Example

```bash
python src/ecg_preprocessing/ecg_data_prepare.py ^
  --dataset_path data ^
  --name ptbl-xl ^
  --fs 250 ^
  --out_root prepared_data ^
  --folds 5 ^
  --test_ratio 0.2 ^
  --balance_mode train ^
  --flatline_seconds 3.0
```

### macOS / Linux Example

```bash
python src/ecg_preprocessing/ecg_data_prepare.py \
  --dataset_path data \
  --name ptbl-xl \
  --fs 250 \
  --out_root prepared_data \
  --folds 5 \
  --test_ratio 0.2 \
  --balance_mode train \
  --flatline_seconds 3.0
```

---

## Manual Split Preparation

### Windows Example

```bash
python src/ecg_preprocessing/ecg_data_prepare.py ^
  --dataset_path data ^
  --name ptbl-xl ^
  --fs 250 ^
  --out_root prepared_data ^
  --split_ratio 0.7 0.2 0.1 ^
  --balance_mode train ^
  --flatline_seconds 3.0
```

### macOS / Linux Example

```bash
python src/ecg_preprocessing/ecg_data_prepare.py \
  --dataset_path data \
  --name ptbl-xl \
  --fs 250 \
  --out_root prepared_data \
  --split_ratio 0.7 0.2 0.1 \
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
ptbl-xl
```

The dataset name can be changed if another experiment name is desired.

---

### `--fs`

Sampling frequency.

Example:

```text
250
```

The interactive runner can run preparation for several selected frequencies one after another.

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

This argument is used for k-fold preparation.

---

### `--split_ratio`

Manual train, validation, and test split ratio.

Example:

```text
0.7 0.2 0.1
```

This argument is used for manual split preparation.

---

### `--test_ratio`

Patient-safe hold-out test ratio.

Example:

```text
0.2
```

In k-fold mode, this creates an optional hold-out test set under:

```text
test/test.pt
```

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

# 14. Prepared Data Output Structure

## K-fold Structure

After k-fold data preparation, the output structure should look like this:

```text
prepared_data/ptbl-xl/250hz/
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
prepared_data/ptbl-xl/
├── 62hz/
├── 100hz/
├── 250hz/
└── 500hz/
```

Each frequency folder contains its own prepared files.

---

## Manual Split Structure

After manual split data preparation, the output structure should look like this:

```text
prepared_data/ptbl-xl/250hz/
├── train.pt
├── val.pt
├── test.pt
└── samples_250hz.csv
```

If an additional hold-out test is created, the folder can also contain:

```text
test/
└── test.pt
```

---

# 15. Manual Training

The interactive runner is recommended, but training can also be run manually.

## K-fold Training

### Windows Example

```bash
python src/train.py ^
  --data_path prepared_data\ptbl-xl\250hz ^
  --model cnn1d ^
  --split_mode kfold ^
  --train_balance downsample ^
  --batch_size 8 ^
  --epochs 50 ^
  --lr 0.001 ^
  --kfolds 5 ^
  --early_stopping_patience 10 ^
  --device cuda
```

### macOS / Linux Example

```bash
python src/train.py \
  --data_path prepared_data/ptbl-xl/250hz \
  --model cnn1d \
  --split_mode kfold \
  --train_balance downsample \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.001 \
  --kfolds 5 \
  --early_stopping_patience 10 \
  --device cuda
```

---

## Manual Split Training

### Windows Example

```bash
python src/train.py ^
  --data_path prepared_data\ptbl-xl\250hz ^
  --model cnn1d ^
  --split_mode manual ^
  --train_balance downsample ^
  --batch_size 8 ^
  --epochs 50 ^
  --lr 0.001 ^
  --early_stopping_patience 10 ^
  --device cuda
```

### macOS / Linux Example

```bash
python src/train.py \
  --data_path prepared_data/ptbl-xl/250hz \
  --model cnn1d \
  --split_mode manual \
  --train_balance downsample \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.001 \
  --early_stopping_patience 10 \
  --device cuda
```

---

## Auto Split Training

Auto mode lets `train.py` detect whether the frequency folder contains k-fold files or manual split files.

```bash
python src/train.py \
  --data_path prepared_data/ptbl-xl/250hz \
  --model cnn1d \
  --split_mode auto \
  --train_balance downsample \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.001 \
  --kfolds 5 \
  --early_stopping_patience 10 \
  --device auto
```

---

## Test-Only Evaluation

```bash
python src/train.py \
  --data_path prepared_data/ptbl-xl/250hz \
  --model cnn1d \
  --split_mode auto \
  --train_balance downsample \
  --batch_size 8 \
  --epochs 1 \
  --lr 0.001 \
  --kfolds 5 \
  --early_stopping_patience 10 \
  --device auto \
  --test_only
```

---

## Training Arguments

### `--data_path`

Path to the prepared data folder for one sampling frequency.

Example:

```text
prepared_data/ptbl-xl/250hz
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

### `--split_mode`

Training split mode.

Options:

```text
auto
kfold
manual
```

Use `auto` when the prepared data folder should be detected automatically.

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

This is mainly used when the split mode is `kfold` or `auto`.

---

### `--early_stopping_patience`

Number of epochs without validation improvement before early stopping.

Example:

```text
10
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

### `--test_only`

Runs final evaluation using existing checkpoints without training new models.

---

# 16. Checkpoint Output Structure

After training, model checkpoints are saved under:

```text
checkpoints/<dataset_name>/<rate>hz/<model>/
```

For example:

```text
checkpoints/ptbl-xl/250hz/cnn1d/
```

The dataset name is inferred from the parent folder of `data_path`.

---

## K-fold Checkpoints

K-fold training creates one checkpoint folder per fold:

```text
checkpoints/ptbl-xl/250hz/cnn1d/
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

Each fold stores:

* `best.pt` — best model checkpoint
* `last.pt` — final model checkpoint
* `metrics.txt` — training and validation metrics
* `roc_val.npz` — validation ROC data

---

## Manual Split Checkpoints

Manual split training saves the best model under:

```text
checkpoints/ptbl-xl/250hz/cnn1d/manual_split/best.pt
```

---

# 17. Testing

If the prepared data contains a test file, the training script can run final test evaluation.

The runner detects test data in either of these locations:

```text
test/test.pt
```

or:

```text
test.pt
```

In k-fold mode, the final test evaluation can use the trained fold checkpoints.

In manual mode, the final test evaluation uses the manual split checkpoint.

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

# 18. Expected Workflow for Examiners

A simple workflow for checking the project is:

## Windows

```bash
git clone <repo>
cd SEARCH_AF_detection_OsloMet_BachelorGroup
python setup_env.py
venv\Scripts\activate
python src\main.py
```

## macOS / Linux

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

For the main thesis experiment, use:

```text
Sampling frequencies: 62 Hz, 100 Hz, 250 Hz, 500 Hz
Split mode: auto or kfold
Preprocessing balance mode: train
Training balance: downsample
Model: cnn1d or cnn_lstm
Action: Train and run final test evaluation
```

---

# 19. Important Notes

* Patient-safe splitting is enforced.
* No patient should appear in more than one split.
* ECG segments inherit the split of their patient.
* Validation and test data should remain natural/unbalanced for the main experiment.
* Training can be balanced using downsampling.
* Sampling frequencies must be integer values.
* Frequencies above 500 Hz are not supported by the interactive runner.
* The 500 Hz setting requires more memory and compute time.
* CNN-LSTM usually needs a smaller batch size than CNN1D.
* K-fold mode trains one model per fold.
* Manual mode trains one model using `train.pt` and `val.pt`.
* Test-only mode requires existing checkpoints.
* The interactive `main.py` runner is the easiest way to run the project.
* Manual commands are provided for reproducibility and advanced use.

---

# 20. Summary

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

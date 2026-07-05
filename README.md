# Breath Sound Classification

This repository contains a deep learning pipeline for classifying respiratory audio signals into Inhale (I) and Exhale (E) classes.
The project was developed for a Kaggle audio classification competition. Given `.wav` files of breath sounds, the goal is to build a binary classification model that predicts whether each sound corresponds to an inhale or an exhale.

---

## Competition Overview

The objective of the competition is to analyze respiratory sound audio files and classify each sample as one of the following two classes:

| Label | Description |
|---|---|
| `I` | Inhale |
| `E` | Exhale |

The official evaluation metric is Accuracy, which measures the proportion of correctly classified samples.

The final submission file must follow this format:

```csv
ID,Target
steth_20180814_09_37_33_E_010,E
steth_20180814_09_38_51_E_006,E
steth_20180814_10_52_40_E_007,E
steth_20180814_10_53_20_I_003,I
```

---

## Pipeline Architecture

The overall pipeline is shown below.

![Breath Sound Classification Pipeline](image/breath_pipeline.png)

The model pipeline consists of the following steps:

1. Load metadata and `.wav` audio files.
2. Load audio using `librosa` with a sampling rate of 16 kHz.
3. Apply waveform-level augmentation during training.
4. Convert each audio signal into a Mel-spectrogram.
5. Apply SpecAugment during training.
6. Extract features using a residual CNN backbone.
7. Use adaptive pooling and a classifier head.
8. Predict either `Inhale (I)` or `Exhale (E)`.

---

## Repository Structure

```text
.
├── README.md
├── config.py
├── data.py
├── engine.py
├── main.py
├── model.py
├── predict.py
├── utils.py
├── requirements.txt
└── image
    └── breath_pipeline.png
```

| File | Description |
|---|---|
| `config.py` | Defines paths, audio parameters, training hyperparameters, and device settings. |
| `data.py` | Handles audio loading, augmentation, Mel-spectrogram conversion, and DataLoader construction. |
| `model.py` | Defines the residual CNN model, `BreathNet`. |
| `engine.py` | Contains training, validation, mixed precision, early stopping, and model checkpoint logic. |
| `predict.py` | Runs inference on the test set and saves the submission file. |
| `utils.py` | Provides utility functions such as seed fixing, zip extraction, and learning curve visualization. |
| `main.py` | Runs the full training and inference pipeline. |
| `requirements.txt` | Lists the required Python packages. |

---

## Dataset

The competition dataset consists of training audio files, test audio files, and metadata CSV files.

```text
train/
└── *.wav

test/
└── *.wav

train.csv
test.csv
```

### `train.csv`

| Column | Description |
|---|---|
| `ID` | Unique identifier of each training audio sample. |
| `Target` | Ground-truth label, either `I` or `E`. |

### `test.csv`

| Column | Description |
|---|---|
| `ID` | Unique identifier of each test audio sample. |

In the current implementation, the labels are internally mapped as follows:

| Target | Encoded Label |
|---|---|
| `I` | `0` |
| `E` | `1` |

> Note:  
> If the competition dataset uses column names such as `file_name` and `label` instead of `ID` and `Target`, update the corresponding column references in `data.py`.

---

## Preprocessing

Each audio file is loaded using `librosa`:

```python
librosa.load(audio_path, sr=16000)
```

The waveform is then converted into a Mel-spectrogram.

| Parameter | Value |
|---|---|
| Sampling rate | `16000` |
| Number of Mel bins | `128` |
| Hop length | `256` |
| Maximum time length | `64` |

The final input tensor shape is:

```text
1 × 128 × 64
```

This means each audio sample is treated as a single-channel spectrogram image before being passed into the CNN model.

---

## Data Augmentation

Data augmentation is applied only to the training set. Validation and test data are not augmented.

### Waveform-level Augmentation

The following waveform-level augmentations are used:

| Augmentation | Description |
|---|---|
| Add noise | Adds small random noise to the waveform. |
| Time stretch | Slightly stretches or compresses the waveform along the time axis. |
| Pitch shift | Slightly shifts the pitch of the audio signal. |

Each augmentation is applied with a probability of `0.3`.

```python
if random.random() < 0.3:
    y = add_noise(y)

if random.random() < 0.3:
    y = time_stretch(y)

if random.random() < 0.3:
    y = pitch_shift(y)
```

### Spectrogram-level Augmentation

After converting the audio into a Mel-spectrogram, SpecAugment is applied.

| Augmentation | Description |
|---|---|
| Time mask | Masks a small region along the time axis. |
| Frequency mask | Masks a small region along the frequency axis. |

The default masking parameters are:

```python
time_mask = 4
freq_mask = 5
```

---

## Model Architecture

The model used in this project is `BreathNet`, a residual CNN designed for Mel-spectrogram-based binary classification.

```text
Input: 1 × 128 × 64

ResidualBlock(1 → 16)
MaxPool2d

ResidualBlock(16 → 32)
MaxPool2d

ResidualBlock(32 → 64)
MaxPool2d

ResidualBlock(64 → 128)
MaxPool2d

AdaptiveAvgPool2d(1, 1)

Flatten
Linear(128 → 64)
BatchNorm1d
ReLU
Dropout(0.3)
Linear(64 → 2)
```

Each residual block contains two convolution layers and a skip connection.

```text
Conv2d
BatchNorm2d
ReLU
Conv2d
BatchNorm2d
Skip Connection
ReLU
```

The model outputs two logits:

| Output Index | Class |
|---|---|
| `0` | `I`, Inhale |
| `1` | `E`, Exhale |

---

## Training Strategy

The training pipeline uses the following components:

| Component | Setting |
|---|---|
| Loss function | `CrossEntropyLoss` |
| Optimizer | `AdamW` |
| Learning rate scheduler | `CosineAnnealingLR` |
| Validation split | Stratified train/validation split |
| Mixed precision | Enabled when CUDA is available |
| Early stopping | Based on validation loss |
| Model checkpoint | Saves the model with the lowest validation loss |

The default training configuration is defined in `config.py`.

```python
batch_size = 64
epochs = 50
lr = 1e-3
weight_decay = 1e-4
patience = 5
val_size = 0.2
threshold = 0.5
```

Although the official competition metric is Accuracy, the validation loop also reports AUC as an additional monitoring metric.


## Key Features

- Binary classification of respiratory audio signals.
- Mel-spectrogram-based CNN input representation.
- Waveform-level augmentation.
- SpecAugment for spectrogram-level regularization.
- Residual CNN backbone.
- Mixed precision training support.
- Stratified train/validation split.
- Validation-loss-based early stopping.
- Automatic Kaggle submission file generation.

---

## Future Improvements

Possible future improvements include:

- Applying K-fold cross validation.
- Tuning the classification threshold.
- Using stronger audio backbones.
- Adding MFCC, chroma, or other acoustic features.
- Applying test-time augmentation.
- Building an ensemble of multiple models.
- Using weighted loss if class imbalance is significant.
- Experimenting with pretrained audio models.

---


The goal of the competition is to classify `.wav` respiratory audio files into two classes: **Inhale (I)** and **Exhale (E)**.

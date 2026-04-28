# CNN + Binary Input — Malware D3FEND Action Recommendation

A Convolutional Neural Network pipeline that predicts MITRE D3FEND defensive actions from malware behavioral features. Input features (ATT&CK TTPs, MBC codes, CAPE signatures) are encoded as **binary (0/1)** vectors, reshaped into 2D grayscale images, and fed into a lightweight 3-layer CNN.

## Overview

This model is part of the capstone thesis *"Behavior Similarity Based Malware Response Action Recommendation"* (IAP391, FPT University, Spring 2026). It serves as one of several deep learning pipelines evaluated for mapping malware behavioral evidence to D3FEND defensive actions.

**Key idea:** Malware behavioral features are arranged by type (Signatures → TTPs → MBCs) into a square image. CNN kernels slide over local neighborhoods to detect co-occurrence patterns within and across feature groups — for example, which CAPE signatures tend to appear together in ransomware samples.

### Input Encoding

Each malware sample is represented as a binary vector of ~478 features (after frequency filtering). The vector is reshaped into a 22×22 single-channel image where each pixel is 0 (feature absent) or 1 (feature present). Features are spatially grouped: signatures occupy the top rows, TTPs the middle, and MBCs the bottom.

### Model Architecture

- **Conv Block 1:** Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Conv Block 2:** Conv2d(32→48, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Conv Block 3:** Conv2d(48→48, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(1×1)
- **Classifier:** Flatten → Dropout(0.45) → Dense(96) → ReLU → Dropout(0.225) → Dense(33)

Total trainable parameters: ~40–60K. Output: sigmoid probabilities for 33 D3FEND actions (multi-label).

### Training Details

- **Dataset:** 943 ransomware samples with LLM-generated silver labels (~3.2 actions per sample)
- **Cross Validation:** 5-fold CV with early stopping (patience=25)
- **Loss:** Focal Loss (γ=2.0) with per-action positive weights for class imbalance
- **Optimizer:** AdamW (lr=7e-4, weight_decay=7e-4) with warmup + cosine annealing
- **Regularization:** Feature dropout augmentation (15% of active features per epoch), label smoothing (0.02)
- **Threshold Tuning:** Per-action optimal thresholds searched on aggregated validation predictions
- **Prediction:** Ensemble of 5 fold models (average logits)

## Project Structure

```
CNN_V2/
├── config.py                   # All hyperparameters and path configuration
├── data_loader.py              # Excel/JSON data loading, vocabulary building, CNN dataset
├── action_loader.py            # D3FEND action label loading from Excel
├── model.py                    # MalwareCNN model definition
├── trainer.py                  # K-fold CV training, Focal Loss, threshold tuning
├── train_cnn.py                # Training entry point
├── predict_cnn.py              # Prediction entry point (ensemble of 5 folds)
├── evaluate_cnn.py             # Evaluation vs ground truth (Precision@k, Recall@k, F1@k)
├── visualize_images_binary.py  # Generate 22×22 visualization images
├── d3fend_fetcher.py           # D3FEND API client with local caching
├── logger.py                   # Logging utility (console + file)
├── requirements.txt            # Python dependencies
├── d3fend_cache.json           # Cached D3FEND technique metadata
├── saved_model_cnn/            # Trained model artifacts (after training)
├── output_cnn/                 # Predictions, evaluations, images
└── logs/                       # Training/prediction/evaluation logs
```

## Directory Setup (Data)

Before training, place your data in the following folders:

```
CNN_V2/
├── train_excel/      # One .xlsx file with columns: filename, sha256, techniqueid, mbc
├── train_reports/    # CAPE JSON sandbox reports (one per sample, matching filenames)
├── train_actions/    # One .xlsx file with columns: Report, SHA256, Action
├── new_excel/        # Excel for new/unseen samples (same format as train_excel)
├── new_reports/      # CAPE JSON reports for new samples
└── ground_truth/     # (Optional) Excel with ground truth actions for evaluation
```

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_cnn.py
```

Outputs saved to `saved_model_cnn/`: 5 fold model weights (`.pt`), vocabulary, action vocab, feature types, per-action thresholds, metrics, and model config.

### 3. Predict on new samples

```bash
python predict_cnn.py
```

Outputs `output_cnn/predictions_cnn.xlsx` with up to 5 ranked D3FEND actions per sample, including confidence scores and D3FEND categories.

### 4. Evaluate predictions

```bash
python evaluate_cnn.py
```

Computes Precision@k, Recall@k, F1@k for k=1..5 against ground truth. Outputs `output_cnn/evaluation_cnn.xlsx`.

### 5. Visualize feature images

```bash
python visualize_images_binary.py
```

Generates 22×22 binary heatmap images for 20 training + 20 test samples, plus 2×3 comparison grids.

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0
- pandas, openpyxl, numpy, scikit-learn, tqdm, requests, matplotlib (for visualization)

## Evaluation Metrics

Following thesis Section 4.2.2:

- **P@k** = |predicted_top_k ∩ ground_truth| / k
- **R@k** = |predicted_top_k ∩ ground_truth| / |ground_truth|
- **F1@k** = harmonic mean of P@k and R@k

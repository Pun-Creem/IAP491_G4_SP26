# MLP — Malware D3FEND Action Recommendation

A Multi-Layer Perceptron (MLP) pipeline that predicts MITRE D3FEND defensive actions from malware behavioral features. Input features (ATT&CK TTPs, MBC codes, CAPE signatures) are encoded as a flat **binary vector** and passed through a 3-layer MLP with BatchNorm.

> **Note:** This repository also contains a Transformer Encoder model (`train_transformer.py`, `predict_transformer.py`, `evaluate_transformer.py`). However, the MLP is the primary model of interest — the Transformer is included for reference but is not the focus of this pipeline.

## Overview

This model is part of the capstone thesis *"Behavior Similarity Based Malware Response Action Recommendation"* (IAP391, FPT University, Spring 2026). The MLP serves as a lightweight deep learning baseline that treats each feature independently (no interaction modeling), relying on its hidden layers to learn non-linear mappings from behavioral features to defensive actions.

### Input Encoding

Each malware sample is represented as a binary vector of ~480 dimensions (vocabulary size with special tokens). For each feature in the vocabulary, the corresponding position is 1 if present in the sample, 0 otherwise. Unlike the Transformer variant (which uses token sequences with embeddings), the MLP operates directly on this fixed-length binary vector.

### MLP Architecture

```
Input (vocab_size ≈ 480)
  → Linear(480, 128) → ReLU → Dropout(0.3) → BatchNorm
  → Linear(128, 64)  → ReLU → Dropout(0.3)
  → Linear(64, 33)   → Sigmoid (per-action probability)
```

Total parameters: ~30K. Output: sigmoid probabilities for 33 D3FEND actions (multi-label classification).

### Training Details

- **Dataset:** 944 ransomware samples with LLM-generated silver labels (~3.2 actions per sample)
- **Cross Validation:** 5-fold CV with early stopping (patience=25)
- **Loss:** Focal Loss (γ=2.0) with per-action positive weights for class imbalance
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4) with warmup (5 epochs) + cosine annealing
- **Regularization:** Feature dropout augmentation (15%), label smoothing (0.02), gradient clipping (max_norm=1.0)
- **Threshold Tuning:** Per-action and global optimal thresholds searched across folds
- **Model Selection:** Best fold model (by tuned F1) is saved; thresholds averaged across folds

## Project Structure

```
DL_V5_MLP/
├── config.py                    # All hyperparameters (MLP + Transformer sections)
├── data_loader.py               # Data loading, vocabulary, MalwareBinaryDataset (MLP) + MalwareDataset (Transformer)
├── action_loader.py             # D3FEND action label loading from Excel
├── model.py                     # MLPBaseline + MalwareActionPredictor (Transformer)
├── trainer.py                   # Training logic for both MLP and Transformer (K-fold CV)
├── train_mlp.py                 # MLP training entry point
├── predict_mlp.py               # MLP prediction entry point
├── evaluate_mlp.py              # MLP evaluation vs ground truth
├── train_transformer.py         # (Reference) Transformer training entry point
├── predict_transformer.py       # (Reference) Transformer prediction entry point
├── evaluate_transformer.py      # (Reference) Transformer evaluation
├── d3fend_fetcher.py            # D3FEND API client with local caching
├── logger.py                    # Logging utility (console + file)
├── requirements.txt             # Python dependencies
├── d3fend_cache.json            # Cached D3FEND technique metadata
├── saved_model_mlp/             # Trained MLP model artifacts
│   ├── model.pt                 # Best fold model weights
│   ├── model_config.json        # Architecture config (input_dim, hidden_dim, etc.)
│   ├── vocab.json               # Feature vocabulary
│   ├── feature_types.json       # Feature type mapping (TTP/MBC/SIG)
│   ├── action_vocab.json        # Action vocabulary + action info
│   ├── thresholds.json          # Per-action + global thresholds
│   └── training_results.json    # CV metrics
├── saved_model/                 # (Reference) Transformer model artifacts
├── output_mlp/                  # MLP predictions and evaluations
├── output/                      # (Reference) Transformer predictions
└── logs/                        # Training/prediction/evaluation logs
```

## Directory Setup (Data)

Before training, place your data in the following folders:

```
DL_V5_MLP/
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

### 2. Train the MLP

```bash
python train_mlp.py
```

Runs 5-fold CV, saves the best model to `saved_model_mlp/`. Logs fold-by-fold F1 scores and overall mean ± std.

### 3. Predict on new samples

```bash
python predict_mlp.py
```

Outputs `output_mlp/predictions_mlp.xlsx` with up to 5 ranked D3FEND actions per sample, including confidence scores and D3FEND tactic categories (Detect, Isolate, Evict, Restore, Harden, Model).

### 4. Evaluate predictions

```bash
python evaluate_mlp.py
```

Computes Precision@k, Recall@k, F1@k for k=1..5 against ground truth. Outputs `output_mlp/evaluation_mlp.xlsx` with Summary and Per Sample sheets.

## Transformer (Reference)

The repository also includes a Transformer Encoder model for comparison:

```bash
python train_transformer.py      # Train Transformer with 5-fold CV
python predict_transformer.py    # Predict with Transformer
python evaluate_transformer.py   # Evaluate Transformer predictions
```

The Transformer uses token embeddings + type embeddings + self-attention + attention pooling. It is heavier (~200K+ params) and designed for learning feature interactions, but may overfit on this small dataset.

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0
- pandas, openpyxl, numpy, scikit-learn, tqdm, requests

## Evaluation Metrics

Following thesis Section 4.2.2:

- **P@k** = |predicted_top_k ∩ ground_truth| / k
- **R@k** = |predicted_top_k ∩ ground_truth| / |ground_truth|
- **F1@k** = harmonic mean of P@k and R@k

All metrics are macro-averaged across matched test samples.

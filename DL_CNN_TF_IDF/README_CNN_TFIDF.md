# CNN + TF-IDF — Malware D3FEND Action Recommendation

A Convolutional Neural Network pipeline that predicts MITRE D3FEND defensive actions from malware behavioral features. Unlike the binary CNN variant, this model applies **TF-IDF weighting** to the feature matrix before reshaping into 2D images — giving the CNN continuous-valued pixel intensities where rare/discriminative features have higher weight.

## Overview

This model is part of the capstone thesis *"Behavior Similarity Based Malware Response Action Recommendation"* (IAP391, FPT University, Spring 2026). It extends the binary CNN by adding two preprocessing steps: TF-IDF transformation and feature category weighting, producing richer input representations for the convolutional layers.

### Input Encoding Pipeline

1. **Binary encoding:** Each sample → binary vector (~478 features, 0/1)
2. **TF-IDF transformation:** `sklearn.TfidfTransformer` converts binary matrix to continuous TF-IDF values. Rare features (low document frequency) get higher weights; common features are down-weighted. L2 normalization is applied per sample.
3. **Category weighting:** TF-IDF values are scaled by feature type weights:
   - Signatures: ×1.0 (dominant discriminative features)
   - TTPs: ×0.0 (zeroed out based on thesis grid search findings)
   - MBCs: ×0.0 (zeroed out based on thesis grid search findings)
4. **2D reshape:** Weighted vector → 22×22 single-channel image (features grouped: SIG → TTP → MBC)

The category weights were determined by an exhaustive grid search over all 66 combinations of (w_sig, w_ttp, w_mbc) summing to 1.0 with step 0.1 — see `grid_search_weights.py`.

### Model Architecture

- **Conv Block 1:** Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Conv Block 2:** Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Conv Block 3:** Conv2d(64→64, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(1×1)
- **Classifier:** Flatten → Dropout(0.4) → Dense(96) → ReLU → Dropout(0.2) → Dense(33)

Total trainable parameters: ~60–80K. Output: sigmoid probabilities for 33 D3FEND actions (multi-label).

### Training Details

- **Dataset:** 943 ransomware samples with LLM-generated silver labels (~3.2 actions per sample)
- **Cross Validation:** 5-fold CV with early stopping (patience=25)
- **Loss:** Focal Loss (γ=2.0) with per-action positive weights for class imbalance
- **Optimizer:** AdamW (lr=7e-4, weight_decay=7e-4) with warmup (5 epochs) + cosine annealing
- **Regularization:** Feature dropout augmentation (5% — lower than binary CNN because TF-IDF values carry meaningful magnitude), label smoothing (0.02)
- **Threshold Tuning:** Per-action optimal thresholds searched on aggregated validation predictions
- **Prediction:** Ensemble of 5 fold models (average logits), per-action thresholds, min 1 / max 5 actions

## Project Structure

```
CNN_TFIDF/
├── config.py                    # Hyperparameters, paths, category weights
├── data_loader.py               # Data loading, TF-IDF pipeline, category weighting, CNN dataset
├── action_loader.py             # D3FEND action label loading from Excel
├── model.py                     # MalwareCNN model definition
├── trainer.py                   # K-fold CV training, Focal Loss, threshold tuning, model saving
├── train_cnn.py                 # Training entry point
├── predict_cnn.py               # Prediction entry point (ensemble of 5 folds)
├── evaluate_cnn.py              # Evaluation vs ground truth (Precision@k, Recall@k, F1@k)
├── grid_search_weights.py       # Exhaustive grid search over 66 category weight combinations
├── visualize_images.py          # Generate 22×22 TF-IDF heatmap images
├── compare_action_images.py     # Compare images of samples sharing the same action set
├── d3fend_fetcher.py            # D3FEND API client with local caching
├── logger.py                    # Logging utility (console + file)
├── requirements.txt             # Python dependencies
├── d3fend_cache.json            # Cached D3FEND technique metadata
├── saved_model_cnn/             # Trained model artifacts
│   ├── model_fold_1.pt … model_fold_5.pt   # 5 fold model weights
│   ├── tfidf_transformer.pkl    # Fitted TF-IDF transformer (for prediction)
│   ├── thresholds.npy           # Per-action optimal thresholds
│   ├── vocab.json               # Feature vocabulary
│   ├── feature_types.json       # Feature type mapping (TTP/MBC/SIG)
│   ├── action_vocab.json        # D3FEND action vocabulary
│   ├── action_info.json         # Action metadata (labels, categories)
│   ├── model_config.json        # CNN architecture config
│   └── metrics.json             # Training metrics
├── output_cnn/                  # Predictions, evaluations, grid search results, images
└── logs/                        # All logs with timestamps
```

## Directory Setup (Data)

Before training, place your data in the following folders:

```
CNN_TFIDF/
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

### 2. (Optional) Run grid search for optimal category weights

```bash
python grid_search_weights.py
```

Tests all 66 weight combinations `(w_sig, w_ttp, w_mbc)` where the sum equals 1.0. Results are saved incrementally to `output_cnn/grid_search_results.xlsx`. After completion, update `WEIGHT_SIG`, `WEIGHT_TTP`, `WEIGHT_MBC` in `config.py` with the best values found.

### 3. Train the model

```bash
python train_cnn.py
```

Fits TF-IDF transformer on training data, applies category weights, trains 5-fold CNN ensemble. All artifacts saved to `saved_model_cnn/` including the fitted `tfidf_transformer.pkl`.

### 4. Predict on new samples

```bash
python predict_cnn.py
```

Loads the saved TF-IDF transformer (fitted on training data), applies it to new samples, runs ensemble prediction. Outputs `output_cnn/predictions_cnn.xlsx` with up to 5 ranked D3FEND actions per sample.

### 5. Evaluate predictions

```bash
python evaluate_cnn.py
```

Computes Precision@k, Recall@k, F1@k for k=1..5 against ground truth. Outputs `output_cnn/evaluation_cnn.xlsx`.

### 6. Visualize feature images

```bash
python visualize_images.py
```

Generates 22×22 TF-IDF intensity heatmap images (inferno colormap) for 20 training + 20 test samples, plus 2×3 comparison grids. Pixel brightness reflects TF-IDF × category weight.

### 7. Compare images by action group

```bash
python compare_action_images.py
```

Finds groups of training samples sharing identical D3FEND action sets, generates comparison grids, and performs signature overlap analysis (core signatures, majority signatures, cross-group comparison). Outputs grid images and `signature_analysis.txt`.

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0
- pandas, openpyxl, numpy, scikit-learn, tqdm, requests, joblib, matplotlib (for visualization)

## Evaluation Metrics

Following thesis Section 4.2.2:

- **P@k** = |predicted_top_k ∩ ground_truth| / k
- **R@k** = |predicted_top_k ∩ ground_truth| / |ground_truth|
- **F1@k** = harmonic mean of P@k and R@k

All metrics are macro-averaged across matched test samples.

## Differences from CNN + Binary

| Aspect | CNN + Binary | CNN + TF-IDF (this model) |
|---|---|---|
| Input values | 0 or 1 | Continuous TF-IDF × category weight |
| Rare feature treatment | Same weight as common | Higher weight (IDF boost) |
| Category weighting | None | Grid-searched (w_sig=1.0, w_ttp=0.0, w_mbc=0.0) |
| Augmentation drop rate | 15% | 5% (TF-IDF magnitudes are informative) |
| Saved artifacts | Model weights, vocab, thresholds | Same + `tfidf_transformer.pkl` |
| Extra scripts | — | `grid_search_weights.py`, `compare_action_images.py` |

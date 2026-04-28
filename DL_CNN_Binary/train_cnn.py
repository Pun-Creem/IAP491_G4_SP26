"""
CNN Training Script.

Usage:
    python train_cnn.py

Directory structure required:
    malware-cnn/
    ├── train_excel/        ← Excel file with TTPs + MBCs (943 samples)
    ├── train_reports/      ← CAPE JSON reports (943 files)
    ├── train_actions/      ← Excel file with D3FEND actions
    ├── train_cnn.py
    └── ...

Output:
    saved_model_cnn/        ← Trained models + artifacts for prediction
"""

import os
import sys
import time
import random

import numpy as np
import torch

import config
from logger import setup_logger, get_logger
from data_loader import load_excel, extract_all_reports, build_vocabulary, MalwareCNNDataset
from action_loader import load_action_labels
from trainer import train_kfold, save_model


def set_seed(seed):
    """Fix ALL random sources for fully reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Setup
    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="train_cnn")
    log = get_logger()

    log.info("=" * 60)
    log.info(" CNN Malware Action Recommendation — Training")
    log.info("=" * 60)
    start_time = time.time()

    # ── Step 1: Load Excel data (TTPs + MBCs) ──
    log.info("\n[Step 1] Loading Excel data...")
    samples = load_excel(config.TRAIN_EXCEL_DIR)

    # ── Step 2: Extract signatures from CAPE JSON reports ──
    log.info("\n[Step 2] Extracting signatures from CAPE reports...")
    filenames = [s["filename"] for s in samples]
    all_signatures = extract_all_reports(config.TRAIN_REPORTS_DIR, filenames)

    # ── Step 3: Build vocabulary with frequency filtering ──
    log.info("\n[Step 3] Building feature vocabulary...")
    vocab, feature_types, image_size = build_vocabulary(
        samples, all_signatures,
        min_freq=config.FEATURE_MIN_FREQ,
        max_freq_ratio=config.FEATURE_MAX_FREQ_RATIO,
    )

    # ── Step 4: Load action labels ──
    log.info("\n[Step 4] Loading D3FEND action labels...")
    action_vocab, action_info, labels = load_action_labels(
        config.TRAIN_ACTIONS_DIR, samples
    )

    # ── Step 5: Create CNN dataset ──
    log.info("\n[Step 5] Creating CNN image dataset...")
    dataset = MalwareCNNDataset(
        samples=samples,
        vocab=vocab,
        feature_types=feature_types,
        image_size=image_size,
        all_signatures=all_signatures,
        labels=labels,
    )

    # ── Step 6: Train with K-Fold CV ──
    log.info("\n[Step 6] Training CNN with K-Fold Cross Validation...")
    num_actions = len(action_vocab)
    fold_models, thresholds, metrics = train_kfold(
        dataset, labels, image_size, num_actions, action_vocab
    )

    # ── Step 7: Save model ──
    log.info("\n[Step 7] Saving model...")
    save_model(
        fold_models, vocab, feature_types, image_size,
        action_vocab, action_info, thresholds, metrics
    )

    elapsed = time.time() - start_time
    log.info(f"\n Training complete in {elapsed:.1f}s")
    log.info(f" Mean F1@0.5: {metrics['mean_f1_at_0.5']:.4f} ± {metrics['std_f1_at_0.5']:.4f}")
    log.info(f" Tuned Macro F1: {metrics['tuned_macro_f1']:.4f}")


if __name__ == "__main__":
    main()

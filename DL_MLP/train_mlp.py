"""
MLP Training Script.

Usage:
    python train_mlp.py

Same data pipeline as Transformer, but uses a simpler MLP model.
MLP uses binary feature vectors (1/0) instead of token sequences.
Better suited for small datasets (~944 samples).
"""

import os
import sys
import json
import time

import numpy as np
import torch

import config
from logger import setup_logger
from data_loader import (
    load_excel,
    extract_all_reports,
    build_vocabulary,
    MalwareBinaryDataset,
)
from action_loader import load_action_labels
from trainer import train_kfold_mlp


def main():
    log = setup_logger(mode="train_mlp")

    log.info("=" * 60)
    log.info("MALWARE DL ACTION RECOMMENDATION - MLP TRAINING")
    log.info("=" * 60)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"PyTorch version: {torch.__version__}")

    # ─────────────────────────────────────────────────────
    # STEP 1: Load data
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 1: Loading data")
    log.info(f"{'─'*40}")

    samples = load_excel(config.TRAIN_EXCEL_DIR)

    filenames = [s["filename"] for s in samples]
    all_categorical = extract_all_reports(
        config.TRAIN_REPORTS_DIR, filenames
    )

    # ─────────────────────────────────────────────────────
    # STEP 2: Load D3FEND action labels
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 2: Loading D3FEND action labels")
    log.info(f"{'─'*40}")

    action_vocab, action_info, labels = load_action_labels(
        config.TRAIN_ACTIONS_DIR, samples
    )

    num_actions = len(action_vocab)
    log.info(f"Number of D3FEND actions (output classes): {num_actions}")

    if num_actions == 0:
        log.info("[ERROR] No D3FEND actions found! Check action Excel file.")
        sys.exit(1)

    labels_np = np.array(labels)
    avg_actions_per_sample = labels_np.sum(axis=1).mean()
    log.info(f"Average actions per sample: {avg_actions_per_sample:.1f}")

    # ─────────────────────────────────────────────────────
    # STEP 3: Build vocabulary
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 3: Building vocabulary")
    log.info(f"{'─'*40}")

    vocab, feature_types = build_vocabulary(
        samples, all_categorical, min_freq=3, max_freq_ratio=0.95
    )

    # ─────────────────────────────────────────────────────
    # STEP 4: Create BINARY dataset (for MLP)
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 4: Creating binary dataset (MLP)")
    log.info(f"{'─'*40}")

    dataset = MalwareBinaryDataset(
        samples=samples,
        vocab=vocab,
        feature_types=feature_types,
        all_categorical=all_categorical,
        labels=labels,
    )

    input_dim = dataset.get_input_dim()
    log.info(f"Dataset size: {len(dataset)}")
    log.info(f"Input dim: {input_dim} (vocab={len(vocab)})")
    log.info(f"Num actions: {num_actions}")

    # ─────────────────────────────────────────────────────
    # STEP 5: Train MLP with K-Fold CV
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 5: Training MLP model")
    log.info(f"{'─'*40}")

    best_model, thresholds, cv_results = train_kfold_mlp(
        dataset=dataset,
        input_dim=input_dim,
        num_actions=num_actions,
        labels=labels,
        device=device,
    )

    # ─────────────────────────────────────────────────────
    # STEP 6: Save everything
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 6: Saving MLP model")
    log.info(f"{'─'*40}")

    os.makedirs(config.MLP_SAVED_MODEL_DIR, exist_ok=True)

    # Save model weights
    model_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "model.pt")
    torch.save(best_model.state_dict(), model_path)
    log.info(f"Model saved: {model_path}")

    # Save config
    model_config = {
        "input_dim": input_dim,
        "num_actions": num_actions,
        "hidden_dim": config.MLP_HIDDEN_DIM,
        "dropout": config.MLP_DROPOUT,
    }
    config_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Save vocab
    vocab_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    # Save feature types
    ftypes_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "feature_types.json")
    with open(ftypes_path, "w") as f:
        json.dump(feature_types, f, indent=2)

    # Save action vocabulary and info
    clean_action_info = {}
    for aid, info in action_info.items():
        clean_action_info[aid] = {
            "label": info.get("label", ""),
            "category": info.get("category", "Unknown"),
            "description": info.get("description", ""),
        }
    actions_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "action_vocab.json")
    with open(actions_path, "w") as f:
        json.dump({
            "action_vocab": action_vocab,
            "action_info": clean_action_info,
        }, f, indent=2, ensure_ascii=False)

    # Save thresholds
    thresh_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # Save training results
    results_path = os.path.join(config.MLP_SAVED_MODEL_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(cv_results, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"MLP TRAINING COMPLETE!")
    log.info(f"  Time: {elapsed:.1f} seconds")
    log.info(f"  Mean F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
    log.info(f"  Model saved to: {config.MLP_SAVED_MODEL_DIR}/")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    main()

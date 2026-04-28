"""
CNN Prediction Script.

Usage:
    python predict_cnn.py

Loads saved CNN model and predicts D3FEND actions for new malware.

Directory structure required:
    malware-cnn/
    ├── new_excel/          ← Excel file with TTPs + MBCs (201 samples)
    ├── new_reports/        ← CAPE JSON reports (201 files)
    ├── saved_model_cnn/    ← From train_cnn.py
    └── predict_cnn.py

Output:
    output_cnn/predictions_cnn.xlsx
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch

import config
from logger import setup_logger, get_logger
from data_loader import load_excel, extract_all_reports, MalwareCNNDataset
from model import MalwareCNN


def load_saved_model(device):
    """Load all saved model artifacts."""
    log = get_logger()
    save_dir = config.SAVED_MODEL_DIR

    if not os.path.isdir(save_dir):
        log.error(f"Saved model not found: {save_dir}")
        log.error("Run 'python train_cnn.py' first!")
        sys.exit(1)

    with open(os.path.join(save_dir, "model_config.json")) as f:
        model_config = json.load(f)

    with open(os.path.join(save_dir, "vocab.json")) as f:
        vocab = json.load(f)
        vocab = {k: int(v) for k, v in vocab.items()}

    with open(os.path.join(save_dir, "feature_types.json")) as f:
        feature_types = json.load(f)
        feature_types = {k: int(v) for k, v in feature_types.items()}

    with open(os.path.join(save_dir, "action_vocab.json")) as f:
        action_vocab = json.load(f)

    with open(os.path.join(save_dir, "action_info.json")) as f:
        action_info = json.load(f)

    thresholds = np.load(os.path.join(save_dir, "thresholds.npy"))

    # Load all fold models
    fold_models = []
    for i in range(config.NUM_FOLDS):
        model = MalwareCNN(
            image_size=model_config["image_size"],
            num_actions=model_config["num_actions"],
            channels=model_config["channels"],
            kernel_size=model_config["kernel_size"],
            classifier_dim=model_config["classifier_dim"],
            dropout=model_config["dropout"],
        ).to(device)

        state_dict = torch.load(
            os.path.join(save_dir, f"model_fold_{i+1}.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        fold_models.append(model)

    log.info(f"Loaded {len(fold_models)} fold models from {save_dir}")

    return fold_models, vocab, feature_types, model_config, action_vocab, action_info, thresholds


def predict_ensemble(fold_models, dataloader, device):
    """
    Ensemble prediction: average logits across all fold models.

    Ensemble of K models reduces variance and improves robustness,
    especially important for small training datasets.
    """
    all_logits = []

    for model in fold_models:
        model.eval()
        model_logits = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                logits = model(images)
                model_logits.append(logits.cpu().numpy())
        all_logits.append(np.concatenate(model_logits, axis=0))

    # Average logits across folds
    ensemble_logits = np.mean(all_logits, axis=0)
    return ensemble_logits


def apply_thresholds(logits, thresholds, action_vocab, action_info):
    """
    Apply per-action thresholds and enforce min/max action constraints.

    Returns list of action dicts per sample (sorted by confidence).
    """
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    n_samples = probs.shape[0]
    results = []

    for i in range(n_samples):
        sample_probs = probs[i]

        # Apply per-action threshold
        selected = []
        for j, aid in enumerate(action_vocab):
            if sample_probs[j] >= thresholds[j]:
                info = action_info.get(aid, {})
                selected.append({
                    "d3fend_id": aid,
                    "label": info.get("label", aid),
                    "category": info.get("category", "Unknown"),
                    "confidence": float(sample_probs[j]),
                })

        # Sort by confidence (descending)
        selected.sort(key=lambda x: x["confidence"], reverse=True)

        # Enforce constraints
        if len(selected) > config.MAX_ACTIONS:
            selected = selected[:config.MAX_ACTIONS]

        if len(selected) < config.MIN_ACTIONS:
            # If nothing passed threshold, take the top-1 by probability
            top_idx = np.argmax(sample_probs)
            aid = action_vocab[top_idx]
            info = action_info.get(aid, {})
            selected = [{
                "d3fend_id": aid,
                "label": info.get("label", aid),
                "category": info.get("category", "Unknown"),
                "confidence": float(sample_probs[top_idx]),
            }]

        results.append(selected)

    return results



def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="predict_cnn")
    log = get_logger()

    log.info("=" * 60)
    log.info(" CNN Malware Action Recommendation — Prediction")
    log.info("=" * 60)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f" Device: {device}")

    # ── Step 1: Load saved model ──
    log.info("\n[Step 1] Loading saved model...")
    (fold_models, vocab, feature_types, model_config,
     action_vocab, action_info, thresholds) = load_saved_model(device)

    image_size = model_config["image_size"]

    # ── Step 2: Load new malware data ──
    log.info("\n[Step 2] Loading new malware data...")
    samples = load_excel(config.NEW_EXCEL_DIR)

    # ── Step 3: Extract signatures from CAPE reports ──
    log.info("\n[Step 3] Extracting signatures from CAPE reports...")
    filenames = [s["filename"] for s in samples]
    all_signatures = extract_all_reports(config.NEW_REPORTS_DIR, filenames)

    # ── Step 4: Create dataset ──
    log.info("\n[Step 4] Creating CNN image dataset...")
    dataset = MalwareCNNDataset(
        samples=samples,
        vocab=vocab,
        feature_types=feature_types,
        image_size=image_size,
        all_signatures=all_signatures,
        labels=None,  # no labels for new data
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ── Step 5: Ensemble prediction ──
    log.info("\n[Step 5] Running ensemble prediction...")
    logits = predict_ensemble(fold_models, dataloader, device)
    predictions = apply_thresholds(logits, thresholds, action_vocab, action_info)

    # ── Step 6: Save results ──
    log.info("\n[Step 6] Saving predictions...")
    output_path = os.path.join(config.OUTPUT_DIR, "predictions_cnn.xlsx")

    rows = []
    for i, s in enumerate(samples):
        actions = predictions[i]

        action_strs = []
        for rank, act in enumerate(actions, 1):
            action_strs.append(
                f"#{rank}: {act['d3fend_id']} - {act['label']} "
                f"({act['category']}) [{act['confidence']:.2f}]"
            )

        row = {
            "filename": s["filename"],
            "sha256": s["sha256"],
            "num_ttps": len(s["ttps"]),
            "num_mbcs": len(s["mbcs"]),
            "num_actions_recommended": len(actions),
        }

        for rank in range(config.MAX_ACTIONS):
            if rank < len(actions):
                act = actions[rank]
                row[f"action_{rank+1}_id"] = act["d3fend_id"]
                row[f"action_{rank+1}_name"] = act["label"]
                row[f"action_{rank+1}_category"] = act["category"]
                row[f"action_{rank+1}_confidence"] = round(act["confidence"], 4)
            else:
                row[f"action_{rank+1}_id"] = ""
                row[f"action_{rank+1}_name"] = ""
                row[f"action_{rank+1}_category"] = ""
                row[f"action_{rank+1}_confidence"] = ""

        row["all_actions_summary"] = " | ".join(action_strs)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_excel(output_path, index=False)
    log.info(f" Predictions saved to: {output_path}")

    # Summary
    actions_per_sample = [len(p) for p in predictions]
    elapsed = time.time() - start_time
    log.info(f"\n Prediction complete in {elapsed:.1f}s")
    log.info(f" Samples: {len(samples)}")
    log.info(f" Actions per sample: min={min(actions_per_sample)}, "
             f"max={max(actions_per_sample)}, mean={np.mean(actions_per_sample):.1f}")


if __name__ == "__main__":
    main()

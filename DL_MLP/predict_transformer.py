"""
Prediction Script.

Usage:
    python predict.py

What it does (fully automated):
    1. Loads saved model from saved_model/
    2. Reads new Excel + CAPE JSON reports from data/new/
    3. Extracts features using same pipeline as training
    4. Predicts top 1-5 D3FEND actions per malware
    5. Exports results to output/predictions.xlsx
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
from data_loader import (
    load_excel,
    extract_all_reports,
    encode_sample,
    MalwareDataset,
)
from model import MalwareActionPredictor


def load_saved_model(device):
    """Load all saved artifacts from training."""
    log = get_logger()
    log.info("Loading saved model...")

    # Check saved_model directory exists
    if not os.path.isdir(config.SAVED_MODEL_DIR):
        log.info(f"[ERROR] Saved model not found: {config.SAVED_MODEL_DIR}")
        log.info("  Run 'python train.py' first!")
        sys.exit(1)

    # Load configs
    with open(os.path.join(config.SAVED_MODEL_DIR, "model_config.json")) as f:
        model_config = json.load(f)

    with open(os.path.join(config.SAVED_MODEL_DIR, "vocab.json")) as f:
        vocab = json.load(f)

    with open(os.path.join(config.SAVED_MODEL_DIR, "feature_types.json")) as f:
        feature_types = json.load(f)
        # Convert string keys back to int values
        feature_types = {k: int(v) for k, v in feature_types.items()}

    with open(os.path.join(config.SAVED_MODEL_DIR, "action_vocab.json")) as f:
        action_data = json.load(f)
        action_vocab = action_data["action_vocab"]
        action_info = action_data["action_info"]

    with open(os.path.join(config.SAVED_MODEL_DIR, "thresholds.json")) as f:
        thresholds = json.load(f)

    # Reconstruct model
    model = MalwareActionPredictor(
        vocab_size=model_config["vocab_size"],
        num_actions=model_config["num_actions"],
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
    ).to(device)

    # Load weights
    model_path = os.path.join(config.SAVED_MODEL_DIR, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    log.info(f"[INFO] Model loaded: {model_config['num_actions']} actions, vocab={model_config['vocab_size']}")

    return {
        "model": model,
        "vocab": vocab,
        "feature_types": feature_types,
        "action_vocab": action_vocab,
        "action_info": action_info,
        "thresholds": thresholds,
        "max_length": model_config["max_length"],
    }


def predict_actions(model, dataset, thresholds, action_vocab, action_info, device):
    """
    Run prediction and select top 1-5 actions per sample.

    Uses per-action thresholds when available, falls back to global threshold.
    """
    model.eval()
    all_probs = []

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    with torch.no_grad():
        for batch in dataloader:
            fids = batch["feature_ids"].to(device)
            tids = batch["type_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(fids, tids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    # Apply thresholds and select top actions
    per_action_thresh = thresholds.get("per_action", None)
    global_thresh = thresholds.get("global", 0.5)

    results = []
    for i in range(len(all_probs)):
        probs = all_probs[i]
        sample_actions = []

        for j in range(len(action_vocab)):
            # Use per-action threshold if available, else global
            if per_action_thresh and j < len(per_action_thresh):
                thresh = per_action_thresh[j]
            else:
                thresh = global_thresh

            if probs[j] >= thresh:
                aid = action_vocab[j]
                info = action_info.get(aid, {})
                sample_actions.append({
                    "d3fend_id": aid,
                    "label": info.get("label", "Unknown"),
                    "category": info.get("category", "Unknown"),
                    "confidence": float(probs[j]),
                })

        # Sort by confidence descending
        sample_actions.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply min/max constraints
        if len(sample_actions) == 0:
            # Always recommend at least MIN_ACTIONS (take highest prob even if below threshold)
            top_indices = np.argsort(probs)[::-1][:config.MIN_ACTIONS]
            for j in top_indices:
                aid = action_vocab[j]
                info = action_info.get(aid, {})
                sample_actions.append({
                    "d3fend_id": aid,
                    "label": info.get("label", "Unknown"),
                    "category": info.get("category", "Unknown"),
                    "confidence": float(probs[j]),
                })

        # Limit to MAX_ACTIONS
        sample_actions = sample_actions[:config.MAX_ACTIONS]

        results.append(sample_actions)

    return results


def export_results(samples, predictions, output_path):
    """Export predictions to Excel."""
    log = get_logger()
    rows = []

    for i, (sample, actions) in enumerate(zip(samples, predictions)):
        # Format actions as readable strings
        action_strs = []
        for rank, act in enumerate(actions, 1):
            action_strs.append(
                f"#{rank}: {act['d3fend_id']} - {act['label']} "
                f"({act['category']}) [{act['confidence']:.2f}]"
            )

        row = {
            "filename": sample["filename"],
            "sha256": sample["sha256"],
            "num_ttps": len(sample["ttps"]),
            "num_mbcs": len(sample["mbcs"]),
            "num_actions_recommended": len(actions),
        }

        # Individual action columns
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

        # All actions as single string
        row["all_actions_summary"] = " | ".join(action_strs)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, sheet_name="Predictions")
    log.info(f"[INFO] Results exported: {output_path}")
    return df


def main():
    log = setup_logger(mode="predict")
    log.info("=" * 60)
    log.info("MALWARE DL ACTION RECOMMENDATION - PREDICTION")
    log.info("=" * 60)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[INFO] Device: {device}")

    # ─────────────────────────────────────────────────────
    # STEP 1: Load saved model
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 1: Loading saved model")
    log.info(f"{'─'*40}")

    saved = load_saved_model(device)

    # Enrich action_info with D3FEND categories/descriptions if missing
    action_info = saved["action_info"]
    needs_enrichment = any(
        "category" not in info or info.get("category") == "Unknown"
        for info in action_info.values()
    )
    if needs_enrichment:
        log.info("Enriching action info from D3FEND API (first run or missing data)...")
        from d3fend_fetcher import enrich_action_info
        enrich_action_info(action_info)
        saved["action_info"] = action_info

    # ─────────────────────────────────────────────────────
    # STEP 2: Load new data
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 2: Loading new data")
    log.info(f"{'─'*40}")

    samples = load_excel(config.NEW_EXCEL_DIR)

    filenames = [s["filename"] for s in samples]
    all_categorical = extract_all_reports(
        config.NEW_REPORTS_DIR, filenames
    )

    # ─────────────────────────────────────────────────────
    # STEP 3: Create dataset
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 3: Building features")
    log.info(f"{'─'*40}")

    dataset = MalwareDataset(
        samples=samples,
        vocab=saved["vocab"],
        feature_types=saved["feature_types"],
        all_categorical=all_categorical,
        max_length=saved["max_length"],
    )

    log.info(f"[INFO] New samples: {len(dataset)}")

    # ─────────────────────────────────────────────────────
    # STEP 4: Predict
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 4: Predicting actions")
    log.info(f"{'─'*40}")

    predictions = predict_actions(
        model=saved["model"],
        dataset=dataset,
        thresholds=saved["thresholds"],
        action_vocab=saved["action_vocab"],
        action_info=saved["action_info"],
        device=device,
    )

    # ─────────────────────────────────────────────────────
    # STEP 5: Export results
    # ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*40}")
    log.info("STEP 5: Exporting results")
    log.info(f"{'─'*40}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, "predictions.xlsx")
    df = export_results(samples, predictions, output_path)

    # Print preview
    log.info(f"\n{'─'*40}")
    log.info("PREVIEW (first 5 samples):")
    log.info(f"{'─'*40}")
    for i in range(min(5, len(samples))):
        log.info(f"\n  [{samples[i]['filename']}]")
        for act in predictions[i]:
            log.info(
                f"    → {act['d3fend_id']} - {act['label']} "
                f"({act['category']}) [confidence: {act['confidence']:.2f}]"
            )

    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"PREDICTION COMPLETE!")
    log.info(f"  Samples: {len(samples)}")
    log.info(f"  Time: {elapsed:.1f} seconds")
    log.info(f"  Output: {output_path}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()

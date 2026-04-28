"""
Action Label Loader for CNN pipeline.

Loads D3FEND defensive actions from LLM-generated Excel file.
Each sample has pre-analyzed actions (1-5 per malware).

Excel format:
    Report | SHA256 | Action
    1_report | ce599... | D3-DNSTA - DNS Traffic Analysis\nD3-PT - Process Termination\n...
"""

import os
import glob
from collections import Counter

import pandas as pd
import numpy as np

import config
from logger import get_logger


def find_excel_file(directory):
    pattern = os.path.join(directory, "*.xlsx")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .xlsx file found in {directory}")
    if len(files) > 1:
        get_logger().warning(f"Multiple .xlsx files in {directory}, using: {files[0]}")
    return files[0]


def load_action_labels(actions_dir, samples):
    """
    Load D3FEND action labels and build multi-label binary vectors.

    Args:
        actions_dir: Directory containing the actions Excel file
        samples: List of sample dicts (from load_excel)

    Returns:
        action_vocab: Sorted list of D3FEND action IDs kept after filtering
        action_info: Dict {d3fend_id: {"label": full_name, ...}}
        labels: List of binary lists, one per sample
    """
    log = get_logger()

    excel_path = find_excel_file(actions_dir)
    log.info(f"Loading actions from: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]

    # Build mapping: report_name → list of action IDs
    report_to_actions = {}
    action_info = {}

    for _, row in df.iterrows():
        report = str(row.get("Report", "")).strip()
        if not report.endswith(".json"):
            report = report + ".json"

        raw_actions = str(row.get("Action", ""))
        actions = []
        for line in raw_actions.split("\n"):
            line = line.strip()
            if not line:
                continue
            if " - " in line:
                parts = line.split(" - ", 1)
                aid = parts[0].strip()
                label = parts[1].strip()
            else:
                aid = line.strip()
                label = aid
            actions.append(aid)
            if aid not in action_info:
                action_info[aid] = {"label": label}

        report_to_actions[report] = actions

    log.info(f"Loaded actions for {len(report_to_actions)} reports")

    # Count action frequency and filter rare ones
    action_counter = Counter()
    for acts in report_to_actions.values():
        for a in acts:
            action_counter[a] += 1

    filtered_actions = {
        aid for aid, count in action_counter.items()
        if count >= config.D3FEND_MIN_SAMPLES
    }

    log.info(f"Total unique actions: {len(action_counter)}")
    log.info(f"After filtering (min {config.D3FEND_MIN_SAMPLES} samples): {len(filtered_actions)}")

    action_vocab = sorted(filtered_actions)
    action_to_idx = {aid: idx for idx, aid in enumerate(action_vocab)}

    # Build label vectors
    labels = []
    matched = 0
    unmatched = 0

    for s in samples:
        fname = s["filename"]
        sample_actions = report_to_actions.get(fname, [])

        if not sample_actions:
            unmatched += 1
        else:
            matched += 1

        label_vector = [0] * len(action_vocab)
        for aid in sample_actions:
            if aid in action_to_idx:
                label_vector[action_to_idx[aid]] = 1
        labels.append(label_vector)

    log.info(f"Matched: {matched}/{len(samples)} samples")
    if unmatched > 0:
        log.warning(f"Unmatched: {unmatched} samples (no actions found)")

    # Stats
    labels_np = np.array(labels)
    actions_per_sample = labels_np.sum(axis=1)
    log.info(f"Actions per sample: min={actions_per_sample.min()}, "
             f"max={actions_per_sample.max()}, mean={actions_per_sample.mean():.1f}")

    # Enrich with D3FEND category info
    from d3fend_fetcher import enrich_action_info
    enrich_action_info(action_info)

    # Log distribution
    log.info("Action distribution:")
    for aid in action_vocab:
        count = action_counter[aid]
        label = action_info.get(aid, {}).get("label", "Unknown")
        category = action_info.get(aid, {}).get("category", "Unknown")
        log.info(f"  {aid} ({category}: {label}): {count} samples")

    return action_vocab, action_info, labels

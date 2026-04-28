"""
Compare Malware Images by Action Group.

Finds training samples that share the same D3FEND action set (ground truth),
generates their TF-IDF weighted 22x22 images, saves comparison grids,
and analyzes signature overlap within and between groups.

Usage:
    python compare_action_images.py

Output:
    output_cnn/images/action_group_A.png
    output_cnn/images/action_group_B.png
    output_cnn/images/signature_analysis.txt  -- detailed overlap analysis
"""

import os
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from logger import setup_logger, get_logger
from data_loader import (
    load_excel, extract_all_reports, build_vocabulary,
    build_binary_matrix, fit_tfidf, apply_category_weights
)

# ── CONFIG ──
N_SAMPLES_PER_GROUP = 6
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_actions(action_str):
    """Parse action string into sorted tuple of action IDs."""
    actions = []
    for line in str(action_str).split('\n'):
        line = line.strip()
        if line and ' - ' in line:
            aid = line.split(' - ')[0].strip()
            actions.append(aid)
    return tuple(sorted(actions))


def load_action_groups(actions_dir):
    """Load action file and group samples by action set."""
    from action_loader import find_excel_file
    excel_path = find_excel_file(actions_dir)
    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]

    groups = defaultdict(list)
    for _, row in df.iterrows():
        report = str(row.get("Report", "")).strip()
        if not report.endswith(".json"):
            report = report + ".json"
        action_set = parse_actions(row.get("Action", ""))
        groups[action_set].append(report)

    return groups


def get_action_names(action_str_series, target_set):
    """Get full action names for a target action set."""
    for action_str in action_str_series:
        parsed = parse_actions(action_str)
        if parsed == target_set:
            names = {}
            for line in str(action_str).split('\n'):
                line = line.strip()
                if line and ' - ' in line:
                    parts = line.split(' - ', 1)
                    names[parts[0].strip()] = parts[1].strip()
            return names
    return {aid: aid for aid in target_set}


def vector_to_image(vec, image_size):
    """Reshape feature vector to 2D image."""
    padded = np.zeros(image_size * image_size, dtype=np.float32)
    padded[:len(vec)] = vec
    return padded.reshape(image_size, image_size)


def save_grid(images, titles, save_path, grid_title, vmax_fixed, action_text):
    """Save comparison grid (2 rows x 3 cols) with action info."""
    n = len(images)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    if rows == 1:
        axes = np.array([axes])

    im = None
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < n:
                im = ax.imshow(images[idx], cmap='inferno', vmin=0, vmax=vmax_fixed,
                               interpolation='nearest', aspect='equal')
                ax.set_title(titles[idx], fontsize=10, fontweight='bold', pad=8)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
        cbar.set_label('TF-IDF Intensity', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(grid_title, fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02, action_text, ha='center', fontsize=9,
             style='italic', wrap=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def analyze_signature_overlap(group_binary_vectors, group_names, vocab_idx_to_name, label):
    """
    Analyze which signatures are shared among samples in a group.
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  SIGNATURE ANALYSIS — {label}")
    lines.append(f"{'='*60}")

    n = len(group_binary_vectors)

    # For each sample, get set of active signature indices
    active_sets = []
    for i, vec in enumerate(group_binary_vectors):
        active = set(np.where(vec > 0)[0])
        # Only keep signature indices
        active = set(idx for idx in active if idx in vocab_idx_to_name)
        active_sets.append(active)
        sig_names = [vocab_idx_to_name[idx] for idx in sorted(active)]
        lines.append(f"\n  {group_names[i]}: {len(active)} active signatures")
        for sname in sig_names:
            lines.append(f"    - {sname}")

    # Core signatures: present in ALL 6 samples
    core = active_sets[0]
    for s in active_sets[1:]:
        core = core.intersection(s)
    core_names = [vocab_idx_to_name[idx] for idx in sorted(core)]

    lines.append(f"\n  --- CORE SIGNATURES (present in ALL {n} samples): {len(core)} ---")
    for sname in core_names:
        lines.append(f"    * {sname}")

    # Majority signatures: present in >= 4 of 6 samples
    majority = set()
    for idx in set.union(*active_sets):
        count = sum(1 for s in active_sets if idx in s)
        if count >= 4:
            majority.add(idx)
    majority_only = majority - core
    majority_names = [vocab_idx_to_name[idx] for idx in sorted(majority_only)]

    lines.append(f"\n  --- MAJORITY SIGNATURES (in >= 4 of {n}, excluding core): {len(majority_only)} ---")
    for idx in sorted(majority_only):
        count = sum(1 for s in active_sets if idx in s)
        lines.append(f"    * {vocab_idx_to_name[idx]} ({count}/{n} samples)")

    # Unique signatures: present in only 1 sample
    unique_count = 0
    for idx in set.union(*active_sets):
        count = sum(1 for s in active_sets if idx in s)
        if count == 1:
            unique_count += 1

    lines.append(f"\n  --- SUMMARY ---")
    all_sigs = set.union(*active_sets)
    lines.append(f"  Total unique signatures across {n} samples: {len(all_sigs)}")
    lines.append(f"  Core (all {n}): {len(core)} ({len(core)/len(all_sigs)*100:.0f}%)")
    lines.append(f"  Majority (>= 4/{n}): {len(majority)} ({len(majority)/len(all_sigs)*100:.0f}%)")
    lines.append(f"  Unique (only 1 sample): {unique_count} ({unique_count/len(all_sigs)*100:.0f}%)")

    return lines, core, majority


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="compare_actions")
    log = get_logger()

    log.info("=" * 60)
    log.info(" COMPARE — Malware Images by Action Group")
    log.info("=" * 60)

    img_dir = os.path.join(config.OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ── Step 1: Load action groups ──
    log.info("\n[Step 1] Loading action groups...")
    groups = load_action_groups(config.TRAIN_ACTIONS_DIR)

    candidates = []
    for action_set, reports in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(action_set) == 4 and len(reports) >= N_SAMPLES_PER_GROUP:
            log.info(f"    {len(reports)} samples: {action_set}")
            candidates.append((action_set, reports))

    if len(candidates) < 2:
        log.error("Cannot find enough groups. Exiting.")
        return

    group_a_set, group_a_reports = candidates[0]
    group_b_set, group_b_reports = candidates[1]

    selected_a = random.sample(group_a_reports, N_SAMPLES_PER_GROUP)
    selected_b = random.sample(group_b_reports, N_SAMPLES_PER_GROUP)

    log.info(f"\n  Group A: {group_a_set} ({len(group_a_reports)} samples)")
    log.info(f"  Selected: {selected_a}")
    log.info(f"\n  Group B: {group_b_set} ({len(group_b_reports)} samples)")
    log.info(f"  Selected: {selected_b}")

    # ── Step 2: Load training data ──
    log.info("\n[Step 2] Loading training data...")
    train_samples = load_excel(config.TRAIN_EXCEL_DIR)
    train_filenames = [s["filename"] for s in train_samples]
    train_signatures = extract_all_reports(config.TRAIN_REPORTS_DIR, train_filenames)

    vocab, feature_types, image_size = build_vocabulary(
        train_samples, train_signatures,
        min_freq=config.FEATURE_MIN_FREQ,
        max_freq_ratio=config.FEATURE_MAX_FREQ_RATIO,
    )

    # Build reverse mapping: index → feature name (only signatures)
    vocab_idx_to_name = {}
    for feat, idx in vocab.items():
        if feature_types.get(feat) == config.TYPE_SIG:
            vocab_idx_to_name[idx] = feat

    # ── Step 3: Build features ──
    log.info("\n[Step 3] Building TF-IDF features...")
    train_binary = build_binary_matrix(train_samples, vocab, train_signatures)
    tfidf_transformer, train_tfidf = fit_tfidf(train_binary)
    train_weighted = apply_category_weights(train_tfidf, vocab, feature_types)

    global_vmax = float(train_weighted.max())
    fname_to_idx = {s["filename"]: i for i, s in enumerate(train_samples)}

    # ── Step 4: Load action names ──
    from action_loader import find_excel_file
    action_df = pd.read_excel(find_excel_file(config.TRAIN_ACTIONS_DIR))
    action_names_a = get_action_names(action_df["Action"], group_a_set)
    action_names_b = get_action_names(action_df["Action"], group_b_set)

    # ── Step 5: Generate grids + analyze ──
    log.info("\n[Step 4] Generating grids and analyzing signatures...")

    all_analysis = []
    group_cores = {}

    for group_label, selected, action_set, action_names in [
        ("A", selected_a, group_a_set, action_names_a),
        ("B", selected_b, group_b_set, action_names_b),
    ]:
        images = []
        titles = []
        binary_vectors = []

        for fname in selected:
            idx = fname_to_idx.get(fname)
            if idx is None:
                continue
            img = vector_to_image(train_weighted[idx], image_size)
            images.append(img)
            name = fname.replace(".json", "")
            n_active = int((train_weighted[idx] > 0).sum())
            titles.append(f"{name} ({n_active} active)")
            binary_vectors.append(train_binary[idx])

        # Save grid
        action_lines = [f"{aid} - {action_names.get(aid, aid)}" for aid in action_set]
        action_text = "Shared actions: " + " | ".join(action_lines)
        grid_title = (f"Group {group_label}: {len(selected)} samples sharing "
                      f"{len(action_set)} identical actions "
                      f"(from {len(groups[action_set])} total)")

        save_path = os.path.join(img_dir, f"action_group_{group_label}.png")
        save_grid(images, titles, save_path, grid_title, global_vmax, action_text)
        log.info(f"  Saved: {save_path}")

        # Signature overlap analysis
        sample_names = [f.replace(".json", "") for f in selected]
        analysis_lines, core, majority = analyze_signature_overlap(
            binary_vectors, sample_names, vocab_idx_to_name,
            f"Group {group_label}: {action_set}"
        )
        all_analysis.extend(analysis_lines)
        group_cores[group_label] = core

    # ── Step 6: Compare cores between groups ──
    all_analysis.append(f"\n{'='*60}")
    all_analysis.append(f"  CROSS-GROUP COMPARISON")
    all_analysis.append(f"{'='*60}")

    core_a = group_cores.get("A", set())
    core_b = group_cores.get("B", set())
    shared_core = core_a.intersection(core_b)
    only_a = core_a - core_b
    only_b = core_b - core_a

    all_analysis.append(f"\n  Group A core signatures: {len(core_a)}")
    all_analysis.append(f"  Group B core signatures: {len(core_b)}")
    all_analysis.append(f"  Shared between both cores: {len(shared_core)}")

    if shared_core:
        all_analysis.append(f"\n  Shared core signatures (same in both groups):")
        for idx in sorted(shared_core):
            all_analysis.append(f"    * {vocab_idx_to_name.get(idx, f'idx_{idx}')}")

    if only_a:
        all_analysis.append(f"\n  Core signatures ONLY in Group A ({len(only_a)}):")
        for idx in sorted(only_a):
            all_analysis.append(f"    * {vocab_idx_to_name.get(idx, f'idx_{idx}')}")

    if only_b:
        all_analysis.append(f"\n  Core signatures ONLY in Group B ({len(only_b)}):")
        for idx in sorted(only_b):
            all_analysis.append(f"    * {vocab_idx_to_name.get(idx, f'idx_{idx}')}")

    all_analysis.append(f"\n  --- CONCLUSION ---")
    if len(shared_core) < min(len(core_a), len(core_b)) * 0.5:
        all_analysis.append(f"  The two groups have DISTINCT core signature patterns.")
        all_analysis.append(f"  This supports CNN's ability to differentiate action groups.")
    else:
        all_analysis.append(f"  The two groups share significant core signatures.")
        all_analysis.append(f"  CNN must rely on subtle differences to differentiate.")

    # Save analysis to file
    analysis_path = os.path.join(img_dir, "signature_analysis.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_analysis))

    # Also print to console
    for line in all_analysis:
        log.info(line)

    log.info(f"\n  Analysis saved to: {analysis_path}")

    # ── Summary ──
    log.info(f"\n{'='*60}")
    log.info(f" COMPLETE")
    log.info(f"  Group A: {group_a_set}")
    log.info(f"  Group B: {group_b_set}")
    log.info(f"  Output images: {img_dir}/action_group_A.png, action_group_B.png")
    log.info(f"  Output analysis: {analysis_path}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()

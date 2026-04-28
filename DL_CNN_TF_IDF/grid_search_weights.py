"""
Grid Search for Feature Category Weights.

Usage:
    python grid_search_weights.py

Searches all 66 combinations of (w_sig, w_ttp, w_mbc) with step=0.1
where w_sig + w_ttp + w_mbc = 1.0.

For each combination:
    1. Apply category weights to TF-IDF matrix
    2. Train CNN with 5-fold CV (on 943 training samples)
    3. Record Tuned Macro F1, Micro F1, Sample F1, Precision, Recall

Does NOT use test set (201 samples). Does NOT save models.
Results are saved incrementally (survives crash).

Output:
    output_cnn/grid_search_results.xlsx  — all 66 results sorted by Macro F1
    logs/grid_search_*.log               — full log
"""

import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch

import config
from logger import setup_logger, get_logger
from data_loader import (
    load_excel, extract_all_reports, build_vocabulary,
    build_binary_matrix, fit_tfidf, MalwareCNNDataset
)
from action_loader import load_action_labels
from trainer import train_kfold


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed):
    """Fix ALL random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# WEIGHT COMBINATIONS
# =============================================================================

def generate_weight_combinations(step=0.1):
    """
    Generate all (w_sig, w_ttp, w_mbc) where sum = 1.0.

    With step=0.1: 11+10+9+8+7+6+5+4+3+2+1 = 66 combinations.
    """
    combos = []
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w_sig = round(i * step, 2)
            w_ttp = round(j * step, 2)
            w_mbc = round(k * step, 2)
            combos.append((w_sig, w_ttp, w_mbc))
    return combos


# =============================================================================
# APPLY WEIGHTS (silent version, no logging)
# =============================================================================

def apply_weights_silent(tfidf_matrix, vocab, feature_types, w_sig, w_ttp, w_mbc):
    """Apply category weights to TF-IDF matrix without logging."""
    weight_map = {
        config.TYPE_SIG: w_sig,
        config.TYPE_TTP: w_ttp,
        config.TYPE_MBC: w_mbc,
    }
    vocab_size = len(vocab)
    col_weights = np.ones(vocab_size, dtype=np.float32)
    for feat, idx in vocab.items():
        ftype = feature_types.get(feat, config.TYPE_TTP)
        col_weights[idx] = weight_map.get(ftype, 1.0)

    return tfidf_matrix * col_weights[np.newaxis, :]


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="grid_search")
    log = get_logger()

    log.info("=" * 60)
    log.info(" GRID SEARCH — Feature Category Weights")
    log.info("=" * 60)
    total_start = time.time()

    # ── Step 1: Load data ONCE (not repeated for each combo) ──
    log.info("\n[Step 1] Loading data (one-time)...")

    samples = load_excel(config.TRAIN_EXCEL_DIR)

    filenames = [s["filename"] for s in samples]
    all_signatures = extract_all_reports(config.TRAIN_REPORTS_DIR, filenames)

    vocab, feature_types, image_size = build_vocabulary(
        samples, all_signatures,
        min_freq=config.FEATURE_MIN_FREQ,
        max_freq_ratio=config.FEATURE_MAX_FREQ_RATIO,
    )

    binary_matrix = build_binary_matrix(samples, vocab, all_signatures)
    tfidf_transformer, tfidf_matrix = fit_tfidf(binary_matrix)

    action_vocab, action_info, labels = load_action_labels(
        config.TRAIN_ACTIONS_DIR, samples
    )

    num_actions = len(action_vocab)

    log.info(f"\n Data loaded: {len(samples)} samples, {len(vocab)} features, "
             f"{num_actions} actions, image {image_size}x{image_size}")

    # ── Step 2: Generate all 66 weight combinations ──
    combos = generate_weight_combinations(step=0.1)
    log.info(f"\n[Step 2] Generated {len(combos)} weight combinations (step=0.1, sum=1.0)")

    # ── Step 3: Run grid search ──
    log.info(f"\n[Step 3] Running grid search...")
    log.info(f" Estimated time: ~{len(combos) * 5} minutes")
    log.info(f"{'='*80}")

    results = []
    output_path = os.path.join(config.OUTPUT_DIR, "grid_search_results.xlsx")

    for idx, (w_sig, w_ttp, w_mbc) in enumerate(combos, 1):
        combo_start = time.time()

        log.info(f"\n[Case {idx:2d}/{len(combos)}] "
                 f"w_sig={w_sig:.1f}, w_ttp={w_ttp:.1f}, w_mbc={w_mbc:.1f}")

        # Reset seed for fair comparison
        set_seed(config.RANDOM_SEED)

        # Apply weights to TF-IDF matrix
        weighted_matrix = apply_weights_silent(
            tfidf_matrix, vocab, feature_types, w_sig, w_ttp, w_mbc
        )

        # Create dataset
        dataset = MalwareCNNDataset(
            feature_matrix=weighted_matrix,
            image_size=image_size,
            labels=labels,
        )

        # Train 5-fold CV
        fold_models, thresholds, metrics = train_kfold(
            dataset, labels, image_size, num_actions, action_vocab
        )

        combo_time = time.time() - combo_start

        # Record results
        result = {
            "case": idx,
            "w_sig": w_sig,
            "w_ttp": w_ttp,
            "w_mbc": w_mbc,
            "tuned_macro_f1": metrics["tuned_macro_f1"],
            "tuned_micro_f1": metrics["tuned_micro_f1"],
            "tuned_sample_f1": metrics["tuned_sample_f1"],
            "tuned_precision": metrics["tuned_precision"],
            "tuned_recall": metrics["tuned_recall"],
            "mean_f1_at_0.5": metrics["mean_f1_at_0.5"],
            "std_f1_at_0.5": metrics["std_f1_at_0.5"],
            "time_seconds": round(combo_time, 1),
        }
        results.append(result)

        log.info(f" Result: Macro F1={metrics['tuned_macro_f1']:.4f}, "
                 f"Micro F1={metrics['tuned_micro_f1']:.4f}, "
                 f"P={metrics['tuned_precision']:.4f}, "
                 f"R={metrics['tuned_recall']:.4f} "
                 f"({combo_time:.0f}s)")

        # Save incrementally (survives crash)
        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values("tuned_macro_f1", ascending=False)
        df_sorted.to_excel(output_path, index=False, sheet_name="Grid Search")

        # Show current best
        best = df_sorted.iloc[0]
        log.info(f" Current best: Case {int(best['case'])} "
                 f"(w_sig={best['w_sig']:.1f}, w_ttp={best['w_ttp']:.1f}, "
                 f"w_mbc={best['w_mbc']:.1f}) → Macro F1={best['tuned_macro_f1']:.4f}")

    # ── Step 4: Final summary ──
    total_time = time.time() - total_start

    df_final = pd.DataFrame(results)
    df_final = df_final.sort_values("tuned_macro_f1", ascending=False)
    df_final.to_excel(output_path, index=False, sheet_name="Grid Search")

    log.info(f"\n{'='*80}")
    log.info(f" GRID SEARCH COMPLETE")
    log.info(f"{'='*80}")
    log.info(f" Total time: {total_time/60:.1f} minutes")
    log.info(f" Total combinations tested: {len(combos)}")
    log.info(f"\n Top 10 configurations:")
    log.info(f" {'Case':>5} {'w_sig':>6} {'w_ttp':>6} {'w_mbc':>6} "
             f"{'Macro F1':>10} {'Micro F1':>10} {'Precision':>10} {'Recall':>10}")
    log.info(f" {'-'*70}")

    for _, row in df_final.head(10).iterrows():
        log.info(f" {int(row['case']):5d} {row['w_sig']:6.1f} {row['w_ttp']:6.1f} "
                 f"{row['w_mbc']:6.1f} {row['tuned_macro_f1']:10.4f} "
                 f"{row['tuned_micro_f1']:10.4f} {row['tuned_precision']:10.4f} "
                 f"{row['tuned_recall']:10.4f}")

    best = df_final.iloc[0]
    log.info(f"\n BEST CONFIG:")
    log.info(f"   w_sig = {best['w_sig']:.1f}")
    log.info(f"   w_ttp = {best['w_ttp']:.1f}")
    log.info(f"   w_mbc = {best['w_mbc']:.1f}")
    log.info(f"   Tuned Macro F1 = {best['tuned_macro_f1']:.4f}")
    log.info(f"\n Next steps:")
    log.info(f"   1. Set WEIGHT_SIG={best['w_sig']:.1f}, WEIGHT_TTP={best['w_ttp']:.1f}, "
             f"WEIGHT_MBC={best['w_mbc']:.1f} in config.py")
    log.info(f"   2. Delete saved_model_cnn/ folder")
    log.info(f"   3. Run: python train_cnn.py")
    log.info(f"   4. Run: python predict_cnn.py")
    log.info(f"   5. Run: python evaluate_cnn.py")
    log.info(f"\n Results saved to: {output_path}")


if __name__ == "__main__":
    main()

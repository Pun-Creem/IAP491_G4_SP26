"""
Evaluate CNN Predictions vs Ground Truth.

Usage:
    python evaluate_cnn.py

Input:
    1. output_cnn/predictions_cnn.xlsx  (from predict_cnn.py)
    2. ground_truth/*.xlsx              (LLM-generated ground truth actions)

Output:
    - Console log with Precision@k, Recall@k, F1@k for k=1..5
    - output_cnn/evaluation_cnn.xlsx (Summary + Per Sample sheets)

Metrics follow Section 4.2.2 of the thesis:
    P_k = |A_k ∩ G| / k
    R_k = |A_k ∩ G| / |G|
    F1_k = 2 * P_k * R_k / (P_k + R_k)
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

import config
from logger import setup_logger, get_logger


# =============================================================================
# PATHS
# =============================================================================

GROUND_TRUTH_DIR = config.GROUND_TRUTH_DIR
PRED_PATH = os.path.join(config.OUTPUT_DIR, "predictions_cnn.xlsx")
EVAL_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "evaluation_cnn.xlsx")


# =============================================================================
# LOADING
# =============================================================================

def find_ground_truth_file():
    if not os.path.isdir(GROUND_TRUTH_DIR):
        print(f"[ERROR] Directory not found: {GROUND_TRUTH_DIR}")
        print(f"  Create 'ground_truth/' folder and place the ground truth Excel file inside.")
        sys.exit(1)
    files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.xlsx"))
    if not files:
        print(f"[ERROR] No .xlsx file found in {GROUND_TRUTH_DIR}")
        sys.exit(1)
    return files[0]


def load_ground_truth(log):
    filepath = find_ground_truth_file()
    df = pd.read_excel(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    fname_col = next((c for c in ['filename', 'report'] if c in df.columns), None)
    action_col = next((c for c in ['action', 'actions'] if c in df.columns), None)

    if not fname_col or not action_col:
        log.info("[ERROR] Ground truth must have FILENAME and ACTION columns.")
        sys.exit(1)

    gt_dict = {}
    for _, row in df.iterrows():
        fname = str(row[fname_col]).strip()
        if not fname.endswith('.json'):
            fname += '.json'

        actions = []
        for line in str(row[action_col]).split('\n'):
            line = line.strip()
            if line.startswith('D3-'):
                actions.append(line.split(' - ')[0].strip())
        gt_dict[fname] = actions

    log.info(f"Ground truth: {filepath}")
    log.info(f"  Samples: {len(gt_dict)}")
    counts = [len(v) for v in gt_dict.values()]
    log.info(f"  Actions/sample: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    return gt_dict


def load_predictions(log):
    if not os.path.exists(PRED_PATH):
        log.error(f"Not found: {PRED_PATH}")
        log.error("Run 'python predict_cnn.py' first!")
        sys.exit(1)
    df = pd.read_excel(PRED_PATH)
    log.info(f"Predictions: {PRED_PATH}")
    log.info(f"  Samples: {len(df)}")
    return df


def parse_pred_actions(row):
    """Extract up to 5 predicted action IDs in ranked order."""
    actions = []
    for i in range(1, 6):
        aid = row.get(f'action_{i}_id', '')
        if pd.notna(aid) and str(aid).strip().startswith('D3-'):
            actions.append(str(aid).strip())
    return actions


# =============================================================================
# EVALUATION (matching thesis Section 4.2.2)
# =============================================================================

def evaluate(pred_df, gt_dict, log):
    """
    Compute Precision@k, Recall@k, F1@k for k = 1..5.

    P_k = |A_k ∩ G| / k
    R_k = |A_k ∩ G| / |G|
    F1_k = 2 * P_k * R_k / (P_k + R_k)

    All metrics are averaged across matched test samples.
    """
    K_VALUES = [1, 2, 3, 4, 5]

    all_precision = {k: [] for k in K_VALUES}
    all_recall = {k: [] for k in K_VALUES}
    all_f1 = {k: [] for k in K_VALUES}

    per_sample_rows = []
    matched = 0

    for _, row in pred_df.iterrows():
        fname = row['filename']
        if fname not in gt_dict:
            continue
        matched += 1

        pred_actions = parse_pred_actions(row)
        true_actions = gt_dict[fname]
        true_set = set(true_actions)
        g = len(true_set)

        sample_row = {
            'filename': fname,
            'predicted_actions': ', '.join(pred_actions),
            'true_actions': ', '.join(true_actions),
            'num_predicted': len(pred_actions),
            'num_true': g,
        }

        for k in K_VALUES:
            a_k = set(pred_actions[:k])
            hits = len(a_k & true_set)

            p_k = hits / k if k > 0 else 0
            r_k = hits / g if g > 0 else 0
            f1_k = 2 * p_k * r_k / (p_k + r_k) if (p_k + r_k) > 0 else 0

            all_precision[k].append(p_k)
            all_recall[k].append(r_k)
            all_f1[k].append(f1_k)

            sample_row[f'P@{k}'] = round(p_k, 4)
            sample_row[f'R@{k}'] = round(r_k, 4)
            sample_row[f'F1@{k}'] = round(f1_k, 4)

        per_sample_rows.append(sample_row)

    # ── Print Results ──
    log.info(f"\n{'=' * 60}")
    log.info(f"CNN — EVALUATION RESULTS")
    log.info(f"{'=' * 60}")
    log.info(f"  Matched samples: {matched}")

    log.info(f"\n  {'k':>3}  {'Precision@k':>12}  {'Recall@k':>10}  {'F1@k':>8}")
    log.info(f"  {'-' * 40}")

    summary_rows = []
    best_f1 = 0
    best_k = 1
    for k in K_VALUES:
        avg_p = np.mean(all_precision[k])
        avg_r = np.mean(all_recall[k])
        avg_f1 = np.mean(all_f1[k])

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_k = k

        log.info(f"  {k:>3}  {avg_p:>11.4f}  {avg_r:>10.4f}  {avg_f1:>8.4f}")

        summary_rows.append({
            'k': k,
            'Precision@k': round(avg_p, 4),
            'Recall@k': round(avg_r, 4),
            'F1@k': round(avg_f1, 4),
        })

    log.info(f"\n  Best F1@k: F1@{best_k} = {best_f1:.4f}")

    # ── Per-sample preview ──
    log.info(f"\n  Per-sample preview (first 10, showing k={best_k}):")
    for s in per_sample_rows[:10]:
        log.info(
            f"    {s['filename']}: "
            f"predict=[{s['predicted_actions']}] | "
            f"truth=[{s['true_actions']}] | "
            f"P@{best_k}={s[f'P@{best_k}']:.2f} "
            f"R@{best_k}={s[f'R@{best_k}']:.2f} "
            f"F1@{best_k}={s[f'F1@{best_k}']:.2f}"
        )

    # ── Export to Excel ──
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    df_summary = pd.DataFrame(summary_rows)
    df_per_sample = pd.DataFrame(per_sample_rows)

    with pd.ExcelWriter(EVAL_OUTPUT_PATH, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_per_sample.to_excel(writer, sheet_name='Per Sample', index=False)

    log.info(f"\n  Results exported: {EVAL_OUTPUT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    setup_logger(log_dir=config.LOG_DIR, mode="evaluate_cnn")
    log = get_logger()

    log.info("=" * 60)
    log.info("CNN — EVALUATE PREDICTIONS VS GROUND TRUTH")
    log.info("=" * 60)

    gt_dict = load_ground_truth(log)
    pred_df = load_predictions(log)
    evaluate(pred_df, gt_dict, log)

    log.info(f"\n{'=' * 60}")
    log.info("EVALUATION COMPLETE!")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()

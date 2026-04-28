"""
Auto calculate Precision, Recall, and F1 from ALL action_score files
found in the output/ folder against ground truth.

Usage: python calc_accuracy_recall_auto.py
  - No file selection needed — automatically finds all *_action_scores.csv in output/
  - Results are saved to accuracy&recall/precision_recall_f1_results.csv
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path

import openpyxl

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")
GROUND_TRUTH_PATH = os.path.join(SCRIPT_DIR, "sample_ground_truth", "sample_ground_truth.xlsx")


def extract_action_code(action_str):
    """Extract D3 action code (e.g., 'D3-EDL') from action string."""
    match = re.match(r"(D3-\w+)", action_str.strip())
    return match.group(1) if match else action_str.strip()


def load_ground_truth(path):
    """Load ground truth from xlsx. Returns dict: sha256 -> set of action codes."""
    wb = openpyxl.load_workbook(path)
    ws = wb.active

    gt = {}
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        sha256, actions_str = row[0], row[1]
        if not sha256 or not actions_str:
            continue
        actions = set()
        for line in actions_str.strip().split("\n"):
            code = extract_action_code(line)
            if code:
                actions.add(code)
        if sha256 in gt:
            gt[sha256] = gt[sha256] | actions
        else:
            gt[sha256] = actions
    wb.close()
    return gt


def load_action_scores(path):
    """Load action scores CSV. Returns dict: sha256 -> list of (rank, action_code, score)."""
    scores = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha256 = row["sha256"]
            rank = int(row["rank"])
            code = extract_action_code(row["action"])
            score = float(row["score"])
            if sha256 not in scores:
                scores[sha256] = []
            scores[sha256].append((rank, code, score))

    for sha256 in scores:
        scores[sha256].sort(key=lambda x: x[0])
    return scores


def calculate_metrics_all_k(gt, scores, top_k_levels):
    """Calculate precision, recall, and F1 at multiple top-K levels per sha256."""
    results = {}
    for sha256, predictions in scores.items():
        if sha256 not in gt:
            continue

        gt_actions = gt[sha256]
        row = {}
        for k in top_k_levels:
            pred_actions = {p[1] for p in predictions[:k]}
            hits = pred_actions & gt_actions
            precision = len(hits) / len(pred_actions) if pred_actions else 0
            recall = len(hits) / len(gt_actions) if gt_actions else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            row[f"precision_top{k}"] = round(precision, 4)
            row[f"recall_top{k}"] = round(recall, 4)
            row[f"f1_top{k}"] = round(f1, 4)
        results[sha256] = row
    return results


def find_all_action_scores(output_dir):
    """Find all *_action_scores.csv files in output/ subfolders."""
    pattern = os.path.join(output_dir, "**", "*_action_scores.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def main():
    setup_logging()
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Ground truth not found: {GROUND_TRUTH_PATH}")
        return

    gt = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"Loaded ground truth: {len(gt)} samples")

    # Auto find all action_score files
    files = find_all_action_scores(OUTPUT_DIR)
    if not files:
        print(f"No action_scores files found in: {OUTPUT_DIR}")
        return

    print(f"Found {len(files)} action_score file(s):")
    for f in files:
        print(f"  - {os.path.relpath(f, OUTPUT_DIR)}")

    top_k_levels = [1, 2, 3, 4, 5]

    all_results = []

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"\nProcessing: {fname}")
        scores = load_action_scores(fpath)
        print(f"  Samples in scores: {len(scores)}")

        # --- Hash validation ---
        score_hashes = set(scores.keys())
        gt_hashes = set(gt.keys())
        matched_hashes = score_hashes & gt_hashes
        unmatched_in_scores = score_hashes - gt_hashes
        missing_in_scores = gt_hashes - score_hashes

        print(f"  Hash check:")
        print(f"    Matched with ground truth: {len(matched_hashes)}")
        if unmatched_in_scores:
            print(f"    In scores but NOT in ground truth ({len(unmatched_in_scores)}):")
            for h in sorted(unmatched_in_scores):
                print(f"      - {h}")
        if missing_in_scores:
            print(f"    In ground truth but NOT in scores ({len(missing_in_scores)}):")
            for h in sorted(missing_in_scores):
                print(f"      - {h}")

        if not matched_hashes:
            print(f"  SKIPPED: no matching hashes between scores and ground truth.")
            continue

        # Filter scores to only matched hashes
        scores = {h: v for h, v in scores.items() if h in matched_hashes}

        results = calculate_metrics_all_k(gt, scores, top_k_levels)

        if not results:
            print("  No matching samples found.")
            continue

        # Print summary
        for k in top_k_levels:
            avg_prec = sum(r[f"precision_top{k}"] for r in results.values()) / len(results)
            avg_rec = sum(r[f"recall_top{k}"] for r in results.values()) / len(results)
            avg_f1 = sum(r[f"f1_top{k}"] for r in results.values()) / len(results)
            print(f"  [top_{k}] Avg Precision: {avg_prec:.4f}, Avg Recall: {avg_rec:.4f}, Avg F1: {avg_f1:.4f}")

        # Build rows
        for sha256, metrics in results.items():
            row = {"sha256": sha256}
            row.update(metrics)
            all_results.append(row)

    if not all_results:
        print("No results to save.")
        return

    # Compute single AVERAGE row across all samples
    avg_row = {"sha256": "AVERAGE"}
    for k in top_k_levels:
        avg_row[f"precision_top{k}"] = round(
            sum(r[f"precision_top{k}"] for r in all_results) / len(all_results), 4
        )
        avg_row[f"recall_top{k}"] = round(
            sum(r[f"recall_top{k}"] for r in all_results) / len(all_results), 4
        )
        avg_row[f"f1_top{k}"] = round(
            sum(r[f"f1_top{k}"] for r in all_results) / len(all_results), 4
        )
    all_results.append(avg_row)

    # Save results
    output_path = os.path.join(SCRIPT_DIR, "precision_recall_f1_results.csv")

    fieldnames = ["sha256"]
    for k in top_k_levels:
        fieldnames.append(f"precision_top{k}")
        fieldnames.append(f"recall_top{k}")
        fieldnames.append(f"f1_top{k}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

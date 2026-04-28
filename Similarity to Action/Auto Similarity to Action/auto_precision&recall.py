#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval Pipeline – Run multi-sample pipeline then calculate Precision/Recall/F1
=============================================================================
1. Snapshot existing output/ subfolders
2. Run auto_sim2action_multi pipeline on all samples
3. Find new output subfolders (created by this run)
4. Collect *_action_scores.csv from new folders
5. Calculate Precision, Recall, F1 at top-K = 1..5
6. Save results to precision&recall/precision_recall_f1_results.csv
7. Delete the new output subfolders

Usage:
  python eval_pipeline.py --sample-dir path/to/json_folder/
  python eval_pipeline.py --samples s1.json s2.json ...
  python eval_pipeline.py               (opens GUI picker)

All other flags are forwarded to the pipeline (same as auto_sim2action_multi.py).
"""

import argparse
import csv
import os
import re
import shutil
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import openpyxl

from auto_sim2action import build_parser, load_config, apply_config, run_pipeline

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
GROUND_TRUTH_PATH = SCRIPT_DIR / "precision&recall" / "sample_ground_truth" / "sample_ground_truth.xlsx"
RESULTS_DIR = SCRIPT_DIR / "precision&recall"
RESULTS_BASE = "precision_recall_f1_results"


# ─────────────────────────────────────────────────────────────────────────────
# Sample collection (same as auto_sim2action_multi.py)
# ─────────────────────────────────────────────────────────────────────────────

def pick_json_files() -> list[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    files = filedialog.askopenfilenames(
        title="Select sample JSON reports",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return list(files)


def collect_samples(args: argparse.Namespace) -> list[str]:
    samples = []
    if args.samples:
        samples.extend(args.samples)
    if args.sample_dir:
        d = Path(args.sample_dir)
        if not d.is_dir():
            print(f"[ERROR] --sample-dir is not a valid directory: {d}")
            sys.exit(1)
        found = sorted(d.glob("*.json"))
        if not found:
            print(f"[ERROR] No .json files found in {d}")
            sys.exit(1)
        samples.extend(str(f) for f in found)
    if not samples:
        print("[GUI] Opening file picker for JSON reports...")
        chosen = pick_json_files()
        if not chosen:
            print("No files selected. Exiting.")
            sys.exit(0)
        samples.extend(chosen)
        print(f"[GUI] Selected {len(samples)} file(s)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Precision / Recall / F1 calculation
# ─────────────────────────────────────────────────────────────────────────────

def extract_action_code(action_str):
    match = re.match(r"(D3-\w+)", action_str.strip())
    return match.group(1) if match else action_str.strip()


def load_ground_truth(path):
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


def unique_output_path(base_dir: Path, base_name: str, ext: str = ".csv") -> Path:
    """Return a path that doesn't collide: base_name.csv, base_name_1.csv, ..."""
    candidate = base_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = base_dir / f"{base_name}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    # ── Parse args (same as auto_sim2action_multi.py) ──
    base = build_parser()
    p = argparse.ArgumentParser(
        description="Eval Pipeline – run multi-sample then calculate Precision/Recall/F1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[base],
        add_help=False,
    )
    p.add_argument("--samples",    nargs="+", default=[], help="Paths to multiple sample JSON reports")
    p.add_argument("--sample-dir", default="",            help="Directory containing sample JSON reports")

    args = p.parse_args()
    cfg = load_config()
    if cfg:
        print(f"[config] Loaded config")
    args = apply_config(args, cfg)

    samples = collect_samples(args)
    total = len(samples)

    # ── Snapshot existing output subfolders ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_folders = {f.name for f in OUTPUT_DIR.iterdir() if f.is_dir()}

    # ── Run pipeline for each sample ──
    passed = 0
    failed = []

    print(f"\n{'='*60}")
    print(f" Eval Pipeline – {total} sample(s)")
    print(f"{'='*60}")

    for i, sample_path in enumerate(samples, 1):
        print(f"\n{'─'*60}")
        print(f" [{i}/{total}] {Path(sample_path).name}")
        print(f"{'─'*60}")

        args.sample = sample_path
        args.out_dir = ""

        try:
            run_pipeline(args)
            passed += 1
        except Exception as e:
            print(f"\n[ERROR] Failed on {Path(sample_path).name}: {e}")
            failed.append(Path(sample_path).name)

    print(f"\n{'='*60}")
    print(f" Pipeline Summary: {passed}/{total} succeeded")
    if failed:
        print(f" Failed : {', '.join(failed)}")
    print(f"{'='*60}")

    # ── Find new output folders ──
    current_folders = {f.name for f in OUTPUT_DIR.iterdir() if f.is_dir()}
    new_folders = current_folders - existing_folders

    if not new_folders:
        print("\n[Eval] No new output folders found. Nothing to evaluate.")
        return

    print(f"\n[Eval] New output folders: {len(new_folders)}")

    # ── Collect action_scores.csv from new folders ──
    score_files = []
    for folder_name in sorted(new_folders):
        folder = OUTPUT_DIR / folder_name
        for f in folder.glob("*_action_scores.csv"):
            score_files.append(f)

    if not score_files:
        print("[Eval] No action_scores files found in new output folders.")
        # Cleanup
        for folder_name in new_folders:
            shutil.rmtree(OUTPUT_DIR / folder_name)
            print(f"  Deleted: output/{folder_name}")
        return

    print(f"[Eval] Found {len(score_files)} action_scores file(s)")

    # ── Load ground truth ──
    if not GROUND_TRUTH_PATH.exists():
        print(f"[ERROR] Ground truth not found: {GROUND_TRUTH_PATH}")
        # Cleanup
        for folder_name in new_folders:
            shutil.rmtree(OUTPUT_DIR / folder_name)
        return

    gt = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"[Eval] Loaded ground truth: {len(gt)} samples")

    # ── Calculate metrics ──
    top_k_levels = [1, 2, 3, 4, 5]
    all_results = []

    for fpath in score_files:
        fname = fpath.name
        print(f"\n[Eval] Processing: {fname}")
        scores = load_action_scores(fpath)
        print(f"  Samples in scores: {len(scores)}")

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
            print(f"  SKIPPED: no matching hashes.")
            continue

        scores = {h: v for h, v in scores.items() if h in matched_hashes}
        results = calculate_metrics_all_k(gt, scores, top_k_levels)

        if not results:
            print("  No matching samples found.")
            continue

        for k in top_k_levels:
            avg_prec = sum(r[f"precision_top{k}"] for r in results.values()) / len(results)
            avg_rec = sum(r[f"recall_top{k}"] for r in results.values()) / len(results)
            avg_f1 = sum(r[f"f1_top{k}"] for r in results.values()) / len(results)
            print(f"  [top_{k}] Avg Precision: {avg_prec:.4f}, Avg Recall: {avg_rec:.4f}, Avg F1: {avg_f1:.4f}")

        for sha256, metrics in results.items():
            row = {"sha256": sha256}
            row.update(metrics)
            all_results.append(row)

    # ── Save results ──
    if all_results:
        # AVERAGE row
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

        output_path = unique_output_path(RESULTS_DIR, RESULTS_BASE)

        fieldnames = ["sha256"]
        for k in top_k_levels:
            fieldnames.append(f"precision_top{k}")
            fieldnames.append(f"recall_top{k}")
            fieldnames.append(f"f1_top{k}")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n[Eval] Results saved to: {output_path}")
    else:
        print("\n[Eval] No results to save.")

    # ── Cleanup: delete new output folders ──
    print(f"\n[Cleanup] Removing {len(new_folders)} output folder(s)...")
    for folder_name in sorted(new_folders):
        folder = OUTPUT_DIR / folder_name
        shutil.rmtree(folder)
        print(f"  Deleted: output/{folder_name}")

    print(f"\n{'='*60}")
    print(f" Eval complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

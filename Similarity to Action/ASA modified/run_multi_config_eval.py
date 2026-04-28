#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run precision_recall_from_topk.py multiple times with different W/beta configs.
================================================================================
Optimized: Sample loading, TTP extraction, neighbor reading, and action mapping
run ONCE. Only action score computation reruns per config.

Reads config_eval_list.csv, and for each row:
  1. [First run only] Load samples, top-k neighbors, map TTPs -> actions (cached)
  2. [Each run] Recompute action scores with different W/beta
  3. Evaluate Precision/Recall/F1
  4. Save results as <run_number>.csv

Usage:
  python run_multi_config_eval.py --sample-dir path/to/jsons --topk-dir path/to/sim_folder
  python run_multi_config_eval.py               (opens GUI picker)
"""

import argparse
import csv
import json
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import openpyxl
import pandas as pd

from auto_sim2action import extract_hash, extract_ttps, load_feature_list
from precision_recall_from_topk import (
    load_config,
    read_similarity_file,
    load_action_report,
    map_sample_actions,
    compute_action_scores,
    extract_action_code,
    load_ground_truth,
    calculate_metrics_all_k,
)

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging

SCRIPT_DIR        = Path(__file__).resolve().parent
CONFIG_FILE       = SCRIPT_DIR / "config_eval.json"
CONFIG_LIST       = SCRIPT_DIR / "config_eval_list.csv"
GROUND_TRUTH_PATH = SCRIPT_DIR / "precision&recall" / "sample_ground_truth" / "sample_ground_truth.xlsx"
RESULTS_DIR       = SCRIPT_DIR / "precision&recall"

# CSV field types for config_eval_list.csv
FLOAT_FIELDS = {"W", "beta"}
INT_FIELDS = {"top_k"}


def parse_csv_value(key, value):
    if key in FLOAT_FIELDS:
        return float(value)
    if key in INT_FIELDS:
        return int(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Cached data per sample (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleCache:
    """Pre-loaded data for one sample — reused across all W/beta configs."""
    sha256: str
    json_name: str
    px: Dict[str, int]                          # TTP -> D3FEND action mapping
    neighbors: List[Tuple[str, float]]          # (neighbor_sha256, similarity)


# ─────────────────────────────────────────────────────────────────────────────
# Directory pickers (one-time)
# ─────────────────────────────────────────────────────────────────────────────

def pick_directory(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Load all samples once
# ─────────────────────────────────────────────────────────────────────────────

def load_all_samples(
    json_files: List[Path],
    sim_files: Dict[str, Path],
    action_ttps_path: Path,
) -> Dict[str, SampleCache]:
    """Load and cache sample data. Returns dict: sha256 -> SampleCache."""
    caches: Dict[str, SampleCache] = {}
    skipped = []

    for i, json_path in enumerate(json_files, 1):
        print(f"\n{'─'*60}")
        print(f" [{i}/{len(json_files)}] {json_path.name}")
        print(f"{'─'*60}")

        # Extract sha256
        try:
            with open(json_path, encoding="utf-8", errors="replace") as f:
                report = json.load(f)
            sha256 = extract_hash(report)
        except Exception as e:
            print(f"  [ERROR] Cannot read JSON: {e}")
            skipped.append(json_path.name)
            continue

        if sha256 == "UNKNOWN_HASH":
            print(f"  [WARN] Could not extract sha256, skipping.")
            skipped.append(json_path.name)
            continue

        print(f"  SHA256: {sha256[:20]}...")

        # Find matching similarity file
        if sha256 not in sim_files:
            print(f"  [WARN] No similarity file found for this hash, skipping.")
            skipped.append(json_path.name)
            continue

        # Read pre-computed top-k neighbors
        neighbors = read_similarity_file(sim_files[sha256])
        if not neighbors:
            print(f"  [WARN] Empty similarity file, skipping.")
            skipped.append(json_path.name)
            continue

        print(f"  Top-{len(neighbors)} neighbors loaded")

        # Extract TTPs -> map to D3FEND actions
        sample_ttps = extract_ttps(report)
        print(f"  Sample TTPs: {len(sample_ttps)}")
        px = map_sample_actions(sample_ttps, action_ttps_path)

        caches[sha256] = SampleCache(
            sha256=sha256,
            json_name=json_path.name,
            px=px,
            neighbors=neighbors,
        )

    if skipped:
        print(f"\n  Skipped: {', '.join(skipped)}")

    return caches


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Recompute scores per config & evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    caches: Dict[str, SampleCache],
    hash_to_acts: Dict[str, Set[str]],
    all_actions: List[str],
    W: float,
    beta: float,
    run_id: str,
    top_k: int = None,
) -> bool:
    """Compute action scores for all cached samples, evaluate P/R/F1, save CSV."""
    all_action_scores: Dict[str, List[Tuple[int, str, float]]] = {}

    for sha256, cache in caches.items():
        neighbors = cache.neighbors
        if top_k is not None and top_k < len(neighbors):
            neighbors = neighbors[:top_k]
        scored = compute_action_scores(
            cache.px, neighbors, hash_to_acts, all_actions, W, beta
        )
        all_action_scores[sha256] = [
            (r, extract_action_code(a), s) for r, a, s in scored
        ]

    if not all_action_scores:
        print(f"  [WARNING] No scores produced for run {run_id}")
        return False

    # Load ground truth
    if not GROUND_TRUTH_PATH.exists():
        print(f"  [ERROR] Ground truth not found: {GROUND_TRUTH_PATH}")
        return False

    gt = load_ground_truth(GROUND_TRUTH_PATH)
    matched = set(all_action_scores.keys()) & set(gt.keys())

    if not matched:
        print(f"  [WARNING] No matching samples with ground truth for run {run_id}")
        return False

    filtered_scores = {h: v for h, v in all_action_scores.items() if h in matched}

    top_k_levels = [1, 2, 3, 4, 5]
    results = calculate_metrics_all_k(gt, filtered_scores, top_k_levels)

    if not results:
        return False

    # Print averages
    for k in top_k_levels:
        avg_prec = sum(r[f"precision_top{k}"] for r in results.values()) / len(results)
        avg_rec  = sum(r[f"recall_top{k}"] for r in results.values()) / len(results)
        avg_f1   = sum(r[f"f1_top{k}"] for r in results.values()) / len(results)
        print(f"  [top_{k}] Precision: {avg_prec:.4f}  Recall: {avg_rec:.4f}  F1: {avg_f1:.4f}")

    # Build result rows
    all_results = []
    for sha256, metrics in results.items():
        row = {"sha256": sha256}
        row.update(metrics)
        all_results.append(row)

    # AVERAGE row
    avg_row = {"sha256": "AVERAGE"}
    for k in top_k_levels:
        avg_row[f"precision_top{k}"] = round(
            sum(r[f"precision_top{k}"] for r in all_results) / len(all_results), 4)
        avg_row[f"recall_top{k}"] = round(
            sum(r[f"recall_top{k}"] for r in all_results) / len(all_results), 4)
        avg_row[f"f1_top{k}"] = round(
            sum(r[f"f1_top{k}"] for r in all_results) / len(all_results), 4)
    all_results.append(avg_row)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{run_id}.csv"

    fieldnames = ["sha256"]
    for k in top_k_levels:
        fieldnames += [f"precision_top{k}", f"recall_top{k}", f"f1_top{k}"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"  [Eval] Saved: {output_path.name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    # ── Read config_eval_list.csv ──
    if not CONFIG_LIST.exists():
        print(f"[ERROR] Config list not found: {CONFIG_LIST}")
        sys.exit(1)

    with open(CONFIG_LIST, newline="", encoding="utf-8") as f:
        config_rows = list(csv.DictReader(f))

    if not config_rows:
        print("[ERROR] config_eval_list.csv is empty.")
        sys.exit(1)

    print(f"[Multi-Config-Eval] Found {len(config_rows)} config(s) in config_eval_list.csv")

    # ── Parse CLI args ──
    parser = argparse.ArgumentParser(
        description="Run precision_recall_from_topk with multiple W/beta configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-dir", default="", help="Directory containing sample JSON reports")
    parser.add_argument("--topk-dir", default="", help="Directory containing top-k similarity CSV/XLSX files")
    args = parser.parse_args()

    # ── Load config_eval.json for data paths ──
    cfg = load_config()
    if cfg:
        print(f"[config] Loaded {CONFIG_FILE.name}")

    # Resolve data paths
    action_ttps_path   = Path(cfg.get("action_ttps",   "data/action_space/action_per_ttps.csv"))
    action_report_path = Path(cfg.get("action_report",  "data/top_k/actionPerReport.xlsx"))
    unique_actions_path = Path(cfg.get("unique_actions", "data/top_k/unique_actions.csv"))
    ttp_unique_path    = Path(cfg.get("ttp_unique",     "data/unique/ttp_unique.csv"))

    # Make relative paths absolute from script dir
    action_ttps_path    = SCRIPT_DIR / action_ttps_path   if not action_ttps_path.is_absolute()   else action_ttps_path
    action_report_path  = SCRIPT_DIR / action_report_path if not action_report_path.is_absolute() else action_report_path
    unique_actions_path = SCRIPT_DIR / unique_actions_path if not unique_actions_path.is_absolute() else unique_actions_path
    ttp_unique_path     = SCRIPT_DIR / ttp_unique_path    if not ttp_unique_path.is_absolute()    else ttp_unique_path

    # ── Select sample directory ──
    sample_dir = args.sample_dir
    if not sample_dir:
        print("[GUI] Select the folder containing sample JSON reports...")
        sample_dir = pick_directory("Select sample JSON folder")
        if not sample_dir:
            print("No folder selected. Exiting.")
            sys.exit(0)
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        print(f"[ERROR] Not a valid directory: {sample_dir}")
        sys.exit(1)

    json_files = sorted(sample_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No .json files found in {sample_dir}")
        sys.exit(1)

    # ── Select top-k similarity directory ──
    topk_dir = args.topk_dir
    if not topk_dir:
        print("[GUI] Select the folder containing top-k similarity files (CSV/XLSX)...")
        topk_dir = pick_directory("Select top-k similarity folder")
        if not topk_dir:
            print("No folder selected. Exiting.")
            sys.exit(0)
    topk_dir = Path(topk_dir)
    if not topk_dir.is_dir():
        print(f"[ERROR] Not a valid directory: {topk_dir}")
        sys.exit(1)

    # Index similarity files by sha256 (filename without extension)
    sim_files: Dict[str, Path] = {}
    for f in topk_dir.iterdir():
        if f.suffix.lower() in (".csv", ".xlsx") and f.is_file():
            sim_files[f.stem] = f

    if not sim_files:
        print(f"[ERROR] No CSV/XLSX files found in {topk_dir}")
        sys.exit(1)

    # ── Check required data files ──
    required = {
        "Action per TTPs":  action_ttps_path,
        "Action per Report": action_report_path,
        "Unique Actions":   unique_actions_path,
        "TTP unique":       ttp_unique_path,
    }
    missing = [(n, p) for n, p in required.items() if not p.exists()]
    if missing:
        print("\n[ERROR] Missing required data files:")
        for n, p in missing:
            print(f"  {n}: {p}")
        sys.exit(1)

    # ── Load shared data (ONE TIME) ──
    print(f"\n{'#'*60}")
    print(f" PHASE 1: Loading shared data & caching samples (one-time)")
    print(f"{'#'*60}")

    print("\n[Data] Loading shared data files...")
    hash_to_acts = load_action_report(action_report_path)
    print(f"  actionPerReport: {len(hash_to_acts)} entries")

    actions_df = pd.read_csv(unique_actions_path)
    all_actions = actions_df["Action"].tolist()
    print(f"  unique_actions: {len(all_actions)} actions")

    ttp_features = load_feature_list(ttp_unique_path)
    print(f"  TTP features: {len(ttp_features)}")

    # ── Cache all samples ──
    print(f"\n[Samples] {len(json_files)} JSON files in {sample_dir.name}")
    print(f"[Top-K]   {topk_dir.name} ({len(sim_files)} files)")

    caches = load_all_samples(json_files, sim_files, action_ttps_path)

    if not caches:
        print("[ERROR] No samples processed successfully.")
        sys.exit(1)

    print(f"\n[PHASE 1 Complete] Cached {len(caches)} sample(s)")

    # ── PHASE 2: For each config, recompute action scores & evaluate ──
    print(f"\n{'#'*60}")
    print(f" PHASE 2: Running {len(config_rows)} W/beta configs")
    print(f"{'#'*60}")

    for i, row in enumerate(config_rows):
        run_id = row.get("run", str(i + 1)).strip()

        # Parse config values (CLI config_eval.json defaults used as fallback)
        W    = float(row.get("W",    cfg.get("W", 0.3)))
        beta = float(row.get("beta", cfg.get("beta", 0.7)))
        top_k_str = row.get("top_k", "").strip()
        top_k = int(top_k_str) if top_k_str else cfg.get("top_k", None)

        print(f"\n{'='*60}")
        print(f" Run {run_id} ({i+1}/{len(config_rows)})")
        print(f"   W={W}  beta={beta}  top_k={top_k if top_k else 'all'}")
        print(f"{'='*60}")

        evaluate_config(caches, hash_to_acts, all_actions, W, beta, run_id, top_k)

    print(f"\n{'='*60}")
    print(f" All {len(config_rows)} runs complete!")
    print(f" Results in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

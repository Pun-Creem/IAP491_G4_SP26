#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Scores from Pre-computed Top-K Similarity
==================================================
Similar to auto_sim2action.py but uses pre-computed top-k similarity files
instead of running the full pipeline (stages 1-5).

For each sample, outputs an action_scores CSV (same format as auto_sim2action.py):
  sha256, rank, action, score

Workflow:
  1. User selects sample folder (JSON reports)
  2. User selects top-k similarity folder (CSV/XLSX files named by sha256)
  3. For each sample:
     a. Extract sha256 & TTPs from JSON report
     b. Find matching similarity file in the chosen folder
     c. Read top-k neighbors (sorted by similarity desc)
     d. Assign neighbor actions from actionPerReport.xlsx
     e. Map sample TTPs -> D3FEND actions (px)
     f. Compute score(a) = W * px(a) + beta * sum(sim_j * yj(a))
     g. Save {sha256}_action_scores.csv to output folder

Config: config_eval.json (only W, beta, and data paths)

Usage:
  python action_from_topk.py
  python action_from_topk.py --sample-dir path/to/jsons --topk-dir path/to/sim_folder
  python action_from_topk.py --sample-dir sample/ --topk-dir "sample top k similarity/report_XGBoost" --out-dir output_topk
"""

import argparse
import csv
import json
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging

# ── Imports from auto_sim2action ──
from auto_sim2action import (
    extract_hash,
    extract_ttps,
    load_feature_list,
)

# ── Shared utilities from precision_recall_from_topk ──
from precision_recall_from_topk import (
    load_config,
    pick_directory,
    read_similarity_file,
    load_action_report,
    map_sample_actions,
    compute_action_scores,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "config_eval.json"
OUTPUT_DIR = SCRIPT_DIR / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Compute action scores from pre-computed Top-K similarity files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-dir", default="", help="Directory containing sample JSON reports")
    parser.add_argument("--topk-dir", default="", help="Directory containing top-k similarity CSV/XLSX files")
    parser.add_argument("--out-dir", default="", help="Output directory (default: output/)")
    parser.add_argument("--W", type=float, default=None, help="Self-weight W")
    parser.add_argument("--beta", type=float, default=None, help="Neighbor weight beta")
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config()
    if cfg:
        print(f"[config] Loaded {CONFIG_FILE.name}")

    # Apply config defaults (CLI overrides config)
    W = args.W if args.W is not None else cfg.get("W", 0.3)
    beta = args.beta if args.beta is not None else cfg.get("beta", 0.7)

    # Resolve data paths
    action_ttps_path = Path(cfg.get("action_ttps", "data/action_space/action_per_ttps.csv"))
    action_report_path = Path(cfg.get("action_report", "data/top_k/actionPerReport.xlsx"))
    unique_actions_path = Path(cfg.get("unique_actions", "data/top_k/unique_actions.csv"))
    ttp_unique_path = Path(cfg.get("ttp_unique", "data/unique/ttp_unique.csv"))

    # Make relative paths absolute from script dir
    action_ttps_path = SCRIPT_DIR / action_ttps_path if not action_ttps_path.is_absolute() else action_ttps_path
    action_report_path = SCRIPT_DIR / action_report_path if not action_report_path.is_absolute() else action_report_path
    unique_actions_path = SCRIPT_DIR / unique_actions_path if not unique_actions_path.is_absolute() else unique_actions_path
    ttp_unique_path = SCRIPT_DIR / ttp_unique_path if not ttp_unique_path.is_absolute() else ttp_unique_path

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

    # ── Output directory ──
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Check required data files ──
    required = {
        "Action per TTPs": action_ttps_path,
        "Action per Report": action_report_path,
        "Unique Actions": unique_actions_path,
        "TTP unique": ttp_unique_path,
    }
    missing = [(n, p) for n, p in required.items() if not p.exists()]
    if missing:
        print("\n[ERROR] Missing required data files:")
        for n, p in missing:
            print(f"  {n}: {p}")
        sys.exit(1)

    # ── Load shared data ──
    print("\n[Data] Loading shared data files...")
    hash_to_acts = load_action_report(action_report_path)
    print(f"  actionPerReport: {len(hash_to_acts)} entries")

    actions_df = pd.read_csv(unique_actions_path)
    all_actions = actions_df["Action"].tolist()
    print(f"  unique_actions: {len(all_actions)} actions")

    ttp_features = load_feature_list(ttp_unique_path)
    print(f"  TTP features: {len(ttp_features)}")

    # ── Print config ──
    print(f"\n{'='*60}")
    print(f" Action Scores from Pre-computed Top-K Similarity")
    print(f"{'='*60}")
    print(f" Samples    : {len(json_files)} JSON files in {sample_dir.name}")
    print(f" Top-K dir  : {topk_dir.name} ({len(sim_files)} files)")
    print(f" Output     : {out_dir}")
    print(f" W={W}  beta={beta}")
    print(f"{'='*60}")

    # ── Process each sample ──
    processed = 0
    skipped = []
    score_files = []

    for i, json_path in enumerate(json_files, 1):
        print(f"\n{'─'*60}")
        print(f" [{i}/{len(json_files)}] {json_path.name}")
        print(f"{'─'*60}")

        # Extract sha256 from JSON
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

        print(f"  Top-{len(neighbors)} neighbors:")
        for j, (n_sha, n_sim) in enumerate(neighbors):
            print(f"    {j+1}. {n_sha[:20]}... | sim={n_sim:.4f}")

        # Extract TTPs from sample -> map to D3FEND actions (px)
        sample_ttps = extract_ttps(report)
        print(f"  Sample TTPs: {len(sample_ttps)}")
        px = map_sample_actions(sample_ttps, action_ttps_path)

        # Compute action scores
        scored = compute_action_scores(px, neighbors, hash_to_acts, all_actions, W, beta)

        # Save action_scores CSV (same format as auto_sim2action.py stage 7)
        stem = json_path.stem
        out_csv = out_dir / f"{stem}_action_scores.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sha256", "rank", "action", "score"])
            for rank, action, score in scored:
                writer.writerow([sha256, rank, action, f"{score:.6f}"])

        score_files.append(out_csv)

        # Print top 10
        print(f"  Top-10 recommended actions:")
        for rank, action, score in scored[:10]:
            print(f"    {rank:2d}. {action:<60s} {score:.4f}")
        print(f"  -> {out_csv.name}")

        processed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f" Summary: {processed}/{len(json_files)} succeeded")
    if skipped:
        print(f" Skipped ({len(skipped)}): {', '.join(skipped)}")
    print(f" Output directory: {out_dir}")
    for sf in score_files:
        print(f"   {sf.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run auto_precision&recall.py multiple times with different weight configs.
==========================================================================
Optimized: Stages 1-4 & 6 (pattern extraction, similarity, merge, action map)
run ONCE per sample. Only Stages 5 & 7 (top-K + action score) rerun per config.

Reads config_list.csv, and for each row:
  1. [First run only] Full pipeline (stages 1-7), cache intermediate files
  2. [Subsequent runs] Reuse cached merged_similarity & mapped_action,
     only recompute top-K (stage 5) and action scores (stage 7)
  3. Evaluate Precision/Recall/F1
  4. Save results as <run_number>.csv

Usage:
  python run_multi_config.py --sample-dir path/to/json_folder/
  python run_multi_config.py --samples s1.json s2.json ...
  python run_multi_config.py               (opens GUI picker)
"""

import argparse
import csv
import json
import shutil
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import openpyxl

from auto_sim2action import (
    build_parser, load_config, apply_config, _resolve_defaults, _check_required_files,
    load_feature_list,
    stage1_extract_patterns,
    stage2_vt_update,
    stage3_similarity,
    stage4_merge_similarity,
    stage5_top_k,
    stage6_map_actions,
    stage7_action_score,
)

# Reuse eval functions from auto_precision&recall
from importlib.util import spec_from_file_location, module_from_spec
_eval_spec = spec_from_file_location(
    "auto_eval", str(Path(__file__).resolve().parent / "auto_precision&recall.py")
)
_eval_mod = module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)

load_ground_truth      = _eval_mod.load_ground_truth
load_action_scores     = _eval_mod.load_action_scores
calculate_metrics_all_k = _eval_mod.calculate_metrics_all_k

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging

SCRIPT_DIR       = Path(__file__).resolve().parent
CONFIG_FILE      = SCRIPT_DIR / "config.json"
CONFIG_LIST      = SCRIPT_DIR / "config_list.csv"
OUTPUT_DIR       = SCRIPT_DIR / "output"
RESULTS_DIR      = SCRIPT_DIR / "precision&recall"
GROUND_TRUTH_PATH = RESULTS_DIR / "sample_ground_truth" / "sample_ground_truth.xlsx"
RESULTS_BASE     = "precision_recall_f1_results"

# CSV field types
BOOL_FIELDS  = {"tfidf_pattern", "no_vt", "keep_process_data"}
INT_FIELDS   = {"top_k"}
FLOAT_FIELDS = {"w_sig", "w_mbc", "w_ttp", "W", "beta"}
STR_FIELDS   = {"metric"}


def parse_csv_value(key, value):
    if key in BOOL_FIELDS:
        return value.strip().upper() in ("TRUE", "1", "YES")
    if key in INT_FIELDS:
        return int(value)
    if key in FLOAT_FIELDS:
        return float(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Cached data per sample (stages 1-4 & 6)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleCache:
    """Intermediate results from stages 1-4 & 6 for one sample."""
    sha256: str
    stem: str
    out_dir: Path
    merged_csv: Path           # Stage 4 output
    mapped_bin_csv: Path       # Stage 6 output
    sim_stem: str              # stem used for similarity files


# ─────────────────────────────────────────────────────────────────────────────
# Input collection (one-time)
# ─────────────────────────────────────────────────────────────────────────────

def collect_samples_once() -> list[str]:
    """Collect input samples ONE TIME. Supports --samples, --sample-dir, or GUI."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--samples",    nargs="+", default=[])
    p.add_argument("--sample-dir", default="")
    args, _ = p.parse_known_args()

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
        print("[GUI] Opening file picker for JSON reports (one-time only)...")
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Select sample JSON reports",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        root.destroy()
        if not files:
            print("No files selected. Exiting.")
            sys.exit(0)
        samples.extend(files)
        print(f"[GUI] Selected {len(samples)} file(s)")

    return [str(Path(s).resolve()) for s in samples]


# ─────────────────────────────────────────────────────────────────────────────
# Build base args (for path resolution / file checks)
# ─────────────────────────────────────────────────────────────────────────────

def build_base_args() -> argparse.Namespace:
    """Parse CLI + config.json to get base args (paths, metric, vt, etc.)."""
    base = build_parser()
    p = argparse.ArgumentParser(parents=[base], add_help=False)
    p.add_argument("--samples",    nargs="+", default=[])
    p.add_argument("--sample-dir", default="")
    args = p.parse_args()
    cfg = load_config()
    if cfg:
        args = apply_config(args, cfg)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1-4 & 6: Run ONCE per sample, return cache
# ─────────────────────────────────────────────────────────────────────────────

def run_stages_1_to_4_and_6(
    sample_path: Path, args: argparse.Namespace
) -> Optional[SampleCache]:
    """Run weight-independent stages once. Returns SampleCache or None on error."""
    stem = sample_path.stem
    tfidf_on = getattr(args, "tfidf_pattern", False)
    folder_name = f"{stem}_tfidf" if tfidf_on else stem
    out_dir = OUTPUT_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_stem = f"{stem}_tfidf" if tfidf_on else stem

    args_copy = argparse.Namespace(**vars(args))
    args_copy.sample = str(sample_path)
    args_copy.out_dir = str(out_dir)
    args_copy = _resolve_defaults(args_copy)
    _check_required_files(args_copy)

    sig_features = load_feature_list(Path(args_copy.sig_unique))
    ttp_features = load_feature_list(Path(args_copy.ttp_unique))
    mbc_features = load_feature_list(Path(args_copy.mbc_unique))

    print(f"\n[Stage 1] Extracting patterns from {sample_path.name}...")
    sig_csv, ttp_csv, mbc_csv, sha256 = stage1_extract_patterns(
        sample_path, sig_features, ttp_features, mbc_features, out_dir
    )

    # Stage 2: VT (optional)
    if args_copy.vt_key and not args_copy.no_vt:
        ttp_csv, mbc_csv = stage2_vt_update(
            sha256, ttp_csv, mbc_csv, args_copy.vt_key, out_dir, stem, sample_path.parent
        )
    else:
        print(f"[Stage 2] Skipped")

    # Stage 3: Similarity
    sig_sim, ttp_sim, mbc_sim = stage3_similarity(
        sig_csv, ttp_csv, mbc_csv,
        Path(args_copy.pattern_sig), Path(args_copy.pattern_ttp), Path(args_copy.pattern_mbc),
        args_copy.metric, out_dir, sim_stem,
        tfidf_pattern=tfidf_on,
    )

    # Stage 4: Merge
    merged_csv = stage4_merge_similarity(sig_sim, ttp_sim, mbc_sim, out_dir, sim_stem)

    # Stage 6: Map actions (weight-independent)
    mapped_bin_csv, _ = stage6_map_actions(
        ttp_csv, Path(args_copy.action_ttps), out_dir, sim_stem
    )

    return SampleCache(
        sha256=sha256,
        stem=stem,
        out_dir=out_dir,
        merged_csv=merged_csv,
        mapped_bin_csv=mapped_bin_csv,
        sim_stem=sim_stem,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 & 7: Rerun per config (weight-dependent)
# ─────────────────────────────────────────────────────────────────────────────

def run_stages_5_and_7(
    cache: SampleCache,
    args: argparse.Namespace,
    w_sig: float, w_mbc: float, w_ttp: float,
    W: float, beta: float, top_k: int,
) -> Optional[Path]:
    """Run only weight-dependent stages using cached data. Returns action_scores path."""
    args_copy = argparse.Namespace(**vars(args))
    args_copy.sample = ""
    args_copy.out_dir = str(cache.out_dir)
    args_copy = _resolve_defaults(args_copy)

    # Stage 5: Top-K (uses w_sig, w_mbc, w_ttp)
    top_k_csv = stage5_top_k(
        cache.merged_csv,
        Path(args_copy.action_report),
        Path(args_copy.unique_actions),
        top_k,
        w_sig, w_mbc, w_ttp,
        cache.out_dir, cache.sim_stem,
    )

    # Stage 7: Action scores (uses W, beta)
    scores_csv = stage7_action_score(
        cache.mapped_bin_csv, top_k_csv,
        W, beta,
        cache.out_dir, cache.sim_stem,
        cache.sha256,
    )

    return scores_csv


# ─────────────────────────────────────────────────────────────────────────────
# Precision / Recall evaluation (adapted from auto_precision&recall.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scores(score_files: List[Path], run_id: str) -> bool:
    """Calculate P/R/F1 and save as <run_id>.csv. Returns True if saved."""
    if not GROUND_TRUTH_PATH.exists():
        print(f"[ERROR] Ground truth not found: {GROUND_TRUTH_PATH}")
        return False

    gt = load_ground_truth(GROUND_TRUTH_PATH)
    top_k_levels = [1, 2, 3, 4, 5]
    all_results = []

    for fpath in score_files:
        scores = load_action_scores(fpath)
        matched = set(scores.keys()) & set(gt.keys())
        if not matched:
            continue
        scores = {h: v for h, v in scores.items() if h in matched}
        results = calculate_metrics_all_k(gt, scores, top_k_levels)
        if not results:
            continue

        for k in top_k_levels:
            avg_prec = sum(r[f"precision_top{k}"] for r in results.values()) / len(results)
            avg_rec  = sum(r[f"recall_top{k}"] for r in results.values()) / len(results)
            avg_f1   = sum(r[f"f1_top{k}"] for r in results.values()) / len(results)
            print(f"  [top_{k}] Precision: {avg_prec:.4f}  Recall: {avg_rec:.4f}  F1: {avg_f1:.4f}")

        for sha256, metrics in results.items():
            row = {"sha256": sha256}
            row.update(metrics)
            all_results.append(row)

    if not all_results:
        return False

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
    # ── Read config_list.csv ──
    if not CONFIG_LIST.exists():
        print(f"[ERROR] Config list not found: {CONFIG_LIST}")
        sys.exit(1)

    with open(CONFIG_LIST, newline="", encoding="utf-8") as f:
        config_rows = list(csv.DictReader(f))

    if not config_rows:
        print("[ERROR] config_list.csv is empty.")
        sys.exit(1)

    print(f"[Multi-Config] Found {len(config_rows)} config(s) in config_list.csv")

    # ── Collect input ONCE ──
    samples = collect_samples_once()
    print(f"[Multi-Config] Input locked: {len(samples)} sample(s)")
    for s in samples:
        print(f"  - {Path(s).name}")

    # ── Build base args from CLI + config.json ──
    args = build_base_args()

    # ── Group configs by (metric, tfidf_pattern) ──
    # These affect Stage 3, so PHASE 1 must run once per unique combination.
    from collections import OrderedDict

    groups: OrderedDict[tuple, list] = OrderedDict()
    for i, row in enumerate(config_rows):
        metric = row.get("metric", getattr(args, "metric", "jaccard")).strip()
        tfidf  = row.get("tfidf_pattern", "").strip().upper() in ("TRUE", "1", "YES") \
                 if "tfidf_pattern" in row and row["tfidf_pattern"].strip() \
                 else getattr(args, "tfidf_pattern", False)
        key = (metric, tfidf)
        groups.setdefault(key, []).append((i, row))

    print(f"[Multi-Config] {len(groups)} unique (metric, tfidf) group(s):")
    for (m, t), rows in groups.items():
        print(f"  - metric={m}, tfidf={'ON' if t else 'OFF'} → {len(rows)} config(s)")

    # ── Process each group ──
    all_caches: list = []  # collect all caches for cleanup

    for (metric, tfidf_on), group_rows in groups.items():
        # ── PHASE 1: Run stages 1-4 & 6 per (metric, tfidf) group ──
        print(f"\n{'#'*60}")
        print(f" PHASE 1: Computing similarities (metric={metric}, tfidf={'ON' if tfidf_on else 'OFF'})")
        print(f"{'#'*60}")

        group_args = argparse.Namespace(**vars(args))
        group_args.metric = metric
        group_args.tfidf_pattern = tfidf_on

        caches: Dict[str, SampleCache] = {}
        failed_samples = []

        for i, sample_path_str in enumerate(samples, 1):
            sample_path = Path(sample_path_str)
            print(f"\n{'─'*60}")
            print(f" [{i}/{len(samples)}] {sample_path.name}")
            print(f"{'─'*60}")

            try:
                cache = run_stages_1_to_4_and_6(sample_path, group_args)
                if cache:
                    caches[sample_path_str] = cache
            except Exception as e:
                print(f"[ERROR] Failed on {sample_path.name}: {e}")
                failed_samples.append(sample_path.name)

        if not caches:
            print(f"[ERROR] No samples processed for group (metric={metric}, tfidf={tfidf_on}).")
            continue

        all_caches.extend(caches.values())
        print(f"\n[PHASE 1 Complete] Cached {len(caches)} sample(s), {len(failed_samples)} failed")

        # ── PHASE 2: For each config in this group, rerun only stages 5 & 7 ──
        print(f"\n{'#'*60}")
        print(f" PHASE 2: Running {len(group_rows)} weight configs (metric={metric}, tfidf={'ON' if tfidf_on else 'OFF'})")
        print(f"{'#'*60}")

        for _, row in group_rows:
            run_id = row.get("run", str(_ + 1)).strip()

            w_sig = float(row.get("w_sig", args.w_sig))
            w_mbc = float(row.get("w_mbc", args.w_mbc))
            w_ttp = float(row.get("w_ttp", args.w_ttp))
            top_k = int(row.get("top_k", args.top_k))
            W     = float(row.get("W", args.W))
            beta  = float(row.get("beta", args.beta))

            print(f"\n{'='*60}")
            print(f" Run {run_id}")
            print(f"   metric={metric}  tfidf={'ON' if tfidf_on else 'OFF'}")
            print(f"   w_sig={w_sig}  w_mbc={w_mbc}  w_ttp={w_ttp}  top_k={top_k}  W={W}  beta={beta}")
            print(f"{'='*60}")

            total_w = round(w_sig + w_mbc + w_ttp, 6)
            if abs(total_w - 1.0) > 1e-5:
                print(f"  [SKIP] Weights must sum to 1.0 (got {total_w})")
                continue

            score_files = []
            for sample_path_str, cache in caches.items():
                try:
                    scores_csv = run_stages_5_and_7(
                        cache, group_args, w_sig, w_mbc, w_ttp, W, beta, top_k
                    )
                    if scores_csv and scores_csv.exists():
                        score_files.append(scores_csv)
                except Exception as e:
                    print(f"  [ERROR] {cache.stem}: {e}")

            if score_files:
                evaluate_scores(score_files, run_id)
            else:
                print(f"  [WARNING] No score files produced for run {run_id}")

    # ── Cleanup: remove cached output folders ──
    print(f"\n{'─'*60}")
    print(f" Cleanup: removing cached output folders...")
    for cache in all_caches:
        if cache.out_dir.exists():
            shutil.rmtree(cache.out_dir)
            print(f"  Deleted: {cache.out_dir.name}")

    print(f"\n{'='*60}")
    print(f" All {len(config_rows)} runs complete!")
    print(f" Results in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

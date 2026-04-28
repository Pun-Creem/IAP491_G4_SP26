#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Similarity to Action – Multi-Sample Runner
=================================================
Run the pipeline on multiple samples at once.
Each sample gets its own output folder.

Usage:
  python auto_sim2action_multi.py --samples s1.json s2.json ...
  python auto_sim2action_multi.py --sample-dir path/to/json_folder/
  python auto_sim2action_multi.py               (opens GUI multi-file picker)

All other flags are forwarded to the single-sample pipeline.
"""

import argparse
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from auto_sim2action import build_parser, load_config, apply_config, run_pipeline

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging


def pick_json_files() -> list[str]:
    """Open GUI to pick multiple JSON report files."""
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
    """Gather sample paths from --samples, --sample-dir, or GUI picker."""
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


def main():
    setup_logging()
    # Build parser that extends the single-sample parser
    base = build_parser()
    p = argparse.ArgumentParser(
        description="Auto Similarity to Action – Multi-Sample Runner",
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
    passed = 0
    failed = []

    print(f"\n{'='*60}")
    print(f" Multi-Sample Pipeline – {total} sample(s)")
    print(f"{'='*60}")

    for i, sample_path in enumerate(samples, 1):
        print(f"\n{'─'*60}")
        print(f" [{i}/{total}] {Path(sample_path).name}")
        print(f"{'─'*60}")

        # Override sample and clear out-dir so each gets its own folder
        args.sample = sample_path
        args.out_dir = ""

        try:
            run_pipeline(args)
            passed += 1
        except Exception as e:
            print(f"\n[ERROR] Failed on {Path(sample_path).name}: {e}")
            failed.append(Path(sample_path).name)

    # Summary
    print(f"\n{'='*60}")
    print(f" Summary: {passed}/{total} succeeded")
    if failed:
        print(f" Failed : {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

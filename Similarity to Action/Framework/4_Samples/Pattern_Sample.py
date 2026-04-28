# -*- coding: utf-8 -*-
"""
Windows JSON Report Scanner -> CSV one-hot matrix (signatures / ttps / mbcs)

- Prompts user to pick a folder containing the pattern space CSVs
  (signature_unique.csv, ttp_unique.csv, mbc_unique.csv)
- Prompts user to pick a folder containing many .json reports
- Scans each file for:
    - hash (from target.file.sha256, fallback sha1/md5)
    - signatures (prefers top-level "signatures")
    - ttps (tries common MITRE/CAPE keys; also crawls common structures)
    - mbcs (tries common keys and signature-embedded mbcs)
- Automatically writes 3 separate CSV files:
    1) <folder_name>_signatture.csv  (columns = signature_unique.csv features)
    2) <folder_name>_ttp.csv         (columns = ttp_unique.csv features)
    3) <folder_name>_mbc.csv         (columns = mbc_unique.csv features)
- CSV format:
    Col A: hash
    Col B..: feature names from pattern space
    Values: 1/0
- No totals row, no percentage row
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None  # type: ignore

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


# -------------------------
# Helpers: safe extraction
# -------------------------

def _as_set_of_strings(x: Any) -> Set[str]:
    out: Set[str] = set()
    if x is None:
        return out
    if isinstance(x, str):
        s = x.strip()
        if s:
            out.add(s)
        return out
    if isinstance(x, dict):
        for k in ("name", "id", "technique", "technique_id", "ttp", "value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
        return out
    if isinstance(x, (list, tuple, set)):
        for it in x:
            out |= _as_set_of_strings(it)
        return out
    return out



def _walk_find_values(obj: Any, keys_wanted: Set[str], max_nodes: int = 200000) -> Dict[str, List[Any]]:
    """
    DFS crawl through dict/list to find values by key name (case-insensitive).
    Safety cap max_nodes to avoid pathological JSON.
    """
    found: Dict[str, List[Any]] = {k: [] for k in keys_wanted}
    stack = [obj]
    visited = 0

    keys_lower = {k.lower(): k for k in keys_wanted}

    while stack:
        cur = stack.pop()
        visited += 1
        if visited > max_nodes:
            break

        if isinstance(cur, dict):
            for k, v in cur.items():
                kl = str(k).lower()
                if kl in keys_lower:
                    found[keys_lower[kl]].append(v)
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    return found



def extract_hash(report: Dict[str, Any]) -> str:
    t = report.get("target", {})
    if isinstance(t, dict):
        f = t.get("file", {})
        if isinstance(f, dict):
            for k in ("sha256", "sha1", "md5"):
                v = f.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    for k in ("sha256", "sha1", "md5", "hash"):
        v = report.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "UNKNOWN_HASH"



def extract_signatures(report: Dict[str, Any]) -> Set[str]:
    """
    ONLY take signatures from top-level 'signatures' (list).
    Do NOT read from 'statistics'.
    """
    sigs: Set[str] = set()

    top = report.get("signatures")
    if isinstance(top, list):
        for it in top:
            if isinstance(it, dict):
                name = it.get("name")
                if isinstance(name, str) and name.strip():
                    sigs.add(name.strip())
            elif isinstance(it, str) and it.strip():
                sigs.add(it.strip())

    return sigs



def extract_ttps(report: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()

    out |= _as_set_of_strings(report.get("ttps"))

    for k in ("mitre_attck", "mitre_attack", "attack", "mitre"):
        blk = report.get(k)
        if isinstance(blk, dict):
            for kk in ("techniques", "ttps", "ttp", "tactics"):
                out |= _as_set_of_strings(blk.get(kk))

    top_sigs = report.get("signatures")
    if isinstance(top_sigs, list):
        for it in top_sigs:
            if isinstance(it, dict):
                out |= _as_set_of_strings(it.get("ttps"))
                out |= _as_set_of_strings(it.get("techniques"))
                out |= _as_set_of_strings(it.get("technique_id"))

    found = _walk_find_values(report, {"ttps", "technique", "technique_id", "techniques"})
    for vlist in found.values():
        for v in vlist:
            out |= _as_set_of_strings(v)

    return {s.strip() for s in out if isinstance(s, str) and s.strip()}



def extract_mbcs(report: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()

    out |= _as_set_of_strings(report.get("mbcs"))

    for k in ("mbc", "mbcs", "mbc_id"):
        out |= _as_set_of_strings(report.get(k))

    for k in ("mitre_attck", "mitre_attack", "attack", "mitre"):
        blk = report.get(k)
        if isinstance(blk, dict):
            out |= _as_set_of_strings(blk.get("mbcs"))
            out |= _as_set_of_strings(blk.get("mbc"))

    top_sigs = report.get("signatures")
    if isinstance(top_sigs, list):
        for it in top_sigs:
            if isinstance(it, dict):
                out |= _as_set_of_strings(it.get("mbcs"))
                out |= _as_set_of_strings(it.get("mbc"))
                out |= _as_set_of_strings(it.get("mbc_id"))

    found = _walk_find_values(report, {"mbcs", "mbc", "mbc_id"})
    for vlist in found.values():
        for v in vlist:
            out |= _as_set_of_strings(v)

    return {s.strip() for s in out if isinstance(s, str) and s.strip()}


# -------------------------
# Pattern space loading
# -------------------------

def load_features_from_csv(csv_path: str) -> List[str]:
    """
    Load feature list from a CSV that has a single column named 'feature'.
    Returns ordered list of feature strings (skips header).
    """
    features: List[str] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("feature", "").strip()
                if val:
                    features.append(val)
    except Exception as e:
        print(f"[!] Không đọc được pattern space CSV: {csv_path} — {e}")
    return features


def load_pattern_space(pattern_folder: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Reads signature_unique.csv, ttp_unique.csv, mbc_unique.csv from pattern_folder.
    Returns (sig_features, ttp_features, mbc_features).
    """
    sig_features = load_features_from_csv(os.path.join(pattern_folder, "signature_unique.csv"))
    ttp_features = load_features_from_csv(os.path.join(pattern_folder, "ttp_unique.csv"))
    mbc_features = load_features_from_csv(os.path.join(pattern_folder, "mbc_unique.csv"))
    return sig_features, ttp_features, mbc_features


# -------------------------
# CSV writing
# -------------------------

def build_csv(out_csv: str, rows: List[Tuple[str, Set[str]]], all_features: List[str]) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["hash"] + all_features)

        for h, feats in rows:
            feats_set = set(feats)
            writer.writerow([h] + [1 if feat in feats_set else 0 for feat in all_features])


# -------------------------
# Main
# -------------------------

def pick_folder(title: str) -> Optional[str]:
    if tk is None:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if folder and os.path.isdir(folder):
        return folder
    return None



def read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return None
    except Exception:
        return None



def main():
    setup_logging()
    # Step 1: pick pattern space folder
    print("Bước 1: Chọn folder chứa pattern space (signature_unique.csv, ttp_unique.csv, mbc_unique.csv)")
    pattern_folder = pick_folder("Chọn folder chứa pattern space CSV (signature/ttp/mbc_unique.csv)")
    if not pattern_folder:
        print("Không chọn folder pattern space hoặc folder không hợp lệ. Thoát.")
        sys.exit(1)

    sig_features, ttp_features, mbc_features = load_pattern_space(pattern_folder)

    if not sig_features and not ttp_features and not mbc_features:
        print("Không load được feature nào từ pattern space. Kiểm tra lại 3 file CSV. Thoát.")
        sys.exit(1)

    print(f"[+] Pattern space đã load:")
    print(f"    - Signature features: {len(sig_features)}")
    print(f"    - TTP features:       {len(ttp_features)}")
    print(f"    - MBC features:       {len(mbc_features)}")

    # Step 2: pick JSON reports folder
    print("Bước 2: Chọn folder chứa các file .json report")
    folder = pick_folder("Chọn folder chứa các file .json report")
    if not folder:
        print("Không chọn folder hoặc folder không hợp lệ. Thoát.")
        sys.exit(1)

    json_files = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(".json") and os.path.isfile(os.path.join(folder, fn))
    ]
    if not json_files:
        print("Folder không có file .json nào. Thoát.")
        sys.exit(1)

    print(f"[+] Tìm thấy {len(json_files)} file .json. Đang scan...")

    per_file: List[Dict[str, Any]] = []
    all_sigs: Set[str] = set()
    all_ttps: Set[str] = set()
    all_mbcs: Set[str] = set()

    for p in sorted(json_files):
        rep = read_json_file(p)
        if rep is None:
            print(f"[!] Bỏ qua (không đọc được JSON): {os.path.basename(p)}")
            continue

        h = extract_hash(rep)
        sigs = extract_signatures(rep)
        ttps = extract_ttps(rep)
        mbcs = extract_mbcs(rep)

        per_file.append({
            "path": p,
            "hash": h,
            "signatures": sigs,
            "ttps": ttps,
            "mbcs": mbcs,
        })

        all_sigs |= sigs
        all_ttps |= ttps
        all_mbcs |= mbcs

    if not per_file:
        print("Không có report hợp lệ để xử lý. Thoát.")
        sys.exit(1)

    print("[+] Scan xong.")
    print(f"    - Reports hợp lệ: {len(per_file)}")
    print(f"    - Unique signatures (trong JSON): {len(all_sigs)}")
    print(f"    - Unique ttps (trong JSON):       {len(all_ttps)}")
    print(f"    - Unique mbcs (trong JSON):       {len(all_mbcs)}")

    # Warn about features found in JSON but not in pattern space
    sig_space_set = set(sig_features)
    ttp_space_set = set(ttp_features)
    mbc_space_set = set(mbc_features)

    out_of_space_sigs = sorted(all_sigs - sig_space_set)
    out_of_space_ttps = sorted(all_ttps - ttp_space_set)
    out_of_space_mbcs = sorted(all_mbcs - mbc_space_set)

    if out_of_space_sigs:
        print(f"\n[!] CẢNH BÁO: {len(out_of_space_sigs)} signature(s) trong JSON KHÔNG có trong pattern space (sẽ bị bỏ qua):")
        for s in out_of_space_sigs:
            print(f"      - {s}")

    if out_of_space_ttps:
        print(f"\n[!] CẢNH BÁO: {len(out_of_space_ttps)} ttp(s) trong JSON KHÔNG có trong pattern space (sẽ bị bỏ qua):")
        for s in out_of_space_ttps:
            print(f"      - {s}")

    if out_of_space_mbcs:
        print(f"\n[!] CẢNH BÁO: {len(out_of_space_mbcs)} mbc(s) trong JSON KHÔNG có trong pattern space (sẽ bị bỏ qua):")
        for s in out_of_space_mbcs:
            print(f"      - {s}")

    if not out_of_space_sigs and not out_of_space_ttps and not out_of_space_mbcs:
        print("[+] Tất cả features trong JSON đều nằm trong pattern space.")

    rows_signatures: List[Tuple[str, Set[str]]] = []
    rows_ttps: List[Tuple[str, Set[str]]] = []
    rows_mbcs: List[Tuple[str, Set[str]]] = []

    for item in per_file:
        rows_signatures.append((item["hash"], item["signatures"]))
        rows_ttps.append((item["hash"], item["ttps"]))
        rows_mbcs.append((item["hash"], item["mbcs"]))

    folder_name = os.path.basename(os.path.normpath(folder))

    out_signatures = os.path.join(folder, f"{folder_name}_signatture.csv")
    out_ttps = os.path.join(folder, f"{folder_name}_ttp.csv")
    out_mbcs = os.path.join(folder, f"{folder_name}_mbc.csv")

    # Use pattern space features as fixed column headers
    print(f"[+] Đang ghi CSV: {out_signatures}")
    build_csv(out_signatures, rows_signatures, sig_features)

    print(f"[+] Đang ghi CSV: {out_ttps}")
    build_csv(out_ttps, rows_ttps, ttp_features)

    print(f"[+] Đang ghi CSV: {out_mbcs}")
    build_csv(out_mbcs, rows_mbcs, mbc_features)

    print("[+] Hoàn tất!")
    print(f"    File signatures: {out_signatures}")
    print(f"    File ttps: {out_ttps}")
    print(f"    File mbcs: {out_mbcs}")


if __name__ == "__main__":
    main()

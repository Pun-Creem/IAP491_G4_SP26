#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework v3 - Malware Defense Recommendation Pipeline
=======================================================
Pipeline đầy đủ từ JSON report đến bảng điểm D3FEND actions.

Stages:
  1. Extract patterns (Signature / TTP / MBC) từ sample JSON report
  2. [Tuỳ chọn] Cập nhật TTP & MBC qua VirusTotal Behaviours API
  3. Tính similarity (Jaccard hoặc Cosine) với dataset patterns
  4. Merge 3 similarity scores → 1 file
  5. Chọn Top-K mẫu tương đồng nhất
  6. Map TTP → D3FEND actions (action_per_ttps.csv)
  7. Tính Action Score có trọng số
  8. Xuất bảng kết quả cuối

Usage:
  python framework.py --sample path/to/sample.json [options]

  Options:
    --sample          Path đến file JSON report của sample cần phân tích (bắt buộc)
    --out-dir         Thư mục lưu kết quả  [default: <script_dir>/output/<sample_name>]
    --metric          jaccard | cosine      [default: jaccard]
    --w-sig           Trọng số Signature    [default: 0.33]
    --w-mbc           Trọng số MBC          [default: 0.33]
    --w-ttp           Trọng số TTP          [default: 0.34]
    --top-k           Số mẫu Top-K          [default: 5]
    --W               Hệ số W (self-weight) [default: 1.0]
    --beta            Hệ số beta (neighbor) [default: 1.0]
    --vt-key          VirusTotal API key    [default: None, bỏ qua bước VT]
    --no-vt           Bỏ qua bước VT dù có --vt-key

  Paths dataset (mặc định tự động dò từ thư mục script):
    --pattern-sig     pattern_signature.csv
    --pattern-ttp     pattern_ttp_update.csv
    --pattern-mbc     pattern_mbc_update.csv
    --sig-unique      unique/signature_unique.csv  (trong 3_Patterns)
    --ttp-unique      unique/ttp_unique.csv
    --mbc-unique      unique/mbc_unique.csv
    --action-ttps     6_Action space/action_per_ttps.csv
    --action-report   7_Top K/actionPerReport.xlsx
    --unique-actions  7_Top K/unique_actions.csv
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"


def load_config() -> Dict[str, Any]:
    """Load config.json nếu tồn tại, trả về dict rỗng nếu không có."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)
    # Flatten paths vào top-level, bỏ _comment
    cfg: Dict[str, Any] = {k: v for k, v in data.items() if not k.startswith("_")}
    for key, val in cfg.pop("paths", {}).items():
        cfg[key] = val
    return cfg


def apply_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """
    Điền giá trị từ config vào args, chỉ với những field mà CLI không truyền
    (tức là còn ở giá trị sentinel None / False / "").
    Priority: CLI > config.json > _resolve_defaults (built-in paths).
    """
    SENTINEL_STR  = ""     # default của các str arg trong parser
    SENTINEL_BOOL = False  # default của store_true

    for key, cfg_val in cfg.items():
        if not hasattr(args, key):
            continue
        cur = getattr(args, key)
        # Ghi đè nếu CLI chưa set (vẫn là giá trị sentinel)
        if isinstance(cfg_val, bool):
            if cur == SENTINEL_BOOL:
                setattr(args, key, cfg_val)
        elif isinstance(cfg_val, (int, float)):
            # Các tham số số có default cứng trong parser — chỉ override qua config
            # nếu người dùng không truyền CLI (không thể phân biệt được 100%,
            # nên ta luôn để config thắng parser-default cho số)
            setattr(args, key, cfg_val)
        else:
            if cur == SENTINEL_STR:
                setattr(args, key, str(cfg_val))
    return args


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


def _walk_find_values(obj: Any, keys_wanted: Set[str], max_nodes: int = 200_000) -> Dict[str, List[Any]]:
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


# ─────────────────────────────────────────────────────────────────────────────
# GUI – File Picker
# ─────────────────────────────────────────────────────────────────────────────

def pick_json_file() -> str:
    """Mở GUI chọn file JSON report."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Chọn file JSON report của sample",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return file_path


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Pattern Transform
# ─────────────────────────────────────────────────────────────────────────────

def apply_tfidf_to_pattern(
    sample_csv: Path,
    dataset_csv: Path,
    out_dir: Path,
    label: str,
) -> tuple:
    """
    Gộp dataset + sample thành 1 matrix, tính TF-IDF, rồi tách lại.
    Trả về (sample_tfidf_csv, dataset_tfidf_csv).
    """
    sample_df  = pd.read_csv(sample_csv)
    dataset_df = pd.read_csv(dataset_csv)

    s_id_col = sample_df.columns[0]
    d_id_col = dataset_df.columns[0]

    # Lấy tập feature chung
    s_features = set(sample_df.columns[1:])
    d_features = set(dataset_df.columns[1:])
    all_features = sorted(s_features | d_features)

    # Chuẩn bị sample row với đầy đủ feature
    for f in all_features:
        if f not in sample_df.columns:
            sample_df[f] = 0
        if f not in dataset_df.columns:
            dataset_df[f] = 0

    # Gộp matrix: dataset trước, sample cuối
    combined_ids = list(dataset_df[d_id_col]) + list(sample_df[s_id_col])
    combined_features = pd.concat([
        dataset_df[all_features].reset_index(drop=True),
        sample_df[all_features].reset_index(drop=True),
    ], ignore_index=True)

    # Convert to numeric
    for col in all_features:
        combined_features[col] = pd.to_numeric(combined_features[col], errors="coerce").fillna(0.0)

    # Tính TF-IDF trên toàn bộ combined matrix
    transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf_matrix = transformer.fit_transform(combined_features.values)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=all_features)

    n_dataset = len(dataset_df)

    # Tách lại dataset và sample
    dataset_tfidf = pd.concat([
        pd.DataFrame({d_id_col: combined_ids[:n_dataset]}),
        tfidf_df.iloc[:n_dataset].reset_index(drop=True),
    ], axis=1)

    sample_tfidf = pd.concat([
        pd.DataFrame({s_id_col: combined_ids[n_dataset:]}),
        tfidf_df.iloc[n_dataset:].reset_index(drop=True),
    ], axis=1)

    # Lưu file
    dataset_out = out_dir / f"tfidf_dataset_{label}.csv"
    sample_out  = out_dir / f"tfidf_sample_{label}.csv"
    dataset_tfidf.to_csv(dataset_out, index=False, encoding="utf-8-sig")
    sample_tfidf.to_csv(sample_out, index=False, encoding="utf-8-sig")

    print(f"  [TF-IDF {label:9s}] Combined {n_dataset}+{len(sample_df)} rows, {len(all_features)} features")

    return sample_out, dataset_out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Pattern Extraction
# ─────────────────────────────────────────────────────────────────────────────

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


def load_feature_list(csv_path: Path) -> List[str]:
    """Đọc cột 'feature' từ CSV unique (signature/ttp/mbc_unique.csv)."""
    features: List[str] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("feature", "").strip()
            if val:
                features.append(val)
    return features


def build_one_hot_row(sample_hash: str, features: List[str], present: Set[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"hash": sample_hash}
    for feat in features:
        row[feat] = 1 if feat in present else 0
    return row


def stage1_extract_patterns(
    json_path: Path,
    sig_features: List[str],
    ttp_features: List[str],
    mbc_features: List[str],
    out_dir: Path,
) -> Tuple[Path, Path, Path, str]:
    """
    Trích xuất pattern từ sample JSON, map vào pattern space.
    Trả về (sig_csv, ttp_csv, mbc_csv, sha256).
    """
    print("\n[Stage 1] Extracting patterns from sample JSON...")
    with open(json_path, encoding="utf-8", errors="replace") as f:
        report = json.load(f)

    sha256 = extract_hash(report)
    sigs   = extract_signatures(report)
    ttps   = extract_ttps(report)
    mbcs   = extract_mbcs(report)

    print(f"  SHA256    : {sha256}")
    print(f"  Signatures: {len(sigs)}")
    print(f"  TTPs      : {len(ttps)}")
    print(f"  MBCs      : {len(mbcs)}")

    stem = json_path.stem  # e.g. "981"

    def write_csv(features, present, suffix):
        out = out_dir / f"{stem}_{suffix}.csv"
        row = build_one_hot_row(sha256, features, present)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["hash"] + features)
            writer.writeheader()
            writer.writerow(row)
        return out

    sig_csv = write_csv(sig_features, sigs,  "signatture")
    ttp_csv = write_csv(ttp_features, ttps,  "ttp_update")
    mbc_csv = write_csv(mbc_features, mbcs,  "mbc_update")

    print(f"  -> {sig_csv.name}")
    print(f"  -> {ttp_csv.name}")
    print(f"  -> {mbc_csv.name}")
    return sig_csv, ttp_csv, mbc_csv, sha256


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – VirusTotal Enrichment (optional)
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
MBC_ID_RE    = re.compile(r"\b[BCDF]\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
MBC_LINE_RE  = re.compile(r"^\s*-\s*(.+?)\s*\[([BCDF]\d{4}(?:\.\d{3})?)\]\s*$", re.IGNORECASE)


def _vt_get(endpoint: str, api_key: str, max_retries: int = 5) -> dict:
    import urllib.request, urllib.error
    url = f"https://www.virustotal.com/api/v3{endpoint}"
    headers = {"x-apikey": api_key, "User-Agent": "framework-v3/1.0"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            if hasattr(exc, "code"):
                if exc.code == 404:
                    return {}
                if exc.code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt, 30)
                    print(f"    [VT] {exc.code} – retry in {wait}s")
                    time.sleep(wait)
                    continue
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"VT request failed after {max_retries} retries: {last_err}")


def _parse_mbc_from_rule_src(rule_src: str) -> Set[str]:
    found: Set[str] = set()
    in_mbc = False
    for raw in rule_src.splitlines():
        line = raw.rstrip("\n")
        if re.match(r"^\s*mbc:\s*$", line, re.IGNORECASE):
            in_mbc = True
            continue
        if in_mbc:
            if re.match(r"^\s*[A-Za-z0-9_&-]+\s*:\s*$", line) and not re.match(r"^\s*-\s*", line):
                in_mbc = False
                continue
            m = MBC_LINE_RE.match(line)
            if m:
                found.add(m.group(2).upper())
    for m in MBC_ID_RE.findall(rule_src):
        found.add(m.upper())
    return found


def _fetch_vt_ttp_mbc(sha256: str, api_key: str) -> Tuple[Set[str], Set[str]]:
    data = _vt_get(f"/files/{sha256}/behaviours", api_key)
    ttps: Set[str] = set()
    mbcs: Set[str] = set()
    for item in data.get("data", []) or []:
        attrs = item.get("attributes", {}) or {}
        for tech in attrs.get("mitre_attack_techniques", []) or []:
            tid = str(tech.get("id", "")).strip().upper()
            if ATTACK_ID_RE.fullmatch(tid):
                ttps.add(tid)
        for sig in attrs.get("signature_matches", []) or []:
            rule_src = str(sig.get("rule_src", "") or "")
            mbcs.update(_parse_mbc_from_rule_src(rule_src))
            for m in ATTACK_ID_RE.findall(rule_src):
                ttps.add(m.upper())
    time.sleep(0.25)
    return ttps, mbcs


def _extract_codes(text: str, pattern: str) -> List[str]:
    if not text:
        return []
    seen: Set[str] = set()
    result: List[str] = []
    for m in re.findall(pattern, text, re.IGNORECASE):
        code = m.upper()
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def stage2_vt_update(
    sha256: str,
    ttp_csv: Path,
    mbc_csv: Path,
    api_key: str,
    out_dir: Path,
    stem: str,
) -> Tuple[Path, Path]:
    """
    Cập nhật TTP & MBC CSV của sample từ VT Behaviours API.
    Trả về (ttp_csv_updated, mbc_csv_updated).
    """
    print(f"\n[Stage 2] VirusTotal enrichment for {sha256[:16]}...")

    vt_ttps, vt_mbcs = _fetch_vt_ttp_mbc(sha256, api_key)
    print(f"  VT TTPs: {len(vt_ttps)} | VT MBCs: {len(vt_mbcs)}")

    def update_csv(src_csv: Path, codes: Set[str], code_re: str, suffix: str) -> Path:
        df = pd.read_csv(src_csv)
        hash_col = "hash" if "hash" in df.columns else df.columns[0]
        for code in codes:
            if code not in df.columns:
                df[code] = 0
        # Set 1 for all codes in this sample row
        for code in codes:
            df.loc[0, code] = 1
        # Re-sort columns: hash first, then rest sorted
        feat_cols = sorted([c for c in df.columns if c != hash_col])
        df = df[[hash_col] + feat_cols]
        out = out_dir / f"{stem}_{suffix}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out

    ttp_out = update_csv(ttp_csv, vt_ttps, r"\bT\d{4}(?:\.\d{3})?\b", "ttp_update_vt")
    mbc_out = update_csv(mbc_csv, vt_mbcs, r"\b[BCDF]\d{4}(?:\.\d{3})?\b", "mbc_update_vt")

    print(f"  -> {ttp_out.name}")
    print(f"  -> {mbc_out.name}")
    return ttp_out, mbc_out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Similarity
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1  = math.sqrt(sum(a * a for a in v1))
    n2  = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def jaccard_similarity(v1: List[float], v2: List[float]) -> float:
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    inter = sum(1 for a, b in zip(b1, b2) if a == 1 and b == 1)
    union = sum(1 for a, b in zip(b1, b2) if a == 1 or b == 1)
    return inter / union if union else 0.0


def _compute_sim_series(
    target_row: List[float],
    dataset_df: pd.DataFrame,
    feature_cols: List[str],
    metric: str,
) -> List[Tuple[str, float]]:
    sim_fn = cosine_similarity if metric == "cosine" else jaccard_similarity
    results = []
    id_col = dataset_df.columns[0]  # hash column
    for _, row in dataset_df.iterrows():
        dataset_id = str(row[id_col])
        vec = [float(row[c]) for c in feature_cols]
        score = sim_fn(target_row, vec)
        results.append((dataset_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def stage3_similarity(
    sig_csv: Path,
    ttp_csv: Path,
    mbc_csv: Path,
    pattern_sig_csv: Path,
    pattern_ttp_csv: Path,
    pattern_mbc_csv: Path,
    metric: str,
    out_dir: Path,
    stem: str,
    tfidf_pattern: bool = False,
) -> Tuple[Path, Path, Path]:
    """
    Tính similarity cho 3 loại pattern. Trả về (sig_sim_csv, ttp_sim_csv, mbc_sim_csv).
    Nếu tfidf_pattern=True, tính TF-IDF trên gộp dataset+sample trước khi tính similarity.
    """
    if tfidf_pattern and metric == "jaccard":
        print("  [TF-IDF] Jaccard binarize vector nên không tương thích TF-IDF -> chuyển sang cosine.")
        metric = "cosine"

    print(f"\n[Stage 3] Computing {metric} similarity...")

    if tfidf_pattern:
        print("  [TF-IDF] Applying TF-IDF transform on combined dataset+sample patterns...")
        sig_csv, pattern_sig_csv = apply_tfidf_to_pattern(sig_csv, pattern_sig_csv, out_dir, "signature")
        ttp_csv, pattern_ttp_csv = apply_tfidf_to_pattern(ttp_csv, pattern_ttp_csv, out_dir, "ttp")
        mbc_csv, pattern_mbc_csv = apply_tfidf_to_pattern(mbc_csv, pattern_mbc_csv, out_dir, "mbc")

    def run_one(target_csv: Path, dataset_csv: Path, label: str) -> Path:
        target_df  = pd.read_csv(target_csv)
        dataset_df = pd.read_csv(dataset_csv)

        t_hash_col = target_df.columns[0]
        d_hash_col = dataset_df.columns[0]

        t_features = list(target_df.columns[1:])
        d_features = list(dataset_df.columns[1:])

        # Intersection of features (target may have extra VT columns)
        common = [f for f in t_features if f in set(d_features)]
        if not common:
            raise ValueError(f"No common features between target and dataset for {label}.")

        target_vec = [float(target_df.iloc[0][c]) for c in common]
        target_id  = str(target_df.iloc[0][t_hash_col])

        dataset_sub = dataset_df[[d_hash_col] + common].copy()
        for c in common:
            dataset_sub[c] = pd.to_numeric(dataset_sub[c], errors="coerce").fillna(0)

        results = _compute_sim_series(target_vec, dataset_sub, common, metric)

        out = out_dir / f"{stem}_{label}_similarity.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["target_id", "dataset_id", "similarity_type", "similarity_score"])
            for dataset_id, score in results:
                writer.writerow([target_id, dataset_id, metric, f"{score:.10f}"])

        print(f"  [{label:9s}] {len(results)} pairs -> {out.name}")
        return out

    sig_sim = run_one(sig_csv, pattern_sig_csv, "signature")
    ttp_sim = run_one(ttp_csv, pattern_ttp_csv, "ttp")
    mbc_sim = run_one(mbc_csv, pattern_mbc_csv, "mbc")
    return sig_sim, ttp_sim, mbc_sim


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Merge Similarity
# ─────────────────────────────────────────────────────────────────────────────

def stage4_merge_similarity(
    sig_sim: Path, ttp_sim: Path, mbc_sim: Path, out_dir: Path, stem: str
) -> Path:
    """
    Merge 3 similarity CSV thành 1 file tổng hợp.
    Trả về merged_csv path.
    """
    print("\n[Stage 4] Merging similarity scores...")

    def load(p: Path, rename_col: str) -> pd.DataFrame:
        df = pd.read_csv(p)
        df["target_id"]       = df["target_id"].astype(str).str.strip()
        df["dataset_id"]      = df["dataset_id"].astype(str).str.strip()
        df["similarity_type"] = df["similarity_type"].astype(str).str.strip().str.lower()
        df["similarity_score"] = pd.to_numeric(df["similarity_score"], errors="coerce")
        return df.rename(columns={"similarity_score": rename_col})

    df_sig = load(sig_sim, "signatures_similarity")
    df_mbc = load(mbc_sim, "mbcs_similarity")
    df_ttp = load(ttp_sim, "ttps_similarity")

    keys = ["target_id", "dataset_id", "similarity_type"]
    merged = (
        df_sig.merge(df_mbc, on=keys, how="outer")
              .merge(df_ttp, on=keys, how="outer")
    )
    col_order = keys + ["signatures_similarity", "mbcs_similarity", "ttps_similarity"]
    merged = merged[col_order].sort_values(keys).reset_index(drop=True)

    out = out_dir / f"{stem}_merged_similarity.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  -> {out.name}  ({len(merged)} rows)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – Top-K Selection
# ─────────────────────────────────────────────────────────────────────────────

def stage5_top_k(
    merged_csv: Path,
    action_report_path: Path,
    unique_actions_path: Path,
    k: int,
    w_sig: float,
    w_mbc: float,
    w_ttp: float,
    out_dir: Path,
    stem: str,
) -> Path:
    """
    Chọn top-K mẫu tương đồng nhất, gán action vector.
    Trả về top_k_csv path.
    """
    print(f"\n[Stage 5] Selecting Top-{k} neighbors (w_sig={w_sig}, w_mbc={w_mbc}, w_ttp={w_ttp})...")

    df = pd.read_csv(merged_csv)
    for col in ("signatures_similarity", "mbcs_similarity", "ttps_similarity"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["weighted_similarity"] = (
        df["signatures_similarity"] * w_sig +
        df["mbcs_similarity"]       * w_mbc +
        df["ttps_similarity"]        * w_ttp
    )

    top_k_df = df.nlargest(k, "weighted_similarity")[["dataset_id", "weighted_similarity"]].reset_index(drop=True)

    print(f"  Top-{k} neighbors:")
    for i, row in top_k_df.iterrows():
        print(f"    {i+1}. {str(row['dataset_id'])[:20]}... | sim={row['weighted_similarity']:.4f}")

    # Load unique actions list
    actions_df = pd.read_csv(unique_actions_path)
    all_actions = actions_df["Action"].tolist()

    # Load actionPerReport
    report_df  = pd.read_excel(action_report_path)
    hash_to_acts: Dict[str, Set[str]] = {}
    for _, row in report_df.iterrows():
        h   = str(row.get("Hash256", "")).strip()
        raw = str(row.get("Action", "")) if pd.notna(row.get("Action")) else ""
        hash_to_acts[h] = {a.strip() for a in raw.split("\n") if a.strip()}

    sim_map = dict(zip(top_k_df["dataset_id"], top_k_df["weighted_similarity"]))
    records = []
    for h in top_k_df["dataset_id"]:
        row_d: Dict[str, Any] = {"sha256": h, "similarity": sim_map[h]}
        acts = hash_to_acts.get(h, set())
        for action in all_actions:
            row_d[action] = 1 if action in acts else 0
        records.append(row_d)

    out_df = pd.DataFrame(records)
    out = out_dir / f"{stem}_top{k}.csv"
    out_df.to_csv(out, index=False)
    print(f"  -> {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – Map Actions (TTP → D3FEND)
# ─────────────────────────────────────────────────────────────────────────────

def stage6_map_actions(
    ttp_csv: Path,
    action_per_ttps_path: Path,
    out_dir: Path,
    stem: str,
) -> Tuple[Path, Path]:
    """
    Map TTP → D3FEND actions cho sample.
    Trả về (mapped_bin_csv, mapped_dupe_csv).
    """
    print("\n[Stage 6] Mapping TTPs -> D3FEND actions...")

    action_df = pd.read_csv(action_per_ttps_path, index_col=0)
    target_df = pd.read_csv(ttp_csv, index_col=0)

    action_names = action_df.columns.tolist()

    results_bin  = []
    results_dupe = []

    for sha256, ttp_row in target_df.iterrows():
        active_ttps = ttp_row[ttp_row == 1].index.tolist()
        valid_ttps  = [t for t in active_ttps if t in action_df.index]

        if valid_ttps:
            action_sum = action_df.loc[valid_ttps].sum(axis=0)
        else:
            action_sum = pd.Series(0, index=action_names)

        row_dupe = {"sha256": sha256}
        row_dupe.update(action_sum.to_dict())
        results_dupe.append(row_dupe)

        action_bin = (action_sum > 0).astype(int)
        row_bin = {"sha256": sha256}
        row_bin.update(action_bin.to_dict())
        results_bin.append(row_bin)

    out_bin  = out_dir / f"{stem}_mapped_action.csv"
    out_dupe = out_dir / f"{stem}_mapped_action_dupe.csv"

    pd.DataFrame(results_bin).to_csv(out_bin,  index=False)
    pd.DataFrame(results_dupe).to_csv(out_dupe, index=False)

    active_actions = sum(1 for v in results_bin[0].values() if isinstance(v, (int, float)) and v > 0) if results_bin else 0
    print(f"  Active actions (binary): {active_actions} / {len(action_names)}")
    print(f"  -> {out_bin.name}")
    print(f"  -> {out_dupe.name}")
    return out_bin, out_dupe


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 – Action Score
# ─────────────────────────────────────────────────────────────────────────────

def _load_mapped_action(path: Path) -> Tuple[str, Dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)
    actions = headers[1:]
    px = {a: float(v) for a, v in zip(actions, row[1:])}
    return row[0], px


def _load_top_k_neighbors(path: Path) -> Tuple[List[Tuple[str, float, Dict[str, float]]], List[str]]:
    neighbors = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        actions = headers[2:]
        for row in reader:
            if not row or not row[0].strip():
                continue
            sha  = row[0]
            sim  = float(row[1])
            yj   = {a: float(v) for a, v in zip(actions, row[2:])}
            neighbors.append((sha, sim, yj))
    return neighbors, actions


def stage7_action_score(
    mapped_csv: Path,
    top_k_csv: Path,
    W: float,
    beta: float,
    out_dir: Path,
    stem: str,
) -> Path:
    """
    Tính action score: score(a) = W * px(a) + beta * Σ sim_j * yj(a)
    Trả về scores_csv path.
    """
    print(f"\n[Stage 7] Computing action scores (W={W}, β={beta})...")

    target_sha, px = _load_mapped_action(mapped_csv)
    neighbors, actions = _load_top_k_neighbors(top_k_csv)

    print(f"  Target    : {target_sha[:20]}...")
    print(f"  Neighbors : {len(neighbors)}")
    print(f"  Actions   : {len(actions)}")

    scores: Dict[str, float] = {}
    for action in actions:
        p = px.get(action, 0.0)
        neighbor_sum = sum(sim * yj.get(action, 0.0) for _, sim, yj in neighbors)
        scores[action] = W * p + beta * neighbor_sum

    # Sort descending by score
    sorted_actions = sorted(actions, key=lambda a: scores[a], reverse=True)

    out = out_dir / f"{stem}_action_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "action", "score"])
        for rank, action in enumerate(sorted_actions, 1):
            writer.writerow([rank, action, f"{scores[action]:.6f}"])

    top10 = [(a, scores[a]) for a in sorted_actions[:10]]
    print("  Top-10 recommended actions:")
    for rank, (action, score) in enumerate(top10, 1):
        print(f"    {rank:2d}. {action:<60s} {score:.4f}")

    print(f"  -> {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Default Path Resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Điền các path mặc định nếu chưa được chỉ định."""
    d = SCRIPT_DIR

    if not args.pattern_sig:
        args.pattern_sig = str(d / "5_Similarity" / "dataset pattern" / "pattern_signature.csv")
    if not args.pattern_ttp:
        args.pattern_ttp = str(d / "5_Similarity" / "dataset pattern" / "pattern_ttp_update.csv")
    if not args.pattern_mbc:
        args.pattern_mbc = str(d / "5_Similarity" / "dataset pattern" / "pattern_mbc_update.csv")
    if not args.sig_unique:
        args.sig_unique = str(d / "3_Patterns" / "unique" / "signature_unique.csv")
    if not args.ttp_unique:
        args.ttp_unique = str(d / "3_Patterns" / "unique" / "ttp_unique.csv")
    if not args.mbc_unique:
        args.mbc_unique = str(d / "3_Patterns" / "unique" / "mbc_unique.csv")
    if not args.action_ttps:
        args.action_ttps = str(d / "6_Action space" / "action_per_ttps.csv")
    if not args.action_report:
        args.action_report = str(d / "7_Top K" / "actionPerReport.xlsx")
    if not args.unique_actions:
        args.unique_actions = str(d / "7_Top K" / "unique_actions.csv")

    return args


def _check_required_files(args: argparse.Namespace):
    required = {
        "Sample JSON": args.sample,
        "Pattern Signature": args.pattern_sig,
        "Pattern TTP": args.pattern_ttp,
        "Pattern MBC": args.pattern_mbc,
        "Signature unique": args.sig_unique,
        "TTP unique": args.ttp_unique,
        "MBC unique": args.mbc_unique,
        "Action per TTPs": args.action_ttps,
        "Action per Report": args.action_report,
        "Unique Actions": args.unique_actions,
    }
    missing = [(name, path) for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print("\n[ERROR] Missing required files:")
        for name, path in missing:
            print(f"  {name}: {path}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace):
    args = _resolve_defaults(args)
    _check_required_files(args)

    # Validate weights
    total_w = round(args.w_sig + args.w_mbc + args.w_ttp, 6)
    if abs(total_w - 1.0) > 1e-5:
        print(f"[ERROR] Weights must sum to 1.0 (got {total_w})")
        sys.exit(1)

    # Setup output directory
    sample_path = Path(args.sample)
    stem = sample_path.stem

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = SCRIPT_DIR / "output" / stem

    out_dir.mkdir(parents=True, exist_ok=True)
    tfidf_on = getattr(args, "tfidf_pattern", False)
    print(f"\n{'='*60}")
    print(f" Framework v3 Pipeline")
    print(f"{'='*60}")
    print(f" Sample  : {sample_path.name}")
    print(f" Output  : {out_dir}")
    print(f" Metric  : {args.metric}")
    print(f" Weights : sig={args.w_sig}  mbc={args.w_mbc}  ttp={args.w_ttp}")
    print(f" Top-K   : {args.top_k}")
    print(f" W={args.W}  β={args.beta}")
    print(f" TF-IDF  : {'ON' if tfidf_on else 'OFF'}")
    print(f"{'='*60}")

    # Load pattern space features
    sig_features = load_feature_list(Path(args.sig_unique))
    ttp_features = load_feature_list(Path(args.ttp_unique))
    mbc_features = load_feature_list(Path(args.mbc_unique))
    print(f"\n Pattern space: {len(sig_features)} sig | {len(ttp_features)} ttp | {len(mbc_features)} mbc")

    # Stage 1: Extract patterns
    sig_csv, ttp_csv, mbc_csv, sha256 = stage1_extract_patterns(
        sample_path, sig_features, ttp_features, mbc_features, out_dir
    )

    # Stage 2: VT enrichment (optional)
    if args.vt_key and not args.no_vt:
        ttp_csv, mbc_csv = stage2_vt_update(sha256, ttp_csv, mbc_csv, args.vt_key, out_dir, stem)
    else:
        if args.no_vt:
            print("\n[Stage 2] Skipped (--no-vt)")
        else:
            print("\n[Stage 2] Skipped (no --vt-key provided)")

    # Stage 3: Similarity (with optional TF-IDF)
    sim_stem = f"{stem}_tfidf" if tfidf_on else stem
    sig_sim, ttp_sim, mbc_sim = stage3_similarity(
        sig_csv, ttp_csv, mbc_csv,
        Path(args.pattern_sig), Path(args.pattern_ttp), Path(args.pattern_mbc),
        args.metric, out_dir, sim_stem,
        tfidf_pattern=tfidf_on,
    )

    # Stage 4: Merge
    merged_csv = stage4_merge_similarity(sig_sim, ttp_sim, mbc_sim, out_dir, sim_stem)

    # Stage 5: Top-K
    top_k_csv = stage5_top_k(
        merged_csv,
        Path(args.action_report),
        Path(args.unique_actions),
        args.top_k,
        args.w_sig, args.w_mbc, args.w_ttp,
        out_dir, sim_stem,
    )

    # Stage 6: Map actions
    mapped_bin_csv, mapped_dupe_csv = stage6_map_actions(
        ttp_csv, Path(args.action_ttps), out_dir, sim_stem
    )

    # Stage 7: Action scores
    scores_csv = stage7_action_score(
        mapped_bin_csv, top_k_csv, args.W, args.beta, out_dir, sim_stem
    )

    print(f"\n{'='*60}")
    print(f" Pipeline complete!")
    print(f" Output directory: {out_dir}")
    print(f"   {sig_csv.name}")
    print(f"   {ttp_csv.name}")
    print(f"   {mbc_csv.name}")
    print(f"   {merged_csv.name}")
    print(f"   {top_k_csv.name}")
    print(f"   {mapped_bin_csv.name}")
    print(f"   {scores_csv.name}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Framework v3 – Malware Defense Recommendation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample",         default="",     help="Path đến sample JSON report (nếu bỏ trống sẽ mở GUI chọn file)")
    p.add_argument("--out-dir",        default="",     help="Thư mục output (mặc định: output/<stem>)")
    p.add_argument("--metric",         default="jaccard", choices=["jaccard", "cosine"])
    p.add_argument("--w-sig",          type=float, default=0.33, help="Trọng số Signature")
    p.add_argument("--w-mbc",          type=float, default=0.33, help="Trọng số MBC")
    p.add_argument("--w-ttp",          type=float, default=0.34, help="Trọng số TTP")
    p.add_argument("--top-k",          type=int,   default=5,    help="Số mẫu top-K")
    p.add_argument("--W",              type=float, default=1.0,  help="Hệ số W (self)")
    p.add_argument("--beta",           type=float, default=1.0,  help="Hệ số beta (neighbor)")
    p.add_argument("--vt-key",         default="",     help="VirusTotal API key")
    p.add_argument("--no-vt",          action="store_true", help="Bỏ qua bước VT")
    p.add_argument("--tfidf-pattern",  action="store_true", help="Tính TF-IDF trên pattern (dataset+sample) trước similarity")
    p.add_argument("--pattern-sig",    default="",     help="Path pattern_signature.csv")
    p.add_argument("--pattern-ttp",    default="",     help="Path pattern_ttp_update.csv")
    p.add_argument("--pattern-mbc",    default="",     help="Path pattern_mbc_update.csv")
    p.add_argument("--sig-unique",     default="",     help="Path signature_unique.csv")
    p.add_argument("--ttp-unique",     default="",     help="Path ttp_unique.csv")
    p.add_argument("--mbc-unique",     default="",     help="Path mbc_unique.csv")
    p.add_argument("--action-ttps",    default="",     help="Path action_per_ttps.csv")
    p.add_argument("--action-report",  default="",     help="Path actionPerReport.xlsx")
    p.add_argument("--unique-actions", default="",     help="Path unique_actions.csv")
    return p


if __name__ == "__main__":
    setup_logging()
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = load_config()
    if cfg:
        print(f"[config] Loaded {CONFIG_FILE.name}")
    args   = apply_config(args, cfg)

    # Nếu không có --sample, mở GUI chọn file JSON
    if not args.sample:
        print("[GUI] Mở cửa sổ chọn file JSON report...")
        chosen = pick_json_file()
        if not chosen:
            print("Đã hủy chọn file. Thoát.")
            sys.exit(0)
        args.sample = chosen
        print(f"[GUI] Đã chọn: {chosen}")

    run_pipeline(args)

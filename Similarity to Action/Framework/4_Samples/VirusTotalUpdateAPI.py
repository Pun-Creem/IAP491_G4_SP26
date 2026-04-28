import csv
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

# =========================
# CONFIG
# =========================
API_KEY = "223b13835ace35e4fc3be592b937c43f109182b2660e612636b35365a5c6deda"
VT_BASE_URL = "https://www.virustotal.com/api/v3"
HEADERS = {"x-apikey": API_KEY}

# Nếu gặp rate limit thì tăng sleep lên
REQUEST_SLEEP_SECONDS = 0.25
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

# Regex lấy:
# - ATT&CK ID: T1055, T1027.002 ...
# - MBC ID: C0007, C0018, F0001.008 ...
ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
MBC_ID_RE = re.compile(r"\b[BCDF]\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

# Parse block kiểu:
# mbc:
#   - Memory::Allocate Memory [C0007]
MBC_LINE_RE = re.compile(r"^\s*-\s*(.+?)\s*\[([BCDF]\d{4}(?:\.\d{3})?)\]\s*$", re.IGNORECASE)


# =========================
# HTTP HELPERS
# =========================
def vt_get(endpoint: str) -> dict:
    url = f"{VT_BASE_URL}{endpoint}"
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504):
                wait_s = min(2 ** attempt, 30)
                print(f"[WARN] {resp.status_code} at {endpoint}, retry sau {wait_s}s...", file=sys.stderr)
                time.sleep(wait_s)
                continue

            # 404: hash không có dữ liệu trên VT
            if resp.status_code == 404:
                print(f"[INFO] Không tìm thấy dữ liệu VT cho endpoint {endpoint}", file=sys.stderr)
                return {}

            # lỗi khác
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"VT API error {resp.status_code} at {endpoint}: {detail}")

        except Exception as e:
            last_err = e
            wait_s = min(2 ** attempt, 30)
            print(f"[WARN] Exception at {endpoint}: {e}. Retry sau {wait_s}s...", file=sys.stderr)
            time.sleep(wait_s)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} retries: {endpoint}. Last error: {last_err}")


# =========================
# CSV + HASH VALIDATION
# =========================
def load_hashes(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, dtype=str)
    if "hash" not in df.columns:
        raise ValueError(f"File {csv_path} không có cột 'hash'")

    hashes = (
        df["hash"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    hashes = [h for h in hashes if h]
    if not hashes:
        raise ValueError(f"File {csv_path} không có hash hợp lệ trong cột 'hash'")

    return hashes


def validate_same_hashes(ttps_hashes: list[str], mbcs_hashes: list[str]) -> list[str]:
    ttps_set = set(ttps_hashes)
    mbcs_set = set(mbcs_hashes)

    only_in_ttps = sorted(ttps_set - mbcs_set)
    only_in_mbcs = sorted(mbcs_set - ttps_set)

    if only_in_ttps or only_in_mbcs:
        print("Hai file KHÔNG có cùng tập hash.", file=sys.stderr)
        if only_in_ttps:
            print(f"  Có trong ttps nhưng không có trong mbcs: {len(only_in_ttps)}", file=sys.stderr)
            print(f"  Ví dụ: {only_in_ttps[:10]}", file=sys.stderr)
        if only_in_mbcs:
            print(f"  Có trong mbcs nhưng không có trong ttps: {len(only_in_mbcs)}", file=sys.stderr)
            print(f"  Ví dụ: {only_in_mbcs[:10]}", file=sys.stderr)
        raise SystemExit(1)

    # Giữ thứ tự theo file ttps
    return ttps_hashes


# =========================
# PARSERS
# =========================
def parse_mbc_from_rule_src(rule_src: str) -> set[str]:
    """
    Parse các dòng trong rule_src như:
      mbc:
        - Memory::Allocate Memory [C0007]
        - Anti-Static Analysis::Software Packing::UPX [F0001.008]
    Kết quả trả về ID MBC: C0007, F0001.008, ...
    """
    found = set()

    # Cách 1: parse theo pattern của từng dòng MBC
    in_mbc_block = False
    for raw_line in rule_src.splitlines():
        line = raw_line.rstrip("\n")

        if re.match(r"^\s*mbc:\s*$", line, re.IGNORECASE):
            in_mbc_block = True
            continue

        if in_mbc_block:
            # kết thúc block khi sang top-level/meta khác
            if re.match(r"^\s*[A-Za-z0-9_&-]+\s*:\s*$", line) and not re.match(r"^\s*-\s*", line):
                in_mbc_block = False
                continue

            m = MBC_LINE_RE.match(line)
            if m:
                found.add(m.group(2).upper())

    # Cách 2: fallback quét regex mọi MBC ID trong rule_src
    for mbc_id in MBC_ID_RE.findall(rule_src):
        found.add(mbc_id.upper())

    return found


def extract_ttps_and_mbcs_from_behaviours(data: dict) -> tuple[set[str], set[str]]:
    """
    Parse cả TTPs lẫn MBCs từ một response duy nhất của
    GET /files/{hash}/behaviours.

    TTPs: lấy từ attributes.mitre_attack_techniques[].id
          + fallback quét ATTACK_ID trong signature_matches[].rule_src
    MBCs: lấy từ signature_matches[].rule_src (parse block mbc:)
          + fallback từ description/name và field mbc/mbcs trực tiếp
    """
    ttps: set[str] = set()
    mbcs: set[str] = set()

    for item in data.get("data", []) or []:
        attrs = item.get("attributes", {}) or {}

        # --- TTPs ---
        for technique in attrs.get("mitre_attack_techniques", []) or []:
            tech_id = str(technique.get("id", "")).strip().upper()
            if ATTACK_ID_RE.fullmatch(tech_id):
                ttps.add(tech_id)

        # --- MBCs + TTP fallback từ signature_matches ---
        for sig in attrs.get("signature_matches", []) or []:
            rule_src = str(sig.get("rule_src", "") or "")

            # MBC từ rule_src
            mbcs.update(parse_mbc_from_rule_src(rule_src))

            # TTP fallback: đôi khi rule_src chứa T-codes
            for match in ATTACK_ID_RE.findall(rule_src):
                ttps.add(match.upper())

            # MBC fallback: description/name
            for key in ("description", "name"):
                val = str(sig.get(key, "") or "")
                for mbc_id in MBC_ID_RE.findall(val):
                    mbcs.add(mbc_id.upper())

        # --- MBC: field mbc/mbcs trực tiếp (nếu API trả về) ---
        for field in ("mbc", "mbcs"):
            direct = attrs.get(field)
            if isinstance(direct, list):
                for item2 in direct:
                    if isinstance(item2, dict):
                        for key in ("id", "mbc_id", "code"):
                            v = str(item2.get(key, "")).strip().upper()
                            if MBC_ID_RE.fullmatch(v):
                                mbcs.add(v)
                    elif isinstance(item2, str):
                        for mbc_id in MBC_ID_RE.findall(item2):
                            mbcs.add(mbc_id.upper())
            elif isinstance(direct, str):
                for mbc_id in MBC_ID_RE.findall(direct):
                    mbcs.add(mbc_id.upper())

    return ttps, mbcs


# =========================
# VT FETCH
# =========================
def fetch_ttps_and_mbcs_for_hash(file_hash: str) -> tuple[set[str], set[str]]:
    # Một request duy nhất — /behaviours chứa cả TTP lẫn MBC
    behaviours_json = vt_get(f"/files/{file_hash}/behaviours")
    ttps, mbcs = extract_ttps_and_mbcs_from_behaviours(behaviours_json)

    time.sleep(REQUEST_SLEEP_SECONDS)
    return ttps, mbcs


# =========================
# OUTPUT WRITER
# =========================
def get_output_path(base_dir: Path, name: str = "vt_update") -> Path:
    candidate = base_dir / f"{name}.csv"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = base_dir / f"{name}_{i}.csv"
        if not candidate.exists():
            return candidate
        i += 1


def write_vt_update(path: Path, hashes: list[str],
                    ttps_map: dict[str, set[str]],
                    mbcs_map: dict[str, set[str]]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["sha256", "ttps", "mbc"])
        for h in hashes:
            ttps_str = "\n".join(sorted(ttps_map.get(h, set())))
            mbc_str = "\n".join(sorted(mbcs_map.get(h, set())))
            writer.writerow([h, ttps_str, mbc_str])


# =========================
# FILE PICKER
# =========================
def pick_csv(title: str, initial_dir: str = "") -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir or "/",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        print(f"[CANCEL] Không có file nào được chọn ({title}). Thoát.", file=sys.stderr)
        raise SystemExit(0)
    return Path(path)


def pick_folder(title: str, initial_dir: str = "") -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title, initialdir=initial_dir or "/")
    root.destroy()
    if not folder:
        print(f"[CANCEL] Không có folder nào được chọn ({title}). Thoát.", file=sys.stderr)
        raise SystemExit(0)
    return Path(folder)


# =========================
# MAIN
# =========================
def main():
    setup_logging()
    # Mở cửa sổ chọn file TTPs
    ttps_path = pick_csv("Chọn file CSV TTPs")
    # Mở cửa sổ chọn file MBCs, bắt đầu từ cùng thư mục
    mbcs_path = pick_csv("Chọn file CSV MBCs", initial_dir=str(ttps_path.parent))
    # Mở cửa sổ chọn folder output
    out_dir = pick_folder("Chọn folder lưu file output", initial_dir=str(ttps_path.parent))

    ttps_hashes = load_hashes(ttps_path)
    mbcs_hashes = load_hashes(mbcs_path)
    hashes = validate_same_hashes(ttps_hashes, mbcs_hashes)

    print(f"[OK] Hai file có cùng tập hash. Tổng số hash: {len(set(hashes))}")

    ttps_map: dict[str, set[str]] = {}
    mbcs_map: dict[str, set[str]] = {}

    total = len(hashes)
    for idx, h in enumerate(hashes, start=1):
        print(f"[{idx}/{total}] Fetch VT cho hash: {h}")
        try:
            ttps, mbcs = fetch_ttps_and_mbcs_for_hash(h)
            ttps_map[h] = ttps
            mbcs_map[h] = mbcs
            print(f"    TTPs={len(ttps)} | MBCs={len(mbcs)}")
        except Exception as e:
            print(f"    [ERROR] {h}: {e}", file=sys.stderr)
            ttps_map[h] = set()
            mbcs_map[h] = set()

    out_path = get_output_path(out_dir)

    write_vt_update(out_path, hashes, ttps_map, mbcs_map)

    print(f"[DONE] Đã ghi: {out_path}")


if __name__ == "__main__":
    main()
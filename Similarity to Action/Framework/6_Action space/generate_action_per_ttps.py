"""
Script: generate_action_per_ttps.py

Đọc TTPs từ ttp_unique.csv, query D3FEND API để mapping TTP → D3FEND actions,
rồi tạo file CSV tương tự action_per_ttps.csv (nếu trùng tên thì thêm _1).
"""

import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

# ── Cấu hình ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
INPUT_TTP    = SCRIPT_DIR / "ttp_unique.csv"
REFERENCE    = SCRIPT_DIR / "action_per_ttps.csv"
OUTPUT_BASE  = SCRIPT_DIR / "action_per_ttps.csv"
API_BASE     = "https://d3fend.mitre.org/api/offensive-technique/attack/{tid}.json"
SLEEP_SEC    = 0.3   # rate-limit courtesy delay between requests

# ── Helpers ──────────────────────────────────────────────────────────────────

def resolve_output_path(base: Path) -> Path:
    """Nếu file đã tồn tại thì trả về <stem>_1<suffix>, v.v."""
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 1
    while True:
        candidate = base.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def read_ttp_list(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["feature"].strip() for row in reader if row["feature"].strip()]


def read_action_columns(path: Path) -> list[str]:
    """Đọc dòng header của file tham chiếu, bỏ cột 'ttps', trả về list action names."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [col.strip() for col in header[1:]]  # bỏ cột đầu 'ttps'


def fetch_d3fend_actions(tid: str) -> set[str]:
    """
    Query D3FEND API cho một ATT&CK technique ID.
    Trả về set các action label dạng "D3-XXX - Full Name" hoặc "D3-XXX".
    """
    url = API_BASE.format(tid=tid)
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "ttp-mapper/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return set()   # TTP không có trong D3FEND
        print(f"  [WARN] HTTP {e.code} for {tid}")
        return set()
    except Exception as e:
        print(f"  [WARN] Request failed for {tid}: {e}")
        return set()

    # Trích xuất tên kỹ thuật phòng thủ từ response JSON
    # Cấu trúc: data["off_to_def"]["results"]["bindings"] → list of dict
    actions: set[str] = set()
    try:
        bindings = data["off_to_def"]["results"]["bindings"]
        for b in bindings:
            # Lấy label và ID ngắn trực tiếp từ API
            label = b.get("def_tech_label", {}).get("value", "")
            short_id = b.get("def_tech_id", {}).get("value", "")  # e.g. "D3-APCA"

            if label and short_id:
                actions.add(f"{short_id} - {label}")
                actions.add(short_id)
            elif short_id:
                actions.add(short_id)
            elif label:
                actions.add(label)
    except (KeyError, TypeError):
        pass

    return actions


def action_matches(action_col: str, d3fend_actions: set[str]) -> int:
    """
    Kiểm tra xem action_col (header) có xuất hiện trong tập d3fend_actions không.
    action_col có thể là "D3-APCA - Application Protocol Command Analysis"
    hoặc chỉ "D3-APCA".
    """
    if action_col in d3fend_actions:
        return 1
    # Thử khớp theo prefix ID (phần trước dấu " - ")
    col_id = action_col.split(" - ")[0].strip()
    if col_id in d3fend_actions:
        return 1
    # Thử khớp ngược: d3fend trả về "D3-XXX - Label", cột chỉ có "D3-XXX - Label"
    for act in d3fend_actions:
        act_id = act.split(" - ")[0].strip()
        if act_id == col_id:
            return 1
    return 0


def normalize_action_id(action: str) -> str:
    """Lấy phần ID (D3-XXX) từ action string."""
    return action.split(" - ")[0].strip()


def find_matching_column(action: str, existing_columns: list[str]) -> str | None:
    """Kiểm tra xem action đã có trong danh sách cột chưa (khớp theo ID)."""
    action_id = normalize_action_id(action)
    for col in existing_columns:
        if normalize_action_id(col) == action_id:
            return col
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    print("=== D3FEND TTP -> Action Mapper ===\n")

    # 1. Đọc danh sách TTP
    ttps = read_ttp_list(INPUT_TTP)
    print(f"Loaded {len(ttps)} TTPs from {INPUT_TTP.name}")

    # 2. Đọc cột action hiện có từ file tham chiếu
    action_columns = read_action_columns(REFERENCE)
    print(f"Loaded {len(action_columns)} existing action columns from {REFERENCE.name}\n")

    # 3. Query tất cả TTPs trước, thu thập action và phát hiện action mới
    print("--- Phase 1: Query D3FEND API cho tất cả TTPs ---")
    ttp_actions_map: dict[str, set[str]] = {}  # ttp -> set of actions from API
    new_actions: list[str] = []  # action mới chưa có trong header (giữ dạng "D3-XXX - Label")

    for i, ttp in enumerate(ttps, 1):
        print(f"[{i:3d}/{len(ttps)}] Querying {ttp} ...", end=" ", flush=True)
        d3fend_actions = fetch_d3fend_actions(ttp)
        ttp_actions_map[ttp] = d3fend_actions

        # Lọc ra các action dạng "D3-XXX - Label" (bỏ các entry chỉ có ID ngắn)
        labeled_actions = {a for a in d3fend_actions if " - " in a}
        short_only = {a for a in d3fend_actions if " - " not in a}

        # Kiểm tra action mới
        for act in labeled_actions:
            if find_matching_column(act, action_columns + new_actions) is None:
                new_actions.append(act)
        # Với action chỉ có ID ngắn mà chưa match
        for act in short_only:
            if find_matching_column(act, action_columns + new_actions) is None:
                new_actions.append(act)

        print(f"-> {len(labeled_actions)} actions found")
        time.sleep(SLEEP_SEC)

    if new_actions:
        print(f"\n*** Phát hiện {len(new_actions)} action MỚI: {new_actions}")
    else:
        print("\nKhông có action mới nào ngoài danh sách hiện có.")

    # 4. Ghép header: cột cũ + cột mới
    all_columns = action_columns + new_actions
    print(f"\nTổng cộng {len(all_columns)} action columns ({len(action_columns)} cũ + {len(new_actions)} mới)\n")

    # 5. Xác định đường dẫn output
    output_path = resolve_output_path(OUTPUT_BASE)
    print(f"Output → {output_path.name}\n")

    # 6. Ghi file CSV
    print("--- Phase 2: Ghi file CSV ---")
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["ttps"] + all_columns)

        for ttp in ttps:
            d3fend_actions = ttp_actions_map[ttp]
            row = [ttp] + [action_matches(col, d3fend_actions) for col in all_columns]
            writer.writerow(row)

    print(f"\nDone! Saved to: {output_path.name}")


if __name__ == "__main__":
    main()

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def pick_csv_file():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Chọn file action_per_ttps CSV đầy đủ nhất",
        initialdir=BASE_DIR,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return path

def get_output_path():
    base = os.path.join(BASE_DIR, "unique_actions.csv")
    if not os.path.exists(base):
        return base
    i = 1
    while True:
        candidate = os.path.join(BASE_DIR, f"unique_actions_{i}.csv")
        if not os.path.exists(candidate):
            return candidate
        i += 1

def parse_action_id(header):
    return header.split(" - ")[0].strip()

def main():
    setup_logging()
    csv_path = pick_csv_file()
    if not csv_path:
        print("Không có file nào được chọn. Thoát.")
        sys.exit(0)

    print(f"Đã chọn: {csv_path}")

    df_ttps = pd.read_csv(csv_path)
    action_headers = [col for col in df_ttps.columns if col != df_ttps.columns[0]]
    print(f"Tổng số action từ header: {len(action_headers)}")

    d3fend_path = os.path.join(BASE_DIR, "d3fend.csv")
    df_d3 = pd.read_csv(d3fend_path)
    d3_lookup = {}
    for _, row in df_d3.iterrows():
        d3_id = str(row.get("ID", "")).strip()
        if d3_id:
            d3_lookup[d3_id] = row.to_dict()

    rows = []
    for i, header in enumerate(action_headers, start=1):
        meta = d3_lookup.get(parse_action_id(header), {})
        rows.append({
            "No": i,
            "Action": header,
            "D3FEND Tactic": meta.get("D3FEND Tactic", "") or "",
            "D3FEND Technique": meta.get("D3FEND Technique", "") or "",
            "D3FEND Technique Level 0": meta.get("D3FEND Technique Level 0", "") or "",
            "D3FEND Technique Level 1": meta.get("D3FEND Technique Level 1", "") or "",
            "Definition": meta.get("Definition", "") or "",
        })

    output_path = get_output_path()
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {output_path}")
    print(f"Tổng số actions: {len(rows)}")

if __name__ == "__main__":
    main()

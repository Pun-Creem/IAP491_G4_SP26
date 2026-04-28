import csv
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict
import re
import os
from pathlib import Path
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from log_utils import setup_logging

def get_valid_d3fend_ids(d3fend_path):
    ids = set()
    with open(d3fend_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_val = row.get('ID', '').strip()
            if id_val:
                ids.add(id_val)
    return ids

def extract_action_id(action_str):
    match = re.match(r'(D3-[A-Z0-9]+)', action_str.strip())
    if match:
        return match.group(1)
    return None

def main():
    setup_logging()
    root = tk.Tk()
    root.withdraw()

    # Prompt user to select input XLSX
    input_path = filedialog.askopenfilename(
        title="Chọn file XLSX đầu vào",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    if not input_path:
        messagebox.showinfo("Thông báo", "Không có file nào được chọn.")
        return

    # Path to d3fend.csv (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    d3fend_path = os.path.join(script_dir, "d3fend.csv")

    if not os.path.exists(d3fend_path):
        messagebox.showerror("Lỗi", f"Không tìm thấy file d3fend.csv tại:\n{d3fend_path}")
        return

    valid_ids = get_valid_d3fend_ids(d3fend_path)

    # Read input XLSX
    try:
        df = pd.read_excel(input_path, sheet_name=0, dtype=str)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không đọc được file Excel:\n{e}")
        return

    # Detect column names (case-insensitive)
    col_map = {c.strip().lower(): c for c in df.columns}
    hash_col = col_map.get('hash256') or col_map.get('hash')
    action_col = col_map.get('action')

    if not hash_col or not action_col:
        messagebox.showerror(
            "Lỗi",
            f"Không tìm thấy cột 'Hash256' hoặc 'Action' trong file.\nCác cột hiện có: {list(df.columns)}"
        )
        return

    # hash -> list of actions not found in d3fend.csv
    missing = defaultdict(list)

    for _, row in df.iterrows():
        hash_val = str(row[hash_col]).strip()
        action_cell = str(row[action_col]).strip()

        if not hash_val or not action_cell or hash_val == 'nan' or action_cell == 'nan':
            continue

        # Each cell may contain multiple actions separated by newlines
        for action_val in action_cell.split('\n'):
            action_val = action_val.strip()
            if not action_val:
                continue
            action_id = extract_action_id(action_val)
            if action_id is None:
                missing[hash_val].append(action_val)
            elif action_id not in valid_ids:
                missing[hash_val].append(action_val)

    if not missing:
        messagebox.showinfo("Kết quả", "Tất cả các action đều tồn tại trong d3fend.csv. Không có gì để ghi ra file.")
        return

    # Write output txt
    output_path = os.path.splitext(input_path)[0] + "_missing_actions.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Hash256\tAction không tồn tại trong D3FEND\n")
        f.write("=" * 80 + "\n")
        for hash_val, actions in sorted(missing.items()):
            for action in actions:
                f.write(f"{hash_val}\t{action}\n")

    messagebox.showinfo(
        "Hoàn thành",
        f"Đã tìm thấy {sum(len(v) for v in missing.values())} action không tồn tại "
        f"từ {len(missing)} hash.\nKết quả đã ghi vào:\n{output_path}"
    )

if __name__ == "__main__":
    main()

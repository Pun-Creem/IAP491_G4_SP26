import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging
setup_logging()

# --- File picker ---
root = tk.Tk()
root.withdraw()

csv_path = filedialog.askopenfilename(
    title="Chọn file merged similarity CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not csv_path:
    print("Không có file được chọn. Thoát.")
    exit()

print(f"File đã chọn: {csv_path}")

# --- Extract target number from filename ---
filename = os.path.basename(csv_path)
match = re.match(r"(\d+)_", filename)
target_number = match.group(1) if match else "target"

# --- Ask weights ---
def get_float(prompt, allow_any=False):
    while True:
        try:
            val = float(input(prompt))
            if not allow_any and not (0 <= val <= 1):
                print("  Giá trị phải trong khoảng [0, 1]. Thử lại.")
                continue
            return val
        except ValueError:
            print("  Vui lòng nhập số thực. Thử lại.")

print("\nNhập trọng số (tổng phải = 1):")
while True:
    w_sig = get_float("  Trọng số Signature: ")
    w_mbc = get_float("  Trọng số MBC: ")
    w_ttp = get_float("  Trọng số TTP: ")
    total = round(w_sig + w_mbc + w_ttp, 6)
    if abs(total - 1.0) < 1e-6:
        break
    print(f"  Tổng = {total} ≠ 1. Vui lòng nhập lại.")

# --- Ask top k ---
while True:
    try:
        k = int(input("\nTop K = "))
        if k > 0:
            break
        print("  K phải > 0.")
    except ValueError:
        print("  Vui lòng nhập số nguyên.")

print(f"\nTrọng số: Signature={w_sig}, MBC={w_mbc}, TTP={w_ttp} | Top K={k}")

# --- Load similarity CSV ---
df = pd.read_csv(csv_path)
df["weighted_similarity"] = (
    df["signatures_similarity"] * w_sig +
    df["mbcs_similarity"] * w_mbc +
    df["ttps_similarity"] * w_ttp
)

# Get top k dataset_ids by weighted similarity
top_k_df = df.nlargest(k, "weighted_similarity")[["dataset_id", "weighted_similarity"]].reset_index(drop=True)
top_k_hashes = top_k_df["dataset_id"].tolist()

print(f"\nTop {k} hashes:")
for i, (h, s) in enumerate(zip(top_k_df["dataset_id"], top_k_df["weighted_similarity"]), 1):
    print(f"  {i}. {h[:16]}... | similarity={s:.4f}")

# --- Load unique actions ---
csv_dir = os.path.normpath(os.path.dirname(csv_path))
script_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
unique_actions_path = os.path.join(script_dir, "unique_actions.csv")
actions_df = pd.read_csv(unique_actions_path)
all_actions = actions_df["Action"].tolist()

# --- Load actionPerReport ---
report_path = os.path.join(script_dir, "actionPerReport.xlsx")
report_df = pd.read_excel(report_path)

# Build dict: hash -> set of actions
hash_to_actions = {}
for _, row in report_df.iterrows():
    h = str(row["Hash256"]).strip()
    raw = str(row["Action"]) if pd.notna(row["Action"]) else ""
    acts = {a.strip() for a in raw.split("\n") if a.strip()}
    hash_to_actions[h] = acts

# --- Build output dataframe ---
sim_map = dict(zip(top_k_df["dataset_id"], top_k_df["weighted_similarity"]))
records = []
for h in top_k_hashes:
    row = {"sha256": h, "similarity": sim_map[h]}
    act_set = hash_to_actions.get(h, set())
    for action in all_actions:
        row[action] = 1 if action in act_set else 0
    records.append(row)

out_df = pd.DataFrame(records)

# --- Save output ---
out_filename = f"{target_number}_top{k}.csv"
out_path = os.path.join(csv_dir, out_filename)
out_df.to_csv(out_path, index=False)

print(f"\nDone! File đầu ra: {out_path}")
print(f"  Số dòng: {len(out_df)} | Số cột: {len(out_df.columns)} (1 sha256 + {len(all_actions)} actions)")

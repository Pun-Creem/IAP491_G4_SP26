import pandas as pd
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

def main():
    setup_logging()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Paths ---
    action_per_ttps_path = os.path.join(script_dir, "action_per_ttps.csv")

    # Hide root Tk window
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # File picker for target CSV
    target_path = filedialog.askopenfilename(
        title="Chọn file sample CSV",
        initialdir=script_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not target_path:
        messagebox.showwarning("Hủy", "Không có file nào được chọn. Thoát.")
        sys.exit(0)

    if not os.path.exists(target_path):
        messagebox.showerror("Lỗi", f"Không tìm thấy file: {target_path}")
        sys.exit(1)

    # Extract leading number from filename for output name
    basename = os.path.basename(target_path)
    match = re.match(r"(\d+)", basename)
    prefix = match.group(1) if match else os.path.splitext(basename)[0]
    output_dir = os.path.dirname(target_path)

    # --- Load data ---
    print(f"Đang đọc {action_per_ttps_path} ...")
    action_df = pd.read_csv(action_per_ttps_path, index_col=0)
    # action_df: index = TTP name, columns = action names, values = 0/1

    print(f"Đang đọc {target_path} ...")
    target_df = pd.read_csv(target_path, index_col=0)
    # target_df: index = sha256, columns = TTP names, values = 0/1

    action_names = action_df.columns.tolist()

    # --- Prompt user ---
    allow_duplicates = messagebox.askyesno(
        title="Cho phép trùng lặp action?",
        message=(
            "Cho phép trùng lặp action?\n\n"
            "Yes - Cộng dồn: action xuất hiện ở nhiều TTP sẽ được cộng tổng\n"
            "No  - Nhị phân: action chỉ tính là 1 dù xuất hiện nhiều lần"
        )
    )

    output_suffix = "_mapped_action_dupe.csv" if allow_duplicates else "_mapped_action.csv"
    output_path = os.path.join(output_dir, f"{prefix}{output_suffix}")

    print(f"\nChế độ: {'Cộng dồn (sum)' if allow_duplicates else 'Nhị phân (0/1)'}")

    # --- Map ---
    results = []

    for sha256, ttp_row in target_df.iterrows():
        # Get TTPs this sample has
        active_ttps = ttp_row[ttp_row == 1].index.tolist()

        # Find intersection with action_df index
        valid_ttps = [t for t in active_ttps if t in action_df.index]

        if valid_ttps:
            # Sum action vectors for all active TTPs
            action_sum = action_df.loc[valid_ttps].sum(axis=0)
        else:
            action_sum = pd.Series(0, index=action_names)

        if not allow_duplicates:
            action_sum = (action_sum > 0).astype(int)

        row = {"sha256": sha256}
        row.update(action_sum.to_dict())
        results.append(row)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)

    print(f"\nDone! Output: {output_path}")
    print(f"  Samples: {len(output_df)}")
    print(f"  Actions: {len(action_names)}")

    messagebox.showinfo(
        "Hoàn tất",
        f"Output: {output_path}\nSamples: {len(output_df)}\nActions: {len(action_names)}"
    )

if __name__ == "__main__":
    main()

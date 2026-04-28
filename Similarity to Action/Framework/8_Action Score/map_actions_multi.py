import pandas as pd
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


def map_file(action_df, target_path):
    action_names = action_df.columns.tolist()
    basename = os.path.basename(target_path)
    match = re.match(r"(\d+)", basename)
    prefix = match.group(1) if match else os.path.splitext(basename)[0]
    output_dir = os.path.dirname(target_path)

    print(f"\nĐang xử lý: {basename}")
    target_df = pd.read_csv(target_path, index_col=0)

    results_sum = []
    results_bin = []

    for sha256, ttp_row in target_df.iterrows():
        active_ttps = ttp_row[ttp_row == 1].index.tolist()
        valid_ttps = [t for t in active_ttps if t in action_df.index]

        if valid_ttps:
            action_sum = action_df.loc[valid_ttps].sum(axis=0)
        else:
            action_sum = pd.Series(0, index=action_names)

        row_sum = {"sha256": sha256}
        row_sum.update(action_sum.to_dict())
        results_sum.append(row_sum)

        action_bin = (action_sum > 0).astype(int)
        row_bin = {"sha256": sha256}
        row_bin.update(action_bin.to_dict())
        results_bin.append(row_bin)

    out_dupe = os.path.join(output_dir, f"{prefix}_mapped_action_dupe.csv")
    out_bin  = os.path.join(output_dir, f"{prefix}_mapped_action.csv")

    pd.DataFrame(results_sum).to_csv(out_dupe, index=False)
    pd.DataFrame(results_bin).to_csv(out_bin,  index=False)

    print(f"  -> {os.path.basename(out_dupe)}")
    print(f"  -> {os.path.basename(out_bin)}")
    return len(target_df)


def main():
    setup_logging()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    action_per_ttps_path = os.path.join(script_dir, "action_per_ttps.csv")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if not os.path.exists(action_per_ttps_path):
        messagebox.showerror("Lỗi", f"Không tìm thấy: {action_per_ttps_path}")
        sys.exit(1)

    print(f"Đang đọc {action_per_ttps_path} ...")
    action_df = pd.read_csv(action_per_ttps_path, index_col=0)

    last_dir = script_dir

    while True:
        target_paths = filedialog.askopenfilenames(
            title="Chọn một hoặc nhiều file sample CSV",
            initialdir=last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not target_paths:
            messagebox.showwarning("Hủy", "Không có file nào được chọn. Thoát.")
            break

        last_dir = os.path.dirname(target_paths[0])

        total_files = len(target_paths)
        total_samples = 0

        for path in target_paths:
            if not os.path.exists(path):
                print(f"  [BỎ QUA] Không tìm thấy: {path}")
                continue
            total_samples += map_file(action_df, path)

        summary = (
            f"Hoàn tất {total_files} file(s), {total_samples} sample(s).\n"
            f"Mỗi file đã tạo cả 2 output:\n"
            f"  *_mapped_action_dupe.csv  (cộng dồn)\n"
            f"  *_mapped_action.csv       (nhị phân)"
        )
        print(f"\n{summary}")

        more = messagebox.askyesno(
            "Tiếp tục?",
            f"{summary}\n\nBạn có muốn xử lý thêm file không?"
        )
        if not more:
            break

    print("\nKết thúc chương trình.")


if __name__ == "__main__":
    main()

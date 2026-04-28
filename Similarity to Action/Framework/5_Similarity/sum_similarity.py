import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


# ====== CẤU HÌNH TÊN CỘT GỐC ======
TARGET_COL = "target_id"
DATASET_COL = "dataset_id"
SIM_TYPE_COL = "similarity_type"
SCORE_COL = "similarity_score"


FILE_ORDER = ["signature", "mbc", "ttp"]
FINAL_COLUMN_ORDER = [
    TARGET_COL,
    DATASET_COL,
    SIM_TYPE_COL,
    "signatures_similarity",
    "mbcs_similarity",
    "ttps_similarity",
]


def select_csv_file(title: str) -> str:
    return filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )



def load_and_validate_csv(file_path: str, source_name: str) -> pd.DataFrame:
    if not file_path:
        raise ValueError(f"Chưa chọn file {source_name}.")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Không đọc được file {source_name}: {file_path}\nLỗi: {e}")

    required_cols = {TARGET_COL, DATASET_COL, SIM_TYPE_COL, SCORE_COL}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"File {source_name} thiếu cột: {', '.join(missing_cols)}\n"
            f"Các cột hiện có: {', '.join(df.columns)}"
        )

    df = df[[TARGET_COL, DATASET_COL, SIM_TYPE_COL, SCORE_COL]].copy()

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df[DATASET_COL] = df[DATASET_COL].astype(str).str.strip()
    df[SIM_TYPE_COL] = df[SIM_TYPE_COL].astype(str).str.strip().str.lower()
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")

    duplicate_mask = df.duplicated(subset=[TARGET_COL, DATASET_COL, SIM_TYPE_COL], keep=False)
    if duplicate_mask.any():
        dup_rows = df.loc[duplicate_mask, [TARGET_COL, DATASET_COL, SIM_TYPE_COL]]
        raise ValueError(
            f"File {source_name} có key bị trùng theo bộ "
            f"({TARGET_COL}, {DATASET_COL}, {SIM_TYPE_COL}).\n"
            f"Ví dụ dòng trùng:\n{dup_rows.head(10).to_string(index=False)}"
        )

    return df



def merge_similarity_files(signature_file: str, mbc_file: str, ttp_file: str) -> pd.DataFrame:
    df_signature = load_and_validate_csv(signature_file, "signature")
    df_mbc = load_and_validate_csv(mbc_file, "mbc")
    df_ttp = load_and_validate_csv(ttp_file, "ttp")

    df_signature = df_signature.rename(columns={SCORE_COL: "signatures_similarity"})
    df_mbc = df_mbc.rename(columns={SCORE_COL: "mbcs_similarity"})
    df_ttp = df_ttp.rename(columns={SCORE_COL: "ttps_similarity"})

    merge_keys = [TARGET_COL, DATASET_COL, SIM_TYPE_COL]

    merged_df = (
        df_signature.merge(df_mbc, on=merge_keys, how="outer")
        .merge(df_ttp, on=merge_keys, how="outer")
    )

    merged_df = merged_df[FINAL_COLUMN_ORDER]
    merged_df = merged_df.sort_values(
        by=[TARGET_COL, DATASET_COL, SIM_TYPE_COL],
        kind="stable",
    ).reset_index(drop=True)

    return merged_df



def main() -> None:
    setup_logging()

    root = tk.Tk()
    root.withdraw()

    try:
        messagebox.showinfo(
            "Bắt đầu",
            "Hãy chọn 3 file CSV theo đúng thứ tự: signature, mbc, ttp.",
        )

        signature_file = select_csv_file("Chọn file CSV cho signature")
        mbc_file = select_csv_file("Chọn file CSV cho mbc")
        ttp_file = select_csv_file("Chọn file CSV cho ttp")

        if not signature_file or not mbc_file or not ttp_file:
            messagebox.showwarning("Thiếu file", "Bạn chưa chọn đủ 3 file CSV theo thứ tự signature, mbc, ttp.")
            return

        merged_df = merge_similarity_files(signature_file, mbc_file, ttp_file)

        output_file = filedialog.asksaveasfilename(
            title="Lưu file kết quả",
            defaultextension=".csv",
            initialfile="merged_similarity_report.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not output_file:
            messagebox.showinfo("Đã hủy", "Bạn chưa chọn nơi lưu file kết quả.")
            return

        merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        messagebox.showinfo(
            "Thành công",
            f"Đã tổng hợp xong file:\n{output_file}\n\n"
            f"Thứ tự file đã nhận: {', '.join(FILE_ORDER)}\n"
            f"Thứ tự cột kết quả: {', '.join(FINAL_COLUMN_ORDER)}\n"
            f"Số dòng kết quả: {len(merged_df)}",
        )

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

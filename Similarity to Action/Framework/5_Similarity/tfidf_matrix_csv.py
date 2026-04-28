import os
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


ID_CANDIDATES = ["hash", "id", "sample_id", "sha256", "md5", "name"]


def pick_csv_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Chọn file CSV matrix",
        filetypes=[("CSV files", "*.csv")],
    )
    root.destroy()
    return file_path


def detect_id_column(df: pd.DataFrame):
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col

    first_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first_col]):
        return first_col

    return None


def build_output_path(input_path: str) -> str:
    folder = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    return os.path.join(folder, f"TF-IDF_{filename}")


def compute_tfidf_matrix(input_path: str) -> str:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("File CSV rỗng.")

    id_col = detect_id_column(df)

    if id_col is not None:
        id_series = df[id_col]
        feature_df = df.drop(columns=[id_col])
    else:
        id_series = None
        feature_df = df.copy()

    if feature_df.empty:
        raise ValueError("Không tìm thấy cột feature để tính TF-IDF.")

    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    if feature_df.isna().all().all():
        raise ValueError("Toàn bộ cột feature không phải số, không thể tính TF-IDF.")

    feature_df = feature_df.fillna(0.0)

    transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf_matrix = transformer.fit_transform(feature_df.values)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_df.columns)

    if id_series is not None:
        output_df = pd.concat([id_series.reset_index(drop=True), tfidf_df], axis=1)
    else:
        output_df = tfidf_df

    output_path = build_output_path(input_path)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def show_message(title: str, message: str, is_error: bool = False):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if is_error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    setup_logging()
    try:
        csv_path = pick_csv_file()
        if not csv_path:
            print("Đã hủy chọn file.")
            sys.exit(0)

        output_path = compute_tfidf_matrix(csv_path)
        msg = f"Đã tạo file TF-IDF:\n{output_path}"
        print(msg)
        show_message("Hoàn tất", msg)
    except Exception as e:
        err = f"Lỗi: {e}\n\n{traceback.format_exc()}"
        print(err)
        show_message("Lỗi", err, is_error=True)
        sys.exit(1)

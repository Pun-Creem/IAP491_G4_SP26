import csv
import math
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


def read_table_file(file_path):
    """
    Đọc file txt/csv/xlsx dạng bảng.
    Yêu cầu:
    - Dòng đầu là header
    - Cột đầu là ID/hash
    - Các cột sau là feature số
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)

    elif ext == ".txt":
        df = pd.read_csv(file_path, sep="\t")

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {file_path}")

    if df.shape[1] < 2:
        raise ValueError("File phải có ít nhất 2 cột.")

    header = list(df.columns)
    feature_names = header[1:]

    rows = []

    for i, row in df.iterrows():

        item_id = str(row.iloc[0]).strip()

        try:
            vector = [float(x) for x in row.iloc[1:]]
        except:
            raise ValueError(
                f"Dòng {i+2} chứa giá trị không phải số."
            )

        rows.append({
            "id": item_id,
            "vector": vector
        })

    if not rows:
        raise ValueError("Không tìm thấy dòng dữ liệu hợp lệ.")

    return feature_names, rows


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def jaccard_similarity(vec1, vec2):

    bin1 = [1 if x > 0 else 0 for x in vec1]
    bin2 = [1 if x > 0 else 0 for x in vec2]

    intersection = sum(1 for a, b in zip(bin1, bin2) if a == 1 and b == 1)
    union = sum(1 for a, b in zip(bin1, bin2) if a == 1 or b == 1)

    if union == 0:
        return 0.0

    return intersection / union


def save_to_csv(output_path, target_id, metric_name, results):

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["target_id", "dataset_id", "similarity_type", "similarity_score"]
        )

        for row in results:
            writer.writerow([
                target_id,
                row["dataset_id"],
                metric_name,
                f"{row['score']:.10f}"
            ])


def choose_metric():

    result = {"metric": None}

    win = tk.Toplevel()
    win.title("Chọn loại similarity")
    win.geometry("300x150")

    label = tk.Label(win, text="Chọn loại similarity", font=("Arial", 11))
    label.pack(pady=15)

    def set_metric(metric_name):
        result["metric"] = metric_name
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)

    tk.Button(
        btn_frame,
        text="Jaccard",
        width=12,
        command=lambda: set_metric("jaccard")
    ).grid(row=0, column=0, padx=10)

    tk.Button(
        btn_frame,
        text="Cosine",
        width=12,
        command=lambda: set_metric("cosine")
    ).grid(row=0, column=1, padx=10)

    win.grab_set()
    win.wait_window()

    return result["metric"]


def main():
    setup_logging()

    root = tk.Tk()
    root.withdraw()

    try:

        messagebox.showinfo(
            "Chọn file target",
            "Hãy chọn file target (csv/xlsx/txt)."
        )

        target_path = filedialog.askopenfilename(
            title="Chọn file target",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.txt"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("Text", "*.txt"),
            ]
        )

        if not target_path:
            return

        messagebox.showinfo(
            "Chọn file dataset",
            "Hãy chọn file dataset report (csv/xlsx/txt)."
        )

        dataset_path = filedialog.askopenfilename(
            title="Chọn file dataset",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.txt"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("Text", "*.txt"),
            ]
        )

        if not dataset_path:
            return

        metric = choose_metric()

        target_features, target_rows = read_table_file(target_path)
        dataset_features, dataset_rows = read_table_file(dataset_path)

        if target_features != dataset_features:
            raise ValueError(
                "Header feature giữa target và dataset không giống nhau."
            )

        if len(target_rows) != 1:
            raise ValueError("File target phải chứa đúng 1 dòng dữ liệu.")

        target_id = target_rows[0]["id"]
        target_vector = target_rows[0]["vector"]

        results = []

        for row in dataset_rows:

            dataset_id = row["id"]
            dataset_vector = row["vector"]

            if metric == "cosine":
                score = cosine_similarity(target_vector, dataset_vector)
            else:
                score = jaccard_similarity(target_vector, dataset_vector)

            results.append({
                "dataset_id": dataset_id,
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        target_base = os.path.splitext(os.path.basename(target_path))[0]

        output_path = os.path.join(
            os.path.dirname(target_path),
            f"{target_base}_similarity.csv"
        )

        save_to_csv(output_path, target_id, metric, results)

        messagebox.showinfo(
            "Hoàn tất",
            f"Đã lưu kết quả:\n{output_path}"
        )

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))


if __name__ == "__main__":
    main()
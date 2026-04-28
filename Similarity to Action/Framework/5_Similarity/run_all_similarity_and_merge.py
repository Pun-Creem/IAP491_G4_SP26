import os
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import sim
import sum_similarity as merger

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


PAIR_CONFIGS = [
    ("signature", "signatures_similarity"),
    ("mbc", "mbcs_similarity"),
    ("ttp", "ttps_similarity"),
]

DATASET_PATTERN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset pattern")

DATASET_FILES = {
    "signature": os.path.join(DATASET_PATTERN_DIR, "pattern_signature.csv"),
    "mbc":       os.path.join(DATASET_PATTERN_DIR, "pattern_mbc_update.csv"),
    "ttp":       os.path.join(DATASET_PATTERN_DIR, "pattern_ttp_update.csv"),
}


FINAL_COLUMN_ORDER = [
    merger.TARGET_COL,
    merger.DATASET_COL,
    merger.SIM_TYPE_COL,
    "signatures_similarity",
    "mbcs_similarity",
    "ttps_similarity",
]


def ask_data_file(title: str) -> str:
    return filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Data files", "*.csv *.xlsx *.xls *.txt"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx *.xls"),
            ("Text", "*.txt"),
            ("All files", "*.*"),
        ],
    )



def run_single_similarity(category_name: str) -> str:
    target_path = ask_data_file(f"Chọn file target cho {category_name}")
    if not target_path:
        raise RuntimeError(f"Đã hủy chọn file target cho {category_name}.")

    dataset_path = DATASET_FILES[category_name]
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dataset cho {category_name}:\n{dataset_path}"
        )

    metric = sim.choose_metric()
    if metric is None:
        raise RuntimeError(f"Đã hủy chọn similarity type cho {category_name}.")

    target_features, target_rows = sim.read_table_file(target_path)
    dataset_features, dataset_rows = sim.read_table_file(dataset_path)

    if target_features != dataset_features:
        raise ValueError(
            f"Header feature giữa target và dataset của {category_name} không giống nhau."
        )

    if len(target_rows) != 1:
        raise ValueError(f"File target của {category_name} phải chứa đúng 1 dòng dữ liệu.")

    target_id = target_rows[0]["id"]
    target_vector = target_rows[0]["vector"]

    results = []
    for row in dataset_rows:
        dataset_id = row["id"]
        dataset_vector = row["vector"]

        if metric == "cosine":
            score = sim.cosine_similarity(target_vector, dataset_vector)
        else:
            score = sim.jaccard_similarity(target_vector, dataset_vector)

        results.append({
            "dataset_id": dataset_id,
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    sim_dir = os.path.join(os.path.dirname(target_path), "sim")
    os.makedirs(sim_dir, exist_ok=True)

    target_base = os.path.splitext(os.path.basename(target_path))[0]
    output_path = os.path.join(sim_dir, f"{target_base}_similarity.csv")

    sim.save_to_csv(output_path, target_id, metric, results)
    target_prefix = os.path.basename(target_path)[:3]
    return output_path, target_prefix, sim_dir



def merge_generated_csvs(csv_map: dict[str, str], output_path: str) -> None:
    df_signature = merger.load_and_validate_csv(csv_map["signature"], "signature")
    df_mbc = merger.load_and_validate_csv(csv_map["mbc"], "mbc")
    df_ttp = merger.load_and_validate_csv(csv_map["ttp"], "ttp")

    df_signature = df_signature.rename(columns={merger.SCORE_COL: "signatures_similarity"})
    df_mbc = df_mbc.rename(columns={merger.SCORE_COL: "mbcs_similarity"})
    df_ttp = df_ttp.rename(columns={merger.SCORE_COL: "ttps_similarity"})

    merge_keys = [merger.TARGET_COL, merger.DATASET_COL, merger.SIM_TYPE_COL]

    merged_df = (
        df_signature.merge(df_mbc, on=merge_keys, how="outer")
        .merge(df_ttp, on=merge_keys, how="outer")
    )

    merged_df = merged_df[FINAL_COLUMN_ORDER]
    merged_df = merged_df.sort_values(
        by=[merger.TARGET_COL, merger.DATASET_COL, merger.SIM_TYPE_COL],
        kind="stable",
    ).reset_index(drop=True)

    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")



def main() -> None:
    setup_logging()

    root = tk.Tk()
    root.withdraw()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        messagebox.showinfo(
            "Bắt đầu",
            "Bạn sẽ lần lượt chọn 3 file TARGET theo thứ tự: signature, mbc, ttp.\n"
            "Dataset tương ứng sẽ được tự động lấy từ thư mục 'dataset pattern'.\n"
            "Mỗi cặp sẽ tạo 1 file CSV tại thư mục chứa file target.\n"
            "Sau đó chương trình sẽ tự gộp 3 file CSV và lưu file tổng hợp tại thư mục chứa script này.",
        )

        generated_csvs = {}
        target_prefix = ""
        merged_sim_dir = ""
        for category_name, _ in PAIR_CONFIGS:
            csv_path, prefix, sim_dir = run_single_similarity(category_name)
            generated_csvs[category_name] = csv_path
            if not target_prefix:
                target_prefix = prefix
                merged_sim_dir = sim_dir
            messagebox.showinfo(
                "Đã tạo file CSV",
                f"Đã tạo xong file cho {category_name}:\n{csv_path}",
            )

        merged_output = os.path.join(merged_sim_dir, f"{target_prefix}_merged_similarity.csv")
        merge_generated_csvs(generated_csvs, merged_output)

        summary = (
            "Đã xử lý xong toàn bộ.\n\n"
            f"signature CSV: {generated_csvs['signature']}\n"
            f"mbc CSV: {generated_csvs['mbc']}\n"
            f"ttp CSV: {generated_csvs['ttp']}\n\n"
            f"File gộp cuối cùng: {merged_output}"
        )
        messagebox.showinfo("Hoàn tất", summary)

    except Exception as exc:
        traceback.print_exc()
        messagebox.showerror("Lỗi", str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()

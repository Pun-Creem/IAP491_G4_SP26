import pandas as pd
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

def extract_hash_from_filename(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return basename

def get_ground_truth(gt_df, file_hash):
    row = gt_df[gt_df['Hash256'] == file_hash]
    if row.empty:
        return None
    actions_str = row.iloc[0]['Action']
    actions = [a.strip().split(' - ')[0].strip() for a in actions_str.split('\n') if a.strip()]
    return actions

def evaluate(gt_actions, pred_df, top_k):
    top_k_preds = pred_df.head(top_k)
    pred_actions = [a.strip().split(' - ')[0].strip() for a in top_k_preds['action'].tolist()]
    hits = len(set(pred_actions) & set(gt_actions))
    precision = hits / top_k if top_k > 0 else 0
    accuracy = hits / len(gt_actions) if len(gt_actions) > 0 else 0
    return accuracy, precision

def main():
    setup_logging()
    print("=== Chọn file Ground Truth (actionPerReport.xlsx) ===")
    gt_path = select_file("Chọn file Ground Truth (actionPerReport)", [("Excel files", "*.xlsx"), ("All files", "*.*")])
    if not gt_path:
        print("Không chọn file ground truth. Thoát.")
        return
    print(f"Ground truth: {gt_path}")
    gt_df = pd.read_excel(gt_path)

    top_k = int(input("Nhập top K: "))
    print(f"Top K = {top_k}")

    results = []

    while True:
        print("\n=== Chọn file Prediction (.csv) ===")
        pred_path = select_file("Chọn file Prediction (.csv)", [("CSV files", "*.csv"), ("All files", "*.*")])
        if not pred_path:
            print("Không chọn file prediction. Bỏ qua.")
        else:
            print(f"Prediction: {pred_path}")
            pred_df = pd.read_csv(pred_path)
            file_hash = extract_hash_from_filename(pred_path)
            gt_actions = get_ground_truth(gt_df, file_hash)

            if gt_actions is None:
                print(f"Không tìm thấy hash '{file_hash}' trong ground truth!")
                results.append({
                    'prediction_file': os.path.basename(pred_path),
                    'accuracy': 'no hash error',
                    'precision': 'no hash error'
                })
            else:
                accuracy, precision = evaluate(gt_actions, pred_df, top_k)
                print(f"  Ground truth actions: {gt_actions}")
                print(f"  Accuracy (Recall@{top_k}): {accuracy:.4f}")
                print(f"  Precision@{top_k}: {precision:.4f}")
                results.append({
                    'prediction_file': os.path.basename(pred_path),
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4)
                })

        cont = input("\nChọn thêm file prediction? (y/n): ").strip().lower()
        if cont != 'y':
            break

    if results:
        out_df = pd.DataFrame(results)
        out_path = os.path.join(os.path.dirname(gt_path), f'evaluation_top{top_k}.csv')
        out_df.to_csv(out_path, index=False)
        print(f"\nKết quả đã lưu tại: {out_path}")
        print(out_df.to_string(index=False))
    else:
        print("Không có kết quả nào.")

if __name__ == '__main__':
    main()

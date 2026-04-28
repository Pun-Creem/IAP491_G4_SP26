import os
import sys
import re
import csv
import glob
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


def pick_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Chọn file mapped_action",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def ask_float(title, prompt, default):
    root = tk.Tk()
    root.withdraw()
    val = simpledialog.askfloat(title, prompt, initialvalue=default, parent=root)
    root.destroy()
    return val if val is not None else default


def find_topx_file(mapped_path):
    dir_path = os.path.dirname(mapped_path)
    basename = os.path.basename(mapped_path)

    # Extract number before "_mapped_action", works with prefixes like "TF-IDF_"
    match = re.search(r"(\d+)_mapped_action", basename)
    if not match:
        raise ValueError(f"Cannot extract target number from filename: {basename}")
    target_num = match.group(1)

    # Find files like {target_num}_top*.csv
    pattern = os.path.join(dir_path, f"{target_num}_top*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No top-x file found matching pattern: {pattern}"
        )
    if len(candidates) > 1:
        print(f"Multiple top-x files found, using: {candidates[0]}")
    return candidates[0]


def load_mapped_action(path):
    """Returns (sha256, dict{action: value}) for the single data row."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)  # only one data row expected
    actions = headers[1:]  # skip sha256 column
    values = row[1:]
    px = {action: float(v) for action, v in zip(actions, values)}
    return row[0], px


def load_topx(path):
    """Returns list of (sha256, similarity, dict{action: value}) for each neighbor."""
    neighbors = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        # headers: sha256, similarity, action1, action2, ...
        actions = headers[2:]
        for row in reader:
            if not row or not row[0].strip():
                continue
            sha = row[0]
            sim = float(row[1])
            yj = {action: float(v) for action, v in zip(actions, row[2:])}
            neighbors.append((sha, sim, yj))
    return neighbors, actions


def compute_scores(px, neighbors, actions, W, beta):
    scores = {}
    for action in actions:
        p = px.get(action, 0.0)
        neighbor_sum = sum(sim * yj.get(action, 0.0) for _, sim, yj in neighbors)
        scores[action] = W * p + beta * neighbor_sum
    return scores


def save_output(scores, actions, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "score"])
        for action in actions:
            writer.writerow([action, scores[action]])
    print(f"Output saved to: {output_path}")


def main():
    setup_logging()
    # 1. Pick mapped_action file
    mapped_path = pick_file()
    if not mapped_path:
        print("No file selected. Exiting.")
        return

    # 2. Ask for W and beta
    W = ask_float("Tham số W", "Nhập giá trị W:", default=1.0)
    beta = ask_float("Tham số β", "Nhập giá trị β (beta):", default=1.0)

    # 3. Auto-find topx file
    try:
        topx_path = find_topx_file(mapped_path)
    except (ValueError, FileNotFoundError) as e:
        messagebox.showerror("Lỗi", str(e))
        return

    print(f"Mapped action file : {mapped_path}")
    print(f"Top-x file         : {topx_path}")
    print(f"W = {W}, β = {beta}")

    # 4. Load data
    target_sha, px = load_mapped_action(mapped_path)
    neighbors, actions = load_topx(topx_path)

    print(f"Target hash        : {target_sha}")
    print(f"Number of neighbors: {len(neighbors)}")
    print(f"Number of actions  : {len(actions)}")

    # 5. Compute scores
    scores = compute_scores(px, neighbors, actions, W, beta)

    # 6. Save output next to the mapped_action file
    base = os.path.splitext(mapped_path)[0]
    output_path = base + "_scores.csv"
    save_output(scores, actions, output_path)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Hoàn thành", f"Đã lưu kết quả tại:\n{output_path}")
    root.destroy()


if __name__ == "__main__":
    main()

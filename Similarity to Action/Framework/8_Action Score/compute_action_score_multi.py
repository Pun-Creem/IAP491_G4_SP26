import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from compute_action_score import (
    pick_file,
    ask_float,
    find_topx_file,
    load_mapped_action,
    load_topx,
    compute_scores,
    save_output,
)

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


def ask_yes_no(title, question):
    root = tk.Tk()
    root.withdraw()
    answer = messagebox.askyesno(title, question)
    root.destroy()
    return answer


def run_once(W, beta):
    mapped_path = pick_file()
    if not mapped_path:
        return False, W, beta  # user cancelled file picker → stop loop

    try:
        topx_path = find_topx_file(mapped_path)
    except (ValueError, FileNotFoundError) as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Lỗi", str(e))
        root.destroy()
        return True, W, beta  # error but keep looping

    print(f"Mapped action file : {mapped_path}")
    print(f"Top-x file         : {topx_path}")
    print(f"W = {W}, β = {beta}")

    target_sha, px = load_mapped_action(mapped_path)
    neighbors, actions = load_topx(topx_path)

    print(f"Target hash        : {target_sha}")
    print(f"Number of neighbors: {len(neighbors)}")
    print(f"Number of actions  : {len(actions)}")

    scores = compute_scores(px, neighbors, actions, W, beta)

    import os
    base = os.path.splitext(mapped_path)[0]
    output_path = base + "_scores.csv"
    save_output(scores, actions, output_path)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Hoàn thành", f"Đã lưu kết quả tại:\n{output_path}")
    root.destroy()

    return True, W, beta


def main():
    setup_logging()
    W = ask_float("Tham số W", "Nhập giá trị W:", default=1.0)
    beta = ask_float("Tham số β", "Nhập giá trị β (beta):", default=1.0)

    while True:
        keep_going, W, beta = run_once(W, beta)

        if not keep_going:
            break

        if not ask_yes_no("Tiếp tục?", "Bạn có muốn tính thêm file khác không?"):
            break

        if ask_yes_no("Đổi tham số?", "Bạn có muốn thay đổi W và β không?"):
            W = ask_float("Tham số W", f"Nhập giá trị W (hiện tại: {W}):", default=W)
            beta = ask_float("Tham số β", f"Nhập giá trị β (hiện tại: {beta}):", default=beta)

    print("Kết thúc.")


if __name__ == "__main__":
    main()

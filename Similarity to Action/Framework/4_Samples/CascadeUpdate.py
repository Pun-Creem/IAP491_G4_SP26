import csv
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


def choose_file(title, filetypes=(("CSV/TSV files", "*.csv *.tsv"), ("All files", "*.*"))):
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if not path:
        raise RuntimeError(f"Chưa chọn file: {title}")
    return path


def normalize_hash_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột hash về 'sha256'
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for candidate in ["sha256", "hash", "sha-256"]:
        if candidate in cols_lower:
            real_col = cols_lower[candidate]
            if real_col != "sha256":
                df = df.rename(columns={real_col: "sha256"})
            return df

    raise ValueError(
        f"Không tìm thấy cột sha256/hash. Các cột hiện có: {list(df.columns)}"
    )


def load_summary_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_hash_column(df)

    # Chuẩn hóa hash về lowercase string
    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()

    # Đảm bảo các cột feature là numeric 0/1 nếu có thể
    for col in df.columns:
        if col != "sha256":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def load_vt_update(path: str) -> pd.DataFrame:
    """
    File VT của bạn thực tế là TSV:
      sha256    ttps    mbc
    Trong đó ttps/mbc có thể chứa nhiều dòng trong 1 ô và có quote.
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=",", quotechar='"')
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("File update rỗng hoặc không đọc được dữ liệu.")

    df = pd.DataFrame(rows)
    df.columns = [str(c).strip() for c in df.columns]

    # Chuẩn hóa tên cột
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["sha256", "hash", "sha-256"]:
            rename_map[c] = "sha256"
        elif cl in ["ttps", "ttp", "tttps"]:
            rename_map[c] = "ttps"
        elif cl == "mbc":
            rename_map[c] = "mbc"

    df = df.rename(columns=rename_map)

    required = {"sha256", "ttps", "mbc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Thiếu cột trong file update: {missing}. Các cột hiện có: {list(df.columns)}"
        )

    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()
    df["ttps"] = df["ttps"].fillna("").astype(str)
    df["mbc"] = df["mbc"].fillna("").astype(str)

    return df


def extract_codes(text: str, pattern: str) -> list[str]:
    """
    Tách code từ text, ví dụ:
      - TTP: T1027, T1059.001
      - MBC: B0002, B0030.002, E1083, OC0006
    Dùng regex để bắt code kể cả khi text có mô tả dài:
      'Command and Control::... [B0030.002]'
    """
    if not text or not str(text).strip():
        return []

    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    # Chuẩn hóa uppercase, unique nhưng giữ thứ tự
    seen = set()
    result = []
    for m in matches:
        code = m.upper()
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def collect_all_codes(df_update: pd.DataFrame, mode: str) -> list[str]:
    """
    Quét toàn bộ file update và thu thập tất cả codes theo mode.
    """
    if mode == "mbc":
        source_col = "mbc"
        code_pattern = r"\b(?:B|C|E|F|OB|OC)\d{4}(?:\.\d{3}|\.m\d{2})?\b"
    else:
        source_col = "ttps"
        code_pattern = r"\bT\d{4}(?:\.\d{3})?\b"

    seen = set()
    result = []
    for text in df_update[source_col]:
        for code in extract_codes(str(text), code_pattern):
            if code not in seen:
                seen.add(code)
                result.append(code)
    return sorted(result)


def ask_new_features(root: tk.Tk, new_codes: list[str], mode: str) -> list[str]:
    """
    Hiện dialog cho user chọn features mới nào sẽ được thêm vào file summary.
    Trả về danh sách codes được chọn.
    """
    mode_label = "TTP" if mode == "ttp" else "MBC"
    dialog = tk.Toplevel(root)
    dialog.title(f"Features mới trong file update ({mode_label})")
    dialog.grab_set()
    dialog.resizable(False, True)

    msg = (
        f"Các feature {mode_label} sau có trong file update nhưng KHÔNG có trong header của file summary.\n"
        f"Chọn feature nào bạn muốn thêm vào (bỏ chọn = bỏ qua feature đó):"
    )
    tk.Label(dialog, text=msg, wraplength=520, justify="left", padx=10, pady=8).pack(fill="x")

    frame_list = tk.Frame(dialog)
    frame_list.pack(fill="both", expand=True, padx=10, pady=4)

    canvas = tk.Canvas(frame_list, width=540, height=min(300, len(new_codes) * 26 + 10))
    scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas)

    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    vars_ = {}
    for code in new_codes:
        var = tk.BooleanVar(value=True)
        vars_[code] = var
        tk.Checkbutton(inner, text=code, variable=var, anchor="w").pack(fill="x", padx=6, pady=1)

    def select_all():
        for v in vars_.values():
            v.set(True)

    def deselect_all():
        for v in vars_.values():
            v.set(False)

    frame_btn_top = tk.Frame(dialog)
    frame_btn_top.pack(pady=4)
    tk.Button(frame_btn_top, text="Chọn tất cả", command=select_all, width=14).pack(side="left", padx=6)
    tk.Button(frame_btn_top, text="Bỏ chọn tất cả", command=deselect_all, width=14).pack(side="left", padx=6)

    result_holder = []

    def on_ok():
        result_holder.extend([code for code, v in vars_.items() if v.get()])
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    frame_btn_bot = tk.Frame(dialog)
    frame_btn_bot.pack(pady=(0, 10))
    tk.Button(frame_btn_bot, text="OK", command=on_ok, width=10, bg="#4CAF50", fg="white").pack(side="left", padx=10)
    tk.Button(frame_btn_bot, text="Hủy (bỏ qua tất cả)", command=on_cancel, width=18).pack(side="left", padx=10)

    dialog.wait_window()
    return result_holder


def ensure_columns(df: pd.DataFrame, codes: list[str], allowed_new: set[str] | None = None) -> pd.DataFrame:
    """
    Nếu chưa có cột thì thêm mới với default = 0.
    Nếu allowed_new được truyền vào, chỉ thêm các cột có trong allowed_new.
    """
    for code in codes:
        if code not in df.columns:
            if allowed_new is None or code in allowed_new:
                df[code] = 0
    return df


def ensure_hash_row(df: pd.DataFrame, hash_value: str) -> pd.DataFrame:
    """
    Nếu hash chưa tồn tại thì thêm dòng mới
    """
    if hash_value not in set(df["sha256"]):
        new_row = {col: 0 for col in df.columns}
        new_row["sha256"] = hash_value
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def update_summary(
    df_summary: pd.DataFrame,
    df_update: pd.DataFrame,
    mode: str,
    allowed_new: set[str] | None = None,
) -> pd.DataFrame:
    """
    mode = 'mbc' hoặc 'ttp'
    allowed_new: tập codes được phép thêm cột mới. None = cho phép tất cả.
    """
    if mode == "mbc":
        source_col = "mbc"
        code_pattern = r"\b(?:B|C|E|F|OB|OC)\d{4}(?:\.\d{3}|\.m\d{2})?\b"
    elif mode == "ttp":
        source_col = "ttps"
        code_pattern = r"\bT\d{4}(?:\.\d{3})?\b"
    else:
        raise ValueError("mode phải là 'mbc' hoặc 'ttp'")

    for _, row in df_update.iterrows():
        hash_value = row["sha256"]
        codes = extract_codes(row[source_col], code_pattern)

        if not hash_value:
            continue

        # Nếu hash chưa có thì thêm
        df_summary = ensure_hash_row(df_summary, hash_value)

        # Nếu thiếu cột thì thêm (chỉ với các code được phép)
        df_summary = ensure_columns(df_summary, codes, allowed_new=allowed_new)

        # Set = 1 cho các code của hash này (chỉ các code đã tồn tại trong cột)
        idx = df_summary.index[df_summary["sha256"] == hash_value]
        if len(idx) == 0:
            continue

        row_idx = idx[0]
        for code in codes:
            if code in df_summary.columns:
                df_summary.at[row_idx, code] = 1

    # Chuẩn hóa lại các cột không phải sha256 thành int
    for col in df_summary.columns:
        if col != "sha256":
            df_summary[col] = pd.to_numeric(df_summary[col], errors="coerce").fillna(0).astype(int)

    return df_summary


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đưa 'sha256' lên đầu, các cột còn lại sort tăng dần
    """
    other_cols = [c for c in df.columns if c != "sha256"]
    other_cols = sorted(other_cols)
    return df[["sha256"] + other_cols]


def main():
    setup_logging()
    root = tk.Tk()
    root.withdraw()

    try:
        vt_update_path = choose_file("Chọn file VirusTotal update")
        summary_mbc_path = choose_file("Chọn file summary MBC")
        summary_ttp_path = choose_file("Chọn file summary TTP")

        print("Đang đọc file update...")
        df_update = load_vt_update(vt_update_path)

        print("Đang đọc file summary MBC...")
        df_mbc = load_summary_csv(summary_mbc_path)

        print("Đang đọc file summary TTP...")
        df_ttp = load_summary_csv(summary_ttp_path)

        # --- Kiểm tra features mới cho MBC ---
        all_mbc_codes = collect_all_codes(df_update, mode="mbc")
        new_mbc_codes = [c for c in all_mbc_codes if c not in df_mbc.columns]
        allowed_mbc: set[str] | None = None
        if new_mbc_codes:
            print(f"Phát hiện {len(new_mbc_codes)} MBC feature mới: {new_mbc_codes}")
            chosen = ask_new_features(root, new_mbc_codes, mode="mbc")
            allowed_mbc = set(chosen)
            skipped = set(new_mbc_codes) - allowed_mbc
            if skipped:
                print(f"Bỏ qua MBC features: {sorted(skipped)}")
            if allowed_mbc:
                print(f"Thêm MBC features: {sorted(allowed_mbc)}")

        # --- Kiểm tra features mới cho TTP ---
        all_ttp_codes = collect_all_codes(df_update, mode="ttp")
        new_ttp_codes = [c for c in all_ttp_codes if c not in df_ttp.columns]
        allowed_ttp: set[str] | None = None
        if new_ttp_codes:
            print(f"Phát hiện {len(new_ttp_codes)} TTP feature mới: {new_ttp_codes}")
            chosen = ask_new_features(root, new_ttp_codes, mode="ttp")
            allowed_ttp = set(chosen)
            skipped = set(new_ttp_codes) - allowed_ttp
            if skipped:
                print(f"Bỏ qua TTP features: {sorted(skipped)}")
            if allowed_ttp:
                print(f"Thêm TTP features: {sorted(allowed_ttp)}")

        print("Đang cập nhật summary MBC...")
        df_mbc_updated = update_summary(df_mbc, df_update, mode="mbc", allowed_new=allowed_mbc)
        df_mbc_updated = reorder_columns(df_mbc_updated)

        print("Đang cập nhật summary TTP...")
        df_ttp_updated = update_summary(df_ttp, df_update, mode="ttp", allowed_new=allowed_ttp)
        df_ttp_updated = reorder_columns(df_ttp_updated)

        base_dir = os.path.dirname(os.path.abspath(vt_update_path))
        mbc_stem = os.path.splitext(os.path.basename(summary_mbc_path))[0]
        ttp_stem = os.path.splitext(os.path.basename(summary_ttp_path))[0]
        mbc_out = os.path.join(base_dir, f"{mbc_stem}_update.csv")
        ttp_out = os.path.join(base_dir, f"{ttp_stem}_update.csv")

        df_mbc_updated.to_csv(mbc_out, index=False, encoding="utf-8-sig")
        df_ttp_updated.to_csv(ttp_out, index=False, encoding="utf-8-sig")

        msg = (
            "Cập nhật xong.\n\n"
            f"File MBC: {mbc_out}\n"
            f"File TTP: {ttp_out}"
        )
        print(msg)
        messagebox.showinfo("Hoàn tất", msg)

    except Exception as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        messagebox.showerror("Lỗi", str(e))


if __name__ == "__main__":
    main()
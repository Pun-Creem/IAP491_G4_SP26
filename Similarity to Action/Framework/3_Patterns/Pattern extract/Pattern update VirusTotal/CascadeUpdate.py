import csv
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from log_utils import setup_logging


def choose_file(title, filetypes=(("CSV/TSV files", "*.csv *.tsv"), ("All files", "*.*"))):
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if not path:
        raise RuntimeError(f"Chưa chọn file: {title}")
    return path


def normalize_hash_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột hash về 'hash'
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for candidate in ["hash", "sha256", "sha-256"]:
        if candidate in cols_lower:
            real_col = cols_lower[candidate]
            if real_col != "hash":
                df = df.rename(columns={real_col: "hash"})
            return df

    raise ValueError(
        f"Không tìm thấy cột hash/sha256. Các cột hiện có: {list(df.columns)}"
    )


def load_summary_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_hash_column(df)

    # Chuẩn hóa hash về lowercase string
    df["hash"] = df["hash"].astype(str).str.strip().str.lower()

    # Đảm bảo các cột feature là numeric 0/1 nếu có thể
    for col in df.columns:
        if col != "hash":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def load_vt_update(path: str) -> pd.DataFrame:
    """
    File VT của bạn thực tế là TSV:
      sha256    ttps    mbc
    Trong đó ttps/mbc có thể chứa nhiều dòng trong 1 ô và có quote.
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
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
            rename_map[c] = "hash"
        elif cl in ["ttps", "ttp", "tttps"]:
            rename_map[c] = "ttps"
        elif cl == "mbc":
            rename_map[c] = "mbc"

    df = df.rename(columns=rename_map)

    required = {"hash", "ttps", "mbc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Thiếu cột trong file update: {missing}. Các cột hiện có: {list(df.columns)}"
        )

    df["hash"] = df["hash"].astype(str).str.strip().str.lower()
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


def ensure_columns(df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """
    Nếu chưa có cột thì thêm mới với default = 0
    """
    for code in codes:
        if code not in df.columns:
            df[code] = 0
    return df


def ensure_hash_row(df: pd.DataFrame, hash_value: str) -> pd.DataFrame:
    """
    Nếu hash chưa tồn tại thì thêm dòng mới
    """
    if hash_value not in set(df["hash"]):
        new_row = {col: 0 for col in df.columns}
        new_row["hash"] = hash_value
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def check_duplicate_hashes(df_update: pd.DataFrame) -> list[str]:
    """
    Tìm các hash xuất hiện nhiều hơn 1 lần trong file VT update.
    Trả về list các hash bị trùng.
    """
    counts = df_update["hash"].value_counts()
    return counts[counts > 1].index.tolist()


def check_missing_hashes(df_summary: pd.DataFrame, update_hash_set: set) -> list[str]:
    """
    Tìm các hash có trong summary nhưng không có trong VT update.
    """
    return [h for h in df_summary["hash"] if h and h not in update_hash_set]


def write_report(report_path: str, missing_hashes: list[str], duplicate_hashes: list[str]):
    """
    Ghi báo cáo ra file txt.
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BÁO CÁO KIỂM TRA CASCADE UPDATE\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"[1] Hash có trong Summary nhưng KHÔNG có trong VT update: {len(missing_hashes)}\n")
        f.write("-" * 60 + "\n")
        if missing_hashes:
            for h in missing_hashes:
                f.write(f"  {h}\n")
        else:
            f.write("  (Không có)\n")

        f.write("\n")
        f.write(f"[2] Hash bị TRÙNG LẶP trong file VT update: {len(duplicate_hashes)}\n")
        f.write("-" * 60 + "\n")
        if duplicate_hashes:
            for h in duplicate_hashes:
                f.write(f"  {h}\n")
        else:
            f.write("  (Không có)\n")


def update_summary(df_summary: pd.DataFrame, df_update: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode = 'mbc' hoặc 'ttp'
    """
    if mode == "mbc":
        source_col = "mbc"
        code_pattern = r"\b(?:B|C|E|F|OB|OC)\d{4}(?:\.\d{3}|\.m\d{2})?\b"
    elif mode == "ttp":
        source_col = "ttps"
        code_pattern = r"\bT\d{4}(?:\.\d{3})?\b"
    else:
        raise ValueError("mode phải là 'mbc' hoặc 'ttp'")

    # Index file update theo hash để lookup nhanh (chỉ lấy dòng đầu tiên nếu trùng)
    update_lookup = df_update.drop_duplicates(subset="hash", keep="first").set_index("hash")

    for row_idx, row in df_summary.iterrows():
        hash_value = row["hash"]
        if not hash_value or hash_value not in update_lookup.index:
            continue

        # Hash trùng → regex trích mã từ VT update (chỉ dòng đầu tiên)
        vt_row = update_lookup.loc[hash_value]
        text = vt_row[source_col]

        codes = extract_codes(text, code_pattern)
        if not codes:
            continue

        # Nếu thiếu cột thì thêm
        df_summary = ensure_columns(df_summary, codes)

        for code in codes:
            df_summary.at[row_idx, code] = 1

    # Chuẩn hóa lại các cột không phải hash thành int
    for col in df_summary.columns:
        if col != "hash":
            df_summary[col] = pd.to_numeric(df_summary[col], errors="coerce").fillna(0).astype(int)

    return df_summary


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đưa 'hash' lên đầu, các cột còn lại sort tăng dần
    """
    other_cols = [c for c in df.columns if c != "hash"]
    other_cols = sorted(other_cols)
    return df[["hash"] + other_cols]


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

        # Kiểm tra hash trùng lặp trong VT update
        print("Đang kiểm tra hash trùng lặp trong VT update...")
        duplicate_hashes = check_duplicate_hashes(df_update)

        # Kiểm tra hash thiếu (có trong summary nhưng không có trong VT update)
        # Gộp hash từ cả 2 file summary để check
        print("Đang kiểm tra hash thiếu trong VT update...")
        update_hash_set = set(df_update["hash"])
        all_summary_hashes = set(df_mbc["hash"]).union(set(df_ttp["hash"]))
        missing_hashes = sorted([h for h in all_summary_hashes if h and h not in update_hash_set])

        print("Đang cập nhật summary MBC...")
        df_mbc_updated = update_summary(df_mbc, df_update, mode="mbc")
        df_mbc_updated = reorder_columns(df_mbc_updated)

        print("Đang cập nhật summary TTP...")
        df_ttp_updated = update_summary(df_ttp, df_update, mode="ttp")
        df_ttp_updated = reorder_columns(df_ttp_updated)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        mbc_out = os.path.join(base_dir, "pattern_mbc_update.csv")
        ttp_out = os.path.join(base_dir, "pattern_ttp_update.csv")
        report_out = os.path.join(base_dir, "cascade_report.txt")

        df_mbc_updated.to_csv(mbc_out, index=False, encoding="utf-8-sig")
        df_ttp_updated.to_csv(ttp_out, index=False, encoding="utf-8-sig")
        write_report(report_out, missing_hashes, duplicate_hashes)

        msg = (
            "Cập nhật xong.\n\n"
            f"File MBC: {mbc_out}\n"
            f"File TTP: {ttp_out}\n"
            f"Báo cáo: {report_out}\n\n"
            f"Hash thiếu trong VT update: {len(missing_hashes)}\n"
            f"Hash trùng lặp trong VT update: {len(duplicate_hashes)}"
        )
        print(msg)
        messagebox.showinfo("Hoàn tất", msg)

    except Exception as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        messagebox.showerror("Lỗi", str(e))


if __name__ == "__main__":
    main()
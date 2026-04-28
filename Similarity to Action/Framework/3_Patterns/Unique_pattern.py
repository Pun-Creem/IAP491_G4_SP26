import csv
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging


def extract_features(input_file: str) -> str:
    basename = os.path.basename(input_file).lower()

    if "ttp" in basename:
        output_name = "ttp_unique.csv"
    elif "mbc" in basename:
        output_name = "mbc_unique.csv"
    elif "signature" in basename:
        output_name = "signature_unique.csv"
    else:
        raise ValueError(
            f"Cannot detect type from filename: {basename!r}.\n"
            "Filename must contain 'ttp', 'mbc', or 'signature'."
        )

    output_file = os.path.join(os.path.dirname(input_file), output_name)

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

    features = headers[1:]  # bo cell dau (hash)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature"])
        for feat in features:
            writer.writerow([feat])

    return output_file, len(features)


def main():
    setup_logging()
    root = tk.Tk()
    root.withdraw()  # an cua so chinh

    input_file = filedialog.askopenfilename(
        title="Select input CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if not input_file:
        messagebox.showinfo("Cancelled", "No file selected.")
        return

    try:
        output_file, count = extract_features(input_file)
        messagebox.showinfo(
            "Done",
            f"Output saved to:\n{output_file}\n\n{count} features extracted."
        )
    except ValueError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # van ho tro truyen tham so tren command line
        for path in sys.argv[1:]:
            try:
                out, count = extract_features(path)
                print(f"Done: '{out}' ({count} features).")
            except Exception as e:
                print(f"Error: {e}")
    else:
        main()

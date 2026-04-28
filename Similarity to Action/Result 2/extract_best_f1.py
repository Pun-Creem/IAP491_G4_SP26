import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_utils import setup_logging


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing CSV files")
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit()
    return folder


def load_csv_files(folder):
    files = {}
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".csv"):
            path = os.path.join(folder, f)
            try:
                df = pd.read_csv(path)
                # Verify it has the expected columns
                if "f1_top1" in df.columns:
                    files[f] = df
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")
    if not files:
        print("No valid CSV files found in the selected folder.")
        sys.exit()
    return files


def get_k_columns():
    """Return list of k values and their column name patterns."""
    ks = []
    for k in range(1, 100):
        if k == 1:
            ks.append(k)
        else:
            ks.append(k)
    return ks


def detect_ks(df):
    """Detect which top_k values exist in the dataframe."""
    ks = []
    for k in range(1, 100):
        if f"f1_top{k}" in df.columns:
            ks.append(k)
    return ks


def compute_means(df, ks):
    """Compute mean precision, recall, f1 for each k."""
    results = {}
    for k in ks:
        p_col = f"precision_top{k}"
        r_col = f"recall_top{k}"
        f_col = f"f1_top{k}"
        if all(c in df.columns for c in [p_col, r_col, f_col]):
            results[k] = {
                "precision": df[p_col].mean(),
                "recall": df[r_col].mean(),
                "f1": df[f_col].mean(),
            }
    return results


def ask_user_tie(tied_ks, means_info, context_label):
    """Ask user to choose among tied k values. Returns list of chosen ks."""
    root = tk.Tk()
    root.title(f"Tie detected - {context_label}")

    msg = f"The following top_k values have the same highest F1:\n\n"
    for k in tied_ks:
        info = means_info[k]
        msg += (
            f"  top_{k}: precision={info['precision']:.4f}, "
            f"recall={info['recall']:.4f}, f1={info['f1']:.4f}\n"
        )
    msg += "\nEnter which k value(s) to use (comma-separated, e.g. '3' or '3,5'):"

    tk.Label(root, text=msg, justify="left", font=("Consolas", 10), padx=10, pady=10).pack()

    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)
    entry.insert(0, ",".join(str(k) for k in tied_ks))

    result = []

    def on_ok():
        text = entry.get().strip()
        chosen = []
        for part in text.split(","):
            part = part.strip()
            if part.isdigit() and int(part) in tied_ks:
                chosen.append(int(part))
        if chosen:
            result.extend(chosen)
            root.destroy()
        else:
            messagebox.showerror("Invalid", f"Please enter valid k values from {tied_ks}")

    tk.Button(root, text="OK", command=on_ok).pack(pady=10)
    root.mainloop()

    if not result:
        print("No valid selection. Exiting.")
        sys.exit()
    return result


def option1_highest_f1_value(files):
    """Per file, find the k with the highest mean F1. If ties, ask user."""
    output_rows = []

    for filename, df in files.items():
        ks = detect_ks(df)
        means = compute_means(df, ks)

        if not means:
            continue

        max_f1 = max(m["f1"] for m in means.values())
        tied_ks = [k for k, m in means.items() if abs(m["f1"] - max_f1) < 1e-9]

        if len(tied_ks) > 1:
            chosen_ks = ask_user_tie(tied_ks, means, context_label=filename)
        else:
            chosen_ks = tied_ks

        for k in chosen_ks:
            output_rows.append(
                {
                    "file": filename,
                    "top_k": k,
                    "precision": round(means[k]["precision"], 4),
                    "recall": round(means[k]["recall"], 4),
                    "f1": round(means[k]["f1"], 4),
                }
            )

    return output_rows


def option2_highest_f1_mean(files):
    """Find the k with the highest mean F1 across ALL files, then output per file."""
    # Detect ks from first file
    first_df = next(iter(files.values()))
    ks = detect_ks(first_df)

    # Compute global mean F1 for each k
    global_f1_sums = {k: 0.0 for k in ks}
    global_counts = {k: 0 for k in ks}
    per_file_means = {}

    for filename, df in files.items():
        file_ks = detect_ks(df)
        means = compute_means(df, file_ks)
        per_file_means[filename] = means
        for k in file_ks:
            if k in means:
                global_f1_sums[k] += means[k]["f1"]
                global_counts[k] += 1

    global_f1_means = {}
    for k in ks:
        if global_counts.get(k, 0) > 0:
            global_f1_means[k] = global_f1_sums[k] / global_counts[k]

    if not global_f1_means:
        print("No valid data found.")
        sys.exit()

    max_f1 = max(global_f1_means.values())
    tied_ks = [k for k, v in global_f1_means.items() if abs(v - max_f1) < 1e-9]

    # Show global means for context
    print("\nGlobal mean F1 per top_k:")
    for k in sorted(global_f1_means.keys()):
        print(f"  top_{k}: mean_f1 = {global_f1_means[k]:.4f}")

    if len(tied_ks) > 1:
        # Build means_info for the tie dialog
        tie_info = {}
        for k in tied_ks:
            all_p, all_r, all_f = [], [], []
            for means in per_file_means.values():
                if k in means:
                    all_p.append(means[k]["precision"])
                    all_r.append(means[k]["recall"])
                    all_f.append(means[k]["f1"])
            tie_info[k] = {
                "precision": sum(all_p) / len(all_p) if all_p else 0,
                "recall": sum(all_r) / len(all_r) if all_r else 0,
                "f1": sum(all_f) / len(all_f) if all_f else 0,
            }
        chosen_ks = ask_user_tie(tied_ks, tie_info, context_label="Global best k")
    else:
        chosen_ks = tied_ks

    print(f"\nSelected top_k: {chosen_ks}")

    output_rows = []
    for filename in files:
        means = per_file_means.get(filename, {})
        for k in chosen_ks:
            if k in means:
                output_rows.append(
                    {
                        "file": filename,
                        "top_k": k,
                        "precision": round(means[k]["precision"], 4),
                        "recall": round(means[k]["recall"], 4),
                        "f1": round(means[k]["f1"], 4),
                    }
                )

    return output_rows


def choose_option():
    root = tk.Tk()
    root.title("Select Option")
    root.geometry("400x200")

    choice = tk.IntVar(value=0)

    tk.Label(root, text="Choose extraction method:", font=("Arial", 12, "bold"), pady=10).pack()

    tk.Radiobutton(
        root, text="Option 1: Highest F1 value (per file)", variable=choice, value=1, font=("Arial", 10)
    ).pack(anchor="w", padx=30)

    tk.Radiobutton(
        root,
        text="Option 2: Highest F1 mean (across all files)",
        variable=choice,
        value=2,
        font=("Arial", 10),
    ).pack(anchor="w", padx=30)

    def on_ok():
        if choice.get() in (1, 2):
            root.destroy()
        else:
            messagebox.showwarning("Warning", "Please select an option.")

    tk.Button(root, text="OK", command=on_ok, width=10).pack(pady=15)
    root.mainloop()

    if choice.get() not in (1, 2):
        print("No option selected. Exiting.")
        sys.exit()
    return choice.get()


def main():
    setup_logging()
    print("=== F1 Extraction Tool ===\n")

    # Step 1: Select folder
    folder = select_folder()
    print(f"Selected folder: {folder}")

    # Step 2: Load CSV files
    files = load_csv_files(folder)
    print(f"Found {len(files)} CSV file(s): {list(files.keys())}\n")

    # Step 3: Choose option
    option = choose_option()
    print(f"Selected option: {option}\n")

    # Step 4: Process
    if option == 1:
        rows = option1_highest_f1_value(files)
    else:
        rows = option2_highest_f1_mean(files)

    if not rows:
        print("No results to output.")
        sys.exit()

    # Step 5: Save output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "extracted_f1_results.csv")

    out_df = pd.DataFrame(rows, columns=["file", "top_k", "precision", "recall", "f1"])
    out_df.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

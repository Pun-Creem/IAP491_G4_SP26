import pandas as pd
import sys
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from log_utils import setup_logging

ACTION_PER_TTPS = "action_per_ttps.csv"
TTP_UNIQUE     = "ttp_unique.csv"
ACTION_REPORT  = "actionPerReport.xlsx"
OUTPUT         = "action_per_ttps.csv"

def main():
    setup_logging()
    apt    = pd.read_csv(ACTION_PER_TTPS)
    ttp_u  = pd.read_csv(TTP_UNIQUE)
    report = pd.read_excel(ACTION_REPORT)

    # 1. Filter rows: only TTPs in ttp_unique
    unique_ttps = set(ttp_u["feature"])
    before_rows = len(apt)
    apt = apt[apt["ttps"].isin(unique_ttps)].copy()
    removed_ttps = before_rows - len(apt)

    # 2. Remove action columns with no active TTP (all zeros)
    action_cols = apt.columns[1:].tolist()
    active_cols = [c for c in action_cols if apt[c].sum() > 0]
    removed_actions = set(action_cols) - set(active_cols)
    apt = apt[["ttps"] + active_cols]

    # 3. Collect all unique actions from actionPerReport
    report_actions = set()
    for val in report["Action"].dropna():
        for a in str(val).split("\n"):
            a = a.strip()
            if a:
                report_actions.add(a)

    # 4. Add missing actions (excluding truncated names already present by ID)
    apt_action_ids = {c.split(" - ")[0].strip() for c in apt.columns[1:]}
    missing = []
    for a in sorted(report_actions):
        aid = a.split(" - ")[0].strip()
        if aid not in apt_action_ids:
            missing.append(a)

    cols_to_add = {a: 0 for a in missing}
    if cols_to_add:
        apt = pd.concat([apt, pd.DataFrame(cols_to_add, index=apt.index)], axis=1)

    apt.to_csv(OUTPUT, index=False)

    # Summary
    print(f"TTPs removed (not in ttp_unique) : {removed_ttps}")
    print(f"Actions removed (no TTP mapping) : {len(removed_actions)}")
    if removed_actions:
        for a in sorted(removed_actions):
            print(f"  - {a}")
    print(f"Actions added (missing from report): {len(missing)}")
    if missing:
        for a in missing:
            print(f"  + {a}")
    print(f"\nSaved '{OUTPUT}'  shape: {apt.shape}")

if __name__ == "__main__":
    main()

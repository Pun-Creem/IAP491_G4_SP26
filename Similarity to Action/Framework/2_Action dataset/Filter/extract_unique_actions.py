import sys
from pathlib import Path
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from log_utils import setup_logging
setup_logging()

df = pd.read_excel('action-space.xlsx')
d3 = pd.read_csv('d3fend.csv').set_index('ID')

all_actions = set()
for cell in df['Action'].dropna():
    for action in cell.split('\n'):
        action = action.strip()
        if action:
            all_actions.add(action)

unique_actions = sorted(all_actions)

print(f"Total unique actions: {len(unique_actions)}\n")
for action in unique_actions:
    print(action)

out_df = pd.DataFrame({'Action': unique_actions})
out_df['ID'] = out_df['Action'].str.split(' - ').str[0]
out_df = out_df.merge(d3.reset_index(), on='ID', how='left').drop(columns='ID')
out_df.index = range(1, len(out_df) + 1)
out_df.index.name = 'No'
out_df.to_excel('unique_actions.xlsx')
print(f"\nSaved to unique_actions.xlsx")

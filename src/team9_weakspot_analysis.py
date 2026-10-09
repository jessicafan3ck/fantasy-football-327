"""
Team 9 weak-spot analysis, EXCLUDING Week 4 (treated as an overperformance
week that would flatter our season averages if included).

For Weeks 1-3 only:
  1. Find the slots where Team 9 underperforms the field.
  2. For each such slot, find the team(s) who performed BEST there.
  3. Diff our strategy vs. theirs: range width, miss direction (over/under),
     and how width relates to whether they hit.

Run: python src/team9_weakspot_analysis.py
"""
import pandas as pd
from scoring import score_predictions

TEAM_NUMBER = 9
EXCLUDE_WEEKS = [4]

df = pd.read_csv('data/class_predictions_wk1-4.csv')
df = df[~df['week'].isin(EXCLUDE_WEEKS)]
scored = score_predictions(df)

us = scored[scored['team'] == TEAM_NUMBER]
field = scored[scored['team'] != TEAM_NUMBER]

print("=" * 100)
print(f"TEAM {TEAM_NUMBER} vs FIELD, Weeks {sorted(scored['week'].unique())} (Week 4 excluded as an overperformance week)")
print("=" * 100)

cmp = pd.DataFrame({
    'us_accuracy':      us.groupby('slot')['accuracy'].mean(),
    'field_accuracy':   field.groupby('slot')['accuracy'].mean(),
    'gap':              us.groupby('slot')['accuracy'].mean() - field.groupby('slot')['accuracy'].mean(),
    'us_width':         us.groupby('slot')['width'].mean(),
    'field_width':      field.groupby('slot')['width'].mean(),
    'us_hit_rate':      us.groupby('slot')['hit'].mean(),
    'field_hit_rate':   field.groupby('slot')['hit'].mean(),
}).round(2).sort_values('gap')

print(cmp)

weak_slots = cmp[cmp['gap'] < 0].index.tolist()
print(f"\nSlots where we're BELOW field average: {weak_slots}\n")

for slot in weak_slots:
    print("=" * 100)
    print(f"SLOT: {slot}  (we: {cmp.loc[slot,'us_accuracy']} acc, field: {cmp.loc[slot,'field_accuracy']} acc)")
    print("=" * 100)

    slot_rows = scored[scored['slot'] == slot].copy()
    # best-performing team(s) at this slot (by avg accuracy across their picks here)
    by_team = slot_rows.groupby('team').agg(
        n=('accuracy', 'size'),
        avg_accuracy=('accuracy', 'mean'),
        avg_width=('width', 'mean'),
        hit_rate=('hit', 'mean'),
        over_rate=('over', 'mean'),
        under_rate=('under', 'mean'),
    ).round(2).sort_values('avg_accuracy', ascending=False)
    print(by_team)

    best_team = by_team.index[0]
    best_rows = slot_rows[slot_rows['team'] == best_team]
    our_rows = slot_rows[slot_rows['team'] == TEAM_NUMBER]

    print(f"\n  Best team at {slot}: Team {best_team} "
          f"(avg_accuracy={by_team.loc[best_team,'avg_accuracy']} vs our "
          f"{by_team.loc[TEAM_NUMBER,'avg_accuracy'] if TEAM_NUMBER in by_team.index else float('nan')})")
    print(f"  Their picks:")
    print(best_rows[['week','player','low','high','actual','width','miss','hit','accuracy']]
          .to_string(index=False))
    print(f"\n  Our picks:")
    print(our_rows[['week','player','low','high','actual','width','miss','hit','accuracy']]
          .to_string(index=False))
    print()

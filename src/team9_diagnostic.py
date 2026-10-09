"""
Team 9 (our team) specific diagnostics vs. the field.
Run: python src/team9_diagnostic.py
"""
import pandas as pd
from scoring import score_predictions, team_week_totals

TEAM_NUMBER = 9

df = pd.read_csv('data/class_predictions_wk1-4.csv')
scored = score_predictions(df)

us = scored[scored['team'] == TEAM_NUMBER]
field = scored[scored['team'] != TEAM_NUMBER]

print("="*90)
print(f"TEAM {TEAM_NUMBER} — every pick, Week 1-4")
print("="*90)
print(us[['week', 'slot', 'player', 'low', 'high', 'actual', 'width', 'miss', 'accuracy']]
      .to_string(index=False))

print()
print("="*90)
print(f"TEAM {TEAM_NUMBER} vs FIELD AVERAGE — by position")
print("="*90)
cmp = pd.DataFrame({
    'us_accuracy': us.groupby('slot')['accuracy'].mean(),
    'field_accuracy': field.groupby('slot')['accuracy'].mean(),
    'us_avg_width': us.groupby('slot')['width'].mean(),
    'field_avg_width': field.groupby('slot')['width'].mean(),
    'us_hit_rate': us.groupby('slot')['hit'].mean(),
    'field_hit_rate': field.groupby('slot')['hit'].mean(),
}).round(2)
print(cmp)

print()
print("="*90)
print(f"TEAM {TEAM_NUMBER} total score by week vs field average that week")
print("="*90)
tw = team_week_totals(scored)
us_tw = tw[tw['team'] == TEAM_NUMBER]
field_avg_tw = tw[tw['team'] != TEAM_NUMBER].groupby('week')['total_score'].mean()
rank = tw.groupby('week')['total_score'].rank(ascending=False)
us_rank = rank[tw['team'] == TEAM_NUMBER]
for i, (_, r) in enumerate(us_tw.iterrows()):
    wk = r['week']
    print(f"Week {int(wk)}: total={r['total_score']:.1f} (acc={r['accuracy_avg']:.1f}, "
          f"pts={r['points']:.1f}, norm={r['norm_points']:.1f}) | field avg={field_avg_tw[wk]:.1f} "
          f"| rank={int(us_rank.iloc[i])}/9")

"""
Exploratory analysis of the 9-team class dataset (Weeks 1-4).
Run: python src/analyze_class_data.py
"""
import pandas as pd
from scoring import score_predictions, team_week_totals

df = pd.read_csv('data/class_predictions_wk1-4.csv')
scored = score_predictions(df)

print("="*90)
print("ACCURACY by slot (avg across all team-weeks)")
print("="*90)
acc_by_slot = scored.groupby('slot').agg(
    avg_width=('width', 'mean'),
    avg_miss=('miss', 'mean'),
    hit_rate=('hit', 'mean'),
    over_rate=('over', 'mean'),
    under_rate=('under', 'mean'),
    avg_accuracy=('accuracy', 'mean'),
).round(2)
print(acc_by_slot)

print()
print("="*90)
print("Width <-> hit-rate correlation by slot")
print("="*90)
corr = scored.groupby('slot').apply(
    lambda g: g['width'].corr(g['hit'].astype(float))
)
print(corr.round(3))

print()
print("="*90)
print("Overshoot vs undershoot: raw points and accuracy by outcome type")
print("="*90)
outcome = scored['over'].map({True: 'OVER', False: None}).combine_first(
    scored['under'].map({True: 'UNDER', False: None})
).fillna('HIT')
print(scored.groupby(outcome).agg(
    n=('actual', 'count'), avg_actual=('actual', 'mean'), avg_accuracy=('accuracy', 'mean')
).round(2))

print()
print("="*90)
print("Season-to-date ranking by Total Score")
print("="*90)
tw = team_week_totals(scored)
season = tw.groupby('team').agg(
    avg_accuracy=('accuracy_avg', 'mean'),
    avg_norm_points=('norm_points', 'mean'),
    avg_total_score=('total_score', 'mean'),
).round(2).sort_values('avg_total_score', ascending=False)
print(season)

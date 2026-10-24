"""
Backtest how much to weight consistency. For each past week W (using ONLY data
from weeks < W, exactly as the live model does), build the candidate tables
once, then re-rank them under different (ACCURACY_WEIGHT, risk-aversion) settings
and score each resulting lineup against what REALLY happened that week.

risk-aversion lam: points_score is computed from a conservative mean
(adj_mean - lam*std) instead of the raw mean, so high-variance boom/bust players
lose their ceiling credit. lam=0 reproduces the old behaviour.

Run: PYTHONPATH=src python3 src/backtest_weights.py
"""
import warnings; warnings.filterwarnings('ignore')
import itertools, numpy as np, pandas as pd
from player_week_stats import (build_offense_weekly, build_kicker_weekly,
                               build_defense_weekly, attach_roster_info)
import optimize_lineup as ol
from optimize_lineup import optimize_week, SCHEDULE_URL
ol.MIN_GAMES = 3   # fixed so Wk 4 (only 3 games of history) has candidates; live model uses 4
from scoring import score_predictions

# best raw point total of any class team that week (for Normalized Points)
CLASS_BEST = {4: 105.9, 5: 146.6, 6: 88.3}


BEST_REF = 110.0   # typical best-team weekly raw points (for the rubric-faithful objective)


def rerank_rubric(c, lam):
    """Rubric-faithful objective: Total = avg(Accuracy, NormPoints) so one slot adds
    exp_acc/12 (6 slots, then /2) and 50*mean/BEST_REF to Total Score.
    lam > 0 prices volatility by swapping mean for (mean - lam*std)."""
    c = c.copy()
    c['combined'] = c['expected_accuracy'] / 12 + 50 * (c['adj_mean'] - lam * c['std']) / BEST_REF
    return c.sort_values('combined', ascending=False)


def rerank(c, w, lam):
    c = c.copy()
    cons = c['adj_mean'] - lam * c['std']
    c['points_score'] = (cons / cons.max() * 100).clip(lower=0)
    c['combined'] = w * c['expected_accuracy'] + (1 - w) * c['points_score']
    return c.sort_values('combined', ascending=False)


def actual_of(slot, row, off, kick, dfn, wk):
    if slot == 'QB':
        d = off[(off.week == wk) & (off.role == 'QB') & (off.player_name == row.player_name)]
    elif slot in ('RB', 'WR', 'TE'):
        d = off[(off.week == wk) & (off.role == 'SKILL') & (off.player_name == row.player_name)]
    elif slot == 'K':
        d = kick[(kick.week == wk) & (kick.player_name == row.player_name)]
    else:
        d = dfn[(dfn.week == wk) & (dfn.team == row.team)]
    return float(d['fantasy_points'].iloc[0]) if len(d) else np.nan


def main():
    import nfl_data_py as nfl
    pbp26 = nfl.import_pbp_data([2026]); pbp25 = nfl.import_pbp_data([2025])
    sched = pd.read_csv(SCHEDULE_URL); s26 = sched[sched.season == 2026]
    off = attach_roster_info(build_offense_weekly(pbp26), nfl.import_seasonal_rosters([2026]))
    kick = build_kicker_weekly(pbp26); dfn = build_defense_weekly(pbp26, s26)
    off25 = build_offense_weekly(pbp25)

    weights = [0.5, 0.65, 0.8]
    lams = [0.0, 0.5, 1.0]
    rub_lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for wk in (4, 5, 6):
        res = optimize_week(wk, off, kick, dfn, off25, s26)
        settings = [(w, lam) for w, lam in itertools.product(weights, lams)] + [('rubric', l) for l in rub_lams]
        for w, lam in settings:
            picks = []
            for slot in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                top = (rerank_rubric(res[slot], lam) if w == 'rubric' else rerank(res[slot], w, lam)).iloc[0]
                act = actual_of(slot, top, off, kick, dfn, wk)
                picks.append(dict(week=wk, w=w, lam=lam, slot=slot, player=top.player_name,
                                  low=top.low, high=top.high, actual=act, std=top['std'], mean=top.adj_mean))
            rows.append(pd.DataFrame(picks))
    allp = pd.concat(rows, ignore_index=True)
    allp['w'] = allp['w'].astype(str)
    allp.to_csv('data/backtest_weights_picks.csv', index=False)
    sc = score_predictions(allp.dropna(subset=['actual']))
    lw = sc.groupby(['week', 'w', 'lam']).agg(acc=('accuracy', 'mean'), pts=('actual', 'sum'),
                                              hits=('hit', 'sum'), avg_std=('std', 'mean')).reset_index()
    lw['norm'] = lw.apply(lambda r: min(r.pts / CLASS_BEST[r.week] * 100, 100), axis=1)
    lw['total'] = (lw.acc + lw.norm) / 2
    out = lw.groupby(['w', 'lam']).agg(acc=('acc', 'mean'), pts=('pts', 'mean'), hits=('hits', 'mean'),
                                       avg_std=('avg_std', 'mean'), total=('total', 'mean')).round(1)
    print(out.to_string())
    print('\nPer-week Total Score estimate:')
    print(lw.pivot_table(index=['w', 'lam'], columns='week', values='total').round(1).to_string())


if __name__ == '__main__':
    main()

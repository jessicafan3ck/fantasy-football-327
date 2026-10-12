"""
Scores saved predictions against what actually happened, using the same
rubric as the class (Accuracy = 100 - width penalty - miss penalty).

A player is only scored once his game is COMPLETE in the data: the game
must have a final score in the schedule AND play-by-play whose final
score matches it. Anything else is reported as pending rather than
scored on partial data (nflverse play-by-play posts hours after the
schedule shows a final score).

Run: PYTHONPATH=src python3 src/evaluate_week.py --week 5
Predictions are read from data/predictions_weekN.csv
(columns: set, week, slot, player, team, low, high).
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from player_week_stats import (
    build_offense_weekly, build_kicker_weekly, build_defense_weekly, attach_roster_info,
)
from scoring import score_predictions

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'


def game_status(pbp_week: pd.DataFrame, sched_week: pd.DataFrame) -> dict:
    """team -> 'final' | 'pending: <reason>'"""
    g = pbp_week.groupby('game_id').agg(home=('home_team', 'first'), away=('away_team', 'first'),
                                        hs=('total_home_score', 'max'), as_=('total_away_score', 'max'))
    status = {}
    for r in sched_week.itertuples():
        for team in (r.home_team, r.away_team):
            if pd.isna(r.home_score):
                status[team] = 'pending: game not final'
                continue
            m = g[(g['home'] == r.home_team) & (g['away'] == r.away_team)]
            if len(m) == 0:
                status[team] = 'pending: final in schedule, play-by-play not posted yet'
            elif m['hs'].iloc[0] == r.home_score and m['as_'].iloc[0] == r.away_score:
                status[team] = 'final'
            else:
                status[team] = 'pending: play-by-play incomplete'
    return status


def main():
    import nfl_data_py as nfl
    ap = argparse.ArgumentParser()
    ap.add_argument('--week', type=int, default=5)
    args = ap.parse_args()

    preds = pd.read_csv(f'data/predictions_week{args.week}.csv')
    pbp = nfl.import_pbp_data([2026])
    ros = nfl.import_seasonal_rosters([2026])
    sched = pd.read_csv(SCHEDULE_URL)
    s26 = sched[sched['season'] == 2026]

    off = attach_roster_info(build_offense_weekly(pbp), ros)
    kick = build_kicker_weekly(pbp)
    dfn = build_defense_weekly(pbp, s26)
    status = game_status(pbp[pbp['week'] == args.week], s26[s26['week'] == args.week])

    def actual_of(r):
        if status.get(r.team) != 'final':
            return np.nan
        w = args.week
        if r.slot == 'QB':
            d = off[(off.week == w) & (off.role == 'QB') & (off.player_name == r.player)]
        elif r.slot in ('RB', 'WR', 'TE'):
            d = off[(off.week == w) & (off.role == 'SKILL') & (off.player_name == r.player)]
        elif r.slot == 'K':
            d = kick[(kick.week == w) & (kick.player_name == r.player)]
        else:
            d = dfn[(dfn.week == w) & (dfn.team == r.team)]
        return float(d['fantasy_points'].iloc[0]) if len(d) else np.nan

    preds['actual'] = [actual_of(r) for r in preds.itertuples()]
    preds['source'] = np.where(preds['actual'].notna(), 'play-by-play', '')

    # Optional hand-entered actuals (e.g. from a box score) for games whose
    # play-by-play hasn't posted yet. Used ONLY where pbp has nothing; swap to
    # play-by-play automatically once it posts, and flag the source.
    import os
    mpath = f'data/manual_actuals_week{args.week}.csv'
    if os.path.exists(mpath):
        man = pd.read_csv(mpath).set_index(['player', 'team'])
        for i, r in preds[preds['actual'].isna()].iterrows():
            if (r['player'], r['team']) in man.index:
                preds.loc[i, 'actual'] = float(man.loc[(r['player'], r['team']), 'actual'])
                preds.loc[i, 'source'] = 'box score (manual)'
    preds['game'] = preds['team'].map(status)
    done = preds.dropna(subset=['actual']).copy()
    scored = score_predictions(done.assign(team=1))   # score_predictions only needs slot/low/high/actual

    print("=" * 100)
    print(f"WEEK {args.week} -- predictions vs. actual (half-PPR league scoring)")
    print("=" * 100)
    scored['source'] = done['source'].values
    show = scored[['set', 'slot', 'player', 'low', 'high', 'actual', 'hit', 'miss', 'accuracy', 'source']].copy()
    show['result'] = np.where(scored['hit'], 'HIT', np.where(scored['over'], 'OVER', 'UNDER'))
    print(show.round(1).to_string(index=False))

    pend = preds[preds['actual'].isna()][['set', 'slot', 'player', 'game']]
    if len(pend):
        print("\nNOT YET SCORED:")
        print(pend.drop_duplicates(['player']).to_string(index=False))

    print("\n" + "-" * 100)
    print("By prediction set (completed slots only -- sets are NOT comparable if they cover different slots)")
    print("-" * 100)
    summ = scored.groupby('set').agg(slots_scored=('slot', 'size'), avg_accuracy=('accuracy', 'mean'),
                                     hit_rate=('hit', 'mean'), avg_width=('width', 'mean'),
                                     raw_points=('actual', 'sum')).round(1)
    summ['slots_total'] = preds.groupby('set').size()
    print(summ)


if __name__ == '__main__':
    main()

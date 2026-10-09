"""
End-to-end demo of the full model: pulls real play-by-play, builds the
weekly fantasy-point tables, runs empirical-Bayes shrinkage + opponent
adjustment, and produces a [Low, High] recommendation -- then BACKTESTS
it against what actually happened.

We predict Week 5 using only information that existed before Week 5 (2026
weeks 1-4 as "current season", all of 2025 as the shrinkage prior), for
the six players Team 9 actually started in Week 4 (reused here purely as
a realistic demo roster). We then compare the recommended range to the
real Week 5 outcome, and to the fixed-width bands the class has been
using.

Run: python src/run_week_projection.py
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from player_week_stats import (
    build_offense_weekly, build_kicker_weekly, build_defense_weekly, attach_roster_info,
)
from model import (
    empirical_bayes_blend, opponent_adjustment_offense, opponent_adjustment_defense,
    recommend_interval,
)
from scoring import SCALE, WIDTH_COST_NUM, MISS_COST_NUM

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'

# Team 9's Week 4 roster -- demo slate. (position is the true roster position,
# not the generic pbp role, so WR/RB/TE are split correctly.)
ROSTER = [
    ('QB', 'Josh Allen', 'BUF'),
    ('RB', 'Bijan Robinson', 'ATL'),
    ('WR', 'Chris Olave', 'NO'),
    ('TE', 'George Kittle', 'SF'),
    ('K',  "Ka'imi Fairbairn", 'HOU'),
    ('DEF', 'MIN', 'MIN'),
]
# what Team 9 actually submitted for Week 4 (their band, re-used in Week 5 as the
# "naive/fixed" comparison), and what Week 4's actual was:
WEEK4_BAND = {
    'Josh Allen': (26, 35), 'Bijan Robinson': (18, 27), 'Chris Olave': (15, 24),
    'George Kittle': (10, 16), "Ka'imi Fairbairn": (5, 11), 'MIN': (7, 13),
}


def project_week(predict_week, off26, kick26, def26, off25, sched26, has_actual):
    """Builds projections for `predict_week` using only weeks < predict_week,
    then (if has_actual) checks them against what actually happened."""
    off_blend = empirical_bayes_blend(off26, off25, predict_week)
    kick_blend = empirical_bayes_blend(kick26, kick26.iloc[0:0], predict_week)
    def_blend = empirical_bayes_blend(def26, def26.iloc[0:0], predict_week)

    off_opp = opponent_adjustment_offense(off26, sched26, predict_week)
    def_opp = opponent_adjustment_defense(off26, sched26, predict_week)

    wk = sched26[sched26['week'] == predict_week][['home_team', 'away_team']]
    opp_of = {}
    for _, r in wk.iterrows():
        opp_of[r['home_team']] = r['away_team']
        opp_of[r['away_team']] = r['home_team']

    print("\n" + "=" * 100)
    label = "BACKTEST" if has_actual else "LIVE PROJECTION"
    print(f"WEEK {predict_week} {label} -- built from weeks 1-{predict_week-1} only (no lookahead)")
    print("=" * 100)

    results = []
    for position, player, team in ROSTER:
        opp = opp_of.get(team)
        if position == 'K':
            row = kick_blend[kick_blend['team'].str.contains(team, case=False, na=False) |
                              kick_blend['player_name'].str.contains(player.split()[-1], case=False, na=False)]
            row = row.sort_values('n_games', ascending=False).head(1)
            ratio = 1.0
        elif position == 'DEF':
            row = def_blend[def_blend['team'] == team]
            opp_row = def_opp[def_opp['offteam'] == opp]
            ratio = float(opp_row['ratio'].iloc[0]) if len(opp_row) else 1.0
        else:
            role = 'QB' if position == 'QB' else 'SKILL'
            row = off_blend[(off_blend['player_name'].str.contains(player.split()[-1], case=False, na=False)) &
                             (off_blend['role'] == role) & (off_blend['team'] == team)]
            opp_row = off_opp[(off_opp['defteam'] == opp) & (off_opp['role'] == role)]
            ratio = float(opp_row['ratio'].iloc[0]) if len(opp_row) else 1.0

        if len(row) == 0:
            print(f"{player:20s} ({position}) -- no current-season data found, skipping")
            continue
        r = row.iloc[0]
        mean, std = r['blended_mean'], r['blended_std']
        adj_mean = mean * ratio
        rec = recommend_interval(adj_mean, std, position)

        actual = None
        if has_actual:
            if position == 'K':
                actual_row = kick26[(kick26['week'] == predict_week) & (kick26['team'] == team)]
            elif position == 'DEF':
                actual_row = def26[(def26['week'] == predict_week) & (def26['team'] == team)]
            else:
                role = 'QB' if position == 'QB' else 'SKILL'
                actual_row = off26[(off26['week'] == predict_week) & (off26['team'] == team) &
                                    (off26['role'] == role) &
                                    (off26['player_name'].str.contains(player.split()[-1], case=False, na=False))]
            actual = actual_row['fantasy_points'].iloc[0] if len(actual_row) else None

        fixed_low, fixed_high = WEEK4_BAND[player]
        results.append({
            'position': position, 'player': player, 'opponent': opp,
            'n_games_used': int(r['n_games']), 'blended_mean': round(mean, 1),
            'opp_ratio': round(ratio, 2), 'adj_mean': round(adj_mean, 1), 'std': round(std, 1),
            'model_low': rec['low'], 'model_high': rec['high'],
            'fixed_low': fixed_low, 'fixed_high': fixed_high,
            'actual': actual,
        })

    res = pd.DataFrame(results)
    print(res.to_string(index=False))

    if has_actual:
        print("\n" + "-" * 100)
        print(f"HIT CHECK -- did each range actually contain the real Week {predict_week} result?")
        print("-" * 100)
        res['scale'] = res['position'].map(SCALE)

        def rubric_accuracy(row, low_col, high_col):
            width = row[high_col] - row[low_col]
            if row['actual'] < row[low_col]:
                miss = row[low_col] - row['actual']
            elif row['actual'] > row[high_col]:
                miss = row['actual'] - row[high_col]
            else:
                miss = 0
            return 100 - width * (WIDTH_COST_NUM / row['scale']) - miss * (MISS_COST_NUM / row['scale'])

        res['model_hit'] = (res['actual'] >= res['model_low']) & (res['actual'] <= res['model_high'])
        res['fixed_hit'] = (res['actual'] >= res['fixed_low']) & (res['actual'] <= res['fixed_high'])
        res['model_width'] = res['model_high'] - res['model_low']
        res['fixed_width'] = res['fixed_high'] - res['fixed_low']
        res['model_accuracy'] = res.apply(lambda r: rubric_accuracy(r, 'model_low', 'model_high'), axis=1)
        res['fixed_accuracy'] = res.apply(lambda r: rubric_accuracy(r, 'fixed_low', 'fixed_high'), axis=1)
        print(res[['player', 'actual', 'model_low', 'model_high', 'model_hit', 'model_accuracy',
                   'fixed_low', 'fixed_high', 'fixed_hit', 'fixed_accuracy']].to_string(index=False))
        print(f"\nModel: hit rate {res['model_hit'].mean():.0%}, avg width {res['model_width'].mean():.1f}, "
              f"avg rubric Accuracy {res['model_accuracy'].mean():.1f}")
        print(f"Fixed: hit rate {res['fixed_hit'].mean():.0%}, avg width {res['fixed_width'].mean():.1f}, "
              f"avg rubric Accuracy {res['fixed_accuracy'].mean():.1f}")
    return res


def main():
    import nfl_data_py as nfl

    print("Pulling 2026 play-by-play + schedule...")
    pbp26 = nfl.import_pbp_data([2026])
    sched = pd.read_csv(SCHEDULE_URL)
    sched26 = sched[sched['season'] == 2026]

    print("Pulling 2025 play-by-play (shrinkage prior)...")
    pbp25 = nfl.import_pbp_data([2025])

    print("Pulling current rosters (for name/position resolution)...")
    rosters26 = nfl.import_seasonal_rosters([2026])

    print("Building weekly fantasy-point tables from play-by-play...")
    off26 = attach_roster_info(build_offense_weekly(pbp26), rosters26)
    kick26 = build_kicker_weekly(pbp26)
    def26 = build_defense_weekly(pbp26, sched26)
    off25 = build_offense_weekly(pbp25)  # prior season, player_id-keyed, no need to re-attach names

    # 1) BACKTEST: predict Week 4 using only weeks 1-3 + the 2025 prior,
    #    then check against Week 4's real, completed result.
    project_week(4, off26, kick26, def26, off25, sched26, has_actual=True)

    # 2) LIVE: predict Week 5 (this weekend's games -- not yet played) using
    #    weeks 1-4 + the 2025 prior. This is the actual recommendation.
    project_week(5, off26, kick26, def26, off25, sched26, has_actual=False)


if __name__ == '__main__':
    main()

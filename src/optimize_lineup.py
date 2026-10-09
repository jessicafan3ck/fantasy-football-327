"""
Lineup optimizer: instead of taking Team 9's roster as fixed, this ranks
EVERY available player at each slot (QB, RB, WR, TE, K, DEF) for the
target week and picks the one that maximizes a combined score built
directly from the two halves of Total Score:
  - Accuracy half: scoring.optimal_interval()'s expected_accuracy for
    that player's projected (mean, std).
  - Points half: the player's opponent-adjusted projected mean, since
    Normalized Points is just our raw point total vs. the field's --
    the single best lever we control is projecting the highest mean,
    not just the safest one.
combined_score = 0.5 * expected_accuracy + 0.5 * points_score, where
points_score re-scales projected mean to a 0-100 range using the spread
of ALL candidates at that slot this week (the same idea as Normalized
Points, just applied within the position instead of across the whole
league).

Run: python src/optimize_lineup.py
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

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
MIN_GAMES = 2          # ignore players with fewer current-season games (too noisy to trust)
TOP_N_PER_SLOT = 5      # how many ranked options to show per slot


def _opponent_map(sched26, week):
    wk = sched26[sched26['week'] == week][['home_team', 'away_team']]
    opp_of = {}
    for _, r in wk.iterrows():
        opp_of[r['home_team']] = r['away_team']
        opp_of[r['away_team']] = r['home_team']
    return opp_of


def _rank_slot(candidates: pd.DataFrame) -> pd.DataFrame:
    """candidates needs: player_name, team, opponent, adj_mean, std, position."""
    if len(candidates) == 0:
        return candidates
    c = candidates.copy()
    recs = c.apply(lambda r: recommend_interval(r['adj_mean'], r['std'], r['position']), axis=1)
    c['low'] = recs.apply(lambda d: d['low'])
    c['high'] = recs.apply(lambda d: d['high'])
    c['expected_accuracy'] = recs.apply(lambda d: d['expected_accuracy'])
    max_mean = c['adj_mean'].max()
    c['points_score'] = (c['adj_mean'] / max_mean * 100).clip(lower=0)
    c['combined_score'] = 0.5 * c['expected_accuracy'] + 0.5 * c['points_score']
    return c.sort_values('combined_score', ascending=False)


def optimize_week(predict_week: int, off26, kick26, def26, off25, sched26):
    off_blend = empirical_bayes_blend(off26, off25, predict_week)
    kick_blend = empirical_bayes_blend(kick26, kick26.iloc[0:0], predict_week)
    def_blend = empirical_bayes_blend(def26, def26.iloc[0:0], predict_week)
    off_opp = opponent_adjustment_offense(off26, sched26, predict_week)
    def_opp = opponent_adjustment_defense(off26, sched26, predict_week)
    opp_of = _opponent_map(sched26, predict_week)

    # pull true roster position back onto off_blend (role 'SKILL' -> RB/WR/TE)
    pos_lookup = off26[['player_id', 'position']].drop_duplicates('player_id') \
        if 'position' in off26.columns else None
    if pos_lookup is not None:
        off_blend = off_blend.merge(pos_lookup, on='player_id', how='left')
    off_blend = off_blend[off_blend['n_games'] >= MIN_GAMES].copy()
    off_blend['opponent'] = off_blend['team'].map(opp_of)
    off_blend = off_blend.dropna(subset=['opponent'])

    results = {}
    for slot, role, pos_filter in [('QB', 'QB', None), ('RB', 'SKILL', 'RB'),
                                    ('WR', 'SKILL', 'WR'), ('TE', 'SKILL', 'TE')]:
        cand = off_blend[off_blend['role'] == role].copy()
        if pos_filter:
            cand = cand[cand['position'] == pos_filter]
        cand = cand.merge(off_opp[off_opp['role'] == role], left_on='opponent', right_on='defteam', how='left')
        cand['ratio'] = cand['ratio'].fillna(1.0)
        cand['adj_mean'] = cand['blended_mean'] * cand['ratio']
        cand['std'] = cand['blended_std']
        cand['position'] = slot
        results[slot] = _rank_slot(cand[['player_name', 'team', 'opponent', 'n_games', 'blended_mean',
                                          'ratio', 'adj_mean', 'std', 'position']])

    kick_blend = kick_blend[kick_blend['n_games'] >= MIN_GAMES].copy()
    kick_blend['opponent'] = kick_blend['team'].map(opp_of)
    kick_blend = kick_blend.dropna(subset=['opponent'])
    kick_blend['ratio'] = 1.0
    kick_blend['adj_mean'] = kick_blend['blended_mean']
    kick_blend['std'] = kick_blend['blended_std']
    kick_blend['position'] = 'K'
    results['K'] = _rank_slot(kick_blend[['player_name', 'team', 'opponent', 'n_games', 'blended_mean',
                                           'ratio', 'adj_mean', 'std', 'position']])

    def_blend = def_blend[def_blend['n_games'] >= MIN_GAMES].copy()
    def_blend['opponent'] = def_blend['team'].map(opp_of)
    def_blend = def_blend.dropna(subset=['opponent'])
    def_blend = def_blend.merge(def_opp, left_on='opponent', right_on='offteam', how='left')
    def_blend['ratio'] = def_blend['ratio'].fillna(1.0)
    def_blend['adj_mean'] = def_blend['blended_mean'] * def_blend['ratio']
    def_blend['std'] = def_blend['blended_std']
    def_blend['position'] = 'DEF'
    results['DEF'] = _rank_slot(def_blend[['player_name', 'team', 'opponent', 'n_games', 'blended_mean',
                                            'ratio', 'adj_mean', 'std', 'position']])
    return results


def main():
    import nfl_data_py as nfl

    print("Pulling data (2026 pbp/schedule/rosters, 2025 pbp prior)...")
    pbp26 = nfl.import_pbp_data([2026])
    sched = pd.read_csv(SCHEDULE_URL)
    sched26 = sched[sched['season'] == 2026]
    pbp25 = nfl.import_pbp_data([2025])
    rosters26 = nfl.import_seasonal_rosters([2026])

    off26 = attach_roster_info(build_offense_weekly(pbp26), rosters26)
    kick26 = build_kicker_weekly(pbp26)
    def26 = build_defense_weekly(pbp26, sched26)
    off25 = build_offense_weekly(pbp25)

    predict_week = 5
    results = optimize_week(predict_week, off26, kick26, def26, off25, sched26)

    best_lineup = []
    print("\n" + "=" * 100)
    print(f"WEEK {predict_week} LINEUP OPTIMIZATION -- ranked by 0.5*expected_accuracy + 0.5*points_score")
    print("=" * 100)
    for slot in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
        c = results[slot]
        print(f"\n--- {slot} (top {TOP_N_PER_SLOT}) ---")
        cols = ['player_name', 'team', 'opponent', 'n_games', 'adj_mean', 'std', 'low', 'high',
                'expected_accuracy', 'points_score', 'combined_score']
        print(c[cols].head(TOP_N_PER_SLOT).round(1).to_string(index=False))
        if len(c):
            top = c.iloc[0]
            best_lineup.append({'slot': slot, 'player': top['player_name'], 'team': top['team'],
                                 'opponent': top['opponent'], 'low': top['low'], 'high': top['high'],
                                 'adj_mean': round(top['adj_mean'], 1),
                                 'combined_score': round(top['combined_score'], 1)})

    print("\n" + "=" * 100)
    print("RECOMMENDED LINEUP")
    print("=" * 100)
    print(pd.DataFrame(best_lineup).to_string(index=False))


if __name__ == '__main__':
    main()

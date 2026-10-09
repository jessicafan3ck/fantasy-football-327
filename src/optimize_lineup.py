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
combined_score = ACCURACY_WEIGHT * expected_accuracy + (1-ACCURACY_WEIGHT) * points_score,
where points_score re-scales projected mean to a 0-100 range using the
spread of ALL candidates at that slot this week (the same idea as
Normalized Points, just applied within the position instead of across
the whole league).

The rubric itself weights Accuracy and Points 50/50 -- but "anything
could happen" week to week, and expected_accuracy already prices in
consistency (optimal_interval has to widen the range for a high-std
player, which costs it Accuracy even at its best achievable interval).
ACCURACY_WEIGHT > 0.5 leans further into that: it explicitly favors the
more PREDICTABLE player over the merely higher-ceiling one, which is
the safer bet when we don't actually know which boom/bust player booms.
We also surface `cv` (std/mean, "coefficient of variation") directly so
consistency is visible, not just baked into one score.

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
    recommend_interval, apply_news_mixture,
)
from news_signals import load_news_signals
import os

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
MIN_GAMES = 3          # require a real track record, not just a low-variance 2-game read --
                        # a player's predictive std already gets inflated for small samples
                        # (see model.py), but that's a correction for ESTIMATION uncertainty,
                        # not a substitute for actually having a track record. 3+ games only.
TOP_N_PER_SLOT = 5      # how many ranked options to show per slot
ACCURACY_WEIGHT = 0.65  # prioritize consistency/Accuracy over ceiling/Points (rubric itself is 0.5/0.5)


def _opponent_map(sched26, week):
    wk = sched26[sched26['week'] == week][['home_team', 'away_team']]
    opp_of = {}
    for _, r in wk.iterrows():
        opp_of[r['home_team']] = r['away_team']
        opp_of[r['away_team']] = r['home_team']
    return opp_of


def _apply_news(row, news: pd.DataFrame):
    """If this player has a news-signal row this week, mix the (mean, std)
    toward the degraded scenario before it ever reaches optimal_interval --
    this is what actually widens/lowers a range for a real, researched
    risk, instead of hand-editing the output."""
    if news is None or len(news) == 0:
        return row['adj_mean'], row['std']
    match = news[news['player'] == row['player_name']]
    if len(match) == 0:
        return row['adj_mean'], row['std']
    r = match.iloc[0]
    return apply_news_mixture(row['adj_mean'], row['std'], r['active_prob_estimate'],
                               r['limited_mean_factor'])


def _rank_slot(candidates: pd.DataFrame, news: pd.DataFrame = None) -> pd.DataFrame:
    """candidates needs: player_name, team, opponent, adj_mean, std, position."""
    if len(candidates) == 0:
        return candidates
    c = candidates.copy()
    if news is not None and len(news):
        mixed = c.apply(lambda r: _apply_news(r, news), axis=1)
        c['adj_mean'] = mixed.apply(lambda t: t[0])
        c['std'] = mixed.apply(lambda t: t[1])
    recs = c.apply(lambda r: recommend_interval(r['adj_mean'], r['std'], r['position']), axis=1)
    c['low'] = recs.apply(lambda d: d['low'])
    c['high'] = recs.apply(lambda d: d['high'])
    c['expected_accuracy'] = recs.apply(lambda d: d['expected_accuracy'])
    max_mean = c['adj_mean'].max()
    c['points_score'] = (c['adj_mean'] / max_mean * 100).clip(lower=0)
    c['cv'] = (c['std'] / c['adj_mean'].abs().clip(lower=0.1)).round(2)  # coefficient of variation
    c['combined_score'] = ACCURACY_WEIGHT * c['expected_accuracy'] + (1 - ACCURACY_WEIGHT) * c['points_score']
    return c.sort_values('combined_score', ascending=False)


def optimize_week(predict_week: int, off26, kick26, def26, off25, sched26, news: pd.DataFrame = None):
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
                                          'ratio', 'adj_mean', 'std', 'position']], news=news)

    kick_blend = kick_blend[kick_blend['n_games'] >= MIN_GAMES].copy()
    kick_blend['opponent'] = kick_blend['team'].map(opp_of)
    kick_blend = kick_blend.dropna(subset=['opponent'])
    kick_blend['ratio'] = 1.0
    kick_blend['adj_mean'] = kick_blend['blended_mean']
    kick_blend['std'] = kick_blend['blended_std']
    kick_blend['position'] = 'K'
    results['K'] = _rank_slot(kick_blend[['player_name', 'team', 'opponent', 'n_games', 'blended_mean',
                                           'ratio', 'adj_mean', 'std', 'position']], news=news)

    def_blend = def_blend[def_blend['n_games'] >= MIN_GAMES].copy()
    def_blend['opponent'] = def_blend['team'].map(opp_of)
    def_blend = def_blend.dropna(subset=['opponent'])
    def_blend = def_blend.merge(def_opp, left_on='opponent', right_on='offteam', how='left')
    def_blend['ratio'] = def_blend['ratio'].fillna(1.0)
    def_blend['adj_mean'] = def_blend['blended_mean'] * def_blend['ratio']
    def_blend['std'] = def_blend['blended_std']
    def_blend['position'] = 'DEF'
    results['DEF'] = _rank_slot(def_blend[['player_name', 'team', 'opponent', 'n_games', 'blended_mean',
                                            'ratio', 'adj_mean', 'std', 'position']], news=news)
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
    news_path = f'data/news_signals_week{predict_week}.csv'
    news = load_news_signals(news_path) if os.path.exists(news_path) else None
    if news is not None:
        print(f"Loaded {len(news)} researched news-signal row(s) from {news_path} "
              f"-- these will widen/shift affected players' ranges before scoring.")
    results = optimize_week(predict_week, off26, kick26, def26, off25, sched26, news=news)

    best_lineup = []
    print("\n" + "=" * 100)
    print(f"WEEK {predict_week} LINEUP OPTIMIZATION -- ranked by {ACCURACY_WEIGHT}*expected_accuracy + "
          f"{round(1-ACCURACY_WEIGHT,2)}*points_score (consistency-weighted)")
    print("=" * 100)
    for slot in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
        c = results[slot]
        print(f"\n--- {slot} (top {TOP_N_PER_SLOT}) ---")
        cols = ['player_name', 'team', 'opponent', 'n_games', 'adj_mean', 'std', 'cv', 'low', 'high',
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

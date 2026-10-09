"""
The actual prediction model: turns weekly fantasy-point history into a
per-player [Low, High] recommendation for next week, optimized directly
against the rubric (via scoring.optimal_interval).

Pipeline, in order:
  1. Empirical-Bayes shrinkage: blend a player's own current-season
     average (small sample, weeks 1-N) toward a prior -- their own LAST
     season average if they have one, else this season's positional
     league average -- weighted by how many current-season games we've
     actually observed. Stops a 2-3 game hot/cold streak from being
     taken at face value.
  2. Opponent adjustment: scale the blended mean by how that player's
     opponent has performed on defense (for offense slots) or by how
     strong the opposing offense has been (for the DEF slot, inverted --
     a tougher opposing offense lowers our defense's expected points).
     This directly targets the QB/DEF weak spots from
     team9_weakspot_analysis.py: those misses were centering errors
     from ignoring matchup, not width problems.
  3. News-signal mixture: if a player has a news_signals row with
     active_prob < 1, blend the "plays/active" distribution with a
     "limited/out" distribution (mean/variance of a two-component
     mixture) rather than assuming full health.
  4. Feed the resulting (mean, std) into scoring.optimal_interval() to
     get the [Low, High] that maximizes expected Accuracy.

VARIANCE FIX (after the Week 4 backtest): the first version of this
model blended each player's own small-sample variance toward either
their own last-season variance (often itself from <4 games) or a
LEAGUE-WIDE variance of raw fantasy_points -- which mixes together
"how spiky is any one player's week-to-week game" with "how much
better are RB1s than backups," a totally different and much bigger
source of spread. That understated true game-to-game volatility and
is exactly why Bijan Robinson's 25.2-point Week 4 (band: 5.9-7.4)
cratered that player's Accuracy to 26.3 while the fixed, wider class
band scored 85.0 on the same game.

Two real fixes, both applied in empirical_bayes_blend now:
  a) The variance PRIOR is now the positional WITHIN-PLAYER variance --
     each player's own deviations from their own mean, pooled across
     all players at that role -- not the variance of raw scores across
     different players. This is the correct measure of "how much does
     a player like this swing week to week," and it's shrunk toward
     much more weakly than the mean (K_VAR=2) specifically so we don't
     wash out real observed boom/bust in a player's own game log.
  b) Predictive-variance inflation: with only n_games observed, our
     estimate of THIS player's own mean is itself uncertain, and that
     parameter uncertainty has to be added back into the variance of
     a predicted future game: Var(new game) = pop_var * (1 + 1/n_eff).
     At n=3 games (Bijan Robinson's case) this alone adds 33% to std,
     which is a real, derivable correction, not a fudge factor.
"""
import numpy as np
import pandas as pd

from scoring import SCALE, optimal_interval

# how many "prior games" of weight to give the MEAN shrinkage target.
# Higher K = trust the prior more / move slower as the season's sample grows.
SHRINKAGE_K = {'QB': 4, 'SKILL': 4, 'K': 6, 'DEF': 5}

# shrinkage weight for VARIANCE specifically -- deliberately much weaker
# than SHRINKAGE_K, because we want to preserve observed boom/bust rather
# than smooth it away toward an average.
VARIANCE_SHRINKAGE_K = 2

# clip opponent-adjustment ratios so one small-sample fluky week can't
# blow the projection up or down unrealistically.
OPPONENT_RATIO_CLIP = (0.75, 1.30)


def _player_current_stats(weekly: pd.DataFrame, upto_week: int) -> pd.DataFrame:
    """Per (player_id, role) mean/var/n using weeks < upto_week of the current season."""
    cur = weekly[weekly['week'] < upto_week]
    g = cur.groupby(['player_id', 'player_name', 'team', 'role'])['fantasy_points']
    stats = g.agg(n='count', mean='mean', var='var').reset_index()
    stats['var'] = stats['var'].fillna(0.0)
    return stats


def _positional_league_stats(weekly: pd.DataFrame, upto_week: int) -> pd.DataFrame:
    """League-wide MEAN per role (fallback mean prior) using weeks < upto_week."""
    cur = weekly[weekly['week'] < upto_week]
    g = cur.groupby('role')['fantasy_points']
    return g.agg(league_mean='mean').reset_index()


def _positional_within_player_variance(weekly: pd.DataFrame, upto_week: int) -> pd.Series:
    """
    The variance PRIOR: pools each player's own deviations from their own
    season-to-date mean, across all players at a role, using weeks <
    upto_week. This measures game-to-game volatility for a player of this
    type, not the (much larger, and irrelevant here) spread between a
    league's stars and backups that a raw across-player variance would
    pick up.
    """
    cur = weekly[weekly['week'] < upto_week].copy()
    cur['player_mean'] = cur.groupby(['player_id', 'role'])['fantasy_points'].transform('mean')
    cur['resid_sq'] = (cur['fantasy_points'] - cur['player_mean']) ** 2
    # players with only 1 game contribute 0 residual and no info -- drop them
    n_games = cur.groupby(['player_id', 'role'])['fantasy_points'].transform('count')
    cur = cur[n_games >= 2]
    pooled = cur.groupby('role').apply(
        lambda g: g['resid_sq'].sum() / max(len(g) - g['player_id'].nunique(), 1)
    )
    return pooled.rename('within_player_var')


def _prior_mean_for_player(player_id: str, role: str, prior_weekly: pd.DataFrame,
                            league_mean_fallback: float) -> float:
    """A player's own prior-season average if they played >=4 games last year,
    else the current-season positional league average."""
    hist = prior_weekly[(prior_weekly['player_id'] == player_id) & (prior_weekly['role'] == role)]
    if len(hist) >= 4:
        return hist['fantasy_points'].mean()
    return league_mean_fallback


def empirical_bayes_blend(current_weekly: pd.DataFrame, prior_weekly: pd.DataFrame,
                           upto_week: int) -> pd.DataFrame:
    """
    Returns per-player blended (mean, std, n_games_observed) using this
    season's weeks < upto_week, shrunk toward a prior as described in the
    module docstring (mean shrunk hard toward a prior; variance shrunk
    gently toward positional within-player volatility, then inflated for
    small-sample parameter uncertainty).
    """
    cur_stats = _player_current_stats(current_weekly, upto_week)
    league_mean = _positional_league_stats(current_weekly, upto_week).set_index('role')
    within_player_var = _positional_within_player_variance(current_weekly, upto_week)

    rows = []
    for _, r in cur_stats.iterrows():
        role = r['role']
        k_mean = SHRINKAGE_K.get(role, 4)
        mean_fallback = (league_mean.loc[role, 'league_mean'] if role in league_mean.index
                          else r['mean'])
        prior_mean = _prior_mean_for_player(r['player_id'], role, prior_weekly, mean_fallback)

        var_prior = within_player_var.get(role, r['var'] or 1.0)
        var_prior = var_prior if var_prior and var_prior > 0 else 1.0

        n = r['n']
        blended_mean = (n * r['mean'] + k_mean * prior_mean) / (n + k_mean)

        # Variance: shrink the player's OWN observed variance only weakly
        # toward the positional within-player prior (preserve real boom/bust),
        # then inflate for parameter (mean-estimation) uncertainty.
        cur_var = r['var'] if n >= 2 and r['var'] > 0 else var_prior
        blended_var = (n * cur_var + VARIANCE_SHRINKAGE_K * var_prior) / (n + VARIANCE_SHRINKAGE_K)
        n_eff = max(n, 1)
        predictive_var = blended_var * (1 + 1 / n_eff)

        rows.append({
            'player_id': r['player_id'], 'player_name': r['player_name'], 'team': r['team'],
            'role': role, 'n_games': n, 'blended_mean': blended_mean,
            'blended_std': np.sqrt(max(predictive_var, 0.01)),
            'prior_mean': prior_mean,
        })
    return pd.DataFrame(rows)


def opponent_adjustment_offense(current_weekly: pd.DataFrame, schedule: pd.DataFrame,
                                 upto_week: int) -> pd.DataFrame:
    """
    Per (defteam, role): ratio of points that defense has allowed to that
    role vs. the league average allowed to that role, using games played
    so far (weeks < upto_week). >1 means an easier-than-average matchup.
    """
    cur = current_weekly[current_weekly['week'] < upto_week].copy()
    sched = schedule[schedule['week'] < upto_week][
        ['season', 'week', 'home_team', 'away_team']]
    home = sched.rename(columns={'home_team': 'team', 'away_team': 'opp'})
    away = sched.rename(columns={'away_team': 'team', 'home_team': 'opp'})
    team_opp = pd.concat([home, away], ignore_index=True)

    cur = cur.merge(team_opp, on=['season', 'week', 'team'], how='left')
    allowed = cur.groupby(['opp', 'role'])['fantasy_points'].mean().reset_index()
    allowed = allowed.rename(columns={'opp': 'defteam', 'fantasy_points': 'allowed_avg'})

    league_allowed = cur.groupby('role')['fantasy_points'].mean().rename('league_allowed_avg')
    allowed = allowed.merge(league_allowed, on='role', how='left')
    allowed['ratio'] = (allowed['allowed_avg'] / allowed['league_allowed_avg']).clip(*OPPONENT_RATIO_CLIP)
    return allowed[['defteam', 'role', 'ratio']]


def opponent_adjustment_defense(current_weekly: pd.DataFrame, schedule: pd.DataFrame,
                                 upto_week: int) -> pd.DataFrame:
    """
    Per offteam: ratio of that offense's average scoring vs. league
    average. For a defense facing them, the adjustment is the INVERSE
    (a stronger-than-average offense lowers our defense's expected
    points), clipped the same way.
    """
    off = current_weekly[(current_weekly['week'] < upto_week) &
                          (current_weekly['role'].isin(['QB', 'SKILL']))]
    team_pts = off.groupby(['team', 'week'])['fantasy_points'].sum().reset_index()
    team_avg = team_pts.groupby('team')['fantasy_points'].mean()
    league_avg = team_avg.mean()
    ratio = (league_avg / team_avg).clip(*OPPONENT_RATIO_CLIP)  # inverted
    out = ratio.reset_index()
    out.columns = ['offteam', 'ratio']
    return out


def apply_news_mixture(mean: float, std: float, active_prob: float,
                        limited_mean_factor: float = 0.35) -> tuple:
    """
    Two-component mixture: with probability active_prob the player plays
    their normal role (mean, std); otherwise they play a diminished role
    (limited_mean_factor * mean, same relative variance). Returns the
    mixture's (mean, std) -- this is what should be fed to
    optimal_interval, not the raw healthy-case numbers.
    """
    if active_prob >= 1.0:
        return mean, std
    m2 = mean * limited_mean_factor
    var1, var2 = std ** 2, (std * limited_mean_factor) ** 2
    mix_mean = active_prob * mean + (1 - active_prob) * m2
    mix_var = (active_prob * var1 + (1 - active_prob) * var2
               + active_prob * (1 - active_prob) * (mean - m2) ** 2)
    return mix_mean, np.sqrt(mix_var)


def recommend_interval(mean: float, std: float, position: str) -> dict:
    scale = SCALE[position]
    low, high, expected_accuracy = optimal_interval(mean, std, scale)
    return {'low': round(low, 1), 'high': round(high, 1),
            'expected_accuracy': round(expected_accuracy, 1)}

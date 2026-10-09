"""
Scoring mechanics for the EXSS 327 Fantasy Football prediction game.

Two independent halves make up a team's weekly Total Score:
  - Accuracy: how well-calibrated the stated [Low, High] range was,
    averaged across the 6 roster slots.
  - Normalized Points: the team's raw Actual point total that week,
    expressed as a percentage of the highest-scoring team that week.

Total Score (week) = average(Accuracy_avg, Normalized Points)

Width Penalty = (High - Low) * (25 / Scale)
Miss Penalty  = distance outside [Low, High] * (60 / Scale)
Accuracy (slot) = 100 - Width Penalty - Miss Penalty

Scale is position-dependent (from the official scoring sheet):
  QB=20, RB=15, WR=14, TE=10, K=9, DEF=8
"""

import pandas as pd
import numpy as np

SCALE = dict(QB=20, RB=15, WR=14, TE=10, K=9, DEF=8)
WIDTH_COST_NUM = 25
MISS_COST_NUM = 60


def score_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with columns: team, week, slot, player, low, high, actual
    Returns the same dataframe with scoring columns added:
      scale, width, miss, over, under, hit,
      width_penalty, miss_penalty, accuracy
    """
    out = df.copy()
    out['scale'] = out['slot'].map(SCALE)
    out['width'] = out['high'] - out['low']
    out['miss'] = np.where(
        out['actual'] < out['low'], out['low'] - out['actual'],
        np.where(out['actual'] > out['high'], out['actual'] - out['high'], 0)
    )
    out['over'] = out['actual'] > out['high']
    out['under'] = out['actual'] < out['low']
    out['hit'] = (~out['over']) & (~out['under'])
    out['width_penalty'] = out['width'] * (WIDTH_COST_NUM / out['scale'])
    out['miss_penalty'] = out['miss'] * (MISS_COST_NUM / out['scale'])
    out['accuracy'] = 100 - out['width_penalty'] - out['miss_penalty']
    return out


def team_week_totals(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a scored (per-slot) dataframe up to team-week level:
    accuracy_avg, raw points, normalized points (vs. the best team that
    week), and the combined Total Score.
    """
    tw = scored_df.groupby(['team', 'week']).agg(
        accuracy_avg=('accuracy', 'mean'),
        points=('actual', 'sum'),
    ).reset_index()
    tw['norm_points'] = tw.groupby('week')['points'].transform(lambda s: s / s.max() * 100)
    tw['total_score'] = (tw['accuracy_avg'] + tw['norm_points']) / 2
    return tw


def optimal_interval(mean: float, std: float, scale: float, n_grid: int = 400):
    """
    Given a player's predicted mean and std (treated as Normal for this
    approximation) and their position's Scale, searches for the
    [low, high] interval that minimizes EXPECTED Accuracy loss:
        E[WidthPenalty + MissPenalty]
    rather than using a fixed percentile heuristic. This is the
    "analytically optimal" range described in the model design doc.

    Returns (low, high, expected_accuracy).
    """
    from scipy.stats import norm

    width_cost = WIDTH_COST_NUM / scale
    miss_cost = MISS_COST_NUM / scale

    best = None
    # search over candidate lows/highs around the mean +/- 4 std
    grid = np.linspace(mean - 4 * std, mean + 4 * std, n_grid)
    for low in grid:
        if low > mean + 2 * std:
            continue
        for high in grid:
            if high <= low:
                continue
            width = high - low
            # E[miss] = E[(low - X) | X<low]*P(X<low) + E[(X-high)|X>high]*P(X>high)
            # closed form for truncated normal expectation of shortfall/excess
            z_low = (low - mean) / std
            z_high = (high - mean) / std
            # E[(low - X)+] for X ~ N(mean, std): std * (phi(z_low) + z_low * Phi(z_low))
            e_under = std * (norm.pdf(z_low) + z_low * norm.cdf(z_low)) if std > 0 else 0
            # E[(X - high)+] for X ~ N(mean, std): std * (phi(z_high) - z_high * (1 - Phi(z_high)))
            e_over = std * (norm.pdf(z_high) - z_high * (1 - norm.cdf(z_high))) if std > 0 else 0
            exp_width_penalty = width * width_cost
            exp_miss_penalty = (e_under + e_over) * miss_cost
            exp_accuracy = 100 - exp_width_penalty - exp_miss_penalty
            if best is None or exp_accuracy > best[2]:
                best = (low, high, exp_accuracy)
    return best

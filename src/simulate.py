"""
Monte Carlo layer on top of the analytic model.

The analytic pipeline (model.py + scoring.optimal_interval) assumes every
player's outcome is Normal(mean, std). Real fantasy scores are skewed and
fat-tailed (esp. DEF and K), and the news-signal "mixture" was collapsed
back into one Normal. This module replaces those assumptions with
simulation, and adds the thing the analytic model cannot: a measure of how
STABLE the lineup picks are when our inputs are themselves uncertain.

What it does
------------
1. SHAPE: builds an empirical, standardized outcome distribution per
   position from last season's game logs (z = (x - player_mean)/player_std,
   starters only, pooled). Outcome = mean + std * z, so skew/tails are
   learned from data instead of assumed symmetric.
2. INTERVALS: solves for the [Low, High] that maximizes expected rubric
   Accuracy against that empirical distribution (and against the true
   two-scenario mixture for players with a news signal, instead of one
   collapsed Normal). Uses the empirical CDF with cumulative sums, so the
   expected shortfall/excess is exact for the sample, not noisy.
3. STABILITY: re-runs the whole lineup ranking N_RESAMPLES times, each time
   drawing the inputs we are unsure of (a player's true mean, his true
   volatility, and the opponent adjustment) from their uncertainty. Reports
   how often each candidate wins his slot. A pick that wins 90% of the time
   is robust; one that wins 35% is a coin flip against its neighbors.
4. LINEUP MC: simulates the chosen lineup N_ITER times (all inputs
   uncertain, news scenarios sampled) and reports the distribution of the
   lineup's Accuracy and raw points, plus per-slot hit rates.

Everything is seeded (--seed), so a given seed reproduces exactly. Run with
two seeds to see Monte Carlo error itself.

Assumptions worth knowing (not hidden):
  - Players are simulated INDEPENDENTLY. Real outcomes in the same game are
    correlated (our Goff/Gibbs/St. Brown/McBride all play DET-ARI), so the
    lineup-level spread reported here is, if anything, too narrow.
  - Uncertainty sizes: SE of a player's mean = std/sqrt(n_games + K);
    volatility df = n_games + VARIANCE_SHRINKAGE_K; opponent-ratio noise
    sd = RATIO_SD (a judgment call, not fit to data).
  - No floor is enforced on simulated scores (a DEF can go below 0 in real
    life too, but QB/RB/WR/TE essentially cannot).

Run: PYTHONPATH=src python3 src/simulate.py [--seed 327] [--resamples 2500] [--iters 100000]
"""
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from player_week_stats import (
    build_offense_weekly, build_kicker_weekly, build_defense_weekly, attach_roster_info,
)
from model import SHRINKAGE_K, VARIANCE_SHRINKAGE_K
from news_signals import load_news_signals
from optimize_lineup import (
    optimize_week, ACCURACY_WEIGHT, SCHEDULE_URL, MIN_GAMES,
)
from scoring import SCALE, WIDTH_COST_NUM, MISS_COST_NUM

PREDICT_WEEK = 5
TOP_K = 8                 # candidates per slot carried into the simulation
RATIO_SD = 0.08           # sd of log-noise on the opponent adjustment (judgment call)
SLOTS = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
# only starters inform the SHAPE of the distribution (bench players are mostly zeros)
SHAPE_MIN_MEAN = {'QB': 10, 'RB': 8, 'WR': 8, 'TE': 6, 'K': 5, 'DEF': -99}
SHAPE_MIN_GAMES = 8


# --------------------------------------------------------------------------
# 1. Empirical outcome shape
# --------------------------------------------------------------------------
def build_shape_pools(off25, kick25, def25) -> dict:
    """Standardized residual pools per slot, sorted, with cumulative sums."""
    pools = {}
    frames = {
        'QB': off25[off25['role'] == 'QB'],
        'RB': off25[(off25['role'] == 'SKILL') & (off25['position'] == 'RB')],
        'WR': off25[(off25['role'] == 'SKILL') & (off25['position'] == 'WR')],
        'TE': off25[(off25['role'] == 'SKILL') & (off25['position'] == 'TE')],
        'K': kick25,
        'DEF': def25,
    }
    for slot, df in frames.items():
        zs = []
        for _, g in df.groupby('player_id'):
            x = g['fantasy_points'].values
            if len(x) < SHAPE_MIN_GAMES or x.mean() < SHAPE_MIN_MEAN[slot] or x.std(ddof=1) <= 0:
                continue
            zs.append((x - x.mean()) / x.std(ddof=1))
        z = np.concatenate(zs)
        z = (z - z.mean()) / z.std()          # exactly mean 0, sd 1
        z.sort()
        cs = np.concatenate([[0.0], np.cumsum(z)])
        skew = float(((z) ** 3).mean())
        pools[slot] = {'z': z, 'cs': cs, 'n': len(z), 'skew': skew,
                       'q01': z[int(0.005 * len(z))], 'q99': z[int(0.995 * len(z)) - 1]}
    return pools


# --------------------------------------------------------------------------
# 2. Simulation-based interval solver (exact expectation over the empirical CDF)
# --------------------------------------------------------------------------
def _shortfall_excess(grid, m, s, pool):
    zq = (grid - m) / s
    idx = np.searchsorted(pool['z'], zq)
    n = pool['n']
    cs_i = pool['cs'][idx]
    e_under = s * (zq * idx - cs_i) / n                       # E[(g - X)+]
    e_over = s * ((pool['cs'][-1] - cs_i) - zq * (n - idx)) / n  # E[(X - g)+]
    return e_under, e_over


def solve_interval(scenarios, scale, pool, n_grid=240):
    """scenarios: list of (weight, mean, std). Returns (low, high, expected_accuracy)."""
    lo = min(m + s * pool['q01'] for _, m, s in scenarios)
    hi = max(m + s * pool['q99'] for _, m, s in scenarios)
    grid = np.linspace(lo, hi, n_grid)
    e_u = np.zeros(n_grid)
    e_o = np.zeros(n_grid)
    for w, m, s in scenarios:
        u, o = _shortfall_excess(grid, m, s, pool)
        e_u += w * u
        e_o += w * o
    width = grid[None, :] - grid[:, None]
    acc = (100 - width * (WIDTH_COST_NUM / scale)
           - (e_u[:, None] + e_o[None, :]) * (MISS_COST_NUM / scale))
    acc = np.where(width > 0, acc, -np.inf)
    i, j = np.unravel_index(np.argmax(acc), acc.shape)
    return float(grid[i]), float(grid[j]), float(acc[i, j])


def scenarios_for(m, s, news_row):
    if news_row is None:
        return [(1.0, m, s)]
    p, f = float(news_row['active_prob_estimate']), float(news_row['limited_mean_factor'])
    if p >= 1.0:
        return [(1.0, m, s)]
    return [(p, m, s), (1 - p, m * f, s * f)]


def exp_mean(scenarios):
    return sum(w * m for w, m, _ in scenarios)


# --------------------------------------------------------------------------
# data context
# --------------------------------------------------------------------------
def build_context():
    import nfl_data_py as nfl
    print("Pulling data (2026 + 2025 play-by-play, rosters, schedule)...")
    pbp26 = nfl.import_pbp_data([2026])
    pbp25 = nfl.import_pbp_data([2025])
    ros26 = nfl.import_seasonal_rosters([2026])
    ros25 = nfl.import_seasonal_rosters([2025])
    sched = pd.read_csv(SCHEDULE_URL)
    s26, s25 = sched[sched['season'] == 2026], sched[sched['season'] == 2025]

    off26 = attach_roster_info(build_offense_weekly(pbp26), ros26)
    kick26 = build_kicker_weekly(pbp26)
    def26 = build_defense_weekly(pbp26, s26)
    off25 = attach_roster_info(build_offense_weekly(pbp25), ros25)
    kick25 = build_kicker_weekly(pbp25)
    def25 = build_defense_weekly(pbp25, s25)
    return dict(off26=off26, kick26=kick26, def26=def26, off25=off25, kick25=kick25,
                def25=def25, s26=s26)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=327)
    ap.add_argument('--resamples', type=int, default=2500)
    ap.add_argument('--iters', type=int, default=100_000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ctx = build_context()
    pools = build_shape_pools(ctx['off25'], ctx['kick25'], ctx['def25'])

    print("\nEmpirical outcome shape from 2025 starters (Normal would be skew 0.00):")
    for s in SLOTS:
        p = pools[s]
        print(f"  {s:3s}  n={p['n']:5d}  skew={p['skew']:+.2f}  "
              f"5th-pct z={np.quantile(p['z'], .05):+.2f}  95th-pct z={np.quantile(p['z'], .95):+.2f}")

    news_path = f'data/news_signals_week{PREDICT_WEEK}.csv'
    news = load_news_signals(news_path) if os.path.exists(news_path) else None

    # Candidates BEFORE news (so the simulation can model the mixture itself),
    # and the production (analytic, news-collapsed) result for comparison.
    cands_nonews = optimize_week(PREDICT_WEEK, ctx['off26'], ctx['kick26'], ctx['def26'],
                                 ctx['off25'], ctx['s26'], news=None)
    cands_prod = optimize_week(PREDICT_WEEK, ctx['off26'], ctx['kick26'], ctx['def26'],
                               ctx['off25'], ctx['s26'], news=news)

    news_by_player = {}
    if news is not None:
        news_by_player = {r['player']: r for _, r in news.iterrows()}

    cand = {}
    for slot in SLOTS:
        df = cands_nonews[slot].head(TOP_K)
        extra = [p for p in news_by_player if p in set(cands_nonews[slot]['player_name'])
                 and p not in set(df['player_name'])]
        if extra:
            df = pd.concat([df, cands_nonews[slot][cands_nonews[slot]['player_name'].isin(extra)]])
        cand[slot] = df.reset_index(drop=True)

    # ---- baseline: simulation-solved interval + score per candidate (no param noise)
    def score_candidate(row, m, s, slot, max_mean_ref=None):
        sc = scenarios_for(m, s, news_by_player.get(row['player_name']))
        lo, hi, ea = solve_interval(sc, SCALE[slot], pools[slot])
        return sc, lo, hi, ea

    base_rows = {}
    for slot in SLOTS:
        rows = []
        for _, r in cand[slot].iterrows():
            sc, lo, hi, ea = score_candidate(r, r['adj_mean'], r['std'], slot)
            rows.append(dict(player=r['player_name'], team=r['team'], opp=r['opponent'],
                             n_games=int(r['n_games']), mean=exp_mean(sc), low=lo, high=hi,
                             exp_acc=ea, news=r['player_name'] in news_by_player))
        b = pd.DataFrame(rows)
        b['points_score'] = (b['mean'] / b['mean'].max() * 100).clip(lower=0)
        b['combined'] = ACCURACY_WEIGHT * b['exp_acc'] + (1 - ACCURACY_WEIGHT) * b['points_score']
        base_rows[slot] = b.sort_values('combined', ascending=False).reset_index(drop=True)

    # ---- stability: resample uncertain inputs, re-rank, count wins
    print(f"\nRe-ranking the lineup under input uncertainty "
          f"({args.resamples} resamples x {sum(len(c) for c in cand.values())} candidates)...")
    wins = {slot: pd.Series(0, index=cand[slot]['player_name']) for slot in SLOTS}
    comb_sum = {slot: pd.Series(0.0, index=cand[slot]['player_name']) for slot in SLOTS}
    for _ in range(args.resamples):
        for slot in SLOTS:
            recs = []
            for _, r in cand[slot].iterrows():
                n = max(int(r['n_games']), 1)
                k = SHRINKAGE_K['QB'] if slot == 'QB' else SHRINKAGE_K.get('SKILL', 4)
                se = r['std'] / np.sqrt(n + k)
                m_r = (r['adj_mean'] + se * rng.standard_normal()) * np.exp(RATIO_SD * rng.standard_normal())
                df_v = n + VARIANCE_SHRINKAGE_K
                s_r = r['std'] * np.sqrt(rng.chisquare(df_v) / df_v)
                sc = scenarios_for(m_r, s_r, news_by_player.get(r['player_name']))
                _, _, ea = solve_interval(sc, SCALE[slot], pools[slot], n_grid=120)
                recs.append((r['player_name'], exp_mean(sc), ea))
            d = pd.DataFrame(recs, columns=['player', 'mean', 'ea'])
            d['pts'] = (d['mean'] / d['mean'].max() * 100).clip(lower=0)
            d['combined'] = ACCURACY_WEIGHT * d['ea'] + (1 - ACCURACY_WEIGHT) * d['pts']
            wins[slot][d.loc[d['combined'].idxmax(), 'player']] += 1
            comb_sum[slot] += d.set_index('player')['combined']

    # ---- print per-slot results
    print("\n" + "=" * 100)
    print(f"WEEK {PREDICT_WEEK} SIMULATION-BASED RANKING  (ranked by mean combined score across resamples; "
          f"win% = share of resamples this player is the best pick at the slot)")
    a_over_b = WIDTH_COST_NUM / MISS_COST_NUM
    print(f"Rubric-implied optimal range: Low at the {a_over_b:.1%} percentile, High at the "
          f"{1 - a_over_b:.1%} percentile of the outcome distribution -> it only ever pays to cover the "
          f"central {1 - 2 * a_over_b:.1%} of outcomes (width cost 25 vs miss cost 60, any position).")
    print("=" * 100)
    lineup = []
    for slot in SLOTS:
        b = base_rows[slot].copy()
        b['win_pct'] = b['player'].map(wins[slot]) / args.resamples * 100
        b['mean_combined'] = b['player'].map(comb_sum[slot]) / args.resamples
        b = b.sort_values('mean_combined', ascending=False).reset_index(drop=True)
        base_rows[slot] = b
        print(f"\n--- {slot} ---")
        show = b[['player', 'team', 'opp', 'n_games', 'mean', 'low', 'high', 'exp_acc',
                  'mean_combined', 'win_pct']].round(1)
        print(show.head(6).to_string(index=False))
        lineup.append((slot, b.iloc[0]))

    # ---- lineup MC
    print("\n" + "=" * 100)
    print(f"LINEUP MONTE CARLO -- {args.iters:,} simulated weeks of the highest-expected-score lineup")
    print("=" * 100)
    prod_interval = {}
    for slot in SLOTS:
        for _, r in cands_prod[slot].iterrows():
            prod_interval[(slot, r['player_name'])] = (r['low'], r['high'])

    acc_sim = np.zeros((args.iters, len(SLOTS)))
    acc_prod = np.zeros((args.iters, len(SLOTS)))
    pts = np.zeros((args.iters, len(SLOTS)))
    hit_sim = np.zeros((args.iters, len(SLOTS)), dtype=bool)
    summary = []
    for k, (slot, t) in enumerate(lineup):
        r = cand[slot][cand[slot]['player_name'] == t['player']].iloc[0]
        n = max(int(r['n_games']), 1)
        kk = SHRINKAGE_K['QB'] if slot == 'QB' else SHRINKAGE_K.get('SKILL', 4)
        se = r['std'] / np.sqrt(n + kk)
        m_r = (r['adj_mean'] + se * rng.standard_normal(args.iters)) * np.exp(RATIO_SD * rng.standard_normal(args.iters))
        df_v = n + VARIANCE_SHRINKAGE_K
        s_r = r['std'] * np.sqrt(rng.chisquare(df_v, args.iters) / df_v)
        nr = news_by_player.get(t['player'])
        if nr is not None and float(nr['active_prob_estimate']) < 1.0:
            degraded = rng.random(args.iters) >= float(nr['active_prob_estimate'])
            f = float(nr['limited_mean_factor'])
            m_r = np.where(degraded, m_r * f, m_r)
            s_r = np.where(degraded, s_r * f, s_r)
        z = pools[slot]['z'][rng.integers(0, pools[slot]['n'], args.iters)]
        x = m_r + s_r * z

        sc = SCALE[slot]
        def acc_of(lo, hi):
            miss = np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0))
            return 100 - (hi - lo) * (WIDTH_COST_NUM / sc) - miss * (MISS_COST_NUM / sc)

        lo_s, hi_s = t['low'], t['high']
        lo_p, hi_p = prod_interval[(slot, t['player'])]
        acc_sim[:, k], acc_prod[:, k], pts[:, k] = acc_of(lo_s, hi_s), acc_of(lo_p, hi_p), x
        hit_sim[:, k] = (x >= lo_s) & (x <= hi_s)
        summary.append(dict(slot=slot, player=t['player'], opp=t['opp'],
                            sim_low=round(lo_s, 1), sim_high=round(hi_s, 1),
                            prod_low=lo_p, prod_high=hi_p,
                            win_pct=round(t['win_pct'], 1),
                            P_hit=round(hit_sim[:, k].mean() * 100, 1),
                            sim_acc=round(acc_sim[:, k].mean(), 1),
                            prod_acc=round(acc_prod[:, k].mean(), 1),
                            p5_pts=round(np.percentile(x, 5), 1), p50_pts=round(np.median(x), 1),
                            p95_pts=round(np.percentile(x, 95), 1)))
    S = pd.DataFrame(summary)
    print(S.to_string(index=False))

    def pct(a):
        return f"{a.mean():.1f}  (5th {np.percentile(a, 5):.1f} | 50th {np.median(a):.1f} | 95th {np.percentile(a, 95):.1f})"
    print(f"\nLineup avg Accuracy, simulation intervals : {pct(acc_sim.mean(axis=1))}")
    print(f"Lineup avg Accuracy, production (Normal)  : {pct(acc_prod.mean(axis=1))}")
    print(f"Lineup total raw points                   : {pct(pts.sum(axis=1))}")
    print(f"Avg # of 6 slots landing inside their range: {hit_sim.sum(axis=1).mean():.2f}")
    se_acc = acc_sim.mean(axis=1).std() / np.sqrt(args.iters)
    print(f"Monte Carlo standard error of the mean lineup Accuracy: +/-{se_acc:.3f} "
          f"(seed={args.seed}, {args.iters:,} iterations)")

    os.makedirs('data', exist_ok=True)
    S.to_csv(f'data/sim_week{PREDICT_WEEK}_lineup.csv', index=False)
    pd.concat([base_rows[s].assign(slot=s) for s in SLOTS]).to_csv(f'data/sim_week{PREDICT_WEEK}_candidates.csv', index=False)


if __name__ == '__main__':
    main()

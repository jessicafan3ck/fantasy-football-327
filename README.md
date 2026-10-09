# Fantasy Football 327

Prediction model project for EXSS 327 Fantasy Football (2026). We are
**Team 9**. Goal: optimize our weekly [Low, High] range + player
selections directly against the game's actual scoring formula, using
real open data plus a news-aggregation layer for the things structured
data can't capture (injury severity trend, role/usage commentary).

## Scoring mechanics (`src/scoring.py`)

Each week, Total Score = average(Accuracy, Normalized Points):
- **Accuracy**: average across the 6 roster slots of
  `100 - WidthPenalty - MissPenalty`, where
  `WidthPenalty = (High-Low) * 25/Scale` and
  `MissPenalty = (distance outside range) * 60/Scale`.
- **Normalized Points**: the team's raw `Actual` point total that
  week, as a percentage of the highest-scoring team that week.
- Scale by position: `QB=20, RB=15, WR=14, TE=10, K=9, DEF=8`.

## Repo layout

```
data/
  class_predictions_wk1-4.csv   # all 9 teams' Week 1-4 picks + actuals (216 rows)
  news_signals_template.csv     # worked example of the news-aggregation schema
  live/                         # output of pull_live_data.py (gitignored by default size, see below)
src/
  scoring.py                    # scoring math + the optimal-interval solver
  analyze_class_data.py         # class-wide EDA: where is accuracy being lost, by position?
  team9_diagnostic.py           # our team's picks vs. the field
  pull_live_data.py             # live 2026 NFL data pull (play-by-play, injuries, snaps, depth charts, NGS, schedule)
  news_signals.py               # loads/merges the news-aggregation layer into projections
```

All scripts are run from the **repo root**, e.g.:
```
pip install -r requirements.txt
python src/analyze_class_data.py
python src/team9_diagnostic.py
python src/pull_live_data.py --season 2026 --outdir data/live
python src/news_signals.py
```

## What we found in the class data (Weeks 1-4, 9 teams, 216 picks)

- **WR is where the whole class loses the most Accuracy** (avg
  Accuracy 51.5, nearly double the miss size of any other position) —
  and widening the range doesn't help there (width↔hit-rate
  correlation ≈ 0). The fix is a better center estimate, not a wider
  band.
- **QB has the strongest positive width↔hit-rate correlation (+0.42)**
  and the cheapest miss cost per point — widening QB ranges reliably
  converts misses into hits, essentially for free.
- **Undershooting happens ~2x as often as overshooting** (107 vs 57
  occurrences) across the class, and while the Accuracy penalty is
  symmetric either direction, overshoot outcomes average **23.9** raw
  points vs. **7.3** for undershoots — meaning overshoots are a much
  smaller real loss once the Points half of the score is considered.
- The two best season-to-date teams independently converged on
  strategies that validate this: Team 2 explicitly widened ranges
  because "the penalty for being incorrect is worse than the penalty
  for width," and Team 7 started feeding Vegas player-prop lines
  (O/U yards, anytime-TD odds) into their projections mid-season and
  immediately became more accurate.

See `src/team9_diagnostic.py` output for our own specific gaps: we're
**ahead of the field at WR/TE** (good center-estimate instincts) but
**behind the field at QB and DEF** — QB because our one highly
volatile starter (Josh Allen) got a fixed-width band that doesn't
match his real week-to-week variance, and DEF because two of our
picks (TEN Wk1, 49ers Wk3) completely cratered to near-zero, which
game-script/blowout risk (visible in betting lines, not matchup
"vibes") would likely have flagged ahead of time.

## Live data sources (`src/pull_live_data.py`)

Confirmed working from this project's environment as of 2026-10-09:

| Data | Source |
|---|---|
| Play-by-play | `nfl_data_py.import_pbp_data()` |
| Full schedule + live results | GitHub mirror of `nflverse/nfldata` `games.csv` |
| Injury reports | `nfl_data_py.import_injuries()` |
| Snap counts | `nfl_data_py.import_snap_counts()` |
| Depth charts | `nfl_data_py.import_depth_charts()` |
| Seasonal rosters | `nfl_data_py.import_seasonal_rosters()` |
| Next Gen Stats (aggregated) | `nfl_data_py.import_ngs_data()` |

**Not available from open sources right now:** betting lines / win
totals (schema exists in `nfl_data_py` but returns empty for 2026 —
would need a paid odds API for live lines), weather (would need NOAA's
API separately), and raw player tracking data (closed; NGS aggregated
stats above is the closest public substitute).

## News-signal aggregation layer (`src/news_signals.py`)

The structured sources above can't capture things like *how* a
"Questionable" tag is trending (practice participation improving vs.
worsening) or beat-reporter confidence language. That requires
actually reading current reporting, which this environment's code
can't do directly (news sites are blocked at the network level) —
only Claude's own search tools can reach them. So each week:

1. Run `pull_live_data.py` and flag any player with a non-trivial
   injury/role status.
2. For each flagged player, research their current practice trend +
   reporting tone and record one row in
   `data/news_signals_week{N}.csv` (schema in
   `news_signals_template.csv`), including an `active_prob_estimate`
   (0-1) synthesized from that reporting.
3. `merge_news_into_projections()` joins that onto the player
   projection table, feeding the mixture-model `active_prob` used in
   the Monte Carlo simulation.

See `news_signals_template.csv` for two real worked examples from
Week 5 2026 (Saquon Barkley — worsening/open-ended, `active_prob≈0.35`;
Chris Olave — planned rest day then returned to practice,
`active_prob≈0.88`) that look identical in a bare status-code table but
are very different risk profiles once you read the actual reporting.

## The model (all run from repo root with `PYTHONPATH=src`)

| File | Role |
|---|---|
| `scoring_rules.py` | The league's exact fantasy-point formula (half-PPR) |
| `player_week_stats.py` | Weekly fantasy points per player/K/DEF computed from play-by-play |
| `model.py` | Empirical-Bayes mean/std, opponent adjustment, news-signal mixture |
| `scoring.optimal_interval()` | Analytic [Low, High] maximizing expected rubric Accuracy |
| `optimize_lineup.py` | Ranks every candidate per slot (`ACCURACY_WEIGHT` = 0.65, `MIN_GAMES` = 3) |
| `simulate.py` | Monte Carlo: empirical outcome shape, lineup-stability resampling, lineup-level simulation |
| `data/news_signals_weekN.csv` | Hand-researched injury/role signals fed into the mixture |

Two facts worth knowing before trusting any range:
- The rubric (width cost 25 vs. miss cost 60) makes the optimal range cover
  only the central ~17% of outcomes (41.7th to 58.3rd percentile). Narrow
  ranges and ~1 of 6 slots landing inside them are expected, not a bug.
- `simulate.py` shows the Normal assumption costs almost nothing (lineup
  Accuracy 72.7 vs 72.6), but several slots (RB, WR, K, DEF) are statistical
  ties between their top candidates. QB and TE are the only clear picks.

Not modeled: correlation between players in the same game (the lineup's
spread is understated if several players share a game).

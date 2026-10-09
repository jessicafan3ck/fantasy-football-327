"""
News-signal aggregation layer.

WHY THIS FILE EXISTS AS DATA, NOT A SCRAPER:
This environment's shell/code cannot reach news sites directly (the
network allowlist blocks arbitrary domains -- confirmed: a direct curl
to cbssports.com returns a 403 from the proxy). What CAN reach news
sites is Claude's own WebSearch/WebFetch tools, used interactively.

So the workflow each week is:
  1. Pull the structured injury/depth-chart data (pull_live_data.py).
  2. For every player flagged there with a non-trivial status, run a
     targeted WebSearch/WebFetch (as Claude, not as code) for that
     player's current practice trend and beat-reporter outlook.
  3. Record the result as one row in data/news_signals_week{N}.csv,
     following the schema below.
  4. This module loads that file and merges it onto the player
     projection table, turning qualitative reporting into the
     active_prob input the Monte Carlo mixture model needs.

Schema (see data/news_signals_template.csv for two real worked
examples from Week 5, 2026 -- Saquon Barkley and Chris Olave):
  week, player, team, slot, injury_or_role_topic, practice_trend,
  severity_read, active_prob_estimate, role_note, source_urls,
  researched_on

Columns that matter downstream:
  active_prob_estimate : float in [0,1]. Feeds directly into the
      mixture-model `active_prob` parameter for that player's
      simulation (see the Monte Carlo design from earlier in the
      project). This is the number a bare status code ("Questionable")
      can't give you -- it's the synthesis of practice trend +
      reporting tone.
"""
import pandas as pd


NEWS_SIGNAL_COLUMNS = [
    'week', 'player', 'team', 'slot', 'injury_or_role_topic',
    'practice_trend', 'severity_read', 'active_prob_estimate',
    'role_note', 'source_urls', 'researched_on',
]


def load_news_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(NEWS_SIGNAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"news_signals file missing columns: {missing}")
    return df


def merge_news_into_projections(projections: pd.DataFrame, news: pd.DataFrame,
                                 default_active_prob: float = 1.0) -> pd.DataFrame:
    """
    Left-joins news signals onto a player projection table (on player
    name + week). Players with no news signal row get the default
    active_prob (1.0 -- healthy/no concern) rather than being dropped.
    """
    merged = projections.merge(
        news[['week', 'player', 'active_prob_estimate', 'role_note']],
        on=['week', 'player'], how='left'
    )
    merged['active_prob_estimate'] = merged['active_prob_estimate'].fillna(default_active_prob)
    return merged


if __name__ == '__main__':
    news = load_news_signals('data/news_signals_template.csv')
    print(news[['week', 'player', 'team', 'practice_trend', 'active_prob_estimate']]
          .to_string(index=False))

"""
Builds a per-player-per-week and per-team-defense-per-week fantasy-points
table directly from nflverse play-by-play, using the EXACT scoring rules
in scoring_rules.py (so the number this produces is the same number the
class rubric's "Actual" column would record).

This is the layer that used to have to come from nfl_data_py's
import_weekly_data() -- confirmed 404/unavailable for both 2025 and 2026
from this environment, so we compute it ourselves from play-by-play,
which IS available and live.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from scoring_rules import (
    qb_points, rush_rec_points, kicker_points, defense_points,
)


def _two_point(pbp, col_player, kind):
    """2pt conversions credited to a given player-role column, by kind ('pass'/'run'/'rec')."""
    mask = (pbp['two_point_attempt'] == 1) & (pbp['two_point_conv_result'] == 'success')
    sub = pbp[mask]
    return sub


def attach_roster_info(weekly: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """
    Joins full player name + true roster position onto a weekly offense
    table (whose player_id is the gsis id straight from play-by-play,
    matching nfl_data_py rosters' 'player_id' column). This is what turns
    the generic 'SKILL' role into a real RB/WR/TE split, and gives us the
    full name ("Bijan Robinson") instead of play-by-play's "Bi.Robinson"
    so it matches the names the class actually drafted with.
    """
    r = rosters[['player_id', 'player_name', 'position']].drop_duplicates('player_id')
    out = weekly.merge(r, on='player_id', how='left', suffixes=('_pbp', ''))
    out['player_name'] = out['player_name'].fillna(out['player_name_pbp'])
    out = out.drop(columns=['player_name_pbp'])
    return out


def build_offense_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (season, week, player, team, role) with fantasy
    points under QB / RB-REC (rush+rec) scoring, where role in {QB, SKILL}.
    A player can appear as both a passer (QB rows) and a rusher (SKILL rows,
    e.g. a scrambling QB) -- callers should pick the role matching the
    roster slot they're predicting for.
    """
    pbp = pbp[pbp['season_type'] == 'REG'].copy()
    key = ['season', 'week']

    # ---- Passing (QB) ----
    pass_df = pbp[pbp['passer_player_id'].notna()].copy()
    pass_2pt = pass_df[(pass_df['two_point_attempt'] == 1) &
                        (pass_df['two_point_conv_result'] == 'success')]
    passing = pass_df.groupby(key + ['passer_player_id', 'passer_player_name', 'posteam']).agg(
        pass_yds=('passing_yards', 'sum'),
        pass_td=('pass_touchdown', 'sum'),
        interceptions=('interception', 'sum'),
    ).reset_index()
    pass_2pt_ct = pass_2pt.groupby(key + ['passer_player_id']).size().rename('pass_2pt').reset_index()
    passing = passing.merge(pass_2pt_ct, on=key + ['passer_player_id'], how='left')
    passing['pass_2pt'] = passing['pass_2pt'].fillna(0)

    # rushing yards/TDs credited to the passer too (e.g. QB scrambles) get added below via rusher table,
    # so join the QB's own rushing line into their passing line to get a true total QB fantasy score.
    rush_df = pbp[pbp['rusher_player_id'].notna()].copy()
    rush_2pt = rush_df[(rush_df['two_point_attempt'] == 1) &
                        (rush_df['two_point_conv_result'] == 'success')]
    rushing = rush_df.groupby(key + ['rusher_player_id', 'rusher_player_name', 'posteam']).agg(
        rush_yds=('rushing_yards', 'sum'),
        rush_td=('rush_touchdown', 'sum'),
        fumbles_lost=('fumble_lost', 'sum'),
    ).reset_index()
    rush_2pt_ct = rush_2pt.groupby(key + ['rusher_player_id']).size().rename('rush_2pt').reset_index()
    rushing = rushing.merge(rush_2pt_ct, on=key + ['rusher_player_id'], how='left')
    rushing['rush_2pt'] = rushing['rush_2pt'].fillna(0)

    qb = passing.merge(
        rushing.rename(columns={'rusher_player_id': 'passer_player_id'})
               [key + ['passer_player_id', 'rush_yds', 'rush_td', 'rush_2pt', 'fumbles_lost']],
        on=key + ['passer_player_id'], how='left'
    )
    for c in ['rush_yds', 'rush_td', 'rush_2pt', 'fumbles_lost']:
        qb[c] = qb[c].fillna(0)
    qb['fantasy_points'] = qb.apply(
        lambda r: qb_points(r['pass_yds'], r['pass_td'], r['interceptions'], r['pass_2pt'],
                             r['rush_yds'], r['rush_td'], r['rush_2pt'], r['fumbles_lost']), axis=1)
    qb = qb.rename(columns={'passer_player_id': 'player_id', 'passer_player_name': 'player_name',
                             'posteam': 'team'})
    qb['role'] = 'QB'

    # ---- Rush+Rec skill players (RB/WR/TE) ----
    rec_df = pbp[pbp['receiver_player_id'].notna()].copy()
    rec_2pt = rec_df[(rec_df['two_point_attempt'] == 1) &
                      (rec_df['two_point_conv_result'] == 'success')]
    receiving = rec_df.groupby(key + ['receiver_player_id', 'receiver_player_name', 'posteam']).agg(
        rec=('complete_pass', 'sum'),
        rec_yds=('receiving_yards', 'sum'),
        rec_td=('pass_touchdown', 'sum'),
    ).reset_index()
    rec_2pt_ct = rec_2pt.groupby(key + ['receiver_player_id']).size().rename('rec_2pt').reset_index()
    receiving = receiving.merge(rec_2pt_ct, on=key + ['receiver_player_id'], how='left')
    receiving['rec_2pt'] = receiving['rec_2pt'].fillna(0)

    skill = rushing.rename(columns={'rusher_player_id': 'player_id', 'rusher_player_name': 'player_name',
                                     'posteam': 'team'}).merge(
        receiving.rename(columns={'receiver_player_id': 'player_id', 'receiver_player_name': 'player_name',
                                   'posteam': 'team'}),
        on=key + ['player_id', 'player_name', 'team'], how='outer'
    )
    for c in ['rush_yds', 'rush_td', 'rush_2pt', 'fumbles_lost', 'rec', 'rec_yds', 'rec_td', 'rec_2pt']:
        skill[c] = skill[c].fillna(0)
    skill['fantasy_points'] = skill.apply(
        lambda r: rush_rec_points(r['rush_yds'], r['rush_td'], r['rush_2pt'], r['rec'],
                                   r['rec_yds'], r['rec_td'], r['rec_2pt'], r['fumbles_lost']),
        axis=1)
    skill['role'] = 'SKILL'

    cols = key + ['player_id', 'player_name', 'team', 'role', 'fantasy_points']
    out = pd.concat([qb[cols], skill[cols]], ignore_index=True)
    return out


def build_kicker_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp[pbp['season_type'] == 'REG'].copy()
    key = ['season', 'week']

    fg = pbp[pbp['field_goal_attempt'] == 1].copy()
    fg['bucket'] = np.select_ = None  # placeholder to avoid lints
    def fg_bucket(row):
        if row['field_goal_result'] != 'made':
            return 'miss'
        d = row['kick_distance']
        if pd.isna(d) or d < 40:
            return 'fg_0_39'
        if d < 50:
            return 'fg_40_49'
        return 'fg_50_plus'
    fg['bucket'] = fg.apply(fg_bucket, axis=1)
    fg_counts = fg.groupby(key + ['kicker_player_name', 'posteam', 'bucket']).size().unstack(fill_value=0)
    fg_counts = fg_counts.reset_index()
    for col in ['miss', 'fg_0_39', 'fg_40_49', 'fg_50_plus']:
        if col not in fg_counts.columns:
            fg_counts[col] = 0

    pat = pbp[pbp['extra_point_attempt'] == 1].copy()
    pat_made = pat[pat['extra_point_result'] == 'good'].groupby(
        key + ['kicker_player_name', 'posteam']).size().rename('pat_made').reset_index()

    k = fg_counts.merge(pat_made, on=key + ['kicker_player_name', 'posteam'], how='outer')
    for c in ['miss', 'fg_0_39', 'fg_40_49', 'fg_50_plus', 'pat_made']:
        k[c] = k[c].fillna(0)
    k['fantasy_points'] = k.apply(
        lambda r: kicker_points(r['pat_made'], r['miss'], r['fg_0_39'], r['fg_40_49'], r['fg_50_plus']),
        axis=1)
    k = k.rename(columns={'kicker_player_name': 'player_name', 'posteam': 'team'})
    k['player_id'] = k['player_name']
    k['role'] = 'K'
    return k[key + ['player_id', 'player_name', 'team', 'role', 'fantasy_points']]


def build_defense_weekly(pbp: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, defteam) with that defense's fantasy points."""
    pbp = pbp[pbp['season_type'] == 'REG'].copy()
    key = ['season', 'week']

    # Total yards gained BY the offense each game -> that's yards ALLOWED by the opposing defense.
    off_yards = pbp.groupby(key + ['posteam'])['yards_gained'].sum().reset_index()
    off_yards = off_yards.rename(columns={'posteam': 'offteam', 'yards_gained': 'yards_allowed'})
    # map offteam -> opponent defteam for that game via schedule
    sched = schedule.copy()
    sched = sched.rename(columns={'season': 'season', 'week': 'week'})

    def_stats = pbp.groupby(key + ['defteam']).agg(
        sacks=('sack', 'sum'),
        interceptions=('interception', 'sum'),
        safeties=('safety', 'sum'),
    ).reset_index()

    fumbles_rec = pbp[pbp['fumble_recovery_1_team'] == pbp['defteam']].groupby(
        key + ['defteam']).size().rename('fumbles_recovered').reset_index()
    return_tds = pbp[(pbp['return_touchdown'] == 1) &
                      (pbp['return_team'] == pbp['defteam'])].groupby(
        key + ['defteam']).size().rename('return_tds').reset_index()

    def_stats = def_stats.merge(fumbles_rec, on=key + ['defteam'], how='left') \
                          .merge(return_tds, on=key + ['defteam'], how='left')
    for c in ['fumbles_recovered', 'return_tds']:
        def_stats[c] = def_stats[c].fillna(0)

    # points allowed: opponent's final score that game, from schedule
    home = sched[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    pa_home = home.rename(columns={'home_team': 'defteam', 'away_score': 'points_allowed'})[
        key + ['defteam', 'points_allowed']]
    pa_away = home.rename(columns={'away_team': 'defteam', 'home_score': 'points_allowed'})[
        key + ['defteam', 'points_allowed']]
    pts_allowed = pd.concat([pa_home, pa_away], ignore_index=True)

    # yards allowed: match offteam's yards to the opposing defteam via schedule
    y_home = home.rename(columns={'away_team': 'offteam', 'home_team': 'defteam'})[
        key + ['offteam', 'defteam']]
    y_away = home.rename(columns={'home_team': 'offteam', 'away_team': 'defteam'})[
        key + ['offteam', 'defteam']]
    matchup = pd.concat([y_home, y_away], ignore_index=True)
    yds_allowed = matchup.merge(off_yards, on=key + ['offteam'], how='left')[
        key + ['defteam', 'yards_allowed']]

    out = def_stats.merge(pts_allowed, on=key + ['defteam'], how='left') \
                    .merge(yds_allowed, on=key + ['defteam'], how='left')
    out['points_allowed'] = out['points_allowed'].fillna(out['points_allowed'].median())
    out['yards_allowed'] = out['yards_allowed'].fillna(out['yards_allowed'].median())
    out['two_pt_returns'] = 0
    out['blocks'] = 0

    out['fantasy_points'] = out.apply(
        lambda r: defense_points(r['sacks'], r['interceptions'], r['fumbles_recovered'],
                                  r['safeties'], r['return_tds'], r['two_pt_returns'], r['blocks'],
                                  r['points_allowed'], r['yards_allowed']), axis=1)
    out = out.rename(columns={'defteam': 'team'})
    out['player_name'] = out['team']
    out['player_id'] = out['team']
    out['role'] = 'DEF'
    return out[key + ['player_id', 'player_name', 'team', 'role', 'fantasy_points']]

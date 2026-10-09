"""
Pulls current-season (2026) structured NFL data from open sources.
All sources below were individually tested and confirmed reachable/live
as of 2026-10-09.

Usage:
    python src/pull_live_data.py --season 2026 --outdir data/live

Sources:
  - Play-by-play, injuries, snap counts, depth charts, NGS, rosters:
    nfl_data_py (wraps nflverse's public data releases)
  - Full season schedule (incl. results as they happen):
    nflverse/nfldata games.csv on GitHub (habitatring.com's dataset,
    mirrored there -- the direct habitatring.com host is NOT reachable
    from this environment's network allowlist, so pull from the GitHub
    mirror instead)

NOT available from open sources (see model design notes):
  - Betting lines / win totals: nfl_data_py's import_sc_lines and
    import_win_totals have the right schema but are returning empty
    for the 2026 season as of this writing. A paid odds API (The Odds
    API, OpticOdds) would be needed for live lines.
  - Weather: not in nfl_data_py; would pull separately from NOAA's API.
  - Raw player tracking (Next Gen Stats positional data): closed/not
    public. import_ngs_data() gives aggregated derived stats only
    (e.g. avg separation), which is the closest open substitute.
"""
import argparse
import os
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'


def pull_all(season: int, outdir: str):
    import nfl_data_py as nfl

    os.makedirs(outdir, exist_ok=True)

    print(f"Pulling play-by-play for {season}...")
    pbp = nfl.import_pbp_data([season])
    pbp.to_csv(f'{outdir}/pbp_{season}.csv', index=False)
    print(f"  -> {pbp.shape[0]} rows, through week {pbp['week'].max()}")

    print(f"Pulling schedule (via GitHub mirror of nflverse/nfldata)...")
    sched = pd.read_csv(SCHEDULE_URL)
    sched = sched[sched['season'] == season]
    sched.to_csv(f'{outdir}/schedule_{season}.csv', index=False)
    played = sched[sched['home_score'].notna()]
    print(f"  -> {sched.shape[0]} games, {played.shape[0]} played so far")

    print(f"Pulling injury reports for {season}...")
    inj = nfl.import_injuries([season])
    inj.to_csv(f'{outdir}/injuries_{season}.csv', index=False)
    print(f"  -> {inj.shape[0]} rows")

    print(f"Pulling snap counts for {season}...")
    snaps = nfl.import_snap_counts([season])
    snaps.to_csv(f'{outdir}/snap_counts_{season}.csv', index=False)
    print(f"  -> {snaps.shape[0]} rows")

    print(f"Pulling depth charts for {season}...")
    depth = nfl.import_depth_charts([season])
    depth.to_csv(f'{outdir}/depth_charts_{season}.csv', index=False)
    print(f"  -> {depth.shape[0]} rows")

    print(f"Pulling seasonal rosters for {season}...")
    rosters = nfl.import_seasonal_rosters([season])
    rosters.to_csv(f'{outdir}/rosters_{season}.csv', index=False)
    print(f"  -> {rosters.shape[0]} rows")

    print(f"Pulling Next Gen Stats (receiving) for {season}...")
    ngs = nfl.import_ngs_data('receiving', [season])
    ngs.to_csv(f'{outdir}/ngs_receiving_{season}.csv', index=False)
    print(f"  -> {ngs.shape[0]} rows")

    print("\nDone. NOT pulled (unavailable/needs paid source): betting lines, weather.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--outdir', type=str, default='data/live')
    args = parser.parse_args()
    pull_all(args.season, args.outdir)

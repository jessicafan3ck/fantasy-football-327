"""
The EXACT league scoring rules for "Actual" points, transcribed from
EXSS_327_Fantasy_Football_2026_-_Scoring.csv. Used to compute fantasy
points for any player/defense from play-by-play aggregates, so our
model predicts the SAME number the rubric actually scores on.
"""

# ---- Offense (QB / RB / WR / TE) ----
PASS_YD = 1 / 25      # 1 pt per 25 passing yards
PASS_TD = 4
INT_THROWN = -2
PASS_2PT = 2

RUSH_YD = 0.1
RUSH_TD = 6
RUSH_2PT = 2

REC_YD = 0.1
REC = 0.5
REC_TD = 6
REC_2PT = 2

# ---- Kicker ----
PAT_MADE = 1
FG_MISS = -1
FG_0_39 = 3
FG_40_49 = 4
FG_50_PLUS = 5

# ---- Team Defense / Special Teams ----
DEF_TD = 6          # any return TD (kickoff/punt/INT/fumble/blocked-kick return)
DEF_2PT_RET = 2
DEF_SAFETY_1PT = 1  # "1pt Safety" line item (rare rule variant)
DEF_SACK = 1
DEF_BLOCK = 2        # blocked punt/PAT/FG (not returned for TD)
DEF_INT = 2
DEF_FUM_REC = 2
DEF_SAFETY = 2
DEF_FUML = -2        # "Total Fumbles Lost" miscellaneous line -- this is against
                      # the opposing offense's own defense entry, kept separate below

PTS_ALLOWED_BRACKETS = [
    (0, 0, 6), (1, 6, 5), (7, 13, 3), (14, 17, 1),
    # 18-27 = 0 (not listed -> implicit 0)
    (28, 34, -1), (35, 45, -3), (46, 999, -5),
]
YDS_ALLOWED_BRACKETS = [
    (0, 99, 5), (100, 199, 3), (200, 299, 2),
    # 300-349 = 0 (not listed -> implicit 0)
    (350, 399, -1), (400, 449, -3), (450, 499, -5),
    (500, 549, -6), (550, 10_000, -7),
]


def _bracket(value, brackets):
    for lo, hi, pts in brackets:
        if lo <= value <= hi:
            return pts
    return 0


def points_allowed_score(points_allowed: float) -> float:
    return _bracket(points_allowed, PTS_ALLOWED_BRACKETS)


def yards_allowed_score(yards_allowed: float) -> float:
    return _bracket(yards_allowed, YDS_ALLOWED_BRACKETS)


def qb_points(pass_yds, pass_td, interceptions, pass_2pt,
              rush_yds=0, rush_td=0, rush_2pt=0, fumbles_lost=0):
    return (pass_yds * PASS_YD + pass_td * PASS_TD + interceptions * INT_THROWN
            + pass_2pt * PASS_2PT + rush_yds * RUSH_YD + rush_td * RUSH_TD
            + rush_2pt * RUSH_2PT + fumbles_lost * DEF_FUML)


def rush_rec_points(rush_yds, rush_td, rush_2pt, rec, rec_yds, rec_td, rec_2pt,
                     fumbles_lost=0):
    return (rush_yds * RUSH_YD + rush_td * RUSH_TD + rush_2pt * RUSH_2PT
            + rec * REC + rec_yds * REC_YD + rec_td * REC_TD + rec_2pt * REC_2PT
            + fumbles_lost * DEF_FUML)


def kicker_points(pat_made, fg_missed, fg_0_39, fg_40_49, fg_50_plus):
    return (pat_made * PAT_MADE + fg_missed * FG_MISS + fg_0_39 * FG_0_39
            + fg_40_49 * FG_40_49 + fg_50_plus * FG_50_PLUS)


def defense_points(sacks, interceptions, fumbles_recovered, safeties,
                    return_tds, two_pt_returns, blocks, points_allowed,
                    yards_allowed):
    return (sacks * DEF_SACK + interceptions * DEF_INT
            + fumbles_recovered * DEF_FUM_REC + safeties * DEF_SAFETY
            + return_tds * DEF_TD + two_pt_returns * DEF_2PT_RET
            + blocks * DEF_BLOCK + points_allowed_score(points_allowed)
            + yards_allowed_score(yards_allowed))

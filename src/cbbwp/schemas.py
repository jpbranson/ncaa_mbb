"""Canonical data contracts shared by the offline and live pipelines.

Every adapter (historical parquet, live ESPN feed, a paid feed later) must emit
`Event` objects. Nothing downstream of an adapter knows where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Rules constants (men's NCAA, 2015-16 rules onward) ----------------------
HALF_SECONDS = 20 * 60          # 1200
REGULATION_SECONDS = 2 * HALF_SECONDS  # 2400
OT_SECONDS = 5 * 60             # 300
TIMEOUTS_AT_TIP = 4             # approximation of the men's allotment


@dataclass(frozen=True, slots=True)
class Event:
    """One play, normalised. `seq` orders events within a game."""
    game_id: int
    seq: int
    period: int                  # 1,2 = halves; 3+ = overtime
    # Seconds left IN THE PERIOD at the play. `None` means the feed gave no
    # readable clock for this play; `state.build_states` resolves that by
    # carrying the period's previous clock forward. An adapter must NOT
    # substitute 0 for "unknown" - see STATE_RULES_VERSION 3 below.
    clock_seconds: Optional[int]
    home_score: int
    away_score: int
    event_type: str              # e.g. "JumpShot", "Timeout", "DefensiveRebound"
    team_id: Optional[int]       # team the event is attributed to
    score_value: int = 0
    scoring_play: bool = False
    shooting_play: bool = False
    text: str = ""


@dataclass(frozen=True, slots=True)
class PregameContext:
    """Known before tip-off. Loaded once per game, never re-fetched mid-game."""
    game_id: int
    home_team_id: int
    away_team_id: int
    neutral_site: bool = False
    # Expected home margin in points (positive = home favoured).
    # Either the negated closing spread, or a model-derived rating differential.
    pregame_exp_margin: float = 0.0
    season: int = 0
    ft_pct_diff: float = 0.0
    exp_points_per_min: float = 3.4


@dataclass(slots=True)
class GameState:
    """A snapshot AFTER one event. One event -> one state row."""
    game_id: int
    seq: int
    period: int
    is_ot: bool
    clock_seconds: int            # left in the period
    game_seconds_remaining: int   # left in regulation, or in the current OT
    home_score: int
    away_score: int
    margin: int                   # home - away
    possession: float             # 1.0 home, 0.0 away, 0.5 unknown
    home_timeouts: int
    away_timeouts: int
    home_fouls: int = 0            # team fouls in the current half
    away_fouls: int = 0
    pregame_exp_margin: float = 0.0
    neutral_site: bool = False
    ft_pct_diff: float = 0.0       # home season-to-date FT% minus away's
    exp_points_per_min: float = 3.4  # combined scoring rate of the two teams


# Column order is part of the contract: the fitted model's coefficients are
# positional. Changing this list requires a new model version.
FEATURE_NAMES = [
    "margin",
    "sqrt_time",
    "margin_per_sqrt_time",
    "possession",
    "pregame_exp_margin",
    "pregame_exp_margin_decayed",
    "is_ot",
    "timeout_diff",
    "bonus_diff",
    "ft_pct_diff",
    "margin_per_sqrt_points_left",
]

# Men's NCAA bonus thresholds, in team fouls per half.
BONUS_FOULS = 7        # 1-and-1
DOUBLE_BONUS_FOULS = 10

# Version of the STATE RULES, as opposed to the feature list above.
#
# FEATURE_NAMES catches someone adding, removing or reordering a column. It does
# NOT catch someone changing what an existing column MEANS - and that is the
# more dangerous edit, because nothing downstream looks any different.
#
# This happened on 2026-09-01: the possession rule was corrected so that made
# three-pointers in 2016-2019 flip possession (they had not, because ESPN typed
# them "Three Point Jump Shot" and the rule keyed on play-type names). The
# feature list was untouched, so the manifest check passed, and a model trained
# on the old meaning would have been served states built with the new one.
#
# Bump this whenever the meaning of any GameState field changes, and refit.
#   1 - original rules, shipped 2026-08-31 (registry/v1)
#   2 - made field goals detected by scoring+shooting flags, not type names
#   3 - an unreadable clock carries the period's previous clock forward instead
#       of silently becoming 0:00 (2026-09-08)
#
# On 3: `clock_to_seconds` mapped both "this play has no clock" and "this clock
# is a format we do not understand" to 0. In the FIRST half that is harmless
# (game_seconds_remaining = 1200, mid-game). In the second half or an overtime
# it means `game_seconds_remaining == 0`, and `endgame.apply` reads that as "the
# game is over" and clamps the published probability to 0.999 - a confident,
# wrong number on a game with ten minutes left, produced by one malformed field.
#
# Measured before making the change: across all ten seasons, 19,462,128
# play-by-play rows, **zero** carry an unreadable clock. So this rule changes no
# training row, and the refit under it reproduced registry/v2 byte for byte
# (published as registry/v3). It is a guard against a feed that changes, not a
# correction of anything in the data - which is exactly why it had to be a
# version bump rather than a quiet edit: nothing about the numbers would have
# revealed it either way.
STATE_RULES_VERSION = 3

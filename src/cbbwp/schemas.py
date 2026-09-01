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
    clock_seconds: int           # seconds left IN THE PERIOD at the play
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
STATE_RULES_VERSION = 2

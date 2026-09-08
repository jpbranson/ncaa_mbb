"""Adapter: hoopR / sportsdataverse historical play-by-play parquet -> Events.

Two paths, deliberately:
  * `load_events` builds canonical Event objects for one or a few games. This is
    the reference path and the one the live pipeline mirrors.
  * `states_lazy` does the same transformation in Polars for tens of millions of
    rows. It exists only for speed, and `tests/test_parity.py` asserts it agrees
    with the reference path row-for-row on real games.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import polars as pl

from ..schemas import Event, HALF_SECONDS, OT_SECONDS
from ..state import TEAM_TIMEOUT_TYPES, FOUL_TYPES, parse_clock

# Columns present in every season 2016-2026 of the hoopR mbb pbp files.
BASE_COLS = [
    "game_id", "game_play_number", "period_number", "clock_display_value",
    "home_score", "away_score", "type_text", "team_id", "score_value",
    "scoring_play", "shooting_play", "home_team_id", "away_team_id",
    "season", "season_type", "game_date",
]

MADE_SHOT_TYPES = ["JumpShot", "LayUpShot", "DunkShot", "TipShot"]


def load_events(path: str, game_id: int) -> tuple[List[Event], int, int]:
    """Reference path: one game's parquet rows -> canonical Events."""
    df = (
        pl.scan_parquet(path)
        .filter(pl.col("game_id") == game_id)
        .select(BASE_COLS)
        .sort("game_play_number")
        .collect()
    )
    if df.is_empty():
        raise KeyError(f"game {game_id} not in {path}")
    home_id = int(df["home_team_id"][0])
    away_id = int(df["away_team_id"][0])
    events = [
        Event(
            game_id=int(r["game_id"]),
            seq=int(r["game_play_number"]),
            period=int(r["period_number"] or 1),
            # None, not 0, when the clock cannot be read: build_states carries
            # the period's previous clock forward (STATE_RULES_VERSION 3).
            clock_seconds=parse_clock(r["clock_display_value"] or ""),
            home_score=int(r["home_score"] or 0),
            away_score=int(r["away_score"] or 0),
            event_type=r["type_text"] or "",
            team_id=None if r["team_id"] is None else int(r["team_id"]),
            score_value=int(r["score_value"] or 0),
            scoring_play=bool(r["scoring_play"]),
            shooting_play=bool(r["shooting_play"]),
        )
        for r in df.iter_rows(named=True)
    ]
    return events, home_id, away_id


# --------------------------------------------------------------------------
# Vectorised bulk path
# --------------------------------------------------------------------------
def _clock_seconds_expr(col: str = "clock_display_value") -> pl.Expr:
    """Vectorised twin of `state.parse_clock`: NULL when the clock is unreadable.

    It used to `.fill_null(0)` here, which is the vectorised half of the bug
    STATE_RULES_VERSION 3 fixes: an unreadable clock silently became 0:00, and
    a 0:00 in the second half reads as a finished game. The null is now carried
    forward in `states_lazy`, matching `state.build_states`.
    """
    parts = pl.col(col).str.split_exact(":", 1)
    mm = parts.struct.field("field_0").cast(pl.Float64, strict=False)
    ss = parts.struct.field("field_1").cast(pl.Float64, strict=False)
    return (
        pl.when(pl.col(col).str.contains(":"))
        .then(mm * 60 + ss.floor())
        .otherwise(pl.col(col).cast(pl.Float64, strict=False).floor())
        .cast(pl.Int32)
    )


def states_lazy(lf: pl.LazyFrame, timeouts_at_tip: int = 4) -> pl.LazyFrame:
    """Vectorised equivalent of state.build_states over many games at once."""
    t = pl.col("type_text")
    actor = (
        pl.when(pl.col("team_id") == pl.col("home_team_id")).then(pl.lit(1.0))
        .when(pl.col("team_id") == pl.col("away_team_id")).then(pl.lit(0.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    other = 1.0 - actor
    made = pl.col("scoring_play").fill_null(False)
    shooting = pl.col("shooting_play").fill_null(False)

    poss_set = (
        # Same rule, same order, as state._possession_after. Made field goals are
        # detected by the scoring/shooting flags, not by play-type name - see the
        # comment there for why the name whitelist was wrong for 2016-2019.
        pl.when(t.str.contains("FreeThrow") & made).then(other)
        .when(made & shooting).then(other)
        .when(t.is_in(["Defensive Rebound", "Offensive Rebound"])).then(actor)
        .when(t.str.contains("Turnover")).then(other)
        .when(t == "Steal").then(actor)
        .when(t == "Jumpball").then(pl.lit(0.5))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )

    is_team_to = t.is_in(list(TEAM_TIMEOUT_TYPES))
    is_foul = t.is_in(list(FOUL_TYPES))
    period = pl.col("period_number").fill_null(1).clip(lower_bound=1).cast(pl.Int32)
    # `_period` is materialised first because the clock's
    # forward-fill has to be windowed on the period, and a window needs a
    # column rather than an expression.
    plen_col = pl.when(pl.col("_period") <= 2).then(pl.lit(HALF_SECONDS)).otherwise(pl.lit(OT_SECONDS))
    # Same rule as state.build_states: carry an unreadable clock forward within
    # its period, and start a period whose first clock is unreadable at full
    # length. tests/test_parity.py holds the two implementations together, and
    # test_state.py pins the rule itself on both paths.
    clock_filled = (pl.col("_raw_clock").forward_fill().over(["game_id", "_period"])
                    .fill_null(plen_col))
    clock = pl.min_horizontal(clock_filled.clip(lower_bound=0), plen_col).cast(pl.Int32)
    gsr = (pl.when(pl.col("_period") <= 1)
           .then(pl.lit(HALF_SECONDS) + pl.col("_clock"))
           .otherwise(pl.col("_clock")))

    allot = pl.lit(timeouts_at_tip) + (pl.col("_period") - 2).clip(lower_bound=0)

    return (
        lf.sort(["game_id", "game_play_number"])
        .with_columns(_period=period, _raw_clock=_clock_seconds_expr())
        .with_columns(_clock=clock)
        .with_columns(
            _gsr=gsr.cast(pl.Int32),
            _poss_set=poss_set,
            _to_home=(is_team_to & (pl.col("team_id") == pl.col("home_team_id"))).cast(pl.Int32),
            _to_away=(is_team_to & (pl.col("team_id") == pl.col("away_team_id"))).cast(pl.Int32),
            _allot=allot,
            _half=pl.when(pl.col("_period") <= 1).then(pl.lit(1)).otherwise(pl.lit(2)),
            _foul_home=(is_foul & (pl.col("team_id") == pl.col("home_team_id"))).cast(pl.Int32),
            _foul_away=(is_foul & (pl.col("team_id") == pl.col("away_team_id"))).cast(pl.Int32),
        )
        .with_columns(
            possession=pl.col("_poss_set").forward_fill().over("game_id").fill_null(0.5),
            home_used=pl.col("_to_home").cum_sum().over("game_id"),
            away_used=pl.col("_to_away").cum_sum().over("game_id"),
            home_fouls=pl.col("_foul_home").cum_sum().over(["game_id", "_half"]),
            away_fouls=pl.col("_foul_away").cum_sum().over(["game_id", "_half"]),
        )
        .with_columns(
            margin=(pl.col("home_score").fill_null(0) - pl.col("away_score").fill_null(0)).cast(pl.Int32),
            home_timeouts=(pl.col("_allot") - pl.col("home_used")).clip(lower_bound=0).cast(pl.Int32),
            away_timeouts=(pl.col("_allot") - pl.col("away_used")).clip(lower_bound=0).cast(pl.Int32),
            is_ot=(pl.col("_period") >= 3),
        )
        .rename({"_period": "period", "_clock": "clock_seconds",
                 "_gsr": "game_seconds_remaining", "game_play_number": "seq"})
        .drop(["_raw_clock", "_poss_set", "_to_home", "_to_away", "_allot",
               "home_used", "away_used", "_half", "_foul_home", "_foul_away"])
    )

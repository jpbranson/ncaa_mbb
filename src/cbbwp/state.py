"""Replayable game-state builder.

`build_states` is a PURE function of the full event list. Feed it the same
events in any arrival order and it produces the same states, so a retroactive
correction from a live feed is handled by simply replaying the game from
event zero (milliseconds).
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from .schemas import (
    Event,
    GameState,
    PregameContext,
    HALF_SECONDS,
    OT_SECONDS,
    TIMEOUTS_AT_TIP,
)

# Fouls that count toward the team-foul total for bonus purposes.
FOUL_TYPES = {"PersonalFoul", "Technical Foul"}

# --- possession rules -------------------------------------------------------
# What the state's `possession` should be AFTER each kind of event.
#   "actor"  -> the team the event is attributed to has the ball
#   "other"  -> the other team has the ball
#   "carry"  -> unchanged from the previous state
#   "unknown"-> 0.5
# NOTE: `_MADE_SHOT_TYPES` is NO LONGER used to decide possession - see
# `_possession_after`, which keys made field goals on the feed's scoring and
# shooting flags. It is kept only to document what the field-goal types look
# like. `_TURNOVER_MARKER` below IS still load-bearing (see `_possession_after`).
_MADE_SHOT_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot"}
_TURNOVER_MARKER = "Turnover"

TEAM_TIMEOUT_TYPES = {"ShortTimeOut", "RegularTimeOut", "TeamTimeOut", "Timeout"}
OFFICIAL_TIMEOUT_TYPES = {"OfficialTVTimeOut", "MediaTimeOut"}


def parse_clock(display: str) -> Optional[int]:
    """'19:48' -> 1188.  '0:23.4' -> 23.  None if it does not parse at all.

    `clock_to_seconds` folds two different facts into 0: "this play carries no
    clock" (normal - ESPN sends administrative rows that way) and "this clock
    string is a shape this code does not understand" (a feed change). The second
    is worth knowing about, because a fabricated 0 in the SECOND HALF reads as
    `game_seconds_remaining == 0`, and the endgame clamp then publishes 0.999
    with ten minutes left on the real clock.

    Nothing downstream of the adapters changes behaviour on this - see
    `adapters/espn.clock_parse_failures`, which counts it and reports it the way
    unknown play types are reported. Carrying the previous clock forward instead
    would change what a GameState MEANS and so requires a STATE_RULES_VERSION
    bump and a refit; that is a deliberate decision, not a bug fix.
    """
    if not display:
        return None
    s = display.strip()
    if not s:
        return None
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + int(float(ss))
        return int(float(s))
    except (ValueError, TypeError):
        return None


def clock_to_seconds(display: str) -> int:
    """'19:48' -> 1188.  '0:23.4' -> 23.  Returns 0 on anything unparseable."""
    v = parse_clock(display)
    return 0 if v is None else v


def period_length(period: int) -> int:
    return HALF_SECONDS if period <= 2 else OT_SECONDS


def game_seconds_remaining(period: int, clock_seconds: int) -> int:
    """Seconds left in regulation; inside overtime, seconds left in that OT.

    Each overtime is treated as its own clock reset (see plan section 8.2).
    """
    if period <= 1:
        return HALF_SECONDS + clock_seconds
    if period == 2:
        return clock_seconds
    return clock_seconds


def _possession_after(ev: Event, home_id: int, away_id: int, prev: float) -> float:
    """Return 1.0 (home has ball), 0.0 (away), or 0.5 (unknown)."""
    t = ev.event_type or ""
    tid = ev.team_id
    actor: Optional[float]
    if tid is None:
        actor = None
    elif tid == home_id:
        actor = 1.0
    elif tid == away_id:
        actor = 0.0
    else:
        actor = None
    other = None if actor is None else 1.0 - actor

    if "FreeThrow" in t:
        return other if (ev.scoring_play and other is not None) else prev
    # A made field goal -> the other team inbounds. A miss leaves the ball live,
    # so possession carries until a rebound resolves it.
    #
    # This is keyed on the feed's own scoring/shooting flags rather than on a
    # list of play-type NAMES, and that is not a style preference. ESPN typed
    # made three-pointers as "Three Point Jump Shot" through 2019 and as
    # "JumpShot" from 2021 onward. A name whitelist therefore missed 324,043
    # made threes - 89% of every made three in 2016-2019 - and left the ball
    # with the team that had just scored. The flags are stable across that
    # rename; the names are not.
    if ev.scoring_play and ev.shooting_play:
        return other if other is not None else prev
    if t == "Defensive Rebound" or t == "Offensive Rebound":
        return actor if actor is not None else prev
    if t == "Dead Ball Rebound":
        return prev
    if _TURNOVER_MARKER in t:
        return other if other is not None else prev
    if t == "Steal":
        return actor if actor is not None else prev
    if t == "Jumpball":
        return 0.5
    return prev  # fouls, blocks, subs, timeouts, period markers


def build_states(
    events: Iterable[Event],
    ctx: PregameContext,
    timeouts_at_tip: int = TIMEOUTS_AT_TIP,
) -> List[GameState]:
    """Replay a game's events into one state per event.

    Pure: no dependence on arrival order, no hidden state, no I/O.
    """
    evs = sorted(events, key=lambda e: e.seq)
    home_id, away_id = ctx.home_team_id, ctx.away_team_id

    states: List[GameState] = []
    # An unreadable clock carries the period's previous clock forward rather
    # than becoming 0:00 (STATE_RULES_VERSION 3). A period whose first play has
    # no readable clock starts at full length, because the period has just
    # started -- carrying 0:00 across the half-time break would be worse than
    # the bug this replaces.
    prev_clock: Optional[int] = None
    prev_period: Optional[int] = None
    poss = 0.5
    home_used = away_used = 0
    home_fouls = away_fouls = 0
    half_of = lambda p: 1 if p <= 1 else 2   # men's fouls reset once, at the break
    cur_half = 1

    for ev in evs:
        period = max(1, ev.period or 1)
        if half_of(period) != cur_half:
            cur_half = half_of(period)
            home_fouls = away_fouls = 0

        if ev.event_type in FOUL_TYPES:
            if ev.team_id == home_id:
                home_fouls += 1
            elif ev.team_id == away_id:
                away_fouls += 1

        if ev.event_type in TEAM_TIMEOUT_TYPES:
            if ev.team_id == home_id:
                home_used += 1
            elif ev.team_id == away_id:
                away_used += 1

        # NCAA grants one extra timeout per overtime period.
        allot = timeouts_at_tip + max(0, period - 2)

        poss = _possession_after(ev, home_id, away_id, poss)
        plen = period_length(period)
        raw = ev.clock_seconds
        if raw is None:
            raw = prev_clock if (prev_period == period and
                                 prev_clock is not None) else plen
        clock = max(0, min(int(raw), plen))
        prev_clock, prev_period = clock, period

        states.append(
            GameState(
                game_id=ev.game_id,
                seq=ev.seq,
                period=period,
                is_ot=period >= 3,
                clock_seconds=clock,
                game_seconds_remaining=game_seconds_remaining(period, clock),
                home_score=ev.home_score,
                away_score=ev.away_score,
                margin=ev.home_score - ev.away_score,
                possession=poss,
                home_timeouts=max(0, allot - home_used),
                away_timeouts=max(0, allot - away_used),
                home_fouls=home_fouls,
                away_fouls=away_fouls,
                pregame_exp_margin=ctx.pregame_exp_margin,
                neutral_site=ctx.neutral_site,
                ft_pct_diff=ctx.ft_pct_diff,
                exp_points_per_min=ctx.exp_points_per_min,
            )
        )
    return states

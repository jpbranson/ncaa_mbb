import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from cbbwp.schemas import Event, PregameContext
from cbbwp.state import build_states, clock_to_seconds, game_seconds_remaining

HOME, AWAY = 10, 20


def ev(seq, period, clock, hs, a_s, typ, team=None, scoring=False, sv=0,
       shooting=False):
    # `shooting` matters: possession after a field goal is decided by the feed's
    # scoring/shooting flags, not by the play-type name. Every real made field
    # goal in the hoopR data carries shooting_play=True, so a fixture that omits
    # it is not modelling the feed.
    return Event(1, seq, period, clock, hs, a_s, typ, team, sv, scoring, shooting)


def test_clock_parsing():
    assert clock_to_seconds("19:48") == 1188
    assert clock_to_seconds("0:23.4") == 23
    assert clock_to_seconds("") == 0
    assert clock_to_seconds(None or "") == 0


def test_an_unreadable_clock_is_none_not_zero():
    """The distinction STATE_RULES_VERSION 3 rests on.

    `clock_to_seconds` folded "no clock" and "a clock we cannot read" into 0,
    and a 0 in the second half means "the game is over" to endgame.apply.
    """
    from cbbwp.state import parse_clock
    assert parse_clock("19:48") == 1188
    assert parse_clock("0:23.4") == 23
    assert parse_clock("42.7") == 42
    assert parse_clock("") is None
    assert parse_clock("  ") is None
    assert parse_clock("nonsense") is None
    # A real 0:00 is a number, not an absence.
    assert parse_clock("0:00") == 0


def test_an_unreadable_clock_carries_the_period_forward():
    """Rules v3. The dangerous case is the second half, where a fabricated
    0:00 would let the endgame clamp publish near-certainty mid-game."""
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 2, 900, 40, 38, "Jumpball"),
        Event(1, 2, 2, None, 40, 38, "Substitution", HOME, 0, False, False),
        ev(3, 2, 880, 42, 38, "JumpShot", HOME, True, 2, shooting=True),
    ], ctx)
    assert [x.clock_seconds for x in s] == [900, 900, 880]
    # ...and so the clamp never sees a finished game.
    assert [x.game_seconds_remaining for x in s] == [900, 900, 880]


def test_a_period_that_opens_unreadable_starts_at_full_length():
    """Carrying 0:00 across the half-time break would be worse than the bug."""
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 3, 40, 38, "JumpShot", HOME, True, 2, shooting=True),
        Event(1, 2, 2, None, 40, 38, "Jumpball", None, 0, False, False),
    ], ctx)
    assert s[0].clock_seconds == 3
    assert s[1].clock_seconds == 1200          # a new half, not 0:03
    assert s[1].game_seconds_remaining == 1200


def test_game_clock_is_regulation_wide_and_ot_resets():
    assert game_seconds_remaining(1, 1200) == 2400   # tip
    assert game_seconds_remaining(1, 0) == 1200      # halftime
    assert game_seconds_remaining(2, 600) == 600
    assert game_seconds_remaining(3, 300) == 300     # OT resets its own clock


def test_replay_is_order_independent():
    evs = [
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 0, 2, "JumpShot", AWAY, True, 2, shooting=True),
        ev(3, 1, 1160, 3, 2, "JumpShot", HOME, True, 3, shooting=True),
    ]
    ctx = PregameContext(1, HOME, AWAY)
    a = build_states(evs, ctx)
    b = build_states(list(reversed(evs)), ctx)
    assert [s.margin for s in a] == [s.margin for s in b] == [0, -2, 1]


def test_possession_rules():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 2, 0, "JumpShot", HOME, True, 2, shooting=True),   # made -> away ball
        ev(3, 1, 1160, 2, 0, "JumpShot", AWAY, False, 2, shooting=True),  # miss -> carry
        ev(4, 1, 1158, 2, 0, "Defensive Rebound", HOME),      # home ball
        ev(5, 1, 1150, 2, 0, "Lost Ball Turnover", HOME),     # -> away ball
        ev(6, 1, 1148, 2, 0, "Steal", AWAY),                  # away ball
    ], ctx)
    assert [x.possession for x in s] == [0.5, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_timeouts_decrement_and_reset_in_overtime():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1000, 0, 0, "ShortTimeOut", HOME),
        ev(3, 2, 100, 50, 50, "RegularTimeOut", HOME),
        ev(4, 3, 300, 50, 50, "Jumpball"),
    ], ctx)
    assert [x.home_timeouts for x in s] == [4, 3, 2, 3]   # +1 allotment in OT
    assert [x.away_timeouts for x in s] == [4, 4, 4, 5]
    assert s[-1].is_ot is True


def test_official_timeouts_do_not_consume_team_timeouts():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([ev(1, 1, 1200, 0, 0, "OfficialTVTimeOut", HOME)], ctx)
    assert s[0].home_timeouts == 4


def test_the_vectorised_path_carries_an_unreadable_clock_the_same_way():
    """Both implementations of rules v3, on the case real data does not contain.

    `test_parity.py` compares the two paths on real games -- where, measured
    across all ten seasons, zero rows carry an unreadable clock. So it cannot
    see this rule at all, and without this test the vectorised half of v3 would
    be shipped untested.
    """
    from cbbwp.adapters.hoopr import states_lazy

    rows = [
        # (play, period, clock string) -- "" and junk are the unreadable cases
        (1, 2, "15:00"), (2, 2, ""), (3, 2, "nonsense"), (4, 2, "14:30"),
        (5, 3, ""),                      # an overtime that opens unreadable
        (6, 3, "4:00"),
    ]
    lf = pl.DataFrame({
        "game_id": [1] * len(rows),
        "game_play_number": [r[0] for r in rows],
        "period_number": [r[1] for r in rows],
        "clock_display_value": [r[2] for r in rows],
        "home_score": [0] * len(rows), "away_score": [0] * len(rows),
        "type_text": ["Substitution"] * len(rows),
        "team_id": [HOME] * len(rows),
        "score_value": [0] * len(rows),
        "scoring_play": [False] * len(rows),
        "shooting_play": [False] * len(rows),
        "home_team_id": [HOME] * len(rows), "away_team_id": [AWAY] * len(rows),
    }).lazy()
    vec = states_lazy(lf).collect().sort("seq")

    ref = build_states(
        [Event(1, p, per, None if not c or c == "nonsense"
               else (int(c.split(":")[0]) * 60 + int(c.split(":")[1])),
               0, 0, "Substitution", HOME, 0, False, False)
         for p, per, c in rows],
        PregameContext(1, HOME, AWAY))

    assert [s.clock_seconds for s in ref] == [900, 900, 900, 870, 300, 240]
    assert list(vec["clock_seconds"]) == [s.clock_seconds for s in ref]
    assert list(vec["game_seconds_remaining"]) == [
        s.game_seconds_remaining for s in ref]

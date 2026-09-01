import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.schemas import Event, PregameContext
from cbbwp.state import build_states, clock_to_seconds, game_seconds_remaining

HOME, AWAY = 10, 20


def ev(seq, period, clock, hs, a_s, typ, team=None, scoring=False, sv=0):
    return Event(1, seq, period, clock, hs, a_s, typ, team, sv, scoring, False)


def test_clock_parsing():
    assert clock_to_seconds("19:48") == 1188
    assert clock_to_seconds("0:23.4") == 23
    assert clock_to_seconds("") == 0
    assert clock_to_seconds(None or "") == 0


def test_game_clock_is_regulation_wide_and_ot_resets():
    assert game_seconds_remaining(1, 1200) == 2400   # tip
    assert game_seconds_remaining(1, 0) == 1200      # halftime
    assert game_seconds_remaining(2, 600) == 600
    assert game_seconds_remaining(3, 300) == 300     # OT resets its own clock


def test_replay_is_order_independent():
    evs = [
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 0, 2, "JumpShot", AWAY, True, 2),
        ev(3, 1, 1160, 3, 2, "JumpShot", HOME, True, 3),
    ]
    ctx = PregameContext(1, HOME, AWAY)
    a = build_states(evs, ctx)
    b = build_states(list(reversed(evs)), ctx)
    assert [s.margin for s in a] == [s.margin for s in b] == [0, -2, 1]


def test_possession_rules():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 2, 0, "JumpShot", HOME, True, 2),      # made -> away ball
        ev(3, 1, 1160, 2, 0, "JumpShot", AWAY, False, 2),     # miss -> carry
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

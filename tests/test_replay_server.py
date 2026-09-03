"""The replay server has to be trustworthy before a dry run means anything.

If this thing reveals plays out of order, or lets the plays array shrink between
polls, then a dry run is testing the simulator's bugs rather than the model's
behaviour - and it would look like a live-path failure, which is the worst kind
of false alarm to chase at tip-off.

These run against whatever is in `tmp/replay/`, which is gitignored, so they
skip when the archive has not been built. That is deliberate: the archive is a
build product (`scripts/archive_replay_games.py`), not source.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import STATUS_FINAL, STATUS_PRE, parse_summary  # noqa: E402

REPLAY_DIR = ROOT / "tmp/replay"
ARCHIVES = sorted(REPLAY_DIR.glob("summary_*.json")) if REPLAY_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not ARCHIVES, reason="no replay archive; run scripts/archive_replay_games.py")


@pytest.fixture(scope="module")
def games():
    from replay_server import ReplayGame
    return [ReplayGame(p) for p in ARCHIVES]


def test_elapsed_seconds_is_monotonic_in_the_archive(games):
    """ESPN's countdown clock must convert to a monotonic ordinate.

    Period plus a counting-DOWN clock cannot be compared directly; getting this
    wrong would reveal the second half before the first.
    """
    from replay_server import elapsed_seconds
    for g in games:
        marks = [elapsed_seconds(p) for p in g.plays]
        assert marks == sorted(marks), f"{g.game_id} not monotonic"
        assert marks[0] <= 60.0, f"{g.game_id} does not start near tip-off"


def test_plays_only_ever_grow(games):
    """The poller tolerates corrections, but a shrinking feed is not realistic."""
    for g in games:
        counts = [g.n_revealed(t) for t in range(0, int(g.duration) + 120, 30)]
        assert counts == sorted(counts), f"{g.game_id} revealed count went backwards"
        # At t=0 the opening tip has happened, so one play is correct. What must
        # not happen is anything being visible BEFORE tip-off.
        assert g.n_revealed(-1.0) == 0, f"{g.game_id} leaked plays before tip"
        assert counts[0] <= 1, f"{g.game_id} revealed {counts[0]} plays at tip"
        assert counts[-1] == len(g.plays), f"{g.game_id} never revealed everything"


def test_status_goes_scheduled_then_in_progress_then_final(games):
    from replay_server import STATUS_IN
    for g in games:
        assert g.state(-1.0) == STATUS_PRE
        assert g.state(g.duration / 2) == STATUS_IN
        assert g.state(g.duration + 1) == STATUS_FINAL


def test_the_finished_replay_equals_the_archive(games):
    """The end of a replay must be the real game, or the dry run proves nothing."""
    for g in games:
        final = g.summary(g.duration + 1)
        assert len(final["plays"]) == len(g.plays)
        events, header = parse_summary(final)
        assert header.status == STATUS_FINAL
        assert len(events) == len(g.plays)


def test_a_mid_game_payload_still_parses_and_is_live(games):
    """Half a game is the case the live path had never actually been given."""
    from cbbwp.adapters.espn import header_from_summary
    for g in games:
        mid = g.summary(g.duration / 2)
        assert 0 < len(mid["plays"]) < len(g.plays)
        h = header_from_summary(mid)
        assert h.is_live and not h.is_final
        events, _ = parse_summary(mid)
        # Dense 1..N numbering must hold on a partial feed too - that is what
        # makes a live state comparable to the same state built offline.
        assert [e.seq for e in events] == list(range(1, len(events) + 1))


def test_the_clock_counts_down_within_a_period(games):
    """A clock that runs backwards would drive the endgame model insane."""
    g = games[0]
    p1, c1 = g.period_and_clock(60)
    p2, c2 = g.period_and_clock(360)
    assert p1 == p2 == 1
    to_s = lambda d: (int(d.split(":")[0]) * 60 + int(d.split(":")[1])
                      if ":" in d else float(d))
    assert to_s(c1) > to_s(c2)


def test_scores_track_the_revealed_plays(games):
    """The header score must agree with the plays shown, not the final score."""
    for g in games:
        mid = g.summary(g.duration / 2)
        comp = (mid["header"]["competitions"])[0]
        shown = {c["homeAway"]: int(c["score"]) for c in comp["competitors"]}
        last = mid["plays"][-1]
        assert shown["home"] == int(last["homeScore"])
        assert shown["away"] == int(last["awayScore"])
        final = g.summary(g.duration + 1)
        fcomp = (final["header"]["competitions"])[0]
        fshown = {c["homeAway"]: int(c["score"]) for c in fcomp["competitors"]}
        assert (fshown["home"], fshown["away"]) != (0, 0)

"""The ESPN (live) adapter must produce exactly what the hoopR (offline) adapter
produces for the same game. This is the train/serve-skew guard for the live path,
and it is the live counterpart of tests/test_parity.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl
import pytest

from cbbwp.adapters import espn
from cbbwp.adapters.hoopr import load_events
from cbbwp.schemas import PregameContext
from cbbwp.state import build_states
from cbbwp.live_context import LiveContextProvider

ROOT = pathlib.Path(__file__).resolve().parents[1]
PBP = str(ROOT / "data/raw/pbp/pbp_2025.parquet")
HAS_DATA = pathlib.Path(PBP).exists()


# --------------------------------------------------------------------------
# Unit tests: no data needed
# --------------------------------------------------------------------------
def test_type_id_beats_feed_text():
    """If ESPN reworded a play type, the trained-on text must still win."""
    p = {"type": {"id": "558", "text": "Jump Shot (new ESPN wording)"}}
    assert espn.play_type_text(p) == "JumpShot"


def test_unknown_type_id_falls_back_to_feed_text():
    p = {"type": {"id": "999999", "text": "Flagrant Foul"}}
    assert espn.play_type_text(p) == "Flagrant Foul"
    # and an unknown type carries possession rather than guessing
    assert espn.play_type_text({"type": {}}) == ""


def test_plays_are_ordered_by_sequence_and_renumbered_densely():
    plays = [
        {"sequenceNumber": "300", "type": {"id": "615"}, "period": {"number": 1},
         "clock": {"displayValue": "18:00"}, "homeScore": 2, "awayScore": 0},
        {"sequenceNumber": "100", "type": {"id": "615"}, "period": {"number": 1},
         "clock": {"displayValue": "20:00"}, "homeScore": 0, "awayScore": 0},
        {"sequenceNumber": "200", "type": {"id": "558"}, "period": {"number": 1},
         "clock": {"displayValue": "19:00"}, "homeScore": 2, "awayScore": 0},
    ]
    evs = espn.events_from_plays(plays, game_id=7)
    assert [e.seq for e in evs] == [1, 2, 3]
    assert [e.clock_seconds for e in evs] == [1200, 1140, 1080]
    assert all(e.game_id == 7 for e in evs)


def test_missing_and_null_fields_do_not_raise():
    evs = espn.events_from_plays([{"type": {"id": "584"}}], game_id=1)
    assert len(evs) == 1
    e = evs[0]
    assert (e.period, e.clock_seconds, e.home_score, e.team_id) == (1, 0, 0, None)


def test_header_parsing():
    h = espn.header_from_summary({"header": {"id": "401", "competitions": [{
        "neutralSite": True,
        "status": {"period": 2, "displayClock": "1:23",
                   "type": {"name": "STATUS_IN_PROGRESS"}},
        "competitors": [
            {"homeAway": "away", "score": "61", "team": {"id": "2", "displayName": "A"}},
            {"homeAway": "home", "score": "64", "team": {"id": "1", "displayName": "H"}},
        ]}]}})
    assert (h.game_id, h.home_team_id, h.away_team_id) == (401, 1, 2)
    assert h.neutral_site and h.is_live and not h.is_final
    assert (h.home_score, h.away_score) == (64, 61)


def test_scoreboard_flattening():
    games = espn.scoreboard_games({"events": [{
        "id": "555", "shortName": "A @ H",
        "competitions": [{"neutralSite": False,
                          "status": {"type": {"name": "STATUS_IN_PROGRESS",
                                              "state": "in", "completed": False}},
                          "competitors": [
                              {"homeAway": "home", "team": {"id": "1"}},
                              {"homeAway": "away", "team": {"id": "2"}}]}]}]})
    assert games == [{"game_id": 555, "name": "A @ H",
                      "status": "STATUS_IN_PROGRESS", "state": "in",
                      "completed": False, "neutral_site": False,
                      "home_team_id": 1, "away_team_id": 2, "start": ""}]


def test_live_context_falls_back_for_unknown_teams():
    p = LiveContextProvider(season=2027, hca=3.5, ratings={1: 6.0},
                            ft_pct={1: 0.75}, ppm={1: 3.6})
    c = p.context_for(99, home_team_id=1, away_team_id=404)
    assert c.pregame_exp_margin == pytest.approx(6.0 - 0.0 + 3.5)
    assert c.ft_pct_diff == pytest.approx(0.75 - 0.700)
    assert not p.known(404)
    # a neutral-site game drops the home-court term
    assert p.context_for(99, 1, 404, neutral_site=True).pregame_exp_margin == \
        pytest.approx(6.0)


# --------------------------------------------------------------------------
# Parity against the offline adapter, on real games
# --------------------------------------------------------------------------
pytestmark_data = pytest.mark.skipif(not HAS_DATA,
                                     reason="pbp_2025.parquet not downloaded yet")


@pytest.fixture(scope="module")
def game_ids():
    return (pl.scan_parquet(PBP).select("game_id").unique().sort("game_id")
            .head(15).collect()["game_id"].to_list())


@pytestmark_data
def test_espn_adapter_matches_hoopr_states(game_ids):
    from espn_fixtures import summary_from_hoopr

    for gid in game_ids:
        ref_events, home_id, away_id = load_events(PBP, gid)
        # shuffled on purpose: the live feed does not promise ordered plays
        payload = summary_from_hoopr(PBP, gid, shuffle_seed=gid % 97)
        live_events, header = espn.parse_summary(payload)

        assert header.home_team_id == home_id
        assert header.away_team_id == away_id
        assert len(live_events) == len(ref_events)

        ctx = PregameContext(gid, home_id, away_id, pregame_exp_margin=2.5,
                             ft_pct_diff=0.02, exp_points_per_min=3.5)
        ref = build_states(ref_events, ctx)
        live = build_states(live_events, ctx)
        for a, b in zip(ref, live):
            assert (a.seq, a.period, a.game_seconds_remaining, a.margin,
                    a.possession, a.home_timeouts, a.away_timeouts,
                    a.home_fouls, a.away_fouls, a.is_ot) == \
                   (b.seq, b.period, b.game_seconds_remaining, b.margin,
                    b.possession, b.home_timeouts, b.away_timeouts,
                    b.home_fouls, b.away_fouls, b.is_ot), (gid, a.seq)


@pytestmark_data
@pytest.mark.skipif(not (ROOT / "registry/v2").exists(),
                    reason="no model registry built yet")
def test_espn_path_gives_identical_win_probabilities(game_ids):
    from espn_fixtures import summary_from_hoopr
    from cbbwp.serve import WinProbabilityService

    svc = WinProbabilityService(ROOT / "registry", "v2")
    for gid in game_ids[:5]:
        ref_events, home_id, away_id = load_events(PBP, gid)
        ctx = PregameContext(gid, home_id, away_id, pregame_exp_margin=2.5,
                             ft_pct_diff=0.02, exp_points_per_min=3.5)
        live_events, _ = espn.parse_summary(
            summary_from_hoopr(PBP, gid, shuffle_seed=1))
        assert svc.score_game(live_events, ctx) == svc.score_game(ref_events, ctx)


@pytestmark_data
def test_partial_feed_is_a_prefix_of_the_finished_game(game_ids):
    """A game polled mid-way must give the same answers for the plays it has."""
    from espn_fixtures import summary_from_hoopr

    gid = game_ids[0]
    full = summary_from_hoopr(PBP, gid)
    ref_events, home_id, away_id = load_events(PBP, gid)
    ctx = PregameContext(gid, home_id, away_id)

    n = len(full["plays"]) // 2
    partial = dict(full, plays=full["plays"][:n],
                   header=full["header"])
    part_events, _ = espn.parse_summary(partial)
    full_events, _ = espn.parse_summary(full)

    a = build_states(part_events, ctx)
    b = build_states(full_events, ctx)[:n]
    assert len(a) == n
    for x, y in zip(a, b):
        assert (x.seq, x.margin, x.possession, x.game_seconds_remaining) == \
               (y.seq, y.margin, y.possession, y.game_seconds_remaining)


def test_serving_refuses_a_model_fit_under_older_state_rules():
    """The guard that the feature-name check could not provide.

    A model fit before the possession fix must not be served states built after
    it. The feature NAMES are identical in both, so without this check the
    mismatch is completely silent - which is exactly what happened on
    2026-09-01 before it was caught.
    """
    from cbbwp.serve import WinProbabilityService
    if not (ROOT / "registry/v1").exists():
        pytest.skip("no v1 artifact kept")
    with pytest.raises(RuntimeError, match="state rules"):
        WinProbabilityService(ROOT / "registry", "v1")


def test_the_current_model_loads():
    from cbbwp.serve import WinProbabilityService
    if not (ROOT / "registry/v2").exists():
        pytest.skip("no v2 artifact built yet")
    svc = WinProbabilityService(ROOT / "registry", "v2")
    assert svc.manifest["state_rules_version"] == 2


def test_default_user_agent_carries_a_contact_url():
    """ESPN's edge 403s bare short user-agents; the contact URL is what passes.

    Measured 2026-09-02: "cbbwp/0.2" got 403 on 15/15 requests, and
    "cbbwp/0.2 (+https://github.com/jpbranson/ncaa_mbb)" got 200 on 15/15, on
    both the scoreboard and summary endpoints. The property that matters is the
    contact URL, not the exact string - so that is what is asserted here, to
    stop a well-meaning tidy-up ("shorten this ugly literal") from silently
    breaking the live path months before anyone runs it again.
    """
    from cbbwp.adapters.espn import DEFAULT_USER_AGENT
    assert "(+http" in DEFAULT_USER_AGENT
    assert len(DEFAULT_USER_AGENT) > 20


def test_user_agent_is_overridable_by_environment(monkeypatch):
    """A WAF change must be a config edit plus a restart, never a code edit."""
    from cbbwp.adapters.espn import DEFAULT_USER_AGENT, EspnClient
    monkeypatch.setenv("CBBWP_USER_AGENT", "someone-else/9.9 (+https://example.org)")
    assert EspnClient().user_agent == "someone-else/9.9 (+https://example.org)"
    assert EspnClient(user_agent="explicit/1").user_agent == "explicit/1"
    monkeypatch.delenv("CBBWP_USER_AGENT")
    assert EspnClient().user_agent == DEFAULT_USER_AGENT


def test_the_client_points_at_espn_unless_told_otherwise(monkeypatch):
    """The replay hook must never become the default.

    `CBBWP_ESPN_BASE` exists so scripts/replay_server.py can stand in for ESPN
    during a dry run. A default pointing anywhere else would mean a live night
    silently scoring simulated games.
    """
    from cbbwp.adapters.espn import SITE_API, EspnClient
    monkeypatch.delenv("CBBWP_ESPN_BASE", raising=False)
    c = EspnClient()
    assert c.base_url == SITE_API
    assert c.is_replay is False

    monkeypatch.setenv("CBBWP_ESPN_BASE", "http://127.0.0.1:8899/")
    r = EspnClient()
    assert r.base_url == "http://127.0.0.1:8899"      # trailing slash trimmed
    assert r.is_replay is True

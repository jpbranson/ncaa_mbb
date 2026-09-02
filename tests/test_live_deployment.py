"""Tests for the deployment surface: config, the API, and ratings freshness.

The freshness tests are the point of this file. A win probability model that
serves confidently from stale ratings looks exactly like one serving from fresh
ones -- there is no visible symptom until someone checks a number by hand. The
project has already been bitten once by a defect with no symptom (the possession
bug), so the staleness signal gets tests rather than trust.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.api import LiveStore, serve_in_thread            # noqa: E402
from cbbwp.config import Settings                            # noqa: E402
from cbbwp.live_context import (DATA_STALE_AFTER_DAYS,        # noqa: E402
                                 LiveContextProvider)
from cbbwp.schemas import STATE_RULES_VERSION                # noqa: E402


# --- config -----------------------------------------------------------------
def test_settings_have_working_defaults_with_no_environment(monkeypatch):
    for k in [k for k in list(__import__("os").environ) if k.startswith("CBBWP_")]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.model_version == "v2"
    assert s.api_port == 8808
    assert s.fixture_dir is None
    assert s.registry.name == "registry"


def test_environment_overrides_are_reported_not_silent(monkeypatch):
    monkeypatch.setenv("CBBWP_MODEL_VERSION", "v99")
    monkeypatch.setenv("CBBWP_API_PORT", "9999")
    s = Settings.from_env()
    assert s.model_version == "v99" and s.api_port == 9999
    # Whatever came from the environment must appear in the startup banner --
    # a service quietly reading the wrong directory is the outage this prevents.
    assert "CBBWP_MODEL_VERSION" in s.describe()


# --- ratings freshness ------------------------------------------------------
def _ctx(latest_game_date: str, generated: str | None = None) -> LiveContextProvider:
    now = generated or dt.datetime.now(dt.timezone.utc).isoformat()
    return LiveContextProvider(season=2027, hca=3.4, ratings={}, ft_pct={}, ppm={},
                               generated=now, latest_game_date=latest_game_date)


def test_a_freshly_written_file_over_stale_data_is_not_called_fresh():
    """The whole reason data_age_days exists.

    A nightly rebuild always writes a file with today's timestamp. If the
    play-by-play behind it has not been refreshed, the ratings are old and
    nothing about the file says so.
    """
    # Ages are measured against the real clock, so build the dates from it --
    # mixing a fictional "now" with the real one is how this test first failed.
    old_game = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat()
    c = _ctx(old_game)
    assert c.age_days < 1                      # the file looks perfectly fresh
    assert c.data_age_days > 25                # the data behind it is not
    assert c.data_age_days > DATA_STALE_AFTER_DAYS
    # ...and in season that is what makes the whole snapshot stale.
    assert LiveContextProvider._in_season(dt.datetime(2027, 1, 20, tzinfo=dt.timezone.utc))


def test_data_staleness_is_not_raised_out_of_season():
    """April to November the newest game is meant to be months old."""
    for when in (dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc),
                 dt.datetime(2026, 9, 2, tzinfo=dt.timezone.utc),
                 dt.datetime(2026, 4, 20, tzinfo=dt.timezone.utc)):
        assert not LiveContextProvider._in_season(when), when


def test_season_window_covers_the_months_games_are_played():
    for when, expected in [
        (dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 1, 15, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 3, 31, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 4, 10, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 4, 16, tzinfo=dt.timezone.utc), False),
        (dt.datetime(2027, 10, 31, tzinfo=dt.timezone.utc), False),
    ]:
        assert LiveContextProvider._in_season(when) is expected, when


def test_preseason_snapshot_with_no_games_reports_unknown_not_stale():
    c = _ctx("")
    assert c.data_age_days is None
    assert c.data_is_stale is False


# --- the API ----------------------------------------------------------------
@pytest.fixture()
def server():
    store = LiveStore(history=4)
    state = {"stale": False, "age": 0.0}
    httpd = serve_in_thread(
        store, {"model_version": "vtest", "ratings_max_age_days": 3},
        lambda: 0.5, "127.0.0.1", 0,
        lambda: state["age"], lambda: state["stale"])
    yield httpd, store, state
    httpd.shutdown()


def _get(httpd, path: str):
    port = httpd.server_address[1]
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_every_response_says_what_produced_it(server):
    httpd, store, _ = server
    store.update({"game_id": 7, "seq": 1, "period": 2,
                  "game_seconds_remaining": 20, "margin": 2, "home_win_prob": 0.7})
    for path in ("/", "/health", "/games", "/games/7"):
        _, body = _get(httpd, path)
        assert body["model_version"] == "vtest", path
        assert body["state_rules_version"] == STATE_RULES_VERSION, path


def test_health_goes_degraded_when_the_data_behind_the_ratings_is_stale(server):
    httpd, _, state = server
    code, body = _get(httpd, "/health")
    assert code == 200 and body["status"] == "ok"
    state["stale"] = True
    state["age"] = 31.0
    code, body = _get(httpd, "/health")
    # 503 so a container healthcheck or a load balancer notices without a human.
    assert code == 503 and body["status"] == "degraded"
    assert "stale data" in body["reason"] or "stale" in body["reason"]
    assert body["data_age_days"] == 31.0


def test_history_is_bounded_so_a_long_game_cannot_grow_without_limit(server):
    httpd, store, _ = server
    for i in range(50):
        store.update({"game_id": 9, "seq": i, "period": 2,
                      "game_seconds_remaining": 100 - i, "margin": 1,
                      "home_win_prob": 0.5})
    _, body = _get(httpd, "/games/9")
    assert len(body["game"]["history"]) == 4
    assert body["game"]["seq"] == 49


def test_unknown_and_malformed_game_ids_are_distinguished(server):
    httpd, _, _ = server
    assert _get(httpd, "/games/12345")[0] == 404
    assert _get(httpd, "/games/not-a-number")[0] == 400
    assert _get(httpd, "/nope")[0] == 404

"""Replay harness (plan 13, phase 5 item 15).

Feed a completed game's events through the LIVE path one poll at a time, as if
they were arriving in real time, and require that the answer for each state
matches what the offline path produces for that same state. This is the check
that catches train/serve skew, and it belongs in CI.
"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import pytest

from cbbwp.schemas import PregameContext
from cbbwp.adapters.hoopr import load_events
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]
PBP = str(ROOT / "data/raw/pbp/pbp_2025.parquet")
REGISTRY = ROOT / "registry"

pytestmark = pytest.mark.skipif(
    not (REGISTRY / "v1").exists() or not pathlib.Path(PBP).exists(),
    reason="no model registry or pbp data built yet")


@pytest.fixture(scope="module")
def svc():
    return WinProbabilityService(REGISTRY, "v1")


@pytest.fixture(scope="module")
def game():
    gid = int(pl.scan_parquet(PBP).select("game_id").unique().sort("game_id")
              .head(1).collect()["game_id"][0])
    return load_events(PBP, gid)


def test_incremental_polling_matches_full_replay(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    offline = svc.score_game(events, ctx)

    # Simulate a poller receiving the feed in irregular chunks.
    seen, live = [], {}
    i = 0
    rng = random.Random(0)
    while i < len(events):
        i += rng.randint(1, 12)
        seen = events[:i]
        for row in svc.score_game(seen, ctx):
            live[row["seq"]] = row["home_win_prob"]

    assert len(live) == len(offline)
    for row in offline:
        assert live[row["seq"]] == pytest.approx(row["home_win_prob"], abs=1e-12), row["seq"]


def test_out_of_order_and_duplicate_events_are_absorbed(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    clean = svc.score_game(events, ctx)

    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    assert svc.score_game(shuffled, ctx) == clean


def test_probabilities_are_bounded_and_finite(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id)
    p = np.array([r["home_win_prob"] for r in svc.score_game(events, ctx)])
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0

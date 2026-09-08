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
    not (REGISTRY / "v2").exists() or not pathlib.Path(PBP).exists(),
    reason="no model registry or pbp data built yet")


@pytest.fixture(scope="module")
def svc():
    return WinProbabilityService(REGISTRY, "v2")


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


def test_arrival_order_does_not_change_the_answer(svc, game):
    """`build_states` is a pure function of the event SET, ordered by seq.

    Renamed 2026-09-08: this used to be called "...and duplicate events are
    absorbed" while testing nothing of the sort. Duplicates now have their own
    test below, which pins what actually happens rather than what the old name
    asserted.
    """
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    clean = svc.score_game(events, ctx)

    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    assert svc.score_game(shuffled, ctx) == clean


def test_a_duplicated_event_repeats_its_state_and_disturbs_nothing_else(svc, game):
    """What a repeated play actually does, stated rather than assumed.

    The live adapter renumbers densely over the feed's array, so it cannot emit
    a duplicate today. If a future feed or adapter ever did, the failure mode
    worth knowing is that the state is REPEATED, not that the game is corrupted:
    every other seq keeps exactly the probability it had. The poller reads
    rows[-1] and the API keys history by seq, so a repeat is inert downstream.
    """
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    clean = svc.score_game(events, ctx)

    mid = len(events) // 2
    doubled = list(events[:mid]) + [events[mid]] + list(events[mid:])
    got = svc.score_game(doubled, ctx)

    assert len(got) == len(clean) + 1
    assert [r["seq"] for r in got].count(events[mid].seq) == 2
    by_seq = {}
    for r in got:
        by_seq.setdefault(r["seq"], r["home_win_prob"])
    for r in clean:
        assert by_seq[r["seq"]] == pytest.approx(r["home_win_prob"], abs=1e-12)


def test_probabilities_are_bounded_and_finite(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id)
    p = np.array([r["home_win_prob"] for r in svc.score_game(events, ctx)])
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0

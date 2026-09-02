"""Tests for the endgame table.

Two kinds. The first check that the solver's structure is what it claims -- the
terminal condition, the symmetry, the monotonicity the plan pre-registered.

The second kind is the one that matters. `test_possession_truth.py` exists
because every check we had compared the code to itself, so a feature that meant
two different things in one training set went unnoticed for four seasons. The
free-throw tests below are written in the same spirit: they assert against the
RULES OF BASKETBALL and against a known feed artifact, so that if ESPN changes
how it labels a free-throw trip again, something fails loudly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from cbbwp import endgame_sim as E

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "registry" / "endgame" / "e1"
PBP26 = ROOT / "data" / "raw" / "pbp" / "pbp_2026.parquet"

needs_table = pytest.mark.skipif(not (TABLE / "table.npz").exists(),
                                 reason="run scripts/build_endgame_table.py first")
needs_pbp = pytest.mark.skipif(not PBP26.exists(), reason="raw play-by-play not fetched")


@pytest.fixture(scope="module")
def table():
    return np.load(TABLE / "table.npz")["table"].astype(np.float64)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((TABLE / "manifest.json").read_text())


@needs_table
def test_time_expired_is_decided_by_the_scoreboard(table):
    for m in range(-6, 7):
        v = table[0, E._mi(m), 6, 8, 1, 1]
        if m > 0:
            assert v == pytest.approx(1.0)
        elif m < 0:
            assert v == pytest.approx(0.0)
        else:
            assert v == pytest.approx(0.5)


@needs_table
def test_a_made_basket_never_lowers_your_win_probability(table):
    """Criterion 3, exhaustively -- every state, not a sample."""
    assert np.diff(table, axis=1).min() >= -1e-6


@needs_table
def test_having_the_ball_is_never_a_disadvantage(table):
    """The same physical state, valued from both sides, must prefer possession."""
    flipped = 1.0 - table[:, ::-1].transpose(0, 1, 3, 2, 5, 4)
    assert (table - flipped).min() >= -1e-6


@needs_table
def test_the_table_is_symmetric_under_swapping_the_teams(table):
    means = [0.667, 0.709, 0.749]
    args = dict(seconds_remaining=np.array([12, 30, 5]), home_fouls=np.array([7, 9, 6]),
                away_fouls=np.array([9, 6, 10]), home_bucket=np.array([0, 1, 2]),
                away_bucket=np.array([2, 1, 0]))
    home = E.lookup_home(table, margin_home=np.array([3.0, -2.0, 1.0]),
                         possession=np.array([1.0, 0.0, 1.0]), **args)
    swapped = E.lookup_home(
        table, seconds_remaining=args["seconds_remaining"], margin_home=np.array([-3.0, 2.0, -1.0]),
        possession=np.array([0.0, 1.0, 0.0]), home_fouls=args["away_fouls"],
        away_fouls=args["home_fouls"], home_bucket=args["away_bucket"],
        away_bucket=args["home_bucket"])
    assert home == pytest.approx(1.0 - swapped, abs=1e-9)
    del means


@needs_table
def test_trailing_by_three_late_is_worse_than_trailing_by_two(table):
    """Two is one possession; three is not. The table has to know that."""
    for t in (5, 10, 20):
        down2 = table[t, E._mi(-2), 6, 8, 1, 1]
        down3 = table[t, E._mi(-3), 6, 8, 1, 1]
        assert down3 < down2 - 0.05, (t, down3, down2)


@needs_table
def test_shipped_table_declares_the_state_rules_it_was_built_under(manifest):
    """Same guard as the model: a table built under different state rules must
    not be served silently alongside code that means something else by them."""
    from cbbwp.schemas import STATE_RULES_VERSION
    assert manifest["state_rules_version"] == STATE_RULES_VERSION


@needs_table
def test_the_table_was_not_fitted_on_the_test_seasons(manifest):
    assert not ({2025, 2026} & set(manifest["seasons_used"]))


# --- the feed-artifact regression -------------------------------------------
@needs_pbp
def test_espn_labels_a_free_throw_trip_by_attempts_taken_not_attempts_awarded():
    """The censoring that made "1 of 1" look like a 54% free-throw rate.

    A MADE one-and-one front end earns a second shot, so ESPN writes the trip as
    "1 of 2" / "2 of 2"; a MISSED one leaves the trip at a single attempt and is
    written "1 of 1". Every made front end therefore leaves the "1 of 1" bucket
    by construction, and reading `scoring_play` off that label conditions on the
    outcome.

    If this test ever fails, ESPN has changed the convention again and every
    free-throw parameter has to be re-derived.
    """
    import polars as pl

    df = (
        pl.scan_parquet(PBP26)
        .select(["game_id", "sequence_number", "type_id", "text", "scoring_play",
                 "athlete_id_1", "clock_minutes", "clock_seconds"])
        .collect()
        .with_columns(pl.col("sequence_number").cast(pl.Int64))
        .sort(["game_id", "sequence_number"])
    )
    df = df.with_columns([
        (pl.col("type_id").cast(pl.Int64) == 540).alias("isft"),
        (pl.col("clock_minutes").cast(pl.Float64).fill_null(0) * 60
         + pl.col("clock_seconds").cast(pl.Float64).fill_null(0)).alias("sec"),
    ])
    df = df.with_columns((pl.col("scoring_play") & ~pl.col("isft")).alias("madefg"))
    andone = None
    for k in range(1, 13):
        t = (pl.col("madefg").shift(k).over("game_id")
             & (pl.col("athlete_id_1").shift(k).over("game_id") == pl.col("athlete_id_1"))
             & ((pl.col("sec").shift(k).over("game_id") - pl.col("sec")).abs() <= 3))
        andone = t if andone is None else (andone | t)
    df = df.with_columns(andone.fill_null(False).alias("andone"))

    ones = df.filter(pl.col("isft") & pl.col("text").str.contains("(?i)free throw 1 of 1"))
    plain = ones.filter(~pl.col("andone"))
    and_ones = ones.filter(pl.col("andone"))

    # And-one free throws are ordinary single shots and convert like them.
    assert len(and_ones) > 5_000
    assert 0.60 < and_ones["scoring_play"].mean() < 0.80

    # The rest are missed one-and-one front ends, and are therefore almost all
    # misses. Anything near a plausible free-throw percentage here would mean
    # the convention had changed.
    assert len(plain) > 2_000
    assert plain["scoring_play"].mean() < 0.20, (
        "'1 of 1' free throws that are not and-ones now convert at a plausible "
        "rate -- ESPN has changed how it labels free-throw trips, and every "
        "free-throw parameter in artifacts/endgame_params.json must be re-derived"
    )

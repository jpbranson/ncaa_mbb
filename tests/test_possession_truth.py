"""Possession rules checked against the RULES OF BASKETBALL, not against each other.

`test_parity.py` asserts the bulk path agrees with the reference path. That is
necessary and it is not sufficient: both were wrong in exactly the same way for
four seasons, so parity passed while 324,043 made three-pointers left the ball
with the team that had just scored.

The bug was possible because the made-shot rule keyed on play-type NAMES, and
ESPN renamed the made-three type between 2019 ("Three Point Jump Shot") and 2021
("JumpShot"). These tests assert the invariant instead: after a made field goal,
the other team has the ball. In every season.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import pytest

from cbbwp.adapters.hoopr import states_lazy, BASE_COLS

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
AVAILABLE = [y for y in SEASONS if (ROOT / f"data/raw/pbp/pbp_{y}.parquet").exists()]

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="no pbp data downloaded")


def _made_fg_possession(year: int, n_games: int = 400) -> pl.DataFrame:
    src = ROOT / f"data/raw/pbp/pbp_{year}.parquet"
    ids = (pl.scan_parquet(src).select("game_id").unique().sort("game_id")
           .head(n_games).collect()["game_id"].to_list())
    lf = (pl.scan_parquet(src).filter(pl.col("game_id").is_in(ids))
          .select(BASE_COLS).drop(["season", "game_date"])
          .filter(pl.col("period_number") <= 8))
    return (states_lazy(lf)
            .filter(pl.col("scoring_play").fill_null(False)
                    & pl.col("shooting_play").fill_null(False)
                    & ~pl.col("type_text").str.contains("FreeThrow")
                    & pl.col("team_id").is_not_null())
            .select(
                home_shot=(pl.col("team_id") == pl.col("home_team_id")),
                poss=pl.col("possession"),
                sv=pl.col("score_value"))
            .collect())


@pytest.mark.parametrize("year", AVAILABLE)
def test_a_made_field_goal_gives_the_ball_to_the_other_team(year):
    df = _made_fg_possession(year)
    assert df.height > 0
    wrong = df.filter(
        (pl.col("home_shot") & (pl.col("poss") == 1.0))
        | (~pl.col("home_shot") & (pl.col("poss") == 0.0))
    ).height
    assert wrong == 0, (
        f"{year}: {wrong:,} of {df.height:,} made field goals left the ball with "
        "the scoring team")


@pytest.mark.parametrize("year", AVAILABLE)
def test_made_threes_specifically_flip_possession(year):
    """The exact case the name whitelist missed for 2016-2019."""
    df = _made_fg_possession(year).filter(pl.col("sv") == 3)
    assert df.height > 0, f"{year}: no made threes found - check the type mapping"
    wrong = df.filter(
        (pl.col("home_shot") & (pl.col("poss") == 1.0))
        | (~pl.col("home_shot") & (pl.col("poss") == 0.0))
    ).height
    assert wrong == 0, f"{year}: {wrong:,} of {df.height:,} made threes kept the ball"


def test_the_rule_does_not_depend_on_the_play_type_name():
    """Rename every play type; possession must be unchanged for made field goals.

    This is the regression guard. If someone reintroduces a name whitelist, the
    renamed feed breaks and this fails.
    """
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20

    def game(shot_type):
        return [
            Event(1, 1, 1, 1200, 0, 0, "Jumpball", None, 0, False, False),
            Event(1, 2, 1, 1180, 3, 0, shot_type, HOME, 3, True, True),
        ]

    ctx = PregameContext(1, HOME, AWAY)
    for name in ("JumpShot", "Three Point Jump Shot", "Some Future ESPN Name"):
        states = build_states(game(name), ctx)
        assert states[-1].possession == 0.0, f"made FG typed {name!r} kept the ball"


def test_a_missed_shot_still_carries_possession():
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20
    states = build_states([
        Event(1, 1, 1, 1200, 0, 0, "Jumpball", None, 0, False, False),
        Event(1, 2, 1, 1190, 0, 0, "Defensive Rebound", HOME, 0, False, False),
        Event(1, 3, 1, 1180, 0, 0, "JumpShot", HOME, 0, False, True),   # miss
    ], PregameContext(1, HOME, AWAY))
    assert states[-1].possession == 1.0   # ball is live, carry


def test_a_made_free_throw_still_flips():
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20
    s = build_states([
        Event(1, 1, 1, 1200, 1, 0, "MadeFreeThrow", HOME, 1, True, True),
    ], PregameContext(1, HOME, AWAY))
    assert s[-1].possession == 0.0

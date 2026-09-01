"""The vectorised bulk path must agree with the canonical state builder.

This is the same class of check as the replay harness (plan 13, phase 5):
two implementations of one definition, asserted equal on real games.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import pytest

from cbbwp.schemas import PregameContext
from cbbwp.state import build_states
from cbbwp.adapters.hoopr import load_events, states_lazy, BASE_COLS

PBP = str(pathlib.Path(__file__).resolve().parents[1] / "data/raw/pbp/pbp_2025.parquet")

pytestmark = pytest.mark.skipif(not pathlib.Path(PBP).exists(),
                                reason="pbp_2025.parquet not downloaded yet")


@pytest.fixture(scope="module")
def sample_game_ids():
    return (
        pl.scan_parquet(PBP)
        .select("game_id").unique().sort("game_id").head(25).collect()["game_id"].to_list()
    )


def test_vectorised_matches_reference(sample_game_ids):
    lf = pl.scan_parquet(PBP).filter(pl.col("game_id").is_in(sample_game_ids)).select(BASE_COLS)
    vec = states_lazy(lf).collect()

    for gid in sample_game_ids:
        events, home_id, away_id = load_events(PBP, gid)
        ref = build_states(events, PregameContext(gid, home_id, away_id))
        v = vec.filter(pl.col("game_id") == gid).sort("seq")
        assert len(ref) == len(v), f"row count differs for {gid}"
        for s, r in zip(ref, v.iter_rows(named=True)):
            assert s.seq == r["seq"]
            assert s.game_seconds_remaining == r["game_seconds_remaining"], (gid, s.seq)
            assert s.margin == r["margin"], (gid, s.seq)
            assert s.possession == r["possession"], (gid, s.seq)
            assert s.home_timeouts == r["home_timeouts"], (gid, s.seq)
            assert s.away_timeouts == r["away_timeouts"], (gid, s.seq)
            assert s.home_fouls == r["home_fouls"], (gid, s.seq)
            assert s.away_fouls == r["away_fouls"], (gid, s.seq)
            assert s.is_ot == r["is_ot"]


def test_vectorised_features_match_reference(sample_game_ids):
    import numpy as np
    from cbbwp.features import build_feature_matrix, feature_exprs
    from cbbwp.schemas import FEATURE_NAMES

    lf = pl.scan_parquet(PBP).filter(pl.col("game_id").is_in(sample_game_ids)).select(BASE_COLS)
    vec = (states_lazy(lf)
           .with_columns(pregame_exp_margin=pl.lit(2.75), neutral_site=pl.lit(False),
                         ft_pct_diff=pl.lit(0.03), exp_points_per_min=pl.lit(3.6))
           .select(["game_id", "seq"] + feature_exprs())
           .collect().sort(["game_id", "seq"]))

    ref_rows = []
    for gid in sample_game_ids:
        events, home_id, away_id = load_events(PBP, gid)
        ref_rows.extend(build_states(events, PregameContext(
            gid, home_id, away_id, pregame_exp_margin=2.75,
            ft_pct_diff=0.03, exp_points_per_min=3.6)))
    ref_rows.sort(key=lambda s: (s.game_id, s.seq))
    Xref = build_feature_matrix(ref_rows)
    Xvec = vec.select(FEATURE_NAMES).to_numpy()
    assert Xref.shape == Xvec.shape
    assert np.allclose(Xref, Xvec, atol=1e-9)

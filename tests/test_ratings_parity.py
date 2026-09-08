"""The live ratings snapshot must be on the scale the model was trained on.

`build_live_context.py` and `ratings.build_all_seasons` both fit the same ridge
model, which made it easy to believe they agreed. They did not: training chains
a prior season over season from 2016 with CARRYOVER at every boundary, and the
live snapshot used to fit the single previous season against an EMPTY prior.

Same fit, different prior, so `pregame_exp_margin` served live was not the
quantity the model learned - by 1.3 points sd over the first fortnight of
November, which is exactly when the pregame term carries the most weight and the
score has not yet absorbed it.

These tests hold the two definitions together.
"""
import pathlib
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.ratings import (CARRYOVER, _fit_ridge, carried_prior,  # noqa: E402
                           season_end_ratings, season_pregame_margins)

GAMES = ROOT / "data/proc/games.parquet"


# --------------------------------------------------------------------------
# Synthetic: no data download needed
# --------------------------------------------------------------------------
def _synthetic(seasons=(2016, 2017, 2018), teams=6, per_pair=1) -> pl.DataFrame:
    """A few seasons of round-robin results with a stable talent ordering."""
    rows = []
    gid = 0
    for si, season in enumerate(seasons):
        day = 0
        for rep in range(per_pair):
            for h in range(teams):
                for a in range(teams):
                    if h == a:
                        continue
                    gid += 1
                    day += 1
                    # Team k is k points better than team 0, plus a home edge.
                    margin = (h - a) * 2 + 3
                    rows.append({
                        "game_id": gid,
                        "season": season,
                        # one game per day keeps the as-of refit deterministic
                        "date": pl.datetime(2000 + si, 11, 1).item() if False else None,
                        "_day": day,
                        "home_id": h,
                        "away_id": a,
                        "margin": margin,
                        "neutral_site": False,
                    })
    df = pl.DataFrame(rows).drop("date")
    # Build real datetimes from the per-season day counter.
    return df.with_columns(
        date=pl.datetime(2000, 1, 1).dt.offset_by(
            pl.format("{}d", pl.col("_day") + 400 * (pl.col("season") - 2016)))
    ).drop("_day")


def test_carried_prior_chains_every_season_not_just_the_last():
    games = _synthetic()
    chained = carried_prior(games, 2018)

    # What the old live path did: the single previous season, empty prior.
    prev = games.filter(pl.col("season") == 2017)
    solo, _ = season_end_ratings(prev, {})
    single = {t: v * CARRYOVER for t, v in solo.items()}

    assert set(chained) == set(single)
    # The chain has seen 2016 as well, so it cannot be the same numbers.
    assert any(abs(chained[t] - single[t]) > 1e-9 for t in chained), (
        "carried_prior produced the single-season prior; the chain is not running")


def test_carried_prior_matches_the_chain_build_all_seasons_walks():
    """The two definitions of "the prior entering season N" must not drift.

    `build_all_seasons` chains inline as it goes; `carried_prior` reconstructs
    the same value for one season. This asserts they agree, which is the whole
    reason the live snapshot is allowed to call the cheap one.
    """
    games = _synthetic()

    prior = {}
    for season in sorted(games["season"].unique().to_list()):
        if season == 2018:
            break
        sg = games.filter(pl.col("season") == season)
        _, final, _ = season_pregame_margins(sg, prior)
        prior = {t: v * CARRYOVER for t, v in final.items()}

    reconstructed = carried_prior(games, 2018)
    assert set(reconstructed) == set(prior)
    for t in prior:
        assert reconstructed[t] == pytest.approx(prior[t], abs=1e-9), t


def test_an_empty_history_is_an_empty_prior():
    games = _synthetic()
    assert carried_prior(games, 2016) == {}


# --------------------------------------------------------------------------
# Against the real data: does the snapshot reproduce the offline term?
# --------------------------------------------------------------------------
@pytest.mark.skipif(not GAMES.exists(),
                    reason="games.parquet not built yet (scripts/build_games.py)")
def test_the_live_prior_tracks_the_offline_pregame_margin_in_november():
    """Early season is where the prior is the whole answer, so measure there.

    Reproduces what `build_live_context.py` does -- chain the prior, fit the
    completed games so far -- and compares the pregame margin it would serve
    against the `pregame_exp_margin` the training pipeline actually stored for
    the games that came next.

    Measured RELATIVE to the old single-season prior, and over several cutoffs,
    on purpose. The absolute error is dominated by something that is not skew:
    the offline column refits every 7 game-days, so a cutoff landing just before
    a refit is compared against ratings up to a week staler than the ones being
    tested, and the absolute number swings between 0.3 and 4.8 points with the
    cutoff alone. The comparison against the old prior is unaffected by that,
    because both sides carry the identical lag.
    """
    import datetime

    games = pl.read_parquet(GAMES)
    season = int(games["season"].max())
    cur = games.filter(pl.col("season") == season).sort("date")
    teams = sorted(set(cur["home_id"].to_list()) | set(cur["away_id"].to_list()))
    start = cur["date"].min()

    def mae(fit_g, nxt, prior):
        r, hca = _fit_ridge(fit_g, teams,
                            {t: v for t, v in prior.items() if t in set(teams)})
        errs = []
        for h, a, neu, off in zip(nxt["home_id"], nxt["away_id"],
                                  nxt["neutral_site"], nxt["pregame_exp_margin"]):
            if off is None:
                continue
            errs.append(abs(r.get(h, 0.0) - r.get(a, 0.0)
                            + (0.0 if neu else hca) - off))
        return sum(errs) / len(errs) if errs else None

    chained_prior_ = carried_prior(games, season)
    prev_season = max(s for s in games["season"].unique().to_list() if s < season)
    solo, _ = season_end_ratings(games.filter(pl.col("season") == prev_season), {})
    single_prior = {t: v * CARRYOVER for t, v in solo.items()}

    chained, single = [], []
    for days in (7, 14, 21, 28):
        cutoff = start + datetime.timedelta(days=days)
        fit_g = cur.filter(pl.col("date") < cutoff)
        nxt = cur.filter(pl.col("date") >= cutoff).head(400)
        if fit_g.height < 100 or nxt.height < 50:
            continue
        c, s = mae(fit_g, nxt, chained_prior_), mae(fit_g, nxt, single_prior)
        if c is None or s is None:
            continue
        chained.append(c)
        single.append(s)
        # The regression guard: reverting to the single-season prior must fail
        # this, at every cutoff, not just on average.
        assert c < s, (
            f"+{days}d: chained prior ({c:.3f}) is no better than the "
            f"single-season one ({s:.3f}) -- has build_live_context stopped "
            "chaining? See cbbwp.ratings.carried_prior")

    if len(chained) < 2:
        pytest.skip("not enough early-season games in this file")
    assert sum(chained) / len(chained) < sum(single) / len(single)

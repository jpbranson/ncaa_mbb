"""As-of-date pregame team ratings - our stand-in for the closing spread.

hoopR carries a betting spread only through ~2023, so we fit our own team
strength model. The output, `pregame_exp_margin`, is on the same scale as a
negated point spread: expected home margin in points.

Leakage discipline (plan 8.2): a game's rating inputs are refit from games
that finished STRICTLY BEFORE that game's date. Nothing a season later, and
nothing from the game itself, can reach the model.

Method: ridge regression of final margin on (home indicator - away indicator)
plus a home-court term, shrunk toward a prior. The prior is last season's final
rating regressed toward the mean, which is what carries a team through the
first few November games when the in-season sample is empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import polars as pl

CARRYOVER = 0.70     # how much of last season's rating survives into this one
RIDGE_LAMBDA = 0.5   # tuned against 2016-23 closing spreads: corr 0.87, HCA 3.8
REFIT_EVERY_DAYS = 7


@dataclass
class SeasonRatings:
    season: int
    ratings: Dict[int, float]   # team_id -> points above average
    hca: float


def _fit_ridge(games: pl.DataFrame, teams: list[int], prior: Dict[int, float],
               lam: float | None = None) -> tuple[Dict[int, float], float]:
    """Ridge fit of margin ~ home_team - away_team + hca, shrunk toward `prior`."""
    lam = RIDGE_LAMBDA if lam is None else lam
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    if games.height == 0:
        return dict(prior), 3.4

    rows = games.height
    X = np.zeros((rows, n_t + 1), dtype=np.float64)
    h = games["home_id"].to_numpy()
    a = games["away_id"].to_numpy()
    X[np.arange(rows), [idx[t] for t in h]] = 1.0
    X[np.arange(rows), [idx[t] for t in a]] = -1.0
    X[:, n_t] = 1.0 - games["neutral_site"].cast(pl.Float64).to_numpy()
    y = games["margin"].to_numpy().astype(np.float64)

    b_prior = np.zeros(n_t + 1)
    for t, v in prior.items():
        if t in idx:
            b_prior[idx[t]] = v
    b_prior[n_t] = 3.4  # prior on home-court advantage, in points

    resid = y - X @ b_prior
    A = X.T @ X
    pen = np.full(n_t + 1, lam)
    pen[n_t] = 1.0          # barely shrink the home-court term
    A[np.diag_indices_from(A)] += pen
    d = np.linalg.solve(A, X.T @ resid)
    b = b_prior + d
    b[:n_t] -= b[:n_t].mean()   # ratings are relative; centre them
    return {t: float(b[idx[t]]) for t in teams}, float(b[n_t])


def season_pregame_margins(games: pl.DataFrame, prior: Dict[int, float], lam: float | None = None) -> tuple[pl.DataFrame, Dict[int, float], float]:
    """For one season, attach `pregame_exp_margin` to every game.

    `games` needs: game_id, date (datetime), home_id, away_id, margin, neutral_site.
    Returns (games+column, end-of-season ratings, fitted home-court advantage).
    """
    games = games.sort("date")
    teams = sorted(set(games["home_id"].to_list()) | set(games["away_id"].to_list()))
    day = games["date"].dt.date()
    games = games.with_columns(_day=day)
    days = sorted(games["_day"].unique().to_list())

    out_ids, out_vals = [], []
    ratings, hca = dict(prior), 3.4
    last_refit_i = -10_000
    for i, d in enumerate(days):
        if i - last_refit_i >= REFIT_EVERY_DAYS or i == 0:
            past = games.filter(pl.col("_day") < d)
            ratings, hca = _fit_ridge(past, teams, prior, lam)
            last_refit_i = i
        todays = games.filter(pl.col("_day") == d)
        for gid, hid, aid, neu in zip(todays["game_id"], todays["home_id"],
                                      todays["away_id"], todays["neutral_site"]):
            out_ids.append(gid)
            out_vals.append(ratings.get(hid, 0.0) - ratings.get(aid, 0.0)
                            + (0.0 if neu else hca))

    final_ratings, final_hca = _fit_ridge(games, teams, prior, lam)
    joined = games.join(
        pl.DataFrame({"game_id": out_ids, "pregame_exp_margin": out_vals}),
        on="game_id", how="left",
    ).drop("_day")
    return joined, final_ratings, final_hca


def season_end_ratings(season_games: pl.DataFrame, prior: Dict[int, float],
                       lam: float | None = None) -> tuple[Dict[int, float], float]:
    """End-of-season ratings for one season, given the prior carried into it.

    Exactly the fit `season_pregame_margins` performs for its own return value,
    lifted out so the live snapshot can reuse it instead of writing a second,
    subtly different version. The team list is that season's teams, which is
    what makes a team that stopped playing drop out of the carry.
    """
    teams = sorted(set(season_games["home_id"].to_list())
                   | set(season_games["away_id"].to_list()))
    return _fit_ridge(season_games, teams, prior, lam)


def carried_prior(games: pl.DataFrame, before_season: int,
                  lam: float | None = None) -> Dict[int, float]:
    """The prior `build_all_seasons` would carry into `before_season`.

    THE POINT OF THIS FUNCTION is that the live path used to build its prior
    from a SINGLE previous season fit against an empty prior, while training
    chains every season from 2016 and applies CARRYOVER at each boundary. Same
    ridge fit, different prior - so `pregame_exp_margin` served live was not the
    quantity the model was trained on. Measured on the real data, the two
    disagreed by 1.3 points sd (max 6.8) over the first fortnight of November,
    which is exactly when the pregame term carries the most weight.

    `tests/test_ratings_parity.py` holds the two definitions together.
    """
    prior: Dict[int, float] = {}
    for s in sorted(x for x in games["season"].unique().to_list()
                    if x < before_season):
        final, _ = season_end_ratings(games.filter(pl.col("season") == s),
                                      prior, lam)
        prior = {t: v * CARRYOVER for t, v in final.items()}
    return prior


def build_all_seasons(games: pl.DataFrame, lam: float | None = None) -> pl.DataFrame:
    """Walk seasons in order, carrying each season's ratings into the next."""
    prior: Dict[int, float] = {}
    frames = []
    for season in sorted(games["season"].unique().to_list()):
        sg = games.filter(pl.col("season") == season)
        out, final, hca = season_pregame_margins(sg, prior, lam)
        frames.append(out)
        prior = {t: v * CARRYOVER for t, v in final.items()}
        print(f"  season {season}: {out.height} games, hca={hca:.2f}, "
              f"rating sd={np.std(list(final.values())):.2f}")
    return pl.concat(frames, how="vertical_relaxed")

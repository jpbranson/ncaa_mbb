"""Feature builder. Imported by BOTH the training pipeline and the live server.

If you change anything here you have changed the model's inputs: bump the model
version and refit. `FEATURE_NAMES` in schemas.py fixes the column order.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .schemas import (GameState, FEATURE_NAMES, REGULATION_SECONDS,
                      BONUS_FOULS, DOUBLE_BONUS_FOULS)

_SQRT_REG = math.sqrt(REGULATION_SECONDS)


def _bonus_level(opp_fouls: int) -> int:
    """0 = no bonus, 1 = one-and-one, 2 = double bonus."""
    if opp_fouls >= DOUBLE_BONUS_FOULS:
        return 2
    if opp_fouls >= BONUS_FOULS:
        return 1
    return 0


def feature_dict(s: GameState) -> Dict[str, float]:
    """Features for one state. The single source of truth for both pipelines."""
    t = max(float(s.game_seconds_remaining), 1.0)
    sqrt_t = math.sqrt(t)
    # Pregame edge is the only information at tip-off and should fade to nothing
    # by the buzzer, because by then the score has absorbed it.
    decay = sqrt_t / _SQRT_REG
    return {
        "margin": float(s.margin),
        "sqrt_time": sqrt_t,
        "margin_per_sqrt_time": float(s.margin) / sqrt_t,
        "possession": float(s.possession),
        "pregame_exp_margin": float(s.pregame_exp_margin),
        "pregame_exp_margin_decayed": float(s.pregame_exp_margin) * decay,
        "is_ot": 1.0 if s.is_ot else 0.0,
        "timeout_diff": float(s.home_timeouts - s.away_timeouts),
        # Bonus level a team SHOOTS in is driven by its opponent's fouls.
        "bonus_diff": float(_bonus_level(s.away_fouls) - _bonus_level(s.home_fouls)),
        "ft_pct_diff": float(s.ft_pct_diff),
        # Pace-aware twin of margin/sqrt(time): two slow teams have fewer
        # chances left than two fast ones with the same clock.
        "margin_per_sqrt_points_left": float(s.margin) / math.sqrt(
            max(s.exp_points_per_min * t / 60.0, 1.0)),
    }


def build_feature_matrix(states: Sequence[GameState]) -> np.ndarray:
    """(n_states, n_features) float64 array in FEATURE_NAMES order."""
    out = np.empty((len(states), len(FEATURE_NAMES)), dtype=np.float64)
    for i, s in enumerate(states):
        d = feature_dict(s)
        for j, name in enumerate(FEATURE_NAMES):
            out[i, j] = d[name]
    return out


def mirror_features(X: np.ndarray, y: np.ndarray):
    """Symmetry augmentation (plan 8.3): swap the two teams, flip the label.

    Forces the model to treat the teams identically except through terms that
    are genuinely home-specific. Free data, and it stabilises the fit.
    """
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    Xm = X.copy()
    for name in ("margin", "margin_per_sqrt_time", "pregame_exp_margin",
                 "pregame_exp_margin_decayed", "timeout_diff", "bonus_diff",
                 "ft_pct_diff", "margin_per_sqrt_points_left"):
        Xm[:, idx[name]] *= -1.0
    Xm[:, idx["possession"]] = 1.0 - Xm[:, idx["possession"]]
    return np.vstack([X, Xm]), np.concatenate([y, 1 - y])


# --------------------------------------------------------------------------
# Vectorised twin of `feature_dict`, for building tens of millions of rows.
# tests/test_parity.py asserts the two agree.
# --------------------------------------------------------------------------
def feature_exprs():
    import polars as pl
    t = pl.max_horizontal(pl.col("game_seconds_remaining").cast(pl.Float64), pl.lit(1.0))
    sqrt_t = t.sqrt()
    return [
        pl.col("margin").cast(pl.Float64).alias("margin"),
        sqrt_t.alias("sqrt_time"),
        (pl.col("margin").cast(pl.Float64) / sqrt_t).alias("margin_per_sqrt_time"),
        pl.col("possession").cast(pl.Float64).alias("possession"),
        pl.col("pregame_exp_margin").cast(pl.Float64).alias("pregame_exp_margin"),
        (pl.col("pregame_exp_margin").cast(pl.Float64) * sqrt_t / _SQRT_REG)
        .alias("pregame_exp_margin_decayed"),
        pl.col("is_ot").cast(pl.Float64).alias("is_ot"),
        (pl.col("home_timeouts") - pl.col("away_timeouts")).cast(pl.Float64).alias("timeout_diff"),
        (_bonus_expr(pl.col("away_fouls")) - _bonus_expr(pl.col("home_fouls")))
        .cast(pl.Float64).alias("bonus_diff"),
        pl.col("ft_pct_diff").cast(pl.Float64).alias("ft_pct_diff"),
        (pl.col("margin").cast(pl.Float64) / pl.max_horizontal(
            pl.col("exp_points_per_min").cast(pl.Float64) * t / 60.0, pl.lit(1.0)).sqrt())
        .alias("margin_per_sqrt_points_left"),
    ]


def _bonus_expr(fouls):
    import polars as pl
    return (pl.when(fouls >= DOUBLE_BONUS_FOULS).then(pl.lit(2))
              .when(fouls >= BONUS_FOULS).then(pl.lit(1))
              .otherwise(pl.lit(0)))

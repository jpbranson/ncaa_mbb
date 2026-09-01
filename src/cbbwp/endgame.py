"""Rule-based overrides the training data cannot teach efficiently (plan 10).

The model does not know the rules; we do. These are constraints, not hacks.
"""
from __future__ import annotations

import numpy as np

# A possession is worth at most 3 points (ignoring 4-point plays, which are rare
# enough that treating them as impossible costs less than the states it saves).
MAX_POINTS_PER_POSSESSION = 3
# Seconds a trailing team needs to score and foul once more.
SECONDS_PER_POSSESSION = 6.0


def max_points_remaining(seconds_remaining: np.ndarray) -> np.ndarray:
    """Optimistic ceiling on what a trailing team can still score."""
    poss = np.floor(np.asarray(seconds_remaining, dtype=np.float64) / SECONDS_PER_POSSESSION) + 1
    return poss * MAX_POINTS_PER_POSSESSION


def apply(p, margin, seconds_remaining, is_ot=None):
    """Clamp probabilities that the rules have already decided."""
    p = np.array(p, dtype=np.float64, copy=True)
    margin = np.asarray(margin)
    t = np.asarray(seconds_remaining, dtype=np.float64)

    # 1. Time expired in the final period: the result is known.
    over = t <= 0
    p[over & (margin > 0)] = 1.0
    p[over & (margin < 0)] = 0.0
    # A tie at 0:00 goes to overtime -> a coin flip, nudged by nothing else here.
    p[over & (margin == 0)] = 0.5

    # 2. Mathematically decided: the trailing team cannot catch up in the
    #    possessions that remain, however well it plays.
    ceiling = max_points_remaining(t)
    decided_home = (~over) & (margin > ceiling)
    decided_away = (~over) & (-margin > ceiling)
    p[decided_home] = 1.0
    p[decided_away] = 0.0
    return p

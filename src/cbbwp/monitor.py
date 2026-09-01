"""Calibration drift monitoring (plan phase 5, item 16).

A win probability model fails quietly. Log loss barely moves when the model
starts saying 0.80 to situations that win 0.74, but that gap is the whole
product. So the check that matters is not "is the loss still good" but
"does what we SAY still match what HAPPENS", sliced by time remaining -
because that is the axis along which this model's error actually varies.

Two thresholds, deliberately both required to fire:
  * statistical  - the gap is larger than sampling noise (|z| > Z_ALERT);
  * practical    - the gap is larger than anyone would care about
                   (|gap| > MIN_GAP).
A million rows will make a 0.3-point gap "significant"; that is not drift,
that is a large sample. Requiring both keeps the alert honest.

THE CLUSTERING POINT, which is the difference between this being useful and
being a weekly false alarm: the ~400 states in one game are NOT independent
observations. They share one outcome. A game the home team won contributes 400
rows all labelled 1, and if the model was 3 points low on that game it is 3
points low on all 400 of them. So the standard error must be computed on the
number of GAMES in a bin, not the number of states - the same principle the
train/test split rests on ("effective sample size is games, not rows").

Treating states as independent inflates z by roughly sqrt(states per game),
which here is about 6x. The first version of this file did exactly that and
reported z = -10.6 for a gap that is really about z = -1.7. Pass `game_ids`
and the correction is automatic; omit them and the report says so, loudly,
rather than quietly overstating its own confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Time buckets, as everywhere else in this package.
TIME_BUCKETS = [
    ("40-20 min", 1200, 2400),
    ("20-10 min", 600, 1200),
    ("10-5 min", 300, 600),
    ("5-2 min", 120, 300),
    ("2-1 min", 60, 120),
    ("1-0 min", 0, 60),
]

Z_ALERT = 3.0        # ~1 false positive per 370 independent checks
MIN_GAP = 0.02       # 2 percentage points: below this nobody would notice
MIN_ROWS = 500       # a decile thinner than this says nothing either way
MIN_GAMES = 100      # ...and neither does one drawn from a handful of games
N_DECILES = 10


@dataclass
class BinReport:
    bucket: str
    decile: int
    n: int              # states
    n_games: int        # independent observations behind those states
    pred: float
    obs: float
    gap: float
    z: float
    alert: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    generated: str = ""
    window: str = ""
    n_rows: int = 0
    n_games: int = 0
    clustered: bool = True
    bins: List[BinReport] = field(default_factory=list)
    bucket_ece: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.alerts

    def as_dict(self) -> dict:
        return {
            "generated": self.generated,
            "window": self.window,
            "n_rows": self.n_rows,
            "n_games": self.n_games,
            "ok": self.ok,
            "clustered": self.clustered,
            "alerts": self.alerts,
            "notes": self.notes,
            "bucket_ece": self.bucket_ece,
            "bins": [b.as_dict() for b in self.bins],
        }


def _z_score(pred: float, obs: float, n_independent: int) -> float:
    """How many standard errors the observed rate sits from the predicted one.

    `n_independent` must be the number of GAMES contributing to the bin, not the
    number of states - see the module docstring. Uses the predicted probability
    for the standard error (the null hypothesis is that the model is right),
    which is the conservative choice near 0 and 1.
    """
    var = pred * (1.0 - pred)
    if n_independent <= 0 or var <= 0:
        return 0.0
    return (obs - pred) / math.sqrt(var / n_independent)


def decile_edges(p: np.ndarray, n_deciles: int = N_DECILES) -> np.ndarray:
    """Equal-COUNT edges. Equal-width bins leave the interesting tails empty."""
    qs = np.linspace(0, 1, n_deciles + 1)[1:-1]
    return np.unique(np.quantile(p, qs))


def check(y: Sequence[int], p: Sequence[float], seconds: Sequence[float],
          game_ids: Optional[Sequence] = None, window: str = "",
          z_alert: float = Z_ALERT, min_gap: float = MIN_GAP,
          min_rows: int = MIN_ROWS, min_games: int = MIN_GAMES,
          generated: str = "") -> DriftReport:
    """Decile calibration check within each time bucket.

    `game_ids` is not optional in spirit: without it every state is treated as
    an independent observation and the z-scores are inflated by roughly the
    square root of the number of states per game. Omitting it is supported for
    synthetic data and unit tests, and the report says so.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(seconds, dtype=np.float64)
    if not (len(y) == len(p) == len(s)):
        raise ValueError("y, p and seconds must be the same length")
    if game_ids is None:
        g = np.arange(len(y))          # every row its own "game": no clustering
        clustered = False
    else:
        g = np.asarray(game_ids)
        if len(g) != len(y):
            raise ValueError("game_ids must be the same length as y")
        clustered = True

    rep = DriftReport(generated=generated, window=window, n_rows=int(len(y)),
                      n_games=int(len(np.unique(g))) if clustered else 0,
                      clustered=clustered)
    if not clustered:
        rep.notes.append(
            "no game_ids supplied: states treated as independent, so z-scores "
            "are optimistic. Fine for synthetic data, wrong for real games.")

    for name, lo, hi in TIME_BUCKETS:
        m = (s >= lo) & (s < hi)
        if m.sum() < min_rows:
            continue
        pb, yb, gb = p[m], y[m], g[m]
        edges = decile_edges(pb)
        idx = np.digitize(pb, edges)
        ece = 0.0
        for d in range(len(edges) + 1):
            dm = idx == d
            n = int(dm.sum())
            if n == 0:
                continue
            n_games = int(len(np.unique(gb[dm])))
            pred = float(pb[dm].mean())
            obs = float(yb[dm].mean())
            gap = obs - pred
            ece += n / len(pb) * abs(gap)
            # The independent-observation count is the number of games.
            z = _z_score(pred, obs, n_games)
            alert = bool(n >= min_rows and n_games >= min_games
                         and abs(z) > z_alert and abs(gap) > min_gap)
            rep.bins.append(BinReport(name, d, n, n_games, pred, obs, gap, z, alert))
            if alert:
                rep.alerts.append(
                    f"{name} decile {d}: model says {pred:.3f}, observed "
                    f"{obs:.3f} over {n:,} states from {n_games:,} games "
                    f"(gap {gap:+.3f}, z {z:+.1f})")
        rep.bucket_ece[name] = float(ece)
    return rep


def format_report(rep: DriftReport) -> str:
    """Human-readable table, for a terminal or an alert email."""
    out = []
    head = f"calibration check  {rep.window}".strip()
    out.append(head)
    out.append(f"{rep.n_rows:,} states"
               + (f" from {rep.n_games:,} games" if rep.n_games else ""))
    for n in rep.notes:
        out.append(f"NOTE: {n}")
    out.append("")
    out.append(f"{'bucket':<12}{'dec':>4}{'states':>10}{'games':>8}"
               f"{'pred':>8}{'obs':>8}{'gap':>8}{'z':>7}  ")
    last = None
    for b in rep.bins:
        sep = "" if b.bucket == last else "\n"
        last = b.bucket
        flag = "  <-- ALERT" if b.alert else ""
        out.append(f"{sep}{b.bucket if sep else '':<12}{b.decile:>4}{b.n:>10,}"
                   f"{b.n_games:>8,}{b.pred:>8.3f}{b.obs:>8.3f}{b.gap:>+8.3f}"
                   f"{b.z:>+7.1f}{flag}")
    out.append("")
    out.append("ECE by bucket: " + "  ".join(
        f"{k} {v:.4f}" for k, v in rep.bucket_ece.items()))
    out.append("")
    if rep.ok:
        out.append("OK - no bucket/decile is both statistically and practically off\n"
                   "     (z computed on games, not states - see monitor.py).")
    else:
        out.append(f"{len(rep.alerts)} ALERT(S):")
        out.extend("  " + a for a in rep.alerts)
    return "\n".join(out)

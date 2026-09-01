"""Probability metrics, always broken out by time remaining (plan 11.2)."""
from __future__ import annotations

import numpy as np

EPS = 1e-6

# (label, lower bound seconds, upper bound seconds)
TIME_BUCKETS = [
    ("40-20 min", 1200, 2400),
    ("20-10 min", 600, 1200),
    ("10-5 min", 300, 600),
    ("5-2 min", 120, 300),
    ("2-0 min", 0, 120),
]


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def accuracy(y, p):
    return float(np.mean((p >= 0.5) == (y == 1)))


def by_time_bucket(y, p, seconds, extra=None):
    """Metrics per bucket. `extra` is an optional dict of other prediction sets."""
    rows = []
    for name, lo, hi in TIME_BUCKETS:
        m = (seconds >= lo) & (seconds < hi) if lo > 0 else (seconds >= lo) & (seconds < hi)
        if m.sum() == 0:
            continue
        row = {"bucket": name, "n": int(m.sum()),
               "log_loss": log_loss(y[m], p[m]), "brier": brier(y[m], p[m]),
               "acc": accuracy(y[m], p[m])}
        if extra:
            for k, v in extra.items():
                ok = m & np.isfinite(v)
                row[f"log_loss_{k}"] = log_loss(y[ok], v[ok]) if ok.sum() else float("nan")
                row[f"n_{k}"] = int(ok.sum())
        rows.append(row)
    return rows


def calibration_table(y, p, n_bins=10):
    """Predicted vs observed, in equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                    "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return out


def ece(y, p, n_bins=20):
    """Expected calibration error: mean |predicted - observed|, size-weighted."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    tot, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(tot)

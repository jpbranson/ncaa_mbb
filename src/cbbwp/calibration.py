"""Time-bucketed probability calibration.

Miscalibration in these models is almost entirely time-dependent, so a single
global calibrator averages two opposite errors and fixes neither (plan 9.3).
We fit one isotonic curve per time bucket on a HELD-OUT season, then blend
between adjacent buckets so the curve never visibly jumps.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

# (lower, upper) seconds remaining, and the anchor point used for blending.
BUCKETS = [(1200, 2400), (600, 1200), (300, 600), (120, 300), (60, 120), (0, 60)]
ANCHORS = np.array([np.sqrt((lo + hi) / 2) for lo, hi in BUCKETS])
CLIP = (0.001, 0.999)


class TimeBucketedCalibrator:
    def __init__(self, buckets=BUCKETS, min_rows=20_000):
        self.buckets = list(buckets)
        self.min_rows = min_rows
        self.models: list[IsotonicRegression | None] = []

    def fit(self, p, y, seconds):
        self.models = []
        for lo, hi in self.buckets:
            m = (seconds >= lo) & (seconds < hi)
            if m.sum() < self.min_rows:
                self.models.append(None)
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p[m], y[m])
            self.models.append(iso)
        return self

    def _apply(self, i, p):
        m = self.models[i]
        return p if m is None else m.predict(p)

    def transform(self, p, seconds):
        """Blend the two nearest bucket calibrators, weighted in sqrt-time."""
        p = np.asarray(p, dtype=np.float64)
        s = np.sqrt(np.maximum(np.asarray(seconds, dtype=np.float64), 0.0))
        # anchors run from long time remaining down to zero
        order = np.argsort(ANCHORS)
        a = ANCHORS[order]
        cols = np.stack([self._apply(order[i], p) for i in range(len(a))], axis=1)
        idx = np.clip(np.searchsorted(a, s), 1, len(a) - 1)
        lo, hi = a[idx - 1], a[idx]
        w = np.clip((s - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
        out = cols[np.arange(len(p)), idx - 1] * (1 - w) + cols[np.arange(len(p)), idx] * w
        return np.clip(out, *CLIP)

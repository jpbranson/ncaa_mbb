"""The live scoring path. Deliberately thin: it calls the SAME state and
feature builders the training pipeline used, then a pinned model artifact.
"""
from __future__ import annotations

import json
import pathlib
import pickle
from typing import Iterable, List

import numpy as np

from .schemas import (Event, PregameContext, FEATURE_NAMES,
                      STATE_RULES_VERSION)
from .state import build_states
from .features import build_feature_matrix
from . import endgame

OVERRIDE_CLIP = 0.999   # never assert more certainty than the feed supports


class WinProbabilityService:
    def __init__(self, registry_dir: str | pathlib.Path, version: str):
        self.dir = pathlib.Path(registry_dir) / version
        self.version = version
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        if self.manifest["features"] != FEATURE_NAMES:
            raise RuntimeError(
                f"model {version} was fit on different features than this code builds; "
                "the feature contract changed - refit or pin an older code version"
            )
        # The names can match while the MEANING has changed underneath them.
        # A model fit before a state-rule change must not be served states built
        # after it. An artifact with no stamp predates the check and is treated
        # as version 1.
        fit_rules = self.manifest.get("state_rules_version", 1)
        if fit_rules != STATE_RULES_VERSION:
            raise RuntimeError(
                f"model {version} was fit with state rules v{fit_rules} but this "
                f"code builds states with v{STATE_RULES_VERSION}. The feature NAMES "
                "still match, so this would have been silent: the model would be "
                "served inputs that mean something different from its training "
                "data. Refit, or pin the code version that matches the artifact."
            )
        kind = self.manifest["kind"]
        if kind == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.dir / "model.txt"))
            self._predict = lambda X: self.model.predict(X)
        else:
            with open(self.dir / "model.pkl", "rb") as f:
                b = pickle.load(f)
            self._predict = lambda X: b["model"].predict_proba(b["scaler"].transform(X))[:, 1]

    def score_game(self, events: Iterable[Event], ctx: PregameContext) -> List[dict]:
        """Replay a game from event zero and return one prediction per state.

        Called on every poll. Replaying from scratch costs milliseconds and makes
        retroactive feed corrections a non-event.
        """
        states = build_states(events, ctx)
        if not states:
            return []
        X = build_feature_matrix(states)
        p = np.asarray(self._predict(X), dtype=np.float64)

        margin = np.array([s.margin for s in states])
        secs = np.array([s.game_seconds_remaining for s in states], dtype=np.float64)
        adj = endgame.apply(p, margin, secs)
        touched = adj != p
        p[touched] = np.clip(adj[touched], 1 - OVERRIDE_CLIP, OVERRIDE_CLIP)

        return [
            {"game_id": s.game_id, "seq": s.seq, "period": s.period,
             "game_seconds_remaining": s.game_seconds_remaining, "margin": s.margin,
             "home_win_prob": float(pi), "model_version": self.version}
            for s, pi in zip(states, p)
        ]

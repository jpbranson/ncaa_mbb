"""Recompute the test-season predictions from the PINNED model, in float64.

`fit_models.py` writes artifacts/test_preds.npz, but artifacts/ is gitignored,
so a restored working copy often has only the float32 parquet export. Metrics
computed from that export land about 0.0001 low on log loss and differ in the
fourth decimal on accuracy and ECE -- enough to look like drift when it is only
storage precision.

Nothing here refits anything. The model in registry/v2 is pinned and hashed, so
re-running it over the same state rows reproduces the original predictions
exactly, at full precision, in a few seconds and a few hundred MB.

    python3 scripts/rebuild_test_preds.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp.schemas import FEATURE_NAMES          # noqa: E402

TEST_SEASONS = [2025, 2026]
OUT = ROOT / "artifacts" / "test_preds.npz"


def main() -> None:
    import lightgbm as lgb
    import pickle

    gbm = lgb.Booster(model_file=str(ROOT / "registry" / "v2" / "model.txt"))
    frames = [pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{s}.parquet")
              for s in TEST_SEASONS]
    te = (pl.concat(frames)
          .select(FEATURE_NAMES + [c for c in
                  ["home_win", "game_seconds_remaining", "espn_wp", "game_id", "season"]
                  if c not in FEATURE_NAMES])
          .collect())
    X = te.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    p_gbm = np.asarray(gbm.predict(X), dtype=np.float64)

    # The logistic baseline is a scikit-learn pickle, and a pickle is only
    # loadable by a compatible scikit-learn. artifacts/lr_v1.pkl was written by
    # 1.8.0 and will not unpickle on 1.7.2 -- which is a good argument for the
    # LightGBM artifact's plain-text format, and a bad property to discover
    # during a rebuild. Fall back to the stored export, but only after checking
    # the rows actually line up.
    p_lr = np.full(len(te), np.nan)
    lr_path = ROOT / "artifacts" / "lr_v1.pkl"
    try:
        with open(lr_path, "rb") as f:
            b = pickle.load(f)
        p_lr = b["model"].predict_proba(b["scaler"].transform(X))[:, 1].astype(np.float64)
        print("logistic: recomputed from lr_v1.pkl")
    except Exception as e:                              # noqa: BLE001
        print(f"logistic: could not load lr_v1.pkl ({type(e).__name__}); "
              "falling back to the stored export")
        pq = ROOT / "artifacts" / "eval_preds.parquet"
        if pq.exists():
            ex = pl.read_parquet(pq)
            same = (len(ex) == len(te)
                    and np.array_equal(ex["game_id"].to_numpy(), te["game_id"].to_numpy()))
            if same:
                p_lr = ex["p_lr"].to_numpy().astype(np.float64)
                print("  rows verified aligned; logistic column carried over (float32 origin)")
            else:
                print("  rows do NOT align; logistic column left as NaN")

    np.savez_compressed(
        OUT,
        y=te["home_win"].to_numpy().astype(np.float64),
        p_lr=p_lr, p_gbm=p_gbm,
        secs=te["game_seconds_remaining"].to_numpy(),
        espn=te["espn_wp"].to_numpy().astype(np.float64),
        game_id=te["game_id"].to_numpy(), season=te["season"].to_numpy())
    print(f"wrote {OUT}  ({len(te):,} rows, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Regenerate the data block behind the published results artifact.

The artifact used to carry numbers pasted in by hand, which is how it ended up
still advertising v1's figures and "17 tests passing" two model versions later.
This makes the data a build product: run it after any refit, paste the one line
it prints into the artifact's trailing <script>, and the page cannot drift from
the model again.

    python3 scripts/build_report_data.py            # writes artifacts/report_data.json

Everything comes from artifacts/eval_preds.parquet (written by fit_models.py)
and the state rows, so it always describes the model that was actually fit.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp.schemas import FEATURE_NAMES          # noqa: E402

OUT = ROOT / "artifacts" / "report_data.json"
CURVE_SEASON = 2026
N_CURVES = 4
MAX_POINTS = 200
EPS = 1e-15

BUCKETS = [("40-20 min", 1200, 2400), ("20-10 min", 600, 1200),
           ("10-5 min", 300, 600), ("5-2 min", 120, 300),
           ("2-1 min", 60, 120), ("1-0 min", 0, 60)]


def _ll(y, p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    # Prefer the float64 predictions. The float32 parquet export lands about
    # 0.0001 low on log loss and differs in the fourth decimal on accuracy and
    # ECE -- small, but enough to make a published page disagree with EXPLAIN.
    npz = ROOT / "artifacts" / "test_preds.npz"
    if npz.exists():
        raw = np.load(npz)
        d = {k: raw[k] for k in raw.files}
        print("source: test_preds.npz (float64)")
    else:
        raise SystemExit("run scripts/rebuild_test_preds.py first -- the float32 "
                         "export is not precise enough for the published figures")
    y = d["y"].astype(np.float64)
    secs = d["secs"].astype(np.float64)
    P = {"model": d["p_gbm"].astype(np.float64),
         "logistic": d["p_lr"].astype(np.float64),
         "espn": d["espn"].astype(np.float64)}
    ok = np.isfinite(P["espn"])

    headline = {}
    for k, p in P.items():
        pf = p[ok].astype(np.float64)
        headline[k] = {"logloss": round(_ll(y[ok], pf), 4),
                       "brier": round(float(((pf - y[ok]) ** 2).mean()), 4),
                       "acc": round(float(((pf >= .5).astype(int) == y[ok]).mean()), 4),
                       "ece": round(_ece(y[ok], pf), 4)}

    buckets = []
    for name, lo, hi in BUCKETS:
        m = ok & (secs >= lo) & (secs < hi)
        buckets.append({"bucket": name, "n": int(m.sum()),
                        **{k: round(_ll(y[m], P[k][m].astype(np.float64)), 4)
                           for k in P}})

    # calibration, 20 equal-width bins on the shipped model
    pg = P["model"].astype(np.float64)
    edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(pg, edges) - 1, 0, 19)
    calib = []
    for b in range(20):
        s = idx == b
        if s.sum() < 50:
            continue
        calib.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(s.sum()),
                      "pred": round(float(pg[s].mean()), 4),
                      "obs": round(float(y[s].mean()), 4)})

    print(f"headline: {headline['model']}")
    print(f"{len(buckets)} buckets, {len(calib)} calibration bins")

    curves = _curves()
    n_ot = sum(1 for c in curves if c["max_x"] > 2400)
    print(f"\n{n_ot} of {len(curves)} highlighted games went to overtime "
          "-- check the artifact caption says so")
    out = {"headline": headline, "buckets": buckets, "calibration": calib,
           "curves": curves,
           "meta": {"test_rows": int(len(y)),
                    "test_games": int(np.unique(d["game_id"]).size),
                    "test_seasons": sorted(int(s) for s in np.unique(d["season"])),
                    "overtime_curves": n_ot}}
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size/1000:.0f} KB)")
    print("\npaste this into the artifact's trailing <script>:")
    print("const D = " + json.dumps(out) + ";")


def _ece(y, p, bins: int = 20) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.sum():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def _curves() -> list[dict]:
    """The highest-movement games of the season, drawn from the shipped model.

    Overtime games dominate this list, which is the point: a curve that handles
    a tied buzzer gracefully is the hardest case, and the eye catches what no
    aggregate metric will.
    """
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / "registry" / "v2" / "model.txt"))

    st = (pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{CURVE_SEASON}.parquet")
          .select(FEATURE_NAMES + [c for c in
                  ["game_id", "seq", "period", "game_seconds_remaining", "espn_wp"]
                  if c not in FEATURE_NAMES])
          .collect().sort(["game_id", "seq"]))
    X = st.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    st = st.with_columns(pl.Series("wp", booster.predict(X)))

    gei = (st.group_by("game_id")
             .agg(pl.col("wp").diff().abs().sum().alias("gei"))
             .sort("gei", descending=True).head(N_CURVES))
    games = pl.read_parquet(ROOT / "data" / "proc" / "games.parquet")
    names = (pl.scan_parquet(ROOT / "data" / "raw" / "pbp" / f"pbp_{CURVE_SEASON}.parquet")
             .select(["game_id", "home_team_name", "away_team_name"])
             .unique(subset="game_id").collect())

    out = []
    for gid, g in gei.iter_rows():
        sub = st.filter(pl.col("game_id") == gid)
        info = games.filter(pl.col("game_id") == gid).row(0, named=True)
        nm = names.filter(pl.col("game_id") == gid)
        home = nm["home_team_name"][0] if len(nm) else f"home {info['home_id']}"
        away = nm["away_team_name"][0] if len(nm) else f"away {info['away_id']}"

        per = sub["period"].to_numpy()
        gsr = sub["game_seconds_remaining"].to_numpy().astype(float)
        # Elapsed seconds. Regulation counts down from 2400; each overtime is its
        # own 300-second clock, so they have to be laid end to end by hand.
        elapsed = np.where(per <= 2, 2400 - gsr,
                           2400 + (per - 3) * 300 + (300 - gsr))
        n_ot = max(0, int(per.max()) - 2)
        max_x = 2400 + n_ot * 300

        keep = np.linspace(0, len(sub) - 1, min(MAX_POINTS, len(sub))).astype(int)
        keep = np.unique(keep)
        out.append({
            "title": f"{away} {info['away_score']} @ {home} {info['home_score']}",
            "date": str(info["date"])[:10],
            "gei": round(float(g), 2),
            "x": [int(v) for v in elapsed[keep]],
            "wp": [round(float(v), 4) for v in sub["wp"].to_numpy()[keep]],
            "espn": [round(float(v), 4) for v in sub["espn_wp"].to_numpy()[keep]],
            "margin": [float(v) for v in sub["margin"].to_numpy()[keep]],
            "max_x": int(max_x),
        })
        print(f"  curve: {out[-1]['title']}  {out[-1]['date']}  movement {g:.2f}")
    return out


if __name__ == "__main__":
    main()

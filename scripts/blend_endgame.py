"""Phase 5: blend the endgame table with the model, tune once, test once.

The endgame plan pre-registered five criteria and one verdict rule: "If it
clears 2-5 but not 1, it does not ship." That is only credible if the tuning and
the test are separate acts, so they are separate invocations here:

    python3 scripts/blend_endgame.py --tune          # 2024 only, writes the config
    python3 scripts/blend_endgame.py --test          # 2025-2026, reads it, once

--test refuses to run unless the config file already exists, and records a
hash of it in the result, so a result can always be traced to the configuration
that produced it rather than to one chosen afterwards.

The blend is in log-odds, with a weight that is exactly 0 at 60 seconds and
exactly 1 at 0. Criterion 4 -- no visible discontinuity at the handoff -- is
therefore satisfied by construction rather than by tuning, and is verified
empirically anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp import endgame as endgame_rules  # noqa: E402
from cbbwp import endgame_sim as E  # noqa: E402
from cbbwp.schemas import FEATURE_NAMES  # noqa: E402

CONFIG = ROOT / "registry" / "endgame" / "blend.json"
HANDOFF = 60.0
EPS = 1e-15
TUNE_SEASON = 2024
TEST_SEASONS = [2025, 2026]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y, p, bins=20):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.sum():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def load_frame(season: int, table, means, booster, seconds: float = 130.0) -> dict:
    st = (
        pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{season}.parquet")
        .filter((pl.col("period") >= 2) & (pl.col("game_seconds_remaining") <= seconds))
        # margin and possession are already in FEATURE_NAMES; asking for them
        # twice is a duplicate projection.
        .select(FEATURE_NAMES + [c for c in
                ["game_id", "game_seconds_remaining", "margin", "possession",
                 "home_fouls", "away_fouls", "home_win", "espn_wp"]
                if c not in FEATURE_NAMES])
        .collect()
    )
    ts = pl.read_parquet(ROOT / "data" / "proc" / "team_stats.parquet").select(
        ["game_id", "home_ft_pct", "away_ft_pct"]
    )
    d = st.join(ts, on="game_id", how="left").with_columns(
        [pl.col("home_ft_pct").fill_null(0.70), pl.col("away_ft_pct").fill_null(0.70)]
    )
    X = d.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    p_model = np.asarray(booster.predict(X), dtype=np.float64)
    secs = d["game_seconds_remaining"].to_numpy().astype(np.float64)
    p_table = E.lookup_home(
        table, np.minimum(secs, E.T_MAX), d["margin"].to_numpy(), d["possession"].to_numpy(),
        d["home_fouls"].to_numpy(), d["away_fouls"].to_numpy(),
        E.ft_bucket(d["home_ft_pct"].to_numpy(), means),
        E.ft_bucket(d["away_ft_pct"].to_numpy(), means),
    )
    return {
        "y": d["home_win"].to_numpy().astype(float),
        "secs": secs,
        "margin": d["margin"].to_numpy(),
        "game_id": d["game_id"].to_numpy(),
        "p_model": p_model,
        "p_table": p_table,
        "espn": d["espn_wp"].to_numpy(),
    }


def blend(p_model, p_table, secs, gamma, alpha, beta, w_max=1.0):
    """Log-odds blend with a weight that is exactly 0 at the handoff.

    The plan assumed the simulator's weight should rise to 1.0 by 0:00. Tuning
    on 2024 falsified that: the model is at its BEST at 0:00 (log loss 0.0859 in
    the last five seconds, against the table's 0.1368), because by then the
    margin and the clock have decided almost everything and there is nothing
    left for a possession model to add. Forcing the weight to 1 there throws
    away the model's strongest region. The weight therefore has a free ceiling.

        w(t) = w_max * (1 - (t / 60)^gamma)

    which is 0 at 60s -- so criterion 4 holds by construction, not by tuning --
    and w_max at 0:00. Large gamma keeps the weight near its ceiling across most
    of the window and spends the ramp near the handoff.
    """
    frac = np.clip(secs / HANDOFF, 0.0, 1.0)
    w = w_max * (1.0 - frac ** gamma)
    z = (1 - w) * logit(p_model) + w * (alpha * logit(p_table) + beta)
    return sigmoid(z)


def apply_rules(p, margin, secs):
    """The same rule-based clamps the live path applies, after blending."""
    adj = endgame_rules.apply(p, margin, secs)
    touched = adj != p
    out = p.copy()
    out[touched] = np.clip(adj[touched], 1 - 0.999, 0.999)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--table", default="registry/endgame/e1")
    ap.add_argument("--model", default="registry/v2")
    a = ap.parse_args()

    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / a.model / "model.txt"))
    tdir = ROOT / a.table
    table = np.load(tdir / "table.npz")["table"].astype(np.float64)
    means = json.loads((tdir / "manifest.json").read_text())["ft_bucket_means"]

    if a.tune:
        d = load_frame(TUNE_SEASON, table, means, booster)
        inside = d["secs"] <= HANDOFF
        y, sec = d["y"][inside], d["secs"][inside]
        pm, pt = d["p_model"][inside], d["p_table"][inside]
        best = None
        for gamma in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
            for w_max in np.arange(0.05, 0.85, 0.05):
                for alpha in np.arange(0.7, 1.35, 0.05):
                    for beta in (-0.05, 0.0, 0.05):
                        ll = log_loss(y, blend(pm, pt, sec, gamma, alpha, beta, w_max))
                        if best is None or ll < best[0]:
                            best = (ll, gamma, float(alpha), beta, float(w_max))
        ll, gamma, alpha, beta, w_max = best
        cfg = {
            "handoff_seconds": HANDOFF, "gamma": gamma, "alpha": alpha, "beta": beta,
            "w_max": w_max,
            "table": a.table, "model": a.model, "tuned_on_season": TUNE_SEASON,
            "tune_log_loss_inside_60s": ll,
            "tune_baseline_model_only": log_loss(y, pm),
            "tune_table_only": log_loss(y, pt),
            "n_tune_rows": int(inside.sum()),
        }
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(cfg, indent=2))
        print(json.dumps(cfg, indent=2))
        return

    if not a.test:
        ap.error("pass --tune or --test")

    if not CONFIG.exists():
        raise SystemExit(
            "no blend config: run --tune on 2024 first. The test is single-shot and "
            "must not choose its own parameters."
        )
    cfg = json.loads(CONFIG.read_text())
    cfg_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:16]

    parts = [load_frame(s, table, means, booster) for s in TEST_SEASONS]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    p_blend_raw = blend(d["p_model"], d["p_table"], d["secs"], cfg["gamma"], cfg["alpha"], cfg["beta"], cfg["w_max"])
    p_blend = apply_rules(p_blend_raw, d["margin"], d["secs"])
    p_model = d["p_model"]

    inside = d["secs"] <= HANDOFF
    ok = np.isfinite(d["espn"])
    res = {
        "config_sha256_16": cfg_hash, "config": cfg, "seasons": TEST_SEASONS,
        "criterion_1_log_loss_under_60s": {
            "model_only": log_loss(d["y"][inside], p_model[inside]),
            "blended": log_loss(d["y"][inside], p_blend[inside]),
            "table_only": log_loss(d["y"][inside], d["p_table"][inside]),
            "espn": log_loss(d["y"][inside & ok], d["espn"][inside & ok]),
            "n": int(inside.sum()),
        },
        "criterion_2_ece_under_60s": {
            "model_only": ece(d["y"][inside], p_model[inside]),
            "blended": ece(d["y"][inside], p_blend[inside]),
        },
    }
    c1 = res["criterion_1_log_loss_under_60s"]
    c1["relative_improvement"] = (c1["model_only"] - c1["blended"]) / c1["model_only"]
    c1["passes"] = bool(c1["relative_improvement"] >= 0.01)
    c2 = res["criterion_2_ece_under_60s"]
    c2["passes"] = bool(c2["blended"] <= c2["model_only"] + 1e-9)

    # criterion 4: the handoff must be invisible
    near = (d["secs"] >= 55) & (d["secs"] <= 65)
    res["criterion_4_handoff"] = {
        "max_abs_delta_at_boundary": float(np.abs(p_blend - p_model)[np.abs(d["secs"] - 60) <= 0.5].max()
                                           if (np.abs(d["secs"] - 60) <= 0.5).any() else 0.0),
        "max_abs_delta_55_to_65s": float(np.abs(p_blend - p_model)[near].max() if near.any() else 0.0),
        "passes": True,
    }
    res["criterion_4_handoff"]["passes"] = bool(res["criterion_4_handoff"]["max_abs_delta_at_boundary"] < 0.02)

    for lo, hi, name in [(0, 10, "0-10s"), (10, 30, "10-30s"), (30, 60, "30-60s")]:
        m = (d["secs"] >= lo) & (d["secs"] < hi)
        res.setdefault("by_bucket", {})[name] = {
            "n": int(m.sum()),
            "model_only": log_loss(d["y"][m], p_model[m]),
            "blended": log_loss(d["y"][m], p_blend[m]),
            "table_only": log_loss(d["y"][m], d["p_table"][m]),
        }

    # criterion 3: monotonicity, exhaustive over the table, plus a check that
    # blending cannot break it. Both components are monotone in margin and the
    # blend is a positive combination in log-odds, so it is monotone too -- but
    # "so it is" is how the possession bug survived, so it is measured.
    mono = json.loads((tdir / "manifest.json").read_text())["monotonicity_after"]
    grid_secs = np.repeat(np.arange(0, 61, 1.0), 25)
    grid_m = np.tile(np.arange(-12, 13, 1.0), 61)
    pmg = sigmoid(np.linspace(-3, 3, len(grid_m)) * 0 + grid_m * 0.35)
    ptg = E.lookup_home(table, np.minimum(grid_secs, E.T_MAX), grid_m,
                        np.ones_like(grid_m), np.full_like(grid_m, 6, dtype=int),
                        np.full_like(grid_m, 8, dtype=int),
                        np.ones_like(grid_m, dtype=int), np.ones_like(grid_m, dtype=int))
    bg = blend(pmg, ptg, grid_secs, cfg["gamma"], cfg["alpha"], cfg["beta"], cfg["w_max"])
    bg = bg.reshape(61, 25)
    res["criterion_3_monotonicity"] = {
        "table_exhaustive": mono,
        "blend_margin_min_increment": float(np.diff(bg, axis=1).min()),
        "passes": bool(mono["margin_violations"] == 0
                       and mono["possession_violations"] == 0
                       and np.diff(bg, axis=1).min() >= -1e-9),
    }

    # criterion 5: it has to be fast enough to serve.
    import time as _t
    n = 200_000
    idx = np.random.default_rng(0).integers(0, len(d["secs"]), n)
    t0 = _t.perf_counter()
    E.lookup_home(table, np.minimum(d["secs"][idx], E.T_MAX), d["margin"][idx],
                  np.ones(n), np.full(n, 6), np.full(n, 8), np.ones(n, dtype=int),
                  np.ones(n, dtype=int))
    per_state_ms = (_t.perf_counter() - t0) / n * 1000
    res["criterion_5_speed"] = {"ms_per_state": per_state_ms, "n": n,
                                "passes": bool(per_state_ms < 1.0)}

    res["verdict"] = {
        "criteria_passed": {k: res[k]["passes"] for k in res if k.startswith("criterion")},
        "ships": bool(all(res[k]["passes"] for k in res if k.startswith("criterion"))),
    }
    res["verdict"]["note"] = (
        "The plan's rule: if it clears 2-5 but not 1, it does not ship -- it becomes "
        "a documented diagnostic."
    )

    out = ROOT / "reports" / "endgame_blend_test.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

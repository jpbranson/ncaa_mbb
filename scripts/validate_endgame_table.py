"""Phase 4: is the table HONEST about the states it describes?

The endgame plan's gate for this phase is deliberately weak -- "beats nothing;
just has to be honest". Phase 4 asks only whether a state the table calls a 20%
state is won about 20% of the time. Whether that is worth shipping is Phase 5's
question, and it is asked once, on 2025-2026.

So this runs on 2024, against a table whose parameters were estimated from
2016-2023 only. 2024 is genuinely out of sample for the table, and it is not one
of the held-out test seasons, so using it here costs nothing later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp import endgame_sim as E  # noqa: E402

EPS = 1e-15


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    acc = float((((p >= 0.5).astype(int)) == y).mean())
    edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 19)
    ece = 0.0
    for b in range(20):
        s = idx == b
        if s.sum():
            ece += s.mean() * abs(p[s].mean() - y[s].mean())
    return {"log_loss": ll, "brier": brier, "accuracy": acc, "ece": float(ece), "n": int(len(p))}


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        s = idx == b
        if s.sum() < 50:
            continue
        out.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(s.sum()),
                    "predicted": float(p[s].mean()), "observed": float(y[s].mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--table", default="registry/endgame/e1_no2024")
    ap.add_argument("--seconds", type=int, default=60)
    a = ap.parse_args()

    tdir = ROOT / a.table
    table = np.load(tdir / "table.npz")["table"].astype(np.float64)
    manifest = json.loads((tdir / "manifest.json").read_text())
    if a.season in manifest["seasons_used"]:
        raise SystemExit(
            f"season {a.season} was used to fit {a.table}; validating on it would prove nothing"
        )
    means = manifest["ft_bucket_means"]

    st = (
        pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{a.season}.parquet")
        .filter((pl.col("period") >= 2) & (pl.col("game_seconds_remaining") <= a.seconds))
        .select(["game_id", "game_seconds_remaining", "margin", "possession",
                 "home_fouls", "away_fouls", "home_win", "espn_wp"])
        .collect()
    )
    ts = pl.read_parquet(ROOT / "data" / "proc" / "team_stats.parquet").select(
        ["game_id", "home_ft_pct", "away_ft_pct"]
    )
    d = st.join(ts, on="game_id", how="left").with_columns(
        [pl.col("home_ft_pct").fill_null(0.70), pl.col("away_ft_pct").fill_null(0.70)]
    )

    p = E.lookup_home(
        table,
        d["game_seconds_remaining"].to_numpy(),
        d["margin"].to_numpy(),
        d["possession"].to_numpy(),
        d["home_fouls"].to_numpy(),
        d["away_fouls"].to_numpy(),
        E.ft_bucket(d["home_ft_pct"].to_numpy(), means),
        E.ft_bucket(d["away_ft_pct"].to_numpy(), means),
    )
    y = d["home_win"].to_numpy().astype(float)

    result = {
        "season": a.season,
        "table": a.table,
        "table_seasons": manifest["seasons_used"],
        "window_seconds": a.seconds,
        "table_alone": metrics(p, y),
        "espn_same_rows": metrics(np.clip(d["espn_wp"].to_numpy(), EPS, 1 - EPS), y),
        "reliability": reliability(p, y),
    }
    print(json.dumps(result, indent=2))
    outp = ROOT / "reports" / f"endgame_validation_{a.season}.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

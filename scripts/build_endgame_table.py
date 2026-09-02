"""Solve the endgame exactly and publish the lookup table.

Endgame plan, Phase 3. Reads the Phase 2 measurements, runs the backward
induction in src/cbbwp/endgame_sim.py, checks every monotonicity property the
plan named -- exhaustively, across the whole table, not sampled -- and writes a
versioned artifact next to the model.

Also writes a small human-readable CSV of canonical states. The plan's third
argument for a table over a live simulation was that a person can read a row and
check it against their own judgement; that only holds if someone actually
prints the rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp import endgame_sim as E  # noqa: E402
from cbbwp.schemas import STATE_RULES_VERSION  # noqa: E402

RAW = ROOT / "data" / "raw" / "pbp"
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]


def ft_bucket_offsets(seasons=TRAIN_SEASONS) -> tuple[list[float], list[float]]:
    """Terciles of team season free-throw percentage, as offsets from the mean.

    The table needs free-throw ability as a small number of buckets. Rather than
    pick cut points, take the terciles the sport actually has and carry the mean
    of each as an offset. Measured on training seasons only.
    """
    rows = []
    for s in seasons:
        d = (
            pl.scan_parquet(RAW / f"pbp_{s}.parquet")
            .select(["team_id", "type_id", "scoring_play"])
            .filter(pl.col("type_id").cast(pl.Int64) == 540)
            .group_by("team_id")
            .agg([pl.len().alias("n"), pl.col("scoring_play").mean().alias("p")])
            .filter(pl.col("n") >= 200)
            .collect()
        )
        rows.append(d)
    d = pl.concat(rows)
    p = d["p"].to_numpy()
    lo, hi = np.quantile(p, [1 / 3, 2 / 3])
    means = [float(p[p <= lo].mean()), float(p[(p > lo) & (p <= hi)].mean()), float(p[p > hi].mean())]
    overall = float(p.mean())
    return [m - overall for m in means], means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="e1")
    ap.add_argument("--params", default=None)
    ap.add_argument("--poss", default=None)
    ap.add_argument("--seasons", nargs="*", type=int, default=None,
                    help="seasons the parameters came from; recorded in the manifest")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seasons = a.seasons or TRAIN_SEASONS
    offsets, bucket_means = ft_bucket_offsets(seasons)
    print(f"free-throw buckets (team season FT%): {[round(x,4) for x in bucket_means]}")
    print(f"offsets from mean:                   {[round(x,4) for x in offsets]}")

    params = E.load_params(
        Path(a.params) if a.params else ROOT / "artifacts" / "endgame_params.json",
        Path(a.poss) if a.poss else ROOT / "artifacts" / "endgame_possessions.json",
        offsets,
    )
    t0 = time.time()
    V = E.solve(params)
    print(f"solved {V.size:,} states in {time.time()-t0:.1f}s")

    report = E.monotonicity_report(V)
    print(json.dumps(report, indent=2))

    V2, moved = E.enforce_margin_monotonicity(V)
    print(f"isotonic projection along margin moved at most {moved:.2e}")
    report_after = E.monotonicity_report(V2)

    out = Path(a.out) if a.out else ROOT / "registry" / "endgame" / a.version
    out.mkdir(parents=True, exist_ok=True)
    table = V2.astype(np.float32)
    np.savez_compressed(out / "table.npz", table=table)

    sha = hashlib.sha256((out / "table.npz").read_bytes()).hexdigest()[:16]
    manifest = {
        "version": a.version,
        "state_rules_version": STATE_RULES_VERSION,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256_16": sha,
        "seasons_used": seasons,
        "shape": list(table.shape),
        "axes": ["seconds_remaining", "margin", "own_fouls", "opp_fouls",
                 "own_ft_bucket", "opp_ft_bucket"],
        "margin_range": [E.MARGIN_MIN, E.MARGIN_MAX],
        "foul_max": E.FOUL_MAX,
        "ft_bucket_means": bucket_means,
        "monotonicity_before": report,
        "monotonicity_after": report_after,
        "isotonic_max_correction": moved,
        "note": "V[t, m, fo, fd, bo, bd] = P(the team WITH THE BALL wins).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # A readable slice: no bonus differential, average free-throw teams.
    mid = E.N_FT_BUCKETS // 2
    rows = []
    for t in (0, 5, 10, 15, 20, 30, 45, 60):
        for m in range(-6, 7):
            for fd, label in ((6, "none"), (8, "1-and-1"), (10, "double")):
                rows.append({
                    "seconds_left": t, "margin": m, "opp_bonus": label,
                    "p_win_with_ball": round(float(table[t, E._mi(m), 6, fd, mid, mid]), 4),
                })
    pl.DataFrame(rows).write_csv(out / "readable.csv")
    print(f"wrote {out}/table.npz  sha {sha}  ({(out/'table.npz').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

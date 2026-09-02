"""Snapshot today's team ratings and season-to-date stats for the live poller.

Uses the SAME ridge fit as the offline pipeline (cbbwp.ratings._fit_ridge) and
the SAME season-to-date formulas as build_team_stats.py, so a live game's
pregame term is on the same scale the model was fit on.

Run daily before the slate:  python3 scripts/build_live_context.py
"""
import sys, pathlib, json, datetime, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from cbbwp.ratings import _fit_ridge, CARRYOVER

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAGUE_FT, PRIOR_FTA, LEAGUE_PPM = 0.700, 40.0, 3.45

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=None,
                help="season to snapshot (default: the latest in games.parquet)")
ap.add_argument("--out", default=str(ROOT / "registry/context_latest.json"))
a = ap.parse_args()

games = pl.read_parquet(ROOT / "data/proc/games.parquet")
season = a.season or int(games["season"].max())
prev = season - 1 if season - 1 != 2020 else season - 2

# --- 1. ratings: fit this season's completed games, prior = last season's ----
prev_g = games.filter(pl.col("season") == prev)
teams_prev = sorted(set(prev_g["home_id"].to_list()) | set(prev_g["away_id"].to_list()))
prev_ratings, _ = _fit_ridge(prev_g, teams_prev, {})
prior = {t: v * CARRYOVER for t, v in prev_ratings.items()}

cur = games.filter(pl.col("season") == season)
teams = sorted(set(cur["home_id"].to_list()) | set(cur["away_id"].to_list()) | set(prior))
ratings, hca = _fit_ridge(cur, teams, prior)
print(f"season {season}: {cur.height} completed games, {len(teams)} teams, hca={hca:.2f}, "
      f"rating sd={np.std(list(ratings.values())):.2f}")
if cur.height == 0:
    print("  (no games yet - ratings are last season's, carried over. Expected in preseason.)")

# --- 2. season-to-date FT% and points per minute -----------------------------
ft_pct, ppm = {}, {}
pbp = ROOT / f"data/raw/pbp/pbp_{season}.parquet"
if cur.height and pbp.exists():
    ft = (pl.scan_parquet(pbp)
          .select(["game_id", "type_text", "scoring_play", "team_id"])
          .filter(pl.col("type_text").str.contains("FreeThrow"))
          .group_by("team_id").agg(fta=pl.len(), ftm=pl.col("scoring_play").sum())
          .collect())
    for tid, fta, ftm in ft.iter_rows():
        if tid is None:
            continue
        ft_pct[int(tid)] = (ftm + LEAGUE_FT * PRIOR_FTA) / (fta + PRIOR_FTA)

    long = pl.concat([
        cur.select(team_id="home_id", pts="home_score", opp="away_score"),
        cur.select(team_id="away_id", pts="away_score", opp="home_score"),
    ]).group_by("team_id").agg(tot=(pl.col("pts") + pl.col("opp")).sum(), g=pl.len())
    for tid, tot, g in long.iter_rows():
        ppm[int(tid)] = (tot + LEAGUE_PPM * 40 * 5) / ((g + 5) * 40)

# The date of the newest completed game these ratings were fit on. Without it,
# the only freshness signal is when this file was written -- so a nightly job
# running over a stale data copy produces a snapshot that reports itself fresh
# and is not. See LiveContextProvider.data_age_days.
latest_game_date = ""
if cur.height and "date" in cur.columns:
    m = cur["date"].max()
    latest_game_date = m.isoformat() if hasattr(m, "isoformat") else str(m)

out = {
    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "latest_game_date": latest_game_date,
    "season": season,
    "hca": hca,
    "n_completed_games": cur.height,
    "ratings": {str(k): round(v, 4) for k, v in ratings.items()},
    "ft_pct": {str(k): round(v, 4) for k, v in ft_pct.items()},
    "ppm": {str(k): round(v, 4) for k, v in ppm.items()},
}
dest = pathlib.Path(a.out)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out))
print(f"wrote {dest}  ({len(ratings)} ratings, {len(ft_pct)} ft, {len(ppm)} ppm)")
print(f"  newest completed game: {latest_game_date or 'none yet (preseason)'}")

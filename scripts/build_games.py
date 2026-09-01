"""Build the game-level table (results + as-of pregame ratings)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from cbbwp.ratings import build_all_seasons

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

frames = []
for y in SEASONS:
    s = (
        pl.scan_parquet(ROOT / f"data/raw/sched/sched_{y}.parquet")
        .select(["game_id", "date", "neutral_site", "home_id", "away_id",
                 "home_score", "away_score", "status_type_completed"])
        .filter(pl.col("status_type_completed")
                & pl.col("home_score").is_not_null()
                & pl.col("away_score").is_not_null()
                & (pl.col("home_score") != pl.col("away_score")))
        .with_columns(
            season=pl.lit(y, dtype=pl.Int32),
            date=pl.col("date").str.to_datetime("%Y-%m-%dT%H:%MZ", strict=False),
            margin=(pl.col("home_score") - pl.col("away_score")).cast(pl.Int32),
            home_win=(pl.col("home_score") > pl.col("away_score")).cast(pl.Int8),
            neutral_site=pl.col("neutral_site").fill_null(False),
        )
        .filter(pl.col("date").is_not_null())
        .unique(subset=["game_id"])
        .collect()
    )
    frames.append(s)
    print(y, s.height)

games = pl.concat(frames, how="vertical_relaxed")
print("total games", games.height)
games = build_all_seasons(games)
games.write_parquet(ROOT / "data/proc/games.parquet")
print("wrote games.parquet", games.height)

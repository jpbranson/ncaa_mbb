"""Replay every game into state rows and write the modelling dataset."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from cbbwp.adapters.hoopr import states_lazy, BASE_COLS
from cbbwp.features import feature_exprs
from cbbwp.schemas import FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
LATE_SECONDS = 300       # keep every state inside the final 5 minutes
EARLY_KEEP_EVERY = 3     # ...and every third state before that (plan 5.5)

games = (pl.read_parquet(ROOT / "data/proc/games.parquet")
         .select("game_id", "season", "home_win", "neutral_site",
                 "pregame_exp_margin", "date", "home_score", "away_score")
         .join(pl.read_parquet(ROOT / "data/proc/team_stats.parquet"),
               on="game_id", how="left"))
out_dir = ROOT / "data/proc/states"
out_dir.mkdir(parents=True, exist_ok=True)

for y in SEASONS:
    t0 = time.time()
    src = ROOT / f"data/raw/pbp/pbp_{y}.parquet"
    cols = set(pl.scan_parquet(src).collect_schema().names())
    sel = list(BASE_COLS) + (["home_win_prob"] if "home_win_prob" in cols else [])
    lf = (pl.scan_parquet(src).select(sel)
          .drop(["season", "game_date"])            # season/date come from games.parquet
          .filter(pl.col("period_number") <= 8))

    st = states_lazy(lf).join(games.lazy(), on="game_id", how="inner")
    st = st.filter(
        pl.col("game_seconds_remaining").is_not_null()
        & (pl.col("margin").abs() < 100)
    )
    # Data-quality gate: ~0.2% of games have a truncated or contradictory feed.
    # A live system would flag these too, so they are excluded everywhere.
    final = (st.group_by("game_id")
             .agg(pl.col("margin").sort_by("seq").last().alias("_last_margin"),
                  pl.col("game_seconds_remaining").sort_by("seq").last().alias("_last_secs"),
                  pl.col("home_win").first().alias("_hw")))
    good = final.filter((pl.col("_last_secs") <= 60)
                        & ((pl.col("_last_margin") > 0).cast(pl.Int8) == pl.col("_hw"))
                        ).select("game_id")
    st = st.join(good, on="game_id", how="inner")
    # Sampling keeps every late-game state and thins the redundant early ones.
    st = st.with_columns(
        _keep=(pl.col("game_seconds_remaining") <= LATE_SECONDS)
        | (pl.col("seq") % EARLY_KEEP_EVERY == 0)
    ).filter(pl.col("_keep"))

    # `margin`, `possession`, `is_ot`, `pregame_exp_margin` come back as features,
    # so they are not repeated here.
    keep = ["game_id", "season", "seq", "period", "game_seconds_remaining",
            "home_timeouts", "away_timeouts", "home_fouls", "away_fouls",
            "neutral_site", "home_win", "date"]
    if "home_win_prob" in cols:
        st = st.rename({"home_win_prob": "espn_wp"})
        keep.append("espn_wp")

    df = st.select(keep + feature_exprs()).collect()
    df.write_parquet(out_dir / f"states_{y}.parquet")
    print(f"{y}: {df.height:>9,} rows  {df['game_id'].n_unique():>5,} games  "
          f"{time.time()-t0:5.1f}s")

"""Season-to-date team stats, as of the day BEFORE each game (no leakage).

Produces per game: ft_pct_diff (home minus away) and exp_points_per_min
(the two teams' combined scoring rate), both from games already played.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
LEAGUE_FT = 0.700          # prior for teams with no games yet
PRIOR_FTA = 40.0           # strength of that prior, in attempts
LEAGUE_PPM = 3.45          # points per minute, both teams combined

games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
    "game_id", "season", "date", "home_id", "away_id", "home_score", "away_score")

per_team = []
for y in SEASONS:
    ft = (
        pl.scan_parquet(ROOT / f"data/raw/pbp/pbp_{y}.parquet")
        .select(["game_id", "type_text", "scoring_play", "team_id"])
        .filter(pl.col("type_text").str.contains("FreeThrow"))
        .group_by(["game_id", "team_id"])
        .agg(fta=pl.len(), ftm=pl.col("scoring_play").sum())
        .collect()
    )
    per_team.append(ft.with_columns(season=pl.lit(y, dtype=pl.Int32)))
ft = pl.concat(per_team)

# long form: one row per (game, team) with that team's FT and points
long = pl.concat([
    games.select("game_id", "season", "date", team_id=pl.col("home_id"),
                 pts=pl.col("home_score"), opp_pts=pl.col("away_score")),
    games.select("game_id", "season", "date", team_id=pl.col("away_id"),
                 pts=pl.col("away_score"), opp_pts=pl.col("home_score")),
]).join(ft.select("game_id", "team_id", "fta", "ftm"), on=["game_id", "team_id"], how="left")

long = long.sort(["team_id", "season", "date"]).with_columns(
    # cum_sum shifted by one game = everything BEFORE this game
    c_fta=pl.col("fta").fill_null(0).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_ftm=pl.col("ftm").fill_null(0).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_pts=(pl.col("pts") + pl.col("opp_pts")).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_g=pl.int_range(pl.len()).over(["team_id", "season"]),
).with_columns(
    ft_pct=(pl.col("c_ftm") + LEAGUE_FT * PRIOR_FTA) / (pl.col("c_fta") + PRIOR_FTA),
    ppm=(pl.col("c_pts") + LEAGUE_PPM * 40 * 5) / ((pl.col("c_g") + 5) * 40),
)

h = long.select("game_id", h_ft="ft_pct", h_ppm="ppm", team_id="team_id")
out = (
    games
    .join(h.rename({"team_id": "home_id"}), on=["game_id", "home_id"], how="left")
    .join(h.rename({"team_id": "away_id", "h_ft": "a_ft", "h_ppm": "a_ppm"}),
          on=["game_id", "away_id"], how="left")
    .select(
        "game_id",
        ft_pct_diff=(pl.col("h_ft") - pl.col("a_ft")).fill_null(0.0),
        exp_points_per_min=((pl.col("h_ppm") + pl.col("a_ppm")) / 2).fill_null(LEAGUE_PPM),
        # The two levels, not just their difference. The endgame table buckets
        # each team's free-throw ability separately, and reconstructing the
        # levels from the difference is not possible. Keeping only the diff here
        # was what would have forced the endgame validation to assume both teams
        # were average -- an assumption the live path would NOT have made, which
        # is exactly the train/serve mismatch this project keeps tripping over.
        home_ft_pct=pl.col("h_ft").fill_null(LEAGUE_FT),
        away_ft_pct=pl.col("a_ft").fill_null(LEAGUE_FT),
    )
)
out.write_parquet(ROOT / "data/proc/team_stats.parquet")
print(out.describe())

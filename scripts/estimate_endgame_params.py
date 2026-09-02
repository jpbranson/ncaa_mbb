"""Estimate every parameter the endgame simulator needs, from play-by-play.

Endgame plan, Phase 2: "No hand-set constants." Every number the simulator uses
is measured here, on TRAINING seasons only (2016-2024). 2025 and 2026 are the
held-out test seasons and this script refuses to read them -- the pre-registered
bar in docs/cbbwp-endgame-plan.md is a single-shot test and stays that way.

Three things this file deliberately does NOT do:

1. It does not key on ESPN's "free throw N of M" text. That text appears only in
   the 2026 feed, so using it would be both unavailable for training seasons and
   a test-set leak. Trips are classified by the OPPONENT'S TEAM FOUL COUNT
   instead -- the rule the sport actually uses, which reproduces the 2026 text
   labels 94.9% of the time.

2. It does not read `scoring_play` on a play labelled "1 of 1" and call the
   result a free-throw percentage. ESPN labels a trip by the attempts actually
   TAKEN, so a MADE one-and-one front end is logged as a two-shot trip and never
   appears as "1 of 1" at all. Conditioning on that label conditions on the
   outcome, which is why the raw figure is 0.537 and the true one is 0.70.
   Trips are rebuilt first; only then is the first shot of each trip read.

3. It does not hold every season in memory at once. hoopR play-by-play is large
   and the device VM is small, so each season is reduced to COUNTS
   (n, successes) and the counts are summed. Means are formed at the end, from
   the totals. This also makes each season's contribution auditable.

Run:
    python3 scripts/estimate_endgame_params.py --season 2016   # ... one at a time
    python3 scripts/estimate_endgame_params.py --combine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbp"
PART = ROOT / "artifacts" / "endgame_parts"
OUT = ROOT / "artifacts" / "endgame_params.json"

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
TEST_SEASONS = {2025, 2026}

FT, FOUL, TECH = 540, 519, 521
OREB, DREB = 586, 587
SHOT_TYPES = [558, 572, 574, 437]
TURNOVER_TYPES = [598, 601, 602]
BONUS_FOULS, DOUBLE_BONUS_FOULS = 7, 10

ENDGAME_SECONDS = 60
WIDE_SECONDS = 180

COLS = [
    "game_id", "sequence_number", "type_id", "text", "scoring_play", "score_value",
    "team_id", "home_team_id", "away_team_id", "athlete_id_1",
    "period_number", "clock_minutes", "clock_seconds",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(season: int) -> pl.DataFrame:
    """One season, ordered, with a season-independent game clock.

    hoopR's seconds-remaining column is named `start_half_seconds_remaining` in
    2016-2022 and `start_period_seconds_remaining` in 2023+. The raw clock
    fields are present in every season, so the clock is rebuilt from those
    rather than picking one of the two names and silently failing on the other.
    """
    df = (
        pl.scan_parquet(RAW / f"pbp_{season}.parquet")
        .select(COLS)
        .collect()
        .with_columns(pl.col("sequence_number").cast(pl.Int64))
        .sort(["game_id", "sequence_number"])
        .with_row_index("i")
    )
    return df.with_columns(
        [
            pl.col("type_id").cast(pl.Int64).alias("tid"),
            pl.when(pl.col("period_number") <= 1).then(1).otherwise(2).alias("half"),
            (
                pl.col("clock_minutes").cast(pl.Float64).fill_null(0) * 60
                + pl.col("clock_seconds").cast(pl.Float64).fill_null(0)
            ).alias("sec"),
        ]
    )


def annotate(df: pl.DataFrame) -> pl.DataFrame:
    """Team fouls, the and-one flag, free-throw trip ids, and elapsed time."""
    df = df.with_columns(
        [
            (pl.col("tid").is_in([FOUL, TECH]) & (pl.col("team_id") == pl.col("home_team_id"))).alias("hf"),
            (pl.col("tid").is_in([FOUL, TECH]) & (pl.col("team_id") == pl.col("away_team_id"))).alias("af"),
            (pl.col("tid") == FT).alias("isft"),
            (pl.col("scoring_play") & (pl.col("tid") != FT)).alias("madefg"),
            pl.col("text").str.contains("(?i)three point").fill_null(False).alias("is3"),
        ]
    )
    # Team fouls reset once, at the half. Counted BEFORE the current row, so the
    # count describes the situation the current event was played under -- which
    # is what decides whether a foul sends the shooter to a one-and-one.
    df = df.with_columns(
        [
            (pl.col("hf").cum_sum().over(["game_id", "half"]) - pl.col("hf").cast(pl.Int32)).alias("home_fouls"),
            (pl.col("af").cum_sum().over(["game_id", "half"]) - pl.col("af").cast(pl.Int32)).alias("away_fouls"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id"))
        .then(pl.col("away_fouls"))
        .otherwise(pl.col("home_fouls"))
        .alias("opp_fouls")
    )

    # And-one: the shooter made a field goal at effectively the same game clock.
    # Substitutions and TV timeouts pad the gap, so the window is events, not
    # rows adjacent -- 12 is comfortably past the longest observed run of subs.
    andone = None
    for k in range(1, 13):
        t = (
            pl.col("madefg").shift(k).over("game_id")
            & (pl.col("athlete_id_1").shift(k).over("game_id") == pl.col("athlete_id_1"))
            & ((pl.col("sec").shift(k).over("game_id") - pl.col("sec")).abs() <= 3)
        )
        andone = t if andone is None else (andone | t)
    df = df.with_columns(andone.fill_null(False).alias("andone"))

    # A trip is a run of free throws by one team at one dead ball.
    ft = pl.col("isft")
    prev_i = pl.when(ft).then(pl.col("i")).otherwise(None).forward_fill().over("game_id").shift(1)
    prev_team = pl.when(ft).then(pl.col("team_id")).otherwise(None).forward_fill().over("game_id").shift(1)
    prev_sec = pl.when(ft).then(pl.col("sec")).otherwise(None).forward_fill().over("game_id").shift(1)
    new_trip = ft & (
        prev_i.is_null()
        | (pl.col("team_id") != prev_team)
        | ((prev_sec - pl.col("sec")).abs() > 4)
        | ((pl.col("i") - prev_i) > 8)
    )
    df = df.with_columns(
        pl.when(ft).then(new_trip.fill_null(True).cast(pl.Int32).cum_sum().over("game_id")).alias("trip")
    )

    # Seconds until the clock next moves. The next ROW is usually a substitution
    # at the same dead ball, so a plain shift(-1) measures nothing and returns a
    # median of 0. Take the first following event whose clock is strictly lower.
    nxt = None
    for k in range(1, 13):
        s = pl.col("sec").shift(-k).over("game_id")
        cand = pl.when(s < pl.col("sec")).then(s).otherwise(None)
        nxt = cand if nxt is None else pl.coalesce([nxt, cand])
    df = df.with_columns((pl.col("sec") - nxt).alias("dt_clock"))

    # Seconds until the next foul, however many dead-ball rows intervene.
    nf = None
    for k in range(1, 25):
        s = pl.col("sec").shift(-k).over("game_id")
        isf = pl.col("tid").shift(-k).over("game_id").is_in([FOUL, TECH])
        cand = pl.when(isf & (s <= pl.col("sec"))).then(s).otherwise(None)
        nf = cand if nf is None else pl.coalesce([nf, cand])
    return df.with_columns((pl.col("sec") - nf).alias("dt_foul"))


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------
def _nk(frame: pl.DataFrame, col: str = "scoring_play") -> dict:
    return {"n": int(len(frame)), "k": int(frame[col].sum()) if len(frame) else 0}


def trip_kind() -> pl.Expr:
    return (
        pl.when(pl.col("andone")).then(pl.lit("and_one"))
        .when(pl.col("n_shots") >= 3).then(pl.lit("shooting_3"))
        .when((pl.col("opp_fouls") >= BONUS_FOULS) & (pl.col("opp_fouls") < DOUBLE_BONUS_FOULS))
        .then(pl.lit("one_and_one"))
        .otherwise(pl.lit("two_shot"))
    )


def count_season(df: pl.DataFrame) -> dict:
    out: dict = {}

    # ---- free throws, by trip kind and shot index -------------------------
    ft = df.filter(pl.col("isft")).with_columns(
        pl.col("i").rank("ordinal").over(["game_id", "trip"]).alias("shot_no")
    )
    sizes = ft.group_by(["game_id", "trip"]).agg(pl.len().alias("n_shots"))
    ft = ft.join(sizes, on=["game_id", "trip"], how="left").with_columns(trip_kind().alias("kind"))

    windows = {
        "all_game": pl.lit(True),
        "last_180s": (pl.col("period_number") >= 2) & (pl.col("sec") <= WIDE_SECONDS),
        "last_60s": (pl.col("period_number") >= 2) & (pl.col("sec") <= ENDGAME_SECONDS),
    }
    out["free_throws"] = {}
    for wname, w in windows.items():
        g = (
            ft.filter(w)
            .group_by(["kind", "shot_no"])
            .agg([pl.len().alias("n"), pl.col("scoring_play").sum().alias("k")])
        )
        out["free_throws"][wname] = {
            f"{r['kind']}_{r['shot_no']}": {"n": int(r["n"]), "k": int(r["k"])}
            for r in g.iter_rows(named=True)
        }

    # ---- how often each kind of trip happens late -------------------------
    trips = (
        ft.group_by(["game_id", "trip"])
        .agg([pl.first("kind").alias("kind"), pl.first("sec").alias("sec"),
              pl.first("period_number").alias("period")])
        .filter((pl.col("period") >= 2) & (pl.col("sec") <= ENDGAME_SECONDS))
    )
    out["late_trip_mix"] = {
        r["kind"]: int(r["n"])
        for r in trips.group_by("kind").agg(pl.len().alias("n")).iter_rows(named=True)
    }

    # ---- rebounding -------------------------------------------------------
    reb = None
    for k in range(1, 4):
        t = pl.col("tid").shift(-k).over("game_id")
        cand = pl.when(t.is_in([OREB, DREB])).then(t).otherwise(None)
        reb = cand if reb is None else pl.coalesce([reb, cand])
    d = df.with_columns(reb.alias("reb")).with_columns((pl.col("reb") == OREB).alias("is_oreb"))

    last_ft = (
        d.filter(pl.col("isft"))
        .with_columns(pl.col("i").rank("ordinal").over(["game_id", "trip"]).alias("shot_no"))
        .join(
            d.filter(pl.col("isft"))
            .group_by(["game_id", "trip"])
            .agg(pl.len().alias("n_shots")),
            on=["game_id", "trip"], how="left",
        )
        .filter(pl.col("shot_no") == pl.col("n_shots"))
        .filter(~pl.col("scoring_play") & pl.col("reb").is_not_null())
    )
    miss_fg = d.filter(pl.col("tid").is_in(SHOT_TYPES) & ~pl.col("scoring_play") & pl.col("reb").is_not_null())
    late = (pl.col("period_number") >= 2) & (pl.col("sec") <= ENDGAME_SECONDS)
    out["rebounds"] = {
        "oreb_after_missed_ft": _nk(last_ft, "is_oreb"),
        "oreb_after_missed_ft_last60": _nk(last_ft.filter(late), "is_oreb"),
        "oreb_after_missed_3": _nk(miss_fg.filter(pl.col("is3")), "is_oreb"),
        "oreb_after_missed_2": _nk(miss_fg.filter(~pl.col("is3")), "is_oreb"),
        "oreb_after_missed_3_last60": _nk(miss_fg.filter(pl.col("is3") & late), "is_oreb"),
        "oreb_after_missed_2_last60": _nk(miss_fg.filter(~pl.col("is3") & late), "is_oreb"),
    }

    # ---- shot mix and accuracy late, by the shooter's own margin ----------
    d = d.with_columns(
        [
            pl.when(pl.col("scoring_play") & (pl.col("team_id") == pl.col("home_team_id")))
            .then(pl.col("score_value")).otherwise(0).alias("hs"),
            pl.when(pl.col("scoring_play") & (pl.col("team_id") == pl.col("away_team_id")))
            .then(pl.col("score_value")).otherwise(0).alias("as_"),
        ]
    )
    d = d.with_columns(
        [
            (pl.col("hs").cum_sum().over("game_id") - pl.col("hs")).alias("h"),
            (pl.col("as_").cum_sum().over("game_id") - pl.col("as_")).alias("a"),
        ]
    ).with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id"))
        .then(pl.col("h") - pl.col("a"))
        .otherwise(pl.col("a") - pl.col("h"))
        .alias("actor_margin")
    )
    shots = d.filter(pl.col("tid").is_in(SHOT_TYPES) & late).with_columns(
        pl.col("actor_margin").clip(-6, 6).alias("m")
    )
    g = shots.group_by("m").agg(
        [
            pl.len().alias("n"),
            pl.col("is3").sum().alias("n3"),
            (pl.col("scoring_play") & pl.col("is3")).sum().alias("k3"),
            (pl.col("scoring_play") & ~pl.col("is3")).sum().alias("k2"),
        ]
    )
    out["late_shots"] = {
        str(r["m"]): {"n": int(r["n"]), "n3": int(r["n3"]), "k3": int(r["k3"]), "k2": int(r["k2"])}
        for r in g.iter_rows(named=True)
    }

    # ---- clock ------------------------------------------------------------
    ends = d.filter(late & pl.col("tid").is_in(SHOT_TYPES + TURNOVER_TYPES))
    out["late_possession_ends"] = {
        "n": int(len(ends)),
        "k_turnover": int(ends["tid"].is_in(TURNOVER_TYPES).sum()) if len(ends) else 0,
    }
    made_ft = d.filter(pl.col("isft") & pl.col("scoring_play") & late)
    dtf = made_ft.filter(pl.col("dt_foul").is_between(0, 20))["dt_foul"]
    dtc = made_ft.filter(pl.col("dt_clock").is_between(0, 30))["dt_clock"]
    out["clock"] = {
        "sec_made_ft_to_next_foul": {
            "n": int(len(dtf)),
            "sum": float(dtf.sum()) if len(dtf) else 0.0,
            "hist": {str(int(b)): int(c) for b, c in
                     zip(*[x.to_list() for x in dtf.cast(pl.Int32).value_counts().sort("dt_foul")])}
            if len(dtf) else {},
        },
        "sec_made_ft_to_clock_move": {
            "n": int(len(dtc)),
            "sum": float(dtc.sum()) if len(dtc) else 0.0,
        },
    }
    return out


# --------------------------------------------------------------------------
# combining
# --------------------------------------------------------------------------
def _merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge(a.get(k), b.get(k)) for k in set(a) | set(b)}
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def combine(parts: list[dict]) -> dict:
    tot: dict = {}
    for p in parts:
        body = {k: v for k, v in p.items() if k != "season"}
        tot = body if not tot else _merge(tot, body)

    def rate(d):
        return {k: {"p": (v["k"] / v["n"]) if v["n"] else None, "n": v["n"]} for k, v in d.items()}

    mix_total = sum(tot["late_trip_mix"].values()) or 1
    late_shots = {
        m: {
            "n": v["n"],
            "p3a": v["n3"] / v["n"] if v["n"] else None,
            "p3": v["k3"] / v["n3"] if v["n3"] else None,
            "p2": v["k2"] / (v["n"] - v["n3"]) if (v["n"] - v["n3"]) else None,
        }
        for m, v in sorted(tot["late_shots"].items(), key=lambda kv: int(kv[0]))
    }
    ends = tot["late_possession_ends"]
    clk = tot["clock"]
    return {
        "seasons": sorted(p["season"] for p in parts),
        "free_throws": {w: rate(d) for w, d in tot["free_throws"].items()},
        "late_trip_mix": {k: v / mix_total for k, v in tot["late_trip_mix"].items()},
        "late_trip_n": mix_total,
        "rebounds": rate(tot["rebounds"]),
        "late_shots_by_actor_margin": late_shots,
        "late_turnover_share": ends["k_turnover"] / ends["n"] if ends["n"] else None,
        "late_possession_ends_n": ends["n"],
        "clock": {
            "mean_sec_made_ft_to_next_foul": clk["sec_made_ft_to_next_foul"]["sum"]
            / (clk["sec_made_ft_to_next_foul"]["n"] or 1),
            "n_ft_to_foul": clk["sec_made_ft_to_next_foul"]["n"],
            "hist_sec_ft_to_foul": dict(sorted(clk["sec_made_ft_to_next_foul"]["hist"].items(),
                                               key=lambda kv: int(kv[0]))),
            "mean_sec_made_ft_to_clock_move": clk["sec_made_ft_to_clock_move"]["sum"]
            / (clk["sec_made_ft_to_clock_move"]["n"] or 1),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="combine just these seasons -- used to hold a season out for Phase 4")
    a = ap.parse_args()
    PART.mkdir(parents=True, exist_ok=True)

    if a.season is not None:
        if a.season in TEST_SEASONS:
            raise SystemExit(f"refusing season {a.season}: 2025-2026 are held out for the single-shot test")
        res = count_season(annotate(load(a.season)))
        res["season"] = a.season
        (PART / f"{a.season}.json").write_text(json.dumps(res))
        print(f"wrote {PART / f'{a.season}.json'}")
        return

    if a.combine:
        parts = [json.loads(p.read_text()) for p in sorted(PART.glob("*.json"))]
        want = set(a.only) if a.only else set(TRAIN_SEASONS)
        parts = [p for p in parts if p["season"] in want]
        missing = want - {p["season"] for p in parts}
        if missing:
            raise SystemExit(f"missing seasons: {sorted(missing)}")
        dest = Path(a.out) if a.out else OUT
        dest.write_text(json.dumps(combine(parts), indent=2))
        print(f"wrote {dest} from {len(parts)} seasons: {sorted(want)}")
        return

    ap.error("pass --season N or --combine")


if __name__ == "__main__":
    main()

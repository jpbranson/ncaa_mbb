"""Possession-level endgame behaviour: what ends a possession, and how long it takes.

The free-throw and shooting parameters (estimate_endgame_params.py) say what
happens when a possession resolves. This says HOW a possession resolves -- above
all, whether the defence intentionally fouls, which is the single decision that
makes an endgame an endgame.

The simulator has to reproduce what teams ACTUALLY do, not what they should do,
because it is predicting real games. So the fouling rule is measured, not
optimised.

Possession runs are taken from data/proc/states (the validated possession rules,
including the 2026-09-01 made-three fix) joined back to the raw feed for the
event type that ended each run. Training seasons only.

Output: artifacts/endgame_possessions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "data" / "proc" / "states"
RAW = ROOT / "data" / "raw" / "pbp"
PART = ROOT / "artifacts" / "endgame_poss_parts"
OUT = ROOT / "artifacts" / "endgame_possessions.json"

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
TEST_SEASONS = {2025, 2026}

FT, FOUL, TECH = 540, 519, 521
SHOT_TYPES = [558, 572, 574, 437]
TURNOVER_TYPES = [598, 601, 602]
WINDOW = 90  # seconds; the simulator owns 60, the extra 30 is for the blend


def runs_for_season(season: int) -> pl.DataFrame:
    st = (
        pl.scan_parquet(STATES / f"states_{season}.parquet")
        .select(["game_id", "seq", "possession", "margin", "game_seconds_remaining", "period"])
        .filter((pl.col("period") >= 2) & (pl.col("game_seconds_remaining") <= WINDOW))
        .collect()
    )
    pbp = (
        pl.scan_parquet(RAW / f"pbp_{season}.parquet")
        .select(["game_id", "game_play_number", "type_id", "scoring_play", "text"])
        .collect()
        .rename({"game_play_number": "seq", "type_id": "tid"})
        .with_columns(pl.col("tid").cast(pl.Int64))
    )
    d = st.join(pbp, on=["game_id", "seq"], how="inner").sort(["game_id", "seq"])

    # A run is a maximal stretch during which ONE team had the ball. 0.5 (jump
    # ball, genuinely unknown) is carried forward rather than treated as its own
    # run.
    d = d.with_columns(pl.col("possession").replace(0.5, None).forward_fill().over("game_id"))

    # Segment on the possession DURING each event, which is the state the
    # PREVIOUS event left behind -- not `possession`, which is the state AFTER
    # this one.
    #
    # This is the difference between measuring endgame fouling and measuring its
    # mirror image. `state._possession_after` flips possession on every MADE free
    # throw, so segmenting on the after-value cut each made free-throw trip out
    # of the fouled team's run and pasted it onto the head of the FOULING team's
    # next run. `fouled_to_line` below - the single most important parameter in
    # the simulator, the one that makes an endgame an endgame - was therefore
    # recorded against the team that committed the foul rather than the team that
    # shot the free throws.
    #
    # The result was inverted and obviously so, once looked at: in the last ten
    # seconds it had a TRAILING offence reaching the line 71-84% of the time and
    # a LEADING offence 10-18%, when in real basketball it is the trailing
    # defence that fouls the leading ball-handler. Fixed 2026-09-08.
    d = d.with_columns(
        pl.col("possession").shift(1).over("game_id").alias("poss_during")
    )
    # An unknown possession at the start of the window is not guessed at. These
    # rows used to fall through the `otherwise` branch below and be recorded
    # silently as the away team's.
    d = d.filter(pl.col("poss_during").is_not_null())
    d = d.with_columns(
        (pl.col("poss_during") != pl.col("poss_during").shift(1).over("game_id"))
        .fill_null(True).cast(pl.Int32).cum_sum().over("game_id").alias("run")
    )
    g = d.group_by(["game_id", "run"]).agg(
        [
            pl.first("poss_during").alias("off_is_home"),
            pl.first("margin").alias("margin_start"),
            pl.first("game_seconds_remaining").alias("t_start"),
            pl.last("game_seconds_remaining").alias("t_end"),
            pl.last("tid").alias("end_tid"),
            pl.col("tid").is_in([FOUL, TECH]).any().alias("saw_foul"),
            pl.col("tid").eq(FT).any().alias("saw_ft"),
            pl.col("tid").is_in(TURNOVER_TYPES).any().alias("saw_to"),
            (pl.col("scoring_play") & pl.col("tid").is_in(SHOT_TYPES)).any().alias("saw_made_fg"),
            pl.len().alias("n_events"),
        ]
    )
    # Margin and fouling are described from the OFFENSE's point of view: the
    # simulator asks "the team with the ball leads by k -- do they get fouled?"
    return g.with_columns(
        [
            pl.when(pl.col("off_is_home") == 1.0)
            .then(pl.col("margin_start"))
            .otherwise(-pl.col("margin_start"))
            .alias("off_margin"),
            (pl.col("t_start") - pl.col("t_end")).alias("dur"),
        ]
    ).filter(pl.col("dur").is_between(0, 45))


def count_season(season: int) -> dict:
    r = runs_for_season(season)
    r = r.with_columns(
        [
            pl.col("off_margin").clip(-8, 8).cast(pl.Int32).alias("m"),
            (pl.col("t_start") // 10 * 10).clip(0, 80).alias("tb"),
            # An intentional foul: the possession reached the line without the
            # offence taking a shot. That is what "they fouled to stop the clock"
            # looks like in a feed that does not label intent.
            (pl.col("saw_ft") & ~pl.col("saw_made_fg")).alias("fouled_to_line"),
        ]
    )
    g = r.group_by(["m", "tb"]).agg(
        [
            pl.len().alias("n"),
            pl.col("fouled_to_line").sum().alias("k_foul"),
            pl.col("saw_to").sum().alias("k_to"),
            pl.col("saw_made_fg").sum().alias("k_made_fg"),
            pl.col("dur").sum().alias("dur_sum"),
            pl.col("dur").filter(pl.col("fouled_to_line")).sum().alias("dur_foul_sum"),
            pl.col("fouled_to_line").sum().alias("dur_foul_n"),
        ]
    )
    return {
        "season": season,
        "cells": {
            f"{r_['m']}|{r_['tb']}": {k: int(r_[k]) for k in
                                      ("n", "k_foul", "k_to", "k_made_fg", "dur_sum",
                                       "dur_foul_sum", "dur_foul_n")}
            for r_ in g.iter_rows(named=True)
        },
    }


def combine(parts: list[dict]) -> dict:
    tot: dict = {}
    for p in parts:
        for key, v in p["cells"].items():
            acc = tot.setdefault(key, {k: 0 for k in v})
            for k, x in v.items():
                acc[k] += x
    cells = {}
    for key, v in sorted(tot.items(), key=lambda kv: (float(kv[0].split("|")[0]), float(kv[0].split("|")[1]))):
        if v["n"] < 30:
            continue
        cells[key] = {
            "n": v["n"],
            "p_fouled_to_line": v["k_foul"] / v["n"],
            "p_turnover": v["k_to"] / v["n"],
            "p_made_fg": v["k_made_fg"] / v["n"],
            "mean_dur": v["dur_sum"] / v["n"],
            "mean_dur_when_fouled": (v["dur_foul_sum"] / v["dur_foul_n"]) if v["dur_foul_n"] else None,
        }
    return {"seasons": sorted(p["season"] for p in parts), "cells": cells,
            "key": "off_margin|seconds_remaining_bucket (offence's point of view)"}


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
            raise SystemExit(f"refusing season {a.season}: held out for the single-shot test")
        (PART / f"{a.season}.json").write_text(json.dumps(count_season(a.season)))
        print("wrote", a.season)
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
        print("wrote", dest)
        return
    ap.error("pass --season N or --combine")


if __name__ == "__main__":
    main()

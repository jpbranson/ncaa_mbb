"""Build ESPN-shaped summary payloads out of hoopR rows.

hoopR IS a scrape of the ESPN feed, so a payload rebuilt from hoopR columns has
the same shape and the same values the live endpoint returns. That lets the
ESPN adapter be tested for train/serve parity with no network at all.

What this proves: the adapter's play ordering, seq numbering, type-id mapping,
field coercion and header parsing all reproduce the offline path exactly.
What it does NOT prove: that ESPN's live payload has no field this rebuild
omits. For that, record real payloads with scripts/record_espn_fixtures.py on a
machine that can reach ESPN and re-run these same assertions against them.
"""
from __future__ import annotations

import pathlib
import random
from typing import List

import polars as pl

from cbbwp.adapters.espn import SYNTHETIC_KEY
from cbbwp.adapters.hoopr import BASE_COLS

EXTRA_COLS = ["type_id"]


def summary_from_hoopr(pbp_path: str, game_id: int, shuffle_seed: int | None = None,
                       status: str = "STATUS_FINAL") -> dict:
    """One game's hoopR rows -> an ESPN `summary` payload."""
    df = (pl.scan_parquet(pbp_path)
          .filter(pl.col("game_id") == game_id)
          .select(BASE_COLS + EXTRA_COLS)
          .sort("game_play_number")
          .collect())
    if df.is_empty():
        raise KeyError(game_id)
    home_id = int(df["home_team_id"][0])
    away_id = int(df["away_team_id"][0])

    plays: List[dict] = []
    for r in df.iter_rows(named=True):
        team = r["team_id"]
        plays.append({
            # ESPN's sequenceNumber is a big opaque string, and -- measured on
            # real payloads 2026-09-03 -- only NEARLY monotonic: roughly one play
            # in forty carries an id lower than its predecessor while the clock
            # runs correctly forwards. This rebuild reproduces that, because a
            # fixture with a perfectly sorted key is what let a sort-by-id bug
            # survive every parity test for months. Anyone who reintroduces that
            # sort will now fail the parity tests immediately.
            "sequenceNumber": str(int(r["game_play_number"]) * 10
                                  - (15 if int(r["game_play_number"]) % 40 == 0 else 0)),
            "type": {"id": str(r["type_id"]) if r["type_id"] is not None else None,
                     "text": r["type_text"]},
            "period": {"number": int(r["period_number"] or 1)},
            "clock": {"displayValue": r["clock_display_value"] or ""},
            "homeScore": int(r["home_score"] or 0),
            "awayScore": int(r["away_score"] or 0),
            "team": None if team is None else {"id": str(int(team))},
            "scoreValue": int(r["score_value"] or 0),
            "scoringPlay": bool(r["scoring_play"]),
            "shootingPlay": bool(r["shooting_play"]),
            "text": "",
        })

    # Shuffling now produces a genuinely disordered feed: the adapter preserves
    # array order, so this no longer round-trips. Used to test that disorder is
    # REPORTED (espn.chronological_inversions), not silently repaired.
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(plays)

    last = df.tail(1)
    return {
        # Provenance, so a rebuilt payload can never be mistaken for a recording
        # of the real feed. scripts/check_espn_fixtures.py refuses to treat a
        # stamped payload as evidence about what ESPN sends today, because every
        # type id in here came from hoopR - the same source the model's type map
        # was built from - so the check could not fail on one if it tried.
        SYNTHETIC_KEY: "rebuilt from hoopR rows by tests/espn_fixtures.py",
        "header": {
            "id": str(game_id),
            "competitions": [{
                "id": str(game_id),
                "neutralSite": False,
                "status": {"period": int(last["period_number"][0] or 2),
                           "displayClock": "0:00",
                           "type": {"name": status,
                                    "completed": status == "STATUS_FINAL"}},
                "competitors": [
                    {"homeAway": "home", "score": str(int(last["home_score"][0] or 0)),
                     "team": {"id": str(home_id), "displayName": "Home Team"}},
                    {"homeAway": "away", "score": str(int(last["away_score"][0] or 0)),
                     "team": {"id": str(away_id), "displayName": "Away Team"}},
                ],
            }],
        },
        "plays": plays,
    }

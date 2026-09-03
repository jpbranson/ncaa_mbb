"""Archive real ESPN games for the replay server to serve back.

Needs network. Run once; the payloads are then reusable offline forever, which
is the point - a dry run of the live path should not depend on ESPN being up,
or on it being basketball season.

    python3 scripts/archive_replay_games.py                # the default set
    python3 scripts/archive_replay_games.py --date 20260307 --limit 10
    python3 scripts/archive_replay_games.py --game 401808285

The default set is chosen, not sampled. A replay is only worth running if it
puts the model somewhere interesting, and "somewhere interesting" for a win
probability model means: games decided in the last possession, at least one
overtime, and a blowout to check the model does not dither when the answer is
obvious. Blowouts matter as much as thrillers here - a model that hedges at 40
points up is as wrong as one that panics at 1 point up.

Payloads land in `tmp/replay/`, which is gitignored: they are ~500KB each and
reproducible from this script, so they are a build product, not source.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import (EspnClient, parse_summary,  # noqa: E402
                                 scoreboard_games)

# (game_id, why it is in the set). All from the 2025-26 season.
DEFAULT_GAMES = [
    (401856600, "2026 national championship, UConn/Michigan, 6-point game"),
    (401808285, "overtime: Arkansas at Missouri, 3 periods"),
    (401820791, "decided by 1: Stanford at NC State"),
    (401822973, "6-point conference game: UConn at Marquette"),
    (401820788, "rivalry, 15-point margin: UNC at Duke"),
    (401856599, "blowout, 18 points: Michigan vs Arizona (Final Four)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "tmp/replay"))
    ap.add_argument("--game", type=int, action="append",
                    help="archive this game id instead of the default set")
    ap.add_argument("--date", help="archive completed games from this slate")
    ap.add_argument("--limit", type=int, default=10, help="with --date")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = EspnClient()
    if c.is_replay:
        print("refusing to archive from a replay server -- unset CBBWP_ESPN_BASE",
              file=sys.stderr)
        return 1

    if a.game:
        wanted = [(g, "requested") for g in a.game]
    elif a.date:
        games = [g for g in scoreboard_games(c.scoreboard(a.date))
                 if g["completed"]]
        wanted = [(g["game_id"], f"{a.date}: {g['name']}")
                  for g in games[:a.limit]]
        print(f"{a.date}: {len(games)} completed games, taking {len(wanted)}")
    else:
        wanted = DEFAULT_GAMES

    ok = 0
    for gid, why in wanted:
        try:
            payload = c.summary(gid)
            events, header = parse_summary(payload)
            if not events:
                print(f"  {gid}  SKIP -- no plays (not started?)")
                continue
            (out / f"summary_{gid}.json").write_text(json.dumps(payload))
            last = events[-1]
            print(f"  {gid}  {len(events):>4} plays  "
                  f"{last.away_score}-{last.home_score}  "
                  f"{max(e.period for e in events)} periods  -- {why}")
            ok += 1
        except Exception as e:                          # noqa: BLE001
            print(f"  {gid}  FAILED -- {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n{ok}/{len(wanted)} archived in {out}")
    if ok:
        print("now: python3 scripts/replay_server.py --speed 60")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

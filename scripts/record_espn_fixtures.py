"""Record real ESPN payloads to disk, so the adapter can be tested against them.

Run this on a machine that can reach ESPN (the sandbox this package was built
in cannot - site.api.espn.com is blocked by egress policy). It saves the raw
JSON exactly as returned; nothing is parsed or normalised, so the fixtures stay
useful even if the adapter changes.

    python3 scripts/record_espn_fixtures.py --date 20261115 --limit 5
    python3 scripts/record_espn_fixtures.py --game 401585555

Then, offline:
    python3 scripts/check_espn_fixtures.py           # sanity-check the payloads
    python3 scripts/live_poller.py --fixture-dir tmp/fixtures --game 401585555 --once
"""
import sys, pathlib, json, argparse, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.adapters.espn import EspnClient, scoreboard_games

ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
ap.add_argument("--game", type=int, default=None, help="record just this game")
ap.add_argument("--limit", type=int, default=5, help="how many games to record")
ap.add_argument("--out", default=str(ROOT / "tmp/fixtures"))
a = ap.parse_args()

out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
c = EspnClient()

if a.game:
    ids = [a.game]
else:
    day = a.date or datetime.datetime.now().strftime("%Y%m%d")
    sb = c.scoreboard(day)
    (out / "scoreboard.json").write_text(json.dumps(sb))
    games = scoreboard_games(sb)
    print(f"scoreboard {day}: {len(games)} games")
    # prefer games that are live or finished - a scheduled game has no plays
    games.sort(key=lambda g: (g["status"] == "STATUS_SCHEDULED", g["game_id"]))
    ids = [g["game_id"] for g in games[:a.limit]]

for gid in ids:
    s = c.summary(gid)
    p = out / f"summary_{gid}.json"
    p.write_text(json.dumps(s))
    print(f"  wrote {p.name}  ({len(s.get('plays') or []):,} plays)")
print(f"\n{len(ids)} fixture(s) in {out}")

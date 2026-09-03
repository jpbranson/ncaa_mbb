"""Serve archived ESPN games back as if they were happening now.

The gap this closes
-------------------
Everything in the live path had only ever been fed FINISHED games: a complete
`plays` array, arriving all at once, with `STATUS_FINAL` on it. A real night
looks nothing like that. The plays array grows between polls, the status starts
scheduled and ends final, and the clock runs. `smoke_live.py` says so in its own
verdict - "UNVALIDATED against a running clock".

This script closes that gap without waiting for November. It speaks ESPN's
protocol - the same two endpoints, the same JSON - and serves archived games
with only the plays that would have occurred by now. Point the real deployment
at it and nothing in the stack knows the difference:

    python3 scripts/archive_replay_games.py            # once, needs network
    python3 scripts/replay_server.py --speed 60        # terminal 1
    CBBWP_ESPN_BASE=http://127.0.0.1:8899 \\
        python3 scripts/serve_live.py --date 20260307  # terminal 2

That is the whole point: the poller uses its real `EspnClient`, over real HTTP,
with its real retry and backoff behaviour. A fixture directory would skip all of
that, which is why `--fixture-dir` is not the same test.

What is faithful, and what is not
---------------------------------
Faithful: payload shape (raw archived JSON, only the plays list truncated),
growing plays, status transitions, `displayClock` and `period`, several games on
one scoreboard progressing independently.

NOT faithful, deliberately:
  * Plays are revealed by their own clock, so they arrive in bursts at whatever
    rate the game had. Real polls see steadier trickle.
  * ESPN's retroactive corrections - a play inserted, rescored or deleted
    minutes later - are not simulated. The poller is built to be immune to those
    by replaying the whole game each poll, so this does not exercise that.
  * No rate limiting, no 403s, no partial outages. `--flaky` injects errors when
    that is what you want to test.

A replay is a rehearsal, not a validation of live play. It cannot tell you ESPN
did not change the feed since the archive was taken.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import STATUS_FINAL, STATUS_PRE  # noqa: E402

# Regulation is two 20-minute halves; overtimes are 5 minutes each.
HALF_SECONDS = 20 * 60
OT_SECONDS = 5 * 60
STATUS_IN = "STATUS_IN_PROGRESS"


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def elapsed_seconds(play: dict) -> float:
    """Seconds of game time from tip-off to this play.

    ESPN gives period plus a COUNTDOWN clock, so this converts to a monotonic
    ordinate. Without it, plays cannot be released in the order they happened.
    """
    period = _int((play.get("period") or {}).get("number"), 1) or 1
    disp = str((play.get("clock") or {}).get("displayValue") or "")
    m = re.match(r"^(\d+):(\d+)", disp)
    if m:
        remaining = int(m.group(1)) * 60 + int(m.group(2))
    else:
        m2 = re.match(r"^(\d+)\.(\d+)$", disp)      # under a minute: "42.7"
        remaining = float(m2.group(1)) if m2 else 0.0
    if period <= 2:
        before = (period - 1) * HALF_SECONDS
        return before + (HALF_SECONDS - remaining)
    before = 2 * HALF_SECONDS + (period - 3) * OT_SECONDS
    return before + (OT_SECONDS - remaining)


def clock_display(remaining: float) -> str:
    """ESPN's own formatting: m:ss, but tenths inside the last minute."""
    remaining = max(0.0, remaining)
    if remaining < 60:
        return f"{remaining:.1f}"
    return f"{int(remaining) // 60}:{int(remaining) % 60:02d}"


class ReplayGame:
    """One archived game, revealed as its own clock advances."""

    def __init__(self, path: pathlib.Path, tip_offset: float = 0.0):
        self.path = path
        self.payload = json.loads(path.read_text())
        self.game_id = int(path.stem.split("_")[1])
        self.tip_offset = tip_offset          # replay seconds before tip-off
        plays = self.payload.get("plays") or []
        # Sort by game time. The archive is already in order, but a replay that
        # depends on the archive being sorted breaks silently if it is not.
        self.plays = sorted(plays, key=elapsed_seconds)
        self.marks = [elapsed_seconds(p) for p in self.plays]
        self.duration = self.marks[-1] if self.marks else 0.0
        self.max_period = max(
            (_int((p.get("period") or {}).get("number"), 1) for p in self.plays),
            default=2)
        hdr = (self.payload.get("header") or {})
        comp = (hdr.get("competitions") or [{}])[0]
        self.name = " @ ".join(
            (c.get("team") or {}).get("abbreviation") or "?"
            for c in sorted(comp.get("competitors") or [],
                            key=lambda c: c.get("homeAway") != "away"))

    def game_clock(self, elapsed: float) -> float:
        """Game seconds elapsed, given replay seconds since this game's tip."""
        return max(0.0, elapsed - self.tip_offset)

    def state(self, elapsed: float) -> str:
        g = self.game_clock(elapsed)
        if elapsed < self.tip_offset:
            return STATUS_PRE
        return STATUS_FINAL if g >= self.duration else STATUS_IN

    def n_revealed(self, elapsed: float) -> int:
        g = self.game_clock(elapsed)
        if elapsed < self.tip_offset:
            return 0
        if g >= self.duration:
            return len(self.plays)
        lo, hi = 0, len(self.marks)
        while lo < hi:                          # bisect_right, no import
            mid = (lo + hi) // 2
            if self.marks[mid] <= g:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def period_and_clock(self, elapsed: float) -> tuple[int, str]:
        g = self.game_clock(elapsed)
        if g >= self.duration:
            return self.max_period, "0.0"
        if g < HALF_SECONDS:
            return 1, clock_display(HALF_SECONDS - g)
        if g < 2 * HALF_SECONDS:
            return 2, clock_display(2 * HALF_SECONDS - g)
        into_ot = g - 2 * HALF_SECONDS
        ot = int(into_ot // OT_SECONDS) + 1
        return 2 + ot, clock_display(OT_SECONDS - (into_ot % OT_SECONDS))

    def summary(self, elapsed: float) -> dict:
        """The archived payload, truncated to what has happened by `elapsed`."""
        n = self.n_revealed(elapsed)
        out = copy.deepcopy(self.payload)
        out["plays"] = self.plays[:n]
        status, (period, disp) = self.state(elapsed), self.period_and_clock(elapsed)
        last = self.plays[n - 1] if n else None
        home = _int((last or {}).get("homeScore"), 0)
        away = _int((last or {}).get("awayScore"), 0)

        comp = ((out.get("header") or {}).get("competitions") or [{}])[0]
        comp["status"] = {
            "period": period, "displayClock": disp,
            "type": {"name": status, "completed": status == STATUS_FINAL,
                     "state": {"STATUS_SCHEDULED": "pre", STATUS_IN: "in",
                               STATUS_FINAL: "post"}[status]},
        }
        for c in comp.get("competitors") or []:
            c["score"] = str(home if c.get("homeAway") == "home" else away)
        return out

    def scoreboard_event(self, elapsed: float) -> dict:
        """This game as one entry in a scoreboard payload."""
        status = self.state(elapsed)
        period, disp = self.period_and_clock(elapsed)
        comp = ((self.payload.get("header") or {}).get("competitions") or [{}])[0]
        n = self.n_revealed(elapsed)
        last = self.plays[n - 1] if n else None
        competitors = []
        for c in comp.get("competitors") or []:
            side = c.get("homeAway")
            score = _int((last or {}).get(
                "homeScore" if side == "home" else "awayScore"), 0)
            competitors.append({"homeAway": side, "score": str(score),
                                "team": c.get("team") or {}})
        return {
            "id": str(self.game_id),
            "shortName": self.name,
            "date": (self.payload.get("header") or {}).get("date") or "",
            "competitions": [{
                "id": str(self.game_id),
                "neutralSite": bool(comp.get("neutralSite")),
                "competitors": competitors,
                "status": {
                    "period": period, "displayClock": disp,
                    "type": {"name": status,
                             "completed": status == STATUS_FINAL,
                             "state": {"STATUS_SCHEDULED": "pre", STATUS_IN: "in",
                                       STATUS_FINAL: "post"}[status]},
                },
            }],
        }


class Replay:
    """The slate: every archived game, on one shared replay clock."""

    def __init__(self, games: list[ReplayGame], speed: float):
        self.games = games
        self.speed = speed
        self.t0 = time.time()
        self.requests = 0
        self.lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        """Game seconds since the replay started."""
        return (time.time() - self.t0) * self.speed

    def by_id(self, gid: int) -> Optional[ReplayGame]:
        return next((g for g in self.games if g.game_id == gid), None)

    def done(self) -> bool:
        e = self.elapsed
        return all(g.state(e) == STATUS_FINAL for g in self.games)


class Handler(BaseHTTPRequestHandler):
    replay: Replay = None           # set on the server before serve_forever
    flaky: float = 0.0

    def log_message(self, *args):   # quiet; the run prints its own progress
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:                       # noqa: N802
        r = self.replay
        with r.lock:
            r.requests += 1
            n = r.requests
        # Deterministic fault injection: every Nth request fails. Deterministic
        # rather than random so a failing dry run can be re-run identically.
        if self.flaky and n % int(1 / self.flaky) == 0:
            self._send(503, {"error": "injected fault"})
            return

        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        e = r.elapsed

        if parsed.path.endswith("/scoreboard"):
            self._send(200, {"events": [g.scoreboard_event(e) for g in r.games]})
        elif parsed.path.endswith("/summary"):
            gid = _int((q.get("event") or ["0"])[0])
            g = r.by_id(gid)
            if g is None:
                self._send(404, {"error": f"no archived game {gid}"})
            else:
                self._send(200, g.summary(e))
        else:
            self._send(404, {"error": f"unhandled path {parsed.path}"})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "tmp/replay"),
                    help="directory of archived summary_*.json")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--speed", type=float, default=60.0,
                    help="game seconds per real second (60 = a half in 20s)")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="game seconds between tip-offs, so games start apart")
    ap.add_argument("--flaky", type=float, default=0.0,
                    help="fail this fraction of requests, e.g. 0.1 for 1 in 10")
    ap.add_argument("--game", type=int, action="append",
                    help="replay only this game id (repeatable)")
    a = ap.parse_args()

    d = pathlib.Path(a.dir)
    paths = sorted(d.glob("summary_*.json"))
    if a.game:
        want = {str(g) for g in a.game}
        paths = [p for p in paths if p.stem.split("_")[1] in want]
    if not paths:
        print(f"no archived games in {d} -- run scripts/archive_replay_games.py",
              file=sys.stderr)
        return 1

    games = [ReplayGame(p, tip_offset=i * a.stagger)
             for i, p in enumerate(paths)]
    replay = Replay(games, a.speed)

    print(f"replaying {len(games)} game(s) at {a.speed}x from {d}")
    for g in games:
        print(f"  {g.game_id}  {g.name:<24} {len(g.plays):>4} plays  "
              f"{g.duration/60:.0f} min of game time"
              + (f"  tip +{g.tip_offset/60:.0f} min" if g.tip_offset else ""))
    longest = max(g.tip_offset + g.duration for g in games)
    print(f"\nfull slate takes {longest / a.speed / 60:.1f} real minutes")
    print(f"serving on http://127.0.0.1:{a.port}\n\npoint the deployment at it:")
    print(f"  CBBWP_ESPN_BASE=http://127.0.0.1:{a.port} "
          f"python3 scripts/serve_live.py\n")

    Handler.replay = replay
    Handler.flaky = a.flaky
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

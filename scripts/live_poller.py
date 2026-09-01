"""Live win-probability poller.

One asyncio task per in-progress game. Each task fetches that game's ESPN
summary, turns the plays into canonical Events, and replays the WHOLE game
through `WinProbabilityService` every poll. Full replay is deliberate: it costs
a few milliseconds and it makes ESPN's retroactive corrections - a play
inserted, rescored, or deleted minutes later - a non-event, because the answer
is always a pure function of the plays currently in the feed.

Usage
-----
    python3 scripts/build_live_context.py          # once a day, before the slate
    python3 scripts/live_poller.py                 # follow tonight's games
    python3 scripts/live_poller.py --date 20261115 # a specific slate
    python3 scripts/live_poller.py --game 401585555 --once   # one game, one poll
    python3 scripts/live_poller.py --fixture-dir tmp/fixtures # offline, no network

Output: a line per changed state to stdout, and JSONL to --out (default
`data/live/wp_YYYYMMDD.jsonl`). Nothing is overwritten; the file is appended to.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import json
import pathlib
import signal
import sys
import time
from typing import Dict, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.adapters.espn import (EspnClient, parse_summary, scoreboard_games,
                                 STATUS_FINAL, STATUS_PRE)
from cbbwp.live_context import LiveContextProvider
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Poll cadence, by seconds remaining in the game. Tight at the end, where the
# probability actually moves, and lazy early, where it barely does.
def poll_interval(secs_remaining: Optional[float], live: bool) -> float:
    if not live:
        return 60.0
    if secs_remaining is None:
        return 30.0
    if secs_remaining <= 120:
        return 5.0
    if secs_remaining <= 300:
        return 10.0
    return 20.0


DISCOVERY_INTERVAL = 120.0    # rescan the scoreboard this often
MAX_CONCURRENT_FETCHES = 8
ERROR_BACKOFF = (5.0, 15.0, 45.0, 90.0)   # per consecutive failure


class Poller:
    def __init__(self, svc: WinProbabilityService, ctx: LiveContextProvider,
                 client: EspnClient, out_path: pathlib.Path, quiet: bool = False,
                 fixture_dir: Optional[pathlib.Path] = None):
        self.svc = svc
        self.ctx = ctx
        self.client = client
        self.out_path = out_path
        self.quiet = quiet
        self.fixture_dir = fixture_dir
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        self.watchers: Dict[int, asyncio.Task] = {}
        self.last_emitted: Dict[int, tuple] = {}
        self.stopping = asyncio.Event()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = out_path.open("a", buffering=1)

    # -- I/O -------------------------------------------------------------
    async def _summary(self, game_id: int) -> dict:
        if self.fixture_dir is not None:
            p = self.fixture_dir / f"summary_{game_id}.json"
            return json.loads(p.read_text())
        async with self.sem:
            return await asyncio.to_thread(self.client.summary, game_id)

    async def _scoreboard(self, date: Optional[str]) -> dict:
        if self.fixture_dir is not None:
            p = self.fixture_dir / "scoreboard.json"
            return json.loads(p.read_text())
        async with self.sem:
            return await asyncio.to_thread(self.client.scoreboard, date)

    def emit(self, row: dict, header) -> None:
        self._fh.write(json.dumps(row) + "\n")
        if self.quiet:
            return
        secs = row["game_seconds_remaining"]
        mm, ss = divmod(int(secs), 60)
        print(f"{header.away_name[:18]:<18} @ {header.home_name[:18]:<18} "
              f"P{row['period']} {mm:>2}:{ss:02d}  "
              f"{row['margin']:>+4}  home {row['home_win_prob']:.3f}", flush=True)

    # -- one game --------------------------------------------------------
    async def watch(self, game_id: int) -> None:
        fails = 0
        while not self.stopping.is_set():
            try:
                summary = await self._summary(game_id)
                events, header = parse_summary(summary)
                fails = 0
            except Exception as e:                     # noqa: BLE001
                fails += 1
                wait = ERROR_BACKOFF[min(fails - 1, len(ERROR_BACKOFF) - 1)]
                if not self.quiet:
                    print(f"[{game_id}] fetch failed ({type(e).__name__}: {e}); "
                          f"retry in {wait:.0f}s", file=sys.stderr, flush=True)
                if fails >= 8:
                    print(f"[{game_id}] giving up after {fails} failures",
                          file=sys.stderr, flush=True)
                    return
                await self._sleep(wait)
                continue

            if events:
                pctx = self.ctx.context_for(
                    game_id, header.home_team_id, header.away_team_id,
                    header.neutral_site)
                rows = self.svc.score_game(events, pctx)
                if rows:
                    last = rows[-1]
                    key = (last["seq"], round(last["home_win_prob"], 6))
                    if self.last_emitted.get(game_id) != key:
                        self.last_emitted[game_id] = key
                        last = dict(last)
                        last["ts"] = datetime.datetime.now(
                            datetime.timezone.utc).isoformat()
                        last["home_team_id"] = header.home_team_id
                        last["away_team_id"] = header.away_team_id
                        last["status"] = header.status
                        self.emit(last, header)
                    secs = last["game_seconds_remaining"]
                else:
                    secs = None
            else:
                secs = None

            if header.is_final:
                if not self.quiet:
                    print(f"[{game_id}] final: {header.away_name} "
                          f"{header.away_score} @ {header.home_name} "
                          f"{header.home_score}", flush=True)
                return
            await self._sleep(poll_interval(secs, header.is_live))

    async def _sleep(self, secs: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.stopping.wait(), timeout=secs)

    # -- discovery -------------------------------------------------------
    async def discover(self, date: Optional[str], once: bool) -> None:
        while not self.stopping.is_set():
            try:
                sb = await self._scoreboard(date)
                games = scoreboard_games(sb)
            except Exception as e:                     # noqa: BLE001
                print(f"scoreboard failed ({type(e).__name__}: {e})",
                      file=sys.stderr, flush=True)
                await self._sleep(30.0)
                continue

            live = [g for g in games
                    if g["status"] not in (STATUS_PRE, STATUS_FINAL)]
            if not self.quiet:
                print(f"scoreboard: {len(games)} games, {len(live)} in progress, "
                      f"{len(self.watchers)} watched", flush=True)
            for g in live:
                gid = g["game_id"]
                t = self.watchers.get(gid)
                if t is None or t.done():
                    self.watchers[gid] = asyncio.create_task(
                        self.watch(gid), name=f"watch-{gid}")
            for gid, t in list(self.watchers.items()):
                if t.done():
                    del self.watchers[gid]
            if once:
                return
            await self._sleep(DISCOVERY_INTERVAL)

    async def run(self, date: Optional[str], game: Optional[int],
                  once: bool) -> None:
        if game is not None:
            await self.watch(game) if not once else await self._one(game)
            return
        await self.discover(date, once)
        if self.watchers:
            await asyncio.gather(*self.watchers.values(),
                                 return_exceptions=True)

    async def _one(self, game_id: int) -> None:
        """A single poll of one game - the smoke test."""
        summary = await self._summary(game_id)
        events, header = parse_summary(summary)
        pctx = self.ctx.context_for(game_id, header.home_team_id,
                                    header.away_team_id, header.neutral_site)
        rows = self.svc.score_game(events, pctx)
        print(f"{header.away_name} @ {header.home_name}  status={header.status}  "
              f"{len(events)} plays -> {len(rows)} states")
        if not self.ctx.known(header.home_team_id):
            print(f"  note: home team id {header.home_team_id} not in the ratings "
                  "snapshot; using league average")
        if not self.ctx.known(header.away_team_id):
            print(f"  note: away team id {header.away_team_id} not in the ratings "
                  "snapshot; using league average")
        for r in rows[-10:]:
            mm, ss = divmod(int(r["game_seconds_remaining"]), 60)
            print(f"  P{r['period']} {mm:>2}:{ss:02d}  margin {r['margin']:>+4}  "
                  f"home {r['home_win_prob']:.4f}")
            self._fh.write(json.dumps(r) + "\n")

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="slate to follow, YYYYMMDD (default: today)")
    ap.add_argument("--game", type=int, help="follow a single game id")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--version", default="v1", help="model registry version")
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    ap.add_argument("--context", default=str(ROOT / "registry/context_latest.json"))
    ap.add_argument("--out", default=None, help="JSONL output path")
    ap.add_argument("--fixture-dir", default=None,
                    help="read saved payloads from this directory instead of the network")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    ctx_path = pathlib.Path(a.context)
    if not ctx_path.exists():
        print(f"no ratings snapshot at {ctx_path}\n"
              "  run: python3 scripts/build_live_context.py", file=sys.stderr)
        return 2
    ctx = LiveContextProvider.load(ctx_path)
    if ctx.is_stale:
        print(f"warning: ratings snapshot is {ctx.age_days:.1f} days old; "
              "re-run scripts/build_live_context.py", file=sys.stderr)

    svc = WinProbabilityService(a.registry, a.version)
    day = a.date or datetime.datetime.now().strftime("%Y%m%d")
    out = pathlib.Path(a.out) if a.out else ROOT / f"data/live/wp_{day}.jsonl"
    poller = Poller(svc, ctx, EspnClient(), out, quiet=a.quiet,
                    fixture_dir=pathlib.Path(a.fixture_dir) if a.fixture_dir else None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, poller.stopping.set)
    try:
        loop.run_until_complete(poller.run(a.date, a.game, a.once))
    except KeyboardInterrupt:
        pass
    finally:
        poller.close()
        loop.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

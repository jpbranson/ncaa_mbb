"""A small web app for watching win probability move, live or on replay.

    python3 scripts/serve_viz.py            # then open http://127.0.0.1:8811

Two things in one page:

  * **Live** - whatever `serve_live.py` is currently tracking, polled and drawn.
  * **Replay** - any past game, from the raw ESPN payloads in `tmp/replay/`, with
    transport controls: play/pause, speed, step forward and back, and a scrubber.

Why replay is precomputed rather than streamed
----------------------------------------------
`scripts/replay_server.py` already replays games, but it does so on a wall
clock, to the poller, forwards only. That is the right shape for rehearsing the
deployment and the wrong shape for *looking* at a game: you cannot step back a
possession, and pausing means stopping time for the poller too.

So this does the opposite. `score_game` replays a whole game in milliseconds and
returns one row per state, so the entire series is computed once, up front, and
the browser scrubs an array. Stepping backward is then free and exact - it is
the same number the model produced going forwards, not a re-derivation.

That difference matters for trust. Nothing here re-computes or interpolates a
probability for display; every number on screen came out of `WinProbabilityService`
exactly as the live path would have produced it.

Endpoints (all JSON except `/`):

    GET  /                    the page
    GET  /api/games           archived games available to replay
    GET  /api/game/<id>       one game, fully scored: a row per play
    GET  /api/fetch/<id>      pull a game from ESPN, archive it, then score it
    GET  /api/live            what the live API is tracking right now
    GET  /api/live/<id>       one live game, with the history so far
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import (EspnClient, is_synthetic_payload,  # noqa: E402
                                 parse_summary)
from cbbwp.config import Settings                           # noqa: E402
from cbbwp.live_context import LiveContextProvider          # noqa: E402
from cbbwp.serve import WinProbabilityService               # noqa: E402

PAGE = ROOT / "web" / "index.html"
# Both directories hold raw ESPN summary payloads; tmp/replay is the curated set
# from archive_replay_games.py, tmp/fixtures whatever the smoke test last saved.
ARCHIVE_DIRS = [ROOT / "tmp" / "replay", ROOT / "tmp" / "fixtures"]


def _archives() -> dict[int, pathlib.Path]:
    """game_id -> payload path. First directory wins on a duplicate."""
    out: dict[int, pathlib.Path] = {}
    for d in ARCHIVE_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("summary_*.json")):
            try:
                gid = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            out.setdefault(gid, p)
    return out


class Scorer:
    """Loads the model once, then scores whole games on demand and caches them."""

    def __init__(self, cfg: Settings):
        self.svc = WinProbabilityService(cfg.registry, cfg.model_version)
        self.ctx = LiveContextProvider.load(cfg.context_path)
        self.cache: dict[int, dict] = {}
        self.lock = threading.Lock()

    def summary_of(self, path: pathlib.Path) -> dict:
        return json.loads(path.read_text())

    def score(self, game_id: int, payload: dict) -> dict:
        """One archived payload -> everything the page needs to draw the game."""
        with self.lock:
            hit = self.cache.get(game_id)
        if hit is not None:
            return hit

        events, header = parse_summary(payload)
        pctx = self.ctx.context_for(game_id, header.home_team_id,
                                    header.away_team_id, header.neutral_site)
        rows = self.svc.score_game(events, pctx)

        # States and events are both dense 1..N over the same plays, so they zip
        # by position. Asserting it is cheap and beats drawing a chart whose
        # probabilities belong to different plays than its captions.
        if len(rows) != len(events):
            raise RuntimeError(
                f"game {game_id}: {len(rows)} states for {len(events)} events")

        plays = []
        for ev, row in zip(events, rows):
            plays.append({
                "seq": row["seq"],
                "period": row["period"],
                "secs": row["game_seconds_remaining"],
                "clock": ev.clock_seconds,
                "margin": row["margin"],
                "wp": round(row["home_win_prob"], 5),
                "home": ev.home_score,
                "away": ev.away_score,
                "type": ev.event_type,
                "text": ev.text,
                # Which side the feed credits the play to; None for the clock,
                # officials and anything else that belongs to neither.
                "team": ("home" if ev.team_id == header.home_team_id
                         else "away" if ev.team_id == header.away_team_id
                         else None),
            })
        out = {
            "game_id": game_id,
            "home_name": header.home_name,
            "away_name": header.away_name,
            "neutral_site": header.neutral_site,
            "status": header.status,
            "periods": max((p["period"] for p in plays), default=2),
            "model_version": self.svc.version,
            "state_rules_version": self.svc.manifest.get("state_rules_version"),
            "synthetic": is_synthetic_payload(payload),
            "plays": plays,
        }
        with self.lock:
            self.cache[game_id] = out
        return out

    def brief(self, game_id: int, path: pathlib.Path) -> dict:
        """Cheap listing entry: names and play count, without scoring."""
        payload = self.summary_of(path)
        events, header = parse_summary(payload)
        last = events[-1] if events else None
        return {
            "game_id": game_id,
            "home_name": header.home_name,
            "away_name": header.away_name,
            "plays": len(events),
            "periods": max((e.period for e in events), default=0),
            "home_score": last.home_score if last else 0,
            "away_score": last.away_score if last else 0,
            "source": path.parent.name,
            # Surfaced in the list: a payload rebuilt from hoopR is a legitimate
            # shape test but not a recording of a real game, and the two should
            # not sit in one list looking identical.
            "synthetic": is_synthetic_payload(payload),
        }


class Handler(BaseHTTPRequestHandler):
    scorer: Scorer = None
    live_url: str = ""

    def log_message(self, *args):
        pass

    def _json(self, code: int, body) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _page(self) -> None:
        if not PAGE.exists():
            self._json(500, {"error": f"missing {PAGE}"})
            return
        raw = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:                       # noqa: N802
        path = urlparse(self.path).path
        s = self.scorer

        if path in ("/", "/index.html"):
            return self._page()

        if path == "/api/games":
            games = []
            for gid, p in sorted(_archives().items()):
                try:
                    games.append(s.brief(gid, p))
                except Exception as e:              # noqa: BLE001
                    games.append({"game_id": gid, "error": f"{type(e).__name__}: {e}"})
            return self._json(200, {"games": games})

        if path.startswith("/api/game/"):
            try:
                gid = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "game id must be an integer"})
            arch = _archives().get(gid)
            if arch is None:
                return self._json(404, {"error": f"no archived game {gid}; "
                                                 f"try /api/fetch/{gid}"})
            try:
                return self._json(200, s.score(gid, s.summary_of(arch)))
            except Exception as e:                  # noqa: BLE001
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        if path.startswith("/api/fetch/"):
            try:
                gid = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "game id must be an integer"})
            try:
                payload = EspnClient().summary(gid)
                events, _ = parse_summary(payload)
                if not events:
                    return self._json(422, {"error": f"game {gid} has no plays "
                                                     "(not started, or not a game)"})
                dest = ROOT / "tmp" / "replay"
                dest.mkdir(parents=True, exist_ok=True)
                (dest / f"summary_{gid}.json").write_text(json.dumps(payload))
                return self._json(200, s.score(gid, payload))
            except urllib.error.HTTPError as e:
                return self._json(502, {"error": f"ESPN returned {e.code}: {e.reason}"})
            except Exception as e:                  # noqa: BLE001
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})

        if path.startswith("/api/live/"):
            try:
                gid = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json(400, {"error": "game id must be an integer"})
            try:
                with urllib.request.urlopen(f"{self.live_url}/games/{gid}",
                                            timeout=3) as r:
                    body = json.loads(r.read().decode())
            except Exception as e:                  # noqa: BLE001
                return self._json(200, {"offline": True, "reason": f"{type(e).__name__}"})
            # The live feed carries team IDs, not names -- the ratings snapshot
            # is keyed by id too. Names are cosmetic, so resolve them from the
            # local archive when we happen to have the game, and otherwise show
            # the id rather than making a network call mid-game for a caption.
            arch = _archives().get(gid)
            if arch is not None:
                try:
                    _, header = parse_summary(s.summary_of(arch))
                    body["home_name"] = header.home_name
                    body["away_name"] = header.away_name
                except Exception:                   # noqa: BLE001
                    pass
            return self._json(200, body)

        if path == "/api/live":
            # Proxied so the page has a single origin and needs no CORS. A live
            # feed simply not running is the normal case, not an error.
            try:
                with urllib.request.urlopen(self.live_url + "/games", timeout=3) as r:
                    return self._json(200, json.loads(r.read().decode()))
            except Exception as e:                  # noqa: BLE001
                return self._json(200, {"games": [], "offline": True,
                                        "reason": f"{type(e).__name__}"})

        return self._json(404, {"error": f"no route {path}"})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--live-url", default=None,
                    help="the running serve_live.py API (default: from config)")
    a = ap.parse_args()

    cfg = Settings.from_env()
    print("cbbwp viz\n" + cfg.describe(), flush=True)
    scorer = Scorer(cfg)
    n = len(_archives())
    print(f"\n{n} archived game(s) available to replay"
          + ("" if n else " -- run scripts/archive_replay_games.py"), flush=True)

    Handler.scorer = scorer
    Handler.live_url = a.live_url or f"http://{cfg.api_host}:{cfg.api_port}"
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"\nopen http://{a.host}:{a.port}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

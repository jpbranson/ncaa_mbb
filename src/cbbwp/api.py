"""A read-only HTTP view of what the poller is currently producing.

Deliberately the standard library and nothing else. This runs beside a live feed
on a laptop or in a small container, and a web framework would be a dependency,
a version to keep current and a CVE feed to watch, in exchange for routing three
endpoints.

    GET /                index of endpoints
    GET /health          liveness, model version, ratings freshness
    GET /games           every game currently tracked, latest state each
    GET /games/<id>      one game, with recent history

**Every response carries `model_version` and `state_rules_version`.** A number
served without saying what produced it cannot be checked later, and this project
has already had one near-miss where a model and the code that fed it disagreed
about what a feature meant.

The store is written by the poller's asyncio thread and read by the HTTP
threads, so it takes a lock. The critical sections are dictionary writes.
"""
from __future__ import annotations

import collections
import datetime
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from .schemas import STATE_RULES_VERSION


class LiveStore:
    """Latest state per game, plus a bounded tail of history."""

    def __init__(self, history: int = 240):
        self._lock = threading.Lock()
        self._latest: dict[int, dict] = {}
        self._history: dict[int, collections.deque] = {}
        self._history_len = history
        self._updates = 0
        self._last_update: Optional[float] = None

    def update(self, row: dict) -> None:
        gid = int(row["game_id"])
        with self._lock:
            self._latest[gid] = row
            d = self._history.get(gid)
            if d is None:
                d = self._history[gid] = collections.deque(maxlen=self._history_len)
            d.append({k: row[k] for k in
                      ("seq", "period", "game_seconds_remaining", "margin",
                       "home_win_prob") if k in row})
            self._updates += 1
            self._last_update = time.time()

    def games(self) -> list[dict]:
        with self._lock:
            rows = list(self._latest.values())
        return sorted(rows, key=lambda r: r.get("game_seconds_remaining", 1e9))

    def game(self, game_id: int) -> Optional[dict]:
        with self._lock:
            latest = self._latest.get(game_id)
            hist = list(self._history.get(game_id, ()))
        if latest is None:
            return None
        return {**latest, "history": hist}

    def stats(self) -> dict:
        with self._lock:
            return {"games_tracked": len(self._latest), "updates": self._updates,
                    "last_update": self._last_update}


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


def make_handler(store: LiveStore, meta: dict,
                 ratings_age_days: Callable[[], Optional[float]],
                 started: float,
                 data_age_days: Callable[[], Optional[float]] = lambda: None,
                 data_is_stale: Callable[[], bool] = lambda: False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cbbwp/" + str(meta.get("model_version", "?"))
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return   # the poller owns stdout; access logs would bury it

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, indent=2, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # A future front end will be a separate origin; this is a read-only
            # public-by-nature feed, so allowing that is not a widening of access.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _envelope(self, extra: dict) -> dict:
            return {"model_version": meta.get("model_version"),
                    "state_rules_version": STATE_RULES_VERSION,
                    "served_at": _iso(time.time()), **extra}

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"

            if path == "/":
                return self._send(200, self._envelope({"endpoints": {
                    "/health": "liveness, model version, ratings freshness",
                    "/games": "every game currently tracked",
                    "/games/{game_id}": "one game, with recent history"}}))

            if path in ("/health", "/healthz"):
                age = ratings_age_days()
                dage = data_age_days()
                file_stale = age is not None and age > meta.get("ratings_max_age_days", 3)
                feed_stale = data_is_stale()
                s = store.stats()
                ok = not (file_stale or feed_stale)
                reason = None
                if feed_stale:
                    reason = ("ratings were fit on stale data -- newest completed "
                              f"game is {dage:.1f} days old; refresh the play-by-play "
                              "before rebuilding the snapshot")
                elif file_stale:
                    reason = "ratings snapshot file is stale; re-run build_live_context.py"
                return self._send(200 if ok else 503, self._envelope({
                    "status": "ok" if ok else "degraded",
                    "reason": reason,
                    "uptime_seconds": round(time.time() - started, 1),
                    "ratings_age_days": None if age is None else round(age, 2),
                    "data_age_days": None if dage is None else round(dage, 2),
                    "games_tracked": s["games_tracked"],
                    "updates": s["updates"],
                    "last_update": _iso(s["last_update"]),
                }))

            if path == "/games":
                return self._send(200, self._envelope({"games": store.games()}))

            if path.startswith("/games/"):
                raw = path.split("/", 2)[2]
                if not raw.isdigit():
                    return self._send(400, self._envelope(
                        {"error": "game id must be numeric", "got": raw}))
                g = store.game(int(raw))
                if g is None:
                    return self._send(404, self._envelope(
                        {"error": "not tracked", "game_id": int(raw)}))
                return self._send(200, self._envelope({"game": g}))

            self._send(404, self._envelope({"error": "no such endpoint",
                                            "path": path}))

    return Handler


def serve_in_thread(store: LiveStore, meta: dict,
                    ratings_age_days: Callable[[], Optional[float]],
                    host: str, port: int,
                    data_age_days: Callable[[], Optional[float]] = lambda: None,
                    data_is_stale: Callable[[], bool] = lambda: False) -> ThreadingHTTPServer:
    """Start the API on a daemon thread and return the server."""
    handler = make_handler(store, meta, ratings_age_days, time.time(),
                           data_age_days, data_is_stale)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="cbbwp-api",
                     daemon=True).start()
    return httpd

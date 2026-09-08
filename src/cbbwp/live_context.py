"""Pregame context for games that have not been played yet.

Offline, `PregameContext` fields come from `games.parquet` / `team_stats.parquet`,
which are built AFTER the fact. A live game has no such row, so the same three
quantities have to be produced from what is known this morning:

    pregame_exp_margin  = rating(home) - rating(away) + (0 if neutral else hca)
    ft_pct_diff         = season-to-date FT% home minus away
    exp_points_per_min  = the two teams' combined scoring rate

`scripts/build_live_context.py` writes the snapshot this class reads. Refresh it
daily (and it is cheap enough to refresh hourly); a stale snapshot degrades
gracefully - it just means yesterday's ratings.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .schemas import PregameContext

DEFAULT_HCA = 3.4
DEFAULT_FT = 0.700
DEFAULT_PPM = 3.45
STALE_AFTER_DAYS = 3
# In season, teams play at least twice a week. Ratings whose newest completed
# game is older than this are being fit on a stale copy of the data, whatever
# the snapshot file's own timestamp says.
DATA_STALE_AFTER_DAYS = 10

# ...but only while the sport is being played. Between mid-April and the start
# of November the newest completed game is MEANT to be months old, and carrying
# the previous season's ratings forward is the documented preseason behaviour.
# A staleness alarm that cries every summer is one nobody reads in January.
SEASON_START_MONTH = 11          # November
SEASON_END_MONTH, SEASON_END_DAY = 4, 15   # through April 15


@dataclass
class LiveContextProvider:
    season: int
    hca: float
    ratings: Dict[int, float]
    ft_pct: Dict[int, float]
    ppm: Dict[int, float]
    generated: str = ""
    latest_game_date: str = ""      # newest COMPLETED game the ratings saw

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "LiveContextProvider":
        d = json.loads(pathlib.Path(path).read_text())
        as_int = lambda m: {int(k): float(v) for k, v in (m or {}).items()}
        return cls(
            season=int(d.get("season", 0)),
            hca=float(d.get("hca", DEFAULT_HCA)),
            ratings=as_int(d.get("ratings")),
            ft_pct=as_int(d.get("ft_pct")),
            ppm=as_int(d.get("ppm")),
            generated=str(d.get("generated", "")),
            latest_game_date=str(d.get("latest_game_date", "")),
        )

    @property
    def age_days(self) -> float:
        if not self.generated:
            return float("inf")
        try:
            t = _dt.datetime.fromisoformat(self.generated)
        except ValueError:
            return float("inf")
        if t.tzinfo is None:
            t = t.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() / 86400.0

    @property
    def data_age_days(self) -> float | None:
        """Days since the newest COMPLETED game these ratings were fit on.

        `age_days` says when the snapshot file was written; this says how
        current the data behind it is. They come apart in the way that matters:
        a nightly job rebuilding from three-week-old parquet writes a file that
        looks perfectly fresh and carries three-week-old ratings. Only this
        property notices.

        None in the preseason, where there are no completed games yet and
        carrying last season's ratings forward is the documented behaviour.
        """
        if not self.latest_game_date:
            return None
        try:
            t = _dt.datetime.fromisoformat(self.latest_game_date)
        except ValueError:
            return None
        if t.tzinfo is None:
            t = t.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() / 86400.0

    @staticmethod
    def _in_season(now: "_dt.datetime | None" = None) -> bool:
        n = now or _dt.datetime.now(_dt.timezone.utc)
        if n.month >= SEASON_START_MONTH or n.month < SEASON_END_MONTH:
            return True
        return n.month == SEASON_END_MONTH and n.day <= SEASON_END_DAY

    @property
    def data_is_stale(self) -> bool:
        """True only when we can tell, it matters, and the answer is bad."""
        d = self.data_age_days
        return (d is not None and d > DATA_STALE_AFTER_DAYS
                and self._in_season())

    @property
    def is_stale(self) -> bool:
        return self.age_days > STALE_AFTER_DAYS or self.data_is_stale

    def context_for(self, game_id: int, home_team_id: int, away_team_id: int,
                    neutral_site: bool = False) -> PregameContext:
        """Best available pregame context. Unknown teams fall back to average.

        An unknown team id is not an error: it is a first-time opponent, a
        non-D1 side, or an id ESPN has just renumbered. Rating 0.0 means
        'league average', which is the right prior for a team we know nothing
        about, and the model's pregame term decays away within a few minutes.
        """
        r_h = self.ratings.get(home_team_id, 0.0)
        r_a = self.ratings.get(away_team_id, 0.0)
        margin = r_h - r_a + (0.0 if neutral_site else self.hca)
        ft_h = self.ft_pct.get(home_team_id, DEFAULT_FT)
        ft_a = self.ft_pct.get(away_team_id, DEFAULT_FT)
        ppm_h = self.ppm.get(home_team_id, DEFAULT_PPM)
        ppm_a = self.ppm.get(away_team_id, DEFAULT_PPM)
        return PregameContext(
            game_id=game_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            neutral_site=neutral_site,
            pregame_exp_margin=float(margin),
            season=self.season,
            ft_pct_diff=float(ft_h - ft_a),
            exp_points_per_min=float((ppm_h + ppm_a) / 2.0),
        )

    def known(self, team_id: int) -> bool:
        return team_id in self.ratings


class ReloadingContextProvider:
    """A `LiveContextProvider` that re-reads its file when it changes on disk.

    The deployment is a poller that runs for weeks and a SEPARATE daily job that
    rebuilds the snapshot (`deploy/install_macos.sh` installs exactly that pair).
    Loading once at startup made that composition silently useless: the running
    process served launch-day ratings for the rest of the season, and `/health`
    watched `ratings_age_days` climb past its own staleness threshold - reporting
    503 and advising a rebuild that cron had already done, to a process that was
    never going to read it.

    So the freshness of the ratings is a property of the FILE, and every reader
    goes through here. Delegates the whole `LiveContextProvider` surface, so it
    is a drop-in for it.

    Thread-safe: the poller reads from the asyncio loop and the API from its own
    HTTP threads. Reads take the lock only long enough to copy a reference; the
    provider itself is never mutated after construction, so callers can use the
    snapshot they got without holding anything.
    """

    # Stat the file at most this often. A stat is microseconds, but the poller
    # asks once per game per poll and there is no reason to make it more often
    # than the rebuild job could possibly produce a new file.
    CHECK_INTERVAL_SECONDS = 5.0

    def __init__(self, path: str | pathlib.Path,
                 check_interval: float = CHECK_INTERVAL_SECONDS,
                 on_reload=None):
        self.path = pathlib.Path(path)
        self._check_interval = check_interval
        self._on_reload = on_reload
        self._lock = threading.Lock()
        self._provider: LiveContextProvider = LiveContextProvider.load(self.path)
        self._mtime: Optional[float] = self._stat_mtime()
        self._last_check = time.monotonic()
        self.reloads = 0
        self.reload_failures = 0

    def _stat_mtime(self) -> Optional[float]:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    @property
    def current(self) -> LiveContextProvider:
        """The freshest snapshot, reloading first if the file has changed."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_check < self._check_interval:
                return self._provider
            self._last_check = now
            mtime = self._stat_mtime()
            if mtime is None or mtime == self._mtime:
                return self._provider
            try:
                # A snapshot rewritten in place can be read half-written. The
                # writer renames into place (build_live_context.py), so this is
                # belt and braces -- but a torn read must never take down a live
                # feed, and the previous ratings are a perfectly good answer.
                fresh = LiveContextProvider.load(self.path)
            except Exception as e:                      # noqa: BLE001
                self.reload_failures += 1
                print(f"warning: could not reload {self.path} "
                      f"({type(e).__name__}: {e}); keeping the previous ratings",
                      file=sys.stderr, flush=True)
                # Do NOT record the mtime: retry on the next check, because the
                # file is probably mid-write rather than permanently broken.
                return self._provider
            self._provider = fresh
            self._mtime = mtime
            self.reloads += 1
            cb = self._on_reload
        if cb is not None:
            try:
                cb(fresh)
            except Exception:                           # noqa: BLE001
                pass
        return fresh

    # -- the LiveContextProvider surface, all delegated to `current` ---------
    def context_for(self, game_id: int, home_team_id: int, away_team_id: int,
                    neutral_site: bool = False) -> PregameContext:
        return self.current.context_for(game_id, home_team_id, away_team_id,
                                        neutral_site)

    def known(self, team_id: int) -> bool:
        return self.current.known(team_id)

    @property
    def season(self) -> int:
        return self.current.season

    @property
    def generated(self) -> str:
        return self.current.generated

    @property
    def latest_game_date(self) -> str:
        return self.current.latest_game_date

    @property
    def age_days(self) -> float:
        return self.current.age_days

    @property
    def data_age_days(self) -> float | None:
        return self.current.data_age_days

    @property
    def data_is_stale(self) -> bool:
        return self.current.data_is_stale

    @property
    def is_stale(self) -> bool:
        return self.current.is_stale

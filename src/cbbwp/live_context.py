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
from dataclasses import dataclass
from typing import Dict

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

"""Adapter: ESPN's live men's college basketball feed -> canonical Events.

This is the live twin of `adapters/hoopr.py`. hoopR is itself a scrape of this
same ESPN feed, so the two adapters must agree exactly - that is what keeps the
live path and the training path on the same definitions.

The one subtlety is play *type*. hoopR stores ESPN's `type.text` verbatim, and
the state builder's possession rules are written against those exact strings.
ESPN occasionally reworks the display text of a play type but almost never its
numeric `type.id`, so this adapter maps id -> the text the model was TRAINED on
and only falls back to whatever text the feed sent when the id is unknown.

`TYPE_ID_TO_TEXT` was extracted from the 2016-2026 hoopR files: every
(type_id, type_text) pair that actually occurs.

Note what this map is and is not for. Possession no longer depends on play-type
NAMES at all - `state._possession_after` keys made field goals on the feed's
scoring/shooting flags, precisely because ESPN renamed the made-three type
between 2019 and 2021 and a name whitelist silently missed 324,043 made threes.
The map survives because the type text is still what the model was trained on
for every OTHER rule (fouls, timeouts, rebounds, turnovers), and because an id
is a more stable key for those than a display string.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from ..schemas import Event, PregameContext
from ..state import clock_to_seconds

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
SCOREBOARD_URL = SITE_API + "/scoreboard"
SUMMARY_URL = SITE_API + "/summary"

# Point the client somewhere other than ESPN. The only intended use is
# scripts/replay_server.py, which speaks ESPN's protocol back to the unmodified
# deployed stack so a dry run exercises the real network path rather than a
# fixture shortcut. Unset in production; there is no default but ESPN.
ESPN_BASE_ENV = "CBBWP_ESPN_BASE"

# ESPN's edge rejects bare short user-agents ("cbbwp/0.2") and browser-claiming
# ones sent without a browser's other headers -- both return 403, deterministically
# (15/15 on 2026-09-02, on both the scoreboard and summary endpoints). A client
# that identifies itself honestly, with a contact URL, passes. Verify with
# scripts/smoke_live.py step 4 before a live night: this is a WAF rule, and WAF
# rules change. Override without editing code by setting CBBWP_USER_AGENT.
DEFAULT_USER_AGENT = "cbbwp/0.2 (+https://github.com/jpbranson/ncaa_mbb)"

# Play type id -> the type_text the model was trained on. See module docstring.
TYPE_ID_TO_TEXT = {
    0: "Not Available",
    91: "Shot",
    215: "Coach's Challenge (Overturned)",
    216: "Coach's Challenge (Stands)",
    402: "End Game",
    412: "End Period",
    437: "TipShot",
    449: "Dead Ball Rebound",
    519: "PersonalFoul",
    521: "Technical Foul",
    540: "MadeFreeThrow",
    558: "JumpShot",
    572: "LayUpShot",
    574: "DunkShot",
    578: "RegularTimeOut",
    579: "ShortTimeOut",
    580: "OfficialTVTimeOut",
    584: "Substitution",
    586: "Offensive Rebound",
    587: "Defensive Rebound",
    598: "Lost Ball Turnover",
    607: "Steal",
    615: "Jumpball",
    618: "Block Shot",
    20437: "TipShot",
    20558: "JumpShot",
    20572: "LayUpShot",
    20574: "DunkShot",
    30558: "Three Point Jump Shot",
}

# Marker stamped into payloads that were REBUILT from hoopR rather than recorded
# from ESPN (tests/espn_fixtures.py). Tooling that claims to tell you something
# about the real feed has to be able to tell the two apart: a rebuilt payload
# carries hoopR's own type ids, so a check for "an id the model never saw"
# cannot fail on one, and a green tick from it means nothing.
SYNTHETIC_KEY = "_cbbwp_synthetic"


def is_synthetic_payload(payload: dict) -> bool:
    """True if this summary was rebuilt from hoopR rather than recorded.

    Prefers the explicit stamp. The name fallback exists because payloads
    written before the stamp are still sitting in `tmp/fixtures` on real
    machines, and silently counting those as evidence is the exact failure this
    function is here to prevent.
    """
    if payload.get(SYNTHETIC_KEY):
        return True
    comps = (payload.get("header") or {}).get("competitions") or [{}]
    names = {((c.get("team") or {}).get("displayName") or "")
             for c in (comps[0].get("competitors") or [])}
    return names == {"Home Team", "Away Team"}


# Statuses ESPN reports. Only IN means the clock is (or may be) running.
STATUS_PRE = "STATUS_SCHEDULED"
STATUS_FINAL = "STATUS_FINAL"


def _int(v, default=None) -> Optional[int]:
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def play_type_text(play: dict) -> str:
    """Canonical type_text for one ESPN play."""
    t = play.get("type") or {}
    tid = _int(t.get("id"))
    if tid is not None and tid in TYPE_ID_TO_TEXT:
        return TYPE_ID_TO_TEXT[tid]
    # Unknown id: fall back to the feed's own text. The state builder treats an
    # unrecognised type as "carry possession", which is the safe default.
    return (t.get("text") or "").strip()


def _sequence_key(play: dict, fallback: int) -> int:
    """ESPN's sequenceNumber, numeric, for ordering. Falls back to feed order."""
    s = _int(play.get("sequenceNumber"))
    return s if s is not None else fallback


def events_from_plays(plays: Sequence[dict], game_id: int) -> List[Event]:
    """ESPN `plays` array -> Events, numbered 1..N like hoopR's game_play_number.

    ESPN's own sequenceNumber is used only for ORDERING; the emitted `seq` is a
    dense 1-based ordinal, exactly as hoopR's game_play_number is, so a state
    built live is directly comparable to the same state built offline.
    """
    ordered = sorted(
        ((_sequence_key(p, i), i, p) for i, p in enumerate(plays)),
        key=lambda x: (x[0], x[1]),
    )
    out: List[Event] = []
    for n, (_key, _i, p) in enumerate(ordered, start=1):
        period = _int((p.get("period") or {}).get("number"), 1) or 1
        clock = (p.get("clock") or {}).get("displayValue") or ""
        team = (p.get("team") or {}).get("id")
        out.append(
            Event(
                game_id=game_id,
                seq=n,
                period=period,
                clock_seconds=clock_to_seconds(clock),
                home_score=_int(p.get("homeScore"), 0) or 0,
                away_score=_int(p.get("awayScore"), 0) or 0,
                event_type=play_type_text(p),
                team_id=_int(team),
                score_value=_int(p.get("scoreValue"), 0) or 0,
                scoring_play=bool(p.get("scoringPlay")),
                shooting_play=bool(p.get("shootingPlay")),
                text=(p.get("text") or ""),
            )
        )
    return out


@dataclass(frozen=True)
class GameHeader:
    """The pregame facts the summary endpoint carries, before ratings."""
    game_id: int
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    neutral_site: bool
    status: str
    period: int
    clock_display: str
    home_score: int
    away_score: int

    @property
    def is_final(self) -> bool:
        return self.status == STATUS_FINAL

    @property
    def is_live(self) -> bool:
        return self.status not in (STATUS_PRE, STATUS_FINAL)


def header_from_summary(summary: dict) -> GameHeader:
    """Pull team ids, neutral-site flag and status out of a summary payload."""
    header = summary.get("header") or {}
    comps = header.get("competitions") or [{}]
    comp = comps[0]
    home = away = None
    for c in comp.get("competitors") or []:
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    if home is None or away is None:
        raise ValueError("summary payload has no home/away competitors")

    st = ((comp.get("status") or {}).get("type") or {})
    return GameHeader(
        game_id=_int(header.get("id") or comp.get("id")) or 0,
        home_team_id=_int((home.get("team") or {}).get("id")) or 0,
        away_team_id=_int((away.get("team") or {}).get("id")) or 0,
        home_name=((home.get("team") or {}).get("displayName") or ""),
        away_name=((away.get("team") or {}).get("displayName") or ""),
        neutral_site=bool(comp.get("neutralSite")),
        status=st.get("name") or "",
        period=_int((comp.get("status") or {}).get("period"), 0) or 0,
        clock_display=str((comp.get("status") or {}).get("displayClock") or ""),
        home_score=_int(home.get("score"), 0) or 0,
        away_score=_int(away.get("score"), 0) or 0,
    )


def parse_summary(summary: dict) -> tuple[List[Event], GameHeader]:
    """One summary payload -> (events, header). No network, no state."""
    h = header_from_summary(summary)
    return events_from_plays(summary.get("plays") or [], h.game_id), h


def scoreboard_games(scoreboard: dict) -> List[dict]:
    """Flatten a scoreboard payload to one dict per game."""
    out = []
    for ev in scoreboard.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        st = ((comp.get("status") or {}).get("type") or {})
        home = away = None
        for c in comp.get("competitors") or []:
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        out.append({
            "game_id": _int(ev.get("id")) or 0,
            "name": ev.get("shortName") or ev.get("name") or "",
            "status": st.get("name") or "",
            "state": st.get("state") or "",
            "completed": bool(st.get("completed")),
            "neutral_site": bool(comp.get("neutralSite")),
            "home_team_id": _int((home or {}).get("team", {}).get("id")) or 0,
            "away_team_id": _int((away or {}).get("team", {}).get("id")) or 0,
            "start": ev.get("date") or "",
        })
    return out


# --------------------------------------------------------------------------
# Network. Kept in one small class so everything above stays testable offline.
# --------------------------------------------------------------------------
class EspnClient:
    """Minimal, dependency-free ESPN reader.

    Deliberately synchronous and tiny: the poller runs these in a thread pool,
    which keeps the asyncio loop free of a third-party HTTP dependency.
    """

    def __init__(self, timeout: float = 10.0, user_agent: str | None = None,
                 base_url: str | None = None):
        self.timeout = timeout
        self.user_agent = user_agent or os.environ.get(
            "CBBWP_USER_AGENT", DEFAULT_USER_AGENT)
        # Read per instance, not at import, so a replay run is one env var and
        # needs no reload of an already-imported module.
        self.base_url = (base_url or os.environ.get(ESPN_BASE_ENV)
                         or SITE_API).rstrip("/")

    @property
    def is_replay(self) -> bool:
        """True when pointed at something other than ESPN itself.

        Anything that reports a dry run should say so with this, so a replay is
        never mistaken for a night of real games in a log or a smoke result.
        """
        return self.base_url != SITE_API

    def _get(self, url: str, params: dict | None = None) -> dict:
        if params:
            q = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Do not retry: a 403 here is a deterministic edge rule, not a blip,
            # so a retry only burns clock during a live game. Fail loudly, and
            # name the value that was refused.
            if e.code in (403, 429):
                raise urllib.error.HTTPError(
                    e.url, e.code, f"{e.reason} -- user-agent {self.user_agent!r} "
                    "was refused; set CBBWP_USER_AGENT", e.headers, e.fp) from None
            raise

    def scoreboard(self, date: str | None = None, groups: str = "50",
                   limit: int = 500) -> dict:
        """`date` is YYYYMMDD. groups=50 is Division I."""
        return self._get(self.base_url + "/scoreboard",
                         {"dates": date, "groups": groups, "limit": limit})

    def summary(self, event_id: int | str) -> dict:
        return self._get(self.base_url + "/summary", {"event": event_id})

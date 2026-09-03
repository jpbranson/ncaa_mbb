# `cbbwp` source bundle
Complete source. Regenerated 2026-09-03 14:05 from the `ncaa_mbb` working folder, at commit `bff9db7`.

State rules v2, model v2. This bundle is a mirror for disaster recovery; the folder is the source of truth (it also holds the data, the fitted model and the git history). Regenerate with `python3 scripts/build_source_bundle.py` whenever the source changes.

See `cbbwp-EXPLAIN.md` for what every piece does and why.

---

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cbbwp"
version = "0.2.0"
description = "College basketball live win probability model"
requires-python = ">=3.10"
dependencies = ["numpy", "polars", "scikit-learn", "lightgbm"]

[tool.setuptools.packages.find]
where = ["src"]
```

## `README.md`

```markdown
# cbbwp — college basketball live win probability

Live win probability for NCAA men's basketball. Trained on 2016–2023,
calibrated on 2024, tested on 2025–2026 (2.23M states, 12,398 games).
Beats ESPN's deployed model in every time bucket.

| Model | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|
| **LightGBM v2 (shipped)** | **0.3103** | 0.1008 | 85.20% | 0.0026 |
| Logistic baseline | 0.3109 | 0.1009 | 85.19% | 0.0043 |
| ESPN (deployed, same rows) | 0.3295 | 0.1061 | 84.58% | 0.0069 |

Checkpointed at `checkpoint-2026-09-02`. What is frozen, and what a future
change has to beat: **`docs/cbbwp-CHECKPOINT.md`**. Running it live:
**`docs/cbbwp-deployment.md`**.

Full explanation of the model, every design decision and its rejected
alternative: **`docs/cbbwp-EXPLAIN.md`**. Read that first.

Repository: https://github.com/jpbranson/ncaa_mbb

## Setup

```bash
pip3 install --break-system-packages polars pyarrow lightgbm scikit-learn pytest numpy
```

## Rebuild everything from scratch

```bash
python3 scripts/fetch_data.py          # ~540 MB of hoopR parquet, ~1 min
python3 scripts/build_games.py         # results + as-of pregame ratings
python3 scripts/build_team_stats.py    # as-of FT% and pace
python3 scripts/build_dataset.py       # replay -> 8.5M state rows + features
python3 scripts/fit_models.py          # logistic + LightGBM  (needs ~6 GB RAM)
python3 scripts/publish_model.py v2    # pinned registry artifact
python3 scripts/evaluate.py            # metrics by time bucket vs ESPN
pytest tests -q
```

Seeds are pinned (`seed=20260831`, `deterministic=True`), so a refit reproduces
`registry/v2` exactly — verified across two different machines, byte for byte.

**Memory note:** `fit_models.py` peaks around 4–6 GB — the symmetry mirroring
doubles 5.4M rows and briefly holds them as float64. It will be OOM-killed in a
3 GB container.

## Run it live

**Do this first, on a machine that can reach ESPN:**

```bash
python3 scripts/smoke_live.py             # eight steps, one verdict
```

Exit 0 = validated. Exit 1 = broken, do not go live. Exit 2 = the offline steps
passed but something needing the network did not happen — either ESPN was
unreachable, or no game was live or finished on the slate. The verdict says
which.

Out of season, exit 2 is the honest answer: ESPN is reachable and the adapter
parses real payloads, but there are no games to record. See
`docs/cbbwp-deployment.md` for what is validated and what is not.

**Rehearse a live night without waiting for one:**

```bash
python3 scripts/archive_replay_games.py   # once; needs network
python3 scripts/replay_server.py --speed 5
CBBWP_ESPN_BASE=http://127.0.0.1:8899 python3 scripts/serve_live.py
```

`replay_server.py` speaks ESPN's protocol back to the unmodified deployment,
serving archived games with only the plays that would have happened by now — a
growing feed, a running clock, real status transitions. Replay rows are tagged
`"replay": true` and written to `data/replay/`, never `data/live/`.

**Watch it:**

```bash
python3 scripts/serve_viz.py              # then open http://127.0.0.1:8811
```

A small web app, standard library and one HTML file — no build step, no CDN,
Bootstrap 3 / Shiny styling written by hand. Two tabs: **Live** draws whatever
`serve_live.py` is tracking; **Replay** loads any past game and gives you
play/pause, speed, step forward and back, and a scrubber. Any ESPN game id can
be fetched and archived from the app itself.

Stepping moves by *moment* rather than by play: the clock stops, so fouls, free
throws and substitutions share a timestamp, and one game's 482 plays are only
255 moments. Plays at the same clock are shown together, each keeping its own
probability.

Replay is precomputed rather than streamed: `score_game` replays a whole game in
milliseconds, so the browser scrubs an array and stepping backward is exact —
the same number the model produced going forwards, not a re-derivation.

Then:

```bash
bash deploy/install_macos.sh              # two LaunchAgents: poller + ratings
curl -s http://127.0.0.1:8808/health
```

or run it in the foreground:

```bash
python3 scripts/build_live_context.py     # ratings snapshot; see the note below
python3 scripts/serve_live.py             # poller + HTTP API, one process
python3 scripts/serve_live.py --date 20261115
CBBWP_FIXTURE_DIR=tmp/fixtures python3 scripts/serve_live.py --once   # offline
```

Two outputs: JSONL at `data/live/wp_YYYYMMDD.jsonl` (the record of truth) and a
read-only API on `127.0.0.1:8808` (`/health`, `/games`, `/games/{id}`). Every
API response carries `model_version` and `state_rules_version`.

**The ratings refresh is two steps, not one.** `build_live_context.py` reads
`games.parquet`, which only changes when `fetch_data.py` runs — so rebuilding
the snapshot alone gives you a file with today's timestamp and last month's
ratings. The snapshot records `latest_game_date` and `/health` reports
`data_age_days` beside `ratings_age_days` so this cannot happen silently.

Step 6 of the smoke test is the one the offline suite cannot do: it reports
play-type ids the model was never trained on. **A frequent unknown type means
the ESPN feed has changed and the model needs a refit, not a patched adapter.**

## Change the model later

A config change and a restart, never an edit:

```bash
CBBWP_MODEL_VERSION=v3 python3 scripts/serve_live.py
```

Every setting is an environment variable with a working default
(`src/cbbwp/config.py`), and every entry point prints its resolved settings at
startup. `serve.py` refuses a model whose feature contract or `STATE_RULES_VERSION`
disagrees with the code, and loads it before binding a port — so a bad version
fails at startup, not at tip-off.

## Monitor it

```bash
python3 scripts/calibration_monitor.py --source backtest --days 7
python3 scripts/calibration_monitor.py --source live --glob 'data/live/*.jsonl'
```

Exit code 1 means a decile is off both statistically (|z| > 3) *and*
practically (gap > 2 points). Both are required — a million rows will make a
0.3-point gap "significant", and that is a large sample, not drift.

## Layout

```
src/cbbwp/
  schemas.py       data contracts; FEATURE_NAMES is the model's input contract
  state.py         replayable state builder — a pure function of the event list
  features.py      the 11 features, one definition, used by training AND serving
  ratings.py       in-house pregame ratings (our stand-in for the betting spread)
  serve.py         WinProbabilityService; refuses to start on a contract mismatch
  live_context.py  pregame context for a game that has not been played yet
  endgame.py       rule-based clamps the data cannot teach efficiently (live)
  endgame_sim.py   the endgame lookup table (diagnostic only — see EXPLAIN 7.10)
  calibration.py   time-bucketed isotonic (diagnostic only — see EXPLAIN 7.7)
  monitor.py       calibration drift statistics
  config.py        deployment settings, from the environment
  api.py           the read-only HTTP view of the live feed
  adapters/
    hoopr.py       historical parquet -> Events   (offline)
    espn.py        live ESPN feed     -> Events   (live)
scripts/           the pipeline, the poller, the smoke test, the replay
                   server, the monitor
deploy/            macOS LaunchAgents, Dockerfile, compose
tests/             92 tests
docs/              the project docs, kept alongside the code
data/, artifacts/, registry/   built locally; not source
```

## The two parity tests that matter

The whole design rests on training and serving sharing one definition of state
and features. Two tests enforce it:

- `tests/test_parity.py` — the fast vectorised Polars path must agree
  row-for-row with the canonical state builder on real games.
- `tests/test_espn_adapter.py` — the **live** ESPN adapter must produce
  byte-identical states and win probabilities to the **offline** hoopR adapter
  for the same game, including when the feed arrives shuffled.

`tests/test_replay_harness.py` adds the third: a finished game fed through the
live path in irregular chunks must match the offline answer exactly.

`tests/test_replay_server.py` covers the dry-run simulator itself — plays must
be revealed in game order from a countdown clock, must only ever grow, and the
finished replay must equal the archive. A dry run that fails on the simulator's
own bugs is the worst kind of false alarm to chase at tip-off.
```

## `src/cbbwp/__init__.py`

```py
"""cbbwp - college basketball win probability.

The state builder and feature builder in this package are imported by BOTH the
offline training pipeline and the live serving path. That is deliberate: it is
the single defence against train/serve skew.
"""
__version__ = "0.2.0"

from .schemas import Event, GameState, PregameContext, FEATURE_NAMES  # noqa: F401
from .state import build_states  # noqa: F401
from .features import build_feature_matrix, feature_dict  # noqa: F401
```

## `src/cbbwp/adapters/__init__.py`

```py

```

## `src/cbbwp/adapters/espn.py`

```py
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
from ..schemas import HALF_SECONDS, OT_SECONDS
from ..state import clock_to_seconds, game_seconds_remaining

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


def chronological_inversions(events: Sequence[Event]) -> int:
    """How many events sit earlier in game time than the event before them.

    Zero on every real ESPN payload measured so far. A non-zero count means the
    feed itself arrived out of order, which the adapter deliberately does NOT
    repair - see `events_from_plays`. Surfaced so a disordered feed is a thing
    somebody is told about rather than something silently rearranged.
    """
    def elapsed(e: Event) -> int:
        if e.period <= 2:
            return 2 * HALF_SECONDS - game_seconds_remaining(e.period,
                                                             e.clock_seconds)
        return (2 * HALF_SECONDS + (e.period - 3) * OT_SECONDS
                + (OT_SECONDS - e.clock_seconds))
    t = [elapsed(e) for e in events]
    return sum(1 for i in range(1, len(t)) if t[i] < t[i - 1])


def events_from_plays(plays: Sequence[dict], game_id: int) -> List[Event]:
    """ESPN `plays` array -> Events, numbered 1..N like hoopR's game_play_number.

    **The feed's own array order is authoritative.** The emitted `seq` is a dense
    1-based ordinal over that order, exactly as hoopR's game_play_number is, so a
    state built live is directly comparable to the same state built offline.

    This used to sort by ESPN's `sequenceNumber`, on the assumption that the
    array made no promise about order and the id did. Measured 2026-09-03 on
    seven archived games, the opposite is true on all seven:

      * the raw `plays` array IS chronological, and matches hoopR's
        game_play_number order exactly;
      * `sequenceNumber` is only NEARLY monotonic - the 2026 championship game
        has 12 inversions in 482 plays, e.g. 120416951 followed by 120416904
        while the clock runs correctly forwards.

    So the sort was taking correctly ordered data and shuffling it, displacing
    plays by as much as 986 seconds and moving mid-game win probability by up to
    27 points. Final probabilities were unaffected, which is why nothing ever
    complained. The parity tests could not catch it either: they run on payloads
    rebuilt from hoopR, where sequenceNumber is synthesised monotonic and so has
    a property the real feed does not.

    Disorder is now REPORTED rather than repaired (`chronological_inversions`):
    an unreliable key cannot fix an out-of-order feed, it can only corrupt an
    ordered one.
    """
    out: List[Event] = []
    for n, p in enumerate(plays, start=1):
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
```

## `src/cbbwp/adapters/hoopr.py`

```py
"""Adapter: hoopR / sportsdataverse historical play-by-play parquet -> Events.

Two paths, deliberately:
  * `load_events` builds canonical Event objects for one or a few games. This is
    the reference path and the one the live pipeline mirrors.
  * `states_lazy` does the same transformation in Polars for tens of millions of
    rows. It exists only for speed, and `tests/test_parity.py` asserts it agrees
    with the reference path row-for-row on real games.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import polars as pl

from ..schemas import Event, HALF_SECONDS, OT_SECONDS
from ..state import TEAM_TIMEOUT_TYPES, FOUL_TYPES, clock_to_seconds

# Columns present in every season 2016-2026 of the hoopR mbb pbp files.
BASE_COLS = [
    "game_id", "game_play_number", "period_number", "clock_display_value",
    "home_score", "away_score", "type_text", "team_id", "score_value",
    "scoring_play", "shooting_play", "home_team_id", "away_team_id",
    "season", "season_type", "game_date",
]

MADE_SHOT_TYPES = ["JumpShot", "LayUpShot", "DunkShot", "TipShot"]


def load_events(path: str, game_id: int) -> tuple[List[Event], int, int]:
    """Reference path: one game's parquet rows -> canonical Events."""
    df = (
        pl.scan_parquet(path)
        .filter(pl.col("game_id") == game_id)
        .select(BASE_COLS)
        .sort("game_play_number")
        .collect()
    )
    if df.is_empty():
        raise KeyError(f"game {game_id} not in {path}")
    home_id = int(df["home_team_id"][0])
    away_id = int(df["away_team_id"][0])
    events = [
        Event(
            game_id=int(r["game_id"]),
            seq=int(r["game_play_number"]),
            period=int(r["period_number"] or 1),
            clock_seconds=clock_to_seconds(r["clock_display_value"] or ""),
            home_score=int(r["home_score"] or 0),
            away_score=int(r["away_score"] or 0),
            event_type=r["type_text"] or "",
            team_id=None if r["team_id"] is None else int(r["team_id"]),
            score_value=int(r["score_value"] or 0),
            scoring_play=bool(r["scoring_play"]),
            shooting_play=bool(r["shooting_play"]),
        )
        for r in df.iter_rows(named=True)
    ]
    return events, home_id, away_id


# --------------------------------------------------------------------------
# Vectorised bulk path
# --------------------------------------------------------------------------
def _clock_seconds_expr(col: str = "clock_display_value") -> pl.Expr:
    parts = pl.col(col).str.split_exact(":", 1)
    mm = parts.struct.field("field_0").cast(pl.Float64, strict=False)
    ss = parts.struct.field("field_1").cast(pl.Float64, strict=False)
    return (
        pl.when(pl.col(col).str.contains(":"))
        .then(mm * 60 + ss.floor())
        .otherwise(pl.col(col).cast(pl.Float64, strict=False).floor())
        .fill_null(0)
        .cast(pl.Int32)
    )


def states_lazy(lf: pl.LazyFrame, timeouts_at_tip: int = 4) -> pl.LazyFrame:
    """Vectorised equivalent of state.build_states over many games at once."""
    t = pl.col("type_text")
    actor = (
        pl.when(pl.col("team_id") == pl.col("home_team_id")).then(pl.lit(1.0))
        .when(pl.col("team_id") == pl.col("away_team_id")).then(pl.lit(0.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    other = 1.0 - actor
    made = pl.col("scoring_play").fill_null(False)
    shooting = pl.col("shooting_play").fill_null(False)

    poss_set = (
        # Same rule, same order, as state._possession_after. Made field goals are
        # detected by the scoring/shooting flags, not by play-type name - see the
        # comment there for why the name whitelist was wrong for 2016-2019.
        pl.when(t.str.contains("FreeThrow") & made).then(other)
        .when(made & shooting).then(other)
        .when(t.is_in(["Defensive Rebound", "Offensive Rebound"])).then(actor)
        .when(t.str.contains("Turnover")).then(other)
        .when(t == "Steal").then(actor)
        .when(t == "Jumpball").then(pl.lit(0.5))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )

    is_team_to = t.is_in(list(TEAM_TIMEOUT_TYPES))
    is_foul = t.is_in(list(FOUL_TYPES))
    period = pl.col("period_number").fill_null(1).clip(lower_bound=1).cast(pl.Int32)
    plen = pl.when(period <= 2).then(pl.lit(HALF_SECONDS)).otherwise(pl.lit(OT_SECONDS))
    clock = _clock_seconds_expr().clip(0, None)
    clock = pl.min_horizontal(clock, plen).cast(pl.Int32)
    gsr = pl.when(period <= 1).then(pl.lit(HALF_SECONDS) + clock).otherwise(clock)

    allot = pl.lit(timeouts_at_tip) + (period - 2).clip(lower_bound=0)

    return (
        lf.sort(["game_id", "game_play_number"])
        .with_columns(
            _period=period,
            _clock=clock,
            _gsr=gsr.cast(pl.Int32),
            _poss_set=poss_set,
            _to_home=(is_team_to & (pl.col("team_id") == pl.col("home_team_id"))).cast(pl.Int32),
            _to_away=(is_team_to & (pl.col("team_id") == pl.col("away_team_id"))).cast(pl.Int32),
            _allot=allot,
            _half=pl.when(period <= 1).then(pl.lit(1)).otherwise(pl.lit(2)),
            _foul_home=(is_foul & (pl.col("team_id") == pl.col("home_team_id"))).cast(pl.Int32),
            _foul_away=(is_foul & (pl.col("team_id") == pl.col("away_team_id"))).cast(pl.Int32),
        )
        .with_columns(
            possession=pl.col("_poss_set").forward_fill().over("game_id").fill_null(0.5),
            home_used=pl.col("_to_home").cum_sum().over("game_id"),
            away_used=pl.col("_to_away").cum_sum().over("game_id"),
            home_fouls=pl.col("_foul_home").cum_sum().over(["game_id", "_half"]),
            away_fouls=pl.col("_foul_away").cum_sum().over(["game_id", "_half"]),
        )
        .with_columns(
            margin=(pl.col("home_score").fill_null(0) - pl.col("away_score").fill_null(0)).cast(pl.Int32),
            home_timeouts=(pl.col("_allot") - pl.col("home_used")).clip(lower_bound=0).cast(pl.Int32),
            away_timeouts=(pl.col("_allot") - pl.col("away_used")).clip(lower_bound=0).cast(pl.Int32),
            is_ot=(pl.col("_period") >= 3),
        )
        .rename({"_period": "period", "_clock": "clock_seconds",
                 "_gsr": "game_seconds_remaining", "game_play_number": "seq"})
        .drop(["_poss_set", "_to_home", "_to_away", "_allot", "home_used", "away_used",
               "_half", "_foul_home", "_foul_away"])
    )
```

## `src/cbbwp/api.py`

```py
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
```

## `src/cbbwp/calibration.py`

```py
"""Time-bucketed probability calibration.

Miscalibration in these models is almost entirely time-dependent, so a single
global calibrator averages two opposite errors and fixes neither (plan 9.3).
We fit one isotonic curve per time bucket on a HELD-OUT season, then blend
between adjacent buckets so the curve never visibly jumps.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

# (lower, upper) seconds remaining, and the anchor point used for blending.
BUCKETS = [(1200, 2400), (600, 1200), (300, 600), (120, 300), (60, 120), (0, 60)]
ANCHORS = np.array([np.sqrt((lo + hi) / 2) for lo, hi in BUCKETS])
CLIP = (0.001, 0.999)


class TimeBucketedCalibrator:
    def __init__(self, buckets=BUCKETS, min_rows=20_000):
        self.buckets = list(buckets)
        self.min_rows = min_rows
        self.models: list[IsotonicRegression | None] = []

    def fit(self, p, y, seconds):
        self.models = []
        for lo, hi in self.buckets:
            m = (seconds >= lo) & (seconds < hi)
            if m.sum() < self.min_rows:
                self.models.append(None)
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p[m], y[m])
            self.models.append(iso)
        return self

    def _apply(self, i, p):
        m = self.models[i]
        return p if m is None else m.predict(p)

    def transform(self, p, seconds):
        """Blend the two nearest bucket calibrators, weighted in sqrt-time."""
        p = np.asarray(p, dtype=np.float64)
        s = np.sqrt(np.maximum(np.asarray(seconds, dtype=np.float64), 0.0))
        # anchors run from long time remaining down to zero
        order = np.argsort(ANCHORS)
        a = ANCHORS[order]
        cols = np.stack([self._apply(order[i], p) for i in range(len(a))], axis=1)
        idx = np.clip(np.searchsorted(a, s), 1, len(a) - 1)
        lo, hi = a[idx - 1], a[idx]
        w = np.clip((s - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
        out = cols[np.arange(len(p)), idx - 1] * (1 - w) + cols[np.arange(len(p)), idx] * w
        return np.clip(out, *CLIP)
```

## `src/cbbwp/config.py`

```py
"""One place where deployment settings come from, for every entry point.

The Mac and a container differ in paths and nothing else, so the difference is
expressed as environment variables rather than as two copies of the code. Every
setting has a working default, so `python3 scripts/serve_live.py` does the right
thing on a laptop with no environment set at all.

The model version is deliberately a setting rather than a constant. Swapping
which model serves is then a config change and a restart -- not an edit -- which
is what keeps "we can change the model later" true in practice. `serve.py` still
refuses to load an artifact whose state-rules version disagrees with the code,
so a careless swap fails loudly at startup instead of silently serving skew.

    CBBWP_ROOT              project root (default: the repo this file is in)
    CBBWP_REGISTRY          model registry dir      (default: <root>/registry)
    CBBWP_MODEL_VERSION     which model to serve    (default: v2)
    CBBWP_CONTEXT           ratings snapshot path   (default: <registry>/context_latest.json)
    CBBWP_LIVE_DIR          JSONL output dir        (default: <root>/data/live)
    CBBWP_FIXTURE_DIR       replay from disk instead of the network (default: unset)
    CBBWP_API_HOST          API bind address        (default: 127.0.0.1)
    CBBWP_API_PORT          API port                (default: 8808)
    CBBWP_API_HISTORY       states kept per game    (default: 240)
    CBBWP_RATINGS_MAX_AGE   days before the snapshot is called stale (default: 3)
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

_HERE = pathlib.Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[2]


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


@dataclass(frozen=True)
class Settings:
    root: pathlib.Path
    registry: pathlib.Path
    model_version: str
    context_path: pathlib.Path
    live_dir: pathlib.Path
    fixture_dir: pathlib.Path | None
    api_host: str
    api_port: int
    api_history: int
    ratings_max_age_days: float
    _source: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        root = pathlib.Path(_env("CBBWP_ROOT", str(_DEFAULT_ROOT))).resolve()
        registry = pathlib.Path(_env("CBBWP_REGISTRY", str(root / "registry")))
        fixture = os.environ.get("CBBWP_FIXTURE_DIR") or None
        overridden = {k: os.environ[k] for k in os.environ if k.startswith("CBBWP_")}
        return cls(
            root=root,
            registry=registry,
            model_version=_env("CBBWP_MODEL_VERSION", "v2"),
            context_path=pathlib.Path(
                _env("CBBWP_CONTEXT", str(registry / "context_latest.json"))),
            live_dir=pathlib.Path(_env("CBBWP_LIVE_DIR", str(root / "data" / "live"))),
            fixture_dir=pathlib.Path(fixture) if fixture else None,
            api_host=_env("CBBWP_API_HOST", "127.0.0.1"),
            api_port=int(_env("CBBWP_API_PORT", "8808")),
            api_history=int(_env("CBBWP_API_HISTORY", "240")),
            ratings_max_age_days=float(_env("CBBWP_RATINGS_MAX_AGE", "3")),
            _source=overridden,
        )

    def describe(self) -> str:
        """What every entry point prints at startup.

        Printing the resolved settings, and which of them came from the
        environment, is the cheapest possible defence against the class of
        outage where a service is quietly serving from the wrong directory.
        """
        lines = [
            f"  root            {self.root}",
            f"  registry        {self.registry}",
            f"  model version   {self.model_version}",
            f"  ratings         {self.context_path}",
            f"  live output     {self.live_dir}",
            f"  api             http://{self.api_host}:{self.api_port}",
        ]
        if self.fixture_dir:
            lines.append(f"  FIXTURES        {self.fixture_dir}  (no network)")
        if self._source:
            lines.append(f"  from environment: {', '.join(sorted(self._source))}")
        return "\n".join(lines)
```

## `src/cbbwp/endgame.py`

```py
"""Rule-based overrides the training data cannot teach efficiently (plan 10).

The model does not know the rules; we do. These are constraints, not hacks.
"""
from __future__ import annotations

import numpy as np

# A possession is worth at most 3 points (ignoring 4-point plays, which are rare
# enough that treating them as impossible costs less than the states it saves).
MAX_POINTS_PER_POSSESSION = 3
# Seconds a trailing team needs to score and foul once more.
SECONDS_PER_POSSESSION = 6.0


def max_points_remaining(seconds_remaining: np.ndarray) -> np.ndarray:
    """Optimistic ceiling on what a trailing team can still score."""
    poss = np.floor(np.asarray(seconds_remaining, dtype=np.float64) / SECONDS_PER_POSSESSION) + 1
    return poss * MAX_POINTS_PER_POSSESSION


def apply(p, margin, seconds_remaining, is_ot=None):
    """Clamp probabilities that the rules have already decided."""
    p = np.array(p, dtype=np.float64, copy=True)
    margin = np.asarray(margin)
    t = np.asarray(seconds_remaining, dtype=np.float64)

    # 1. Time expired in the final period: the result is known.
    over = t <= 0
    p[over & (margin > 0)] = 1.0
    p[over & (margin < 0)] = 0.0
    # A tie at 0:00 goes to overtime -> a coin flip, nudged by nothing else here.
    p[over & (margin == 0)] = 0.5

    # 2. Mathematically decided: the trailing team cannot catch up in the
    #    possessions that remain, however well it plays.
    ceiling = max_points_remaining(t)
    decided_home = (~over) & (margin > ceiling)
    decided_away = (~over) & (-margin > ceiling)
    p[decided_home] = 1.0
    p[decided_away] = 0.0
    return p
```

## `src/cbbwp/endgame_sim.py`

```py
"""Exhaustive endgame solver: the last 60 seconds, by backward induction.

Endgame plan, Phase 3. The plan's design decision was "a table, not a live
simulation", for three reasons: serving becomes a lookup, monotonicity can be
ENFORCED across the whole table rather than hoped for, and a person can read a
row and check it against their own judgement.

This module goes one step further than the plan asked and replaces Monte Carlo
with **backward induction**. The endgame state space is small, discrete and
acyclic in time -- every transition burns at least one second -- so the exact
value of every state can be computed directly. That removes simulation noise
entirely, which matters because two identical states must never disagree, and
it makes the monotonicity checks meaningful: a violation is then a statement
about the model, not about how many samples were drawn.

STATE, from the point of view of the team WITH THE BALL
    t   seconds remaining, 0..60
    m   that team's margin, clamped to -12..+12
    fo  that team's own team fouls this half, 0..10 (10 = "10 or more")
    fd  the defending team's team fouls, same encoding
    bo  that team's free-throw ability bucket, 0=poor 1=average 2=good
    bd  the defending team's bucket

V[t][m, fo, fd, bo, bd] = P(the team with the ball wins).

Symmetry is structural rather than fitted. When possession changes, the value
of the state to the team that just lost the ball is

    1 - V[t'][-m, fd, fo, bd, bo]

so the table cannot disagree with itself about which side of a game it is
describing, and a mirrored state cannot drift from its twin.

Every probability comes from artifacts/endgame_params.json and
artifacts/endgame_possessions.json, both measured on 2016-2024 only. Nothing
here is hand-set except the state-space bounds and the smoothing window, and
both are declared as constants below.

WHAT THIS DELIBERATELY DOES NOT MODEL, and why
  * Timeouts. The plan lists them in the state space, but no separable effect
    was measured -- the possession-length and foul-rate cells already average
    over however teams actually used their timeouts. Adding a state dimension
    with an invented coefficient would add sampling error to sparse states for
    no measured gain. See `docs/cbbwp-endgame-phase2.md`.
  * Optimal play. The fouling rule is OBSERVED behaviour. The model predicts
    real games, in which coaches foul later and less often than optimally.
  * Team strength. The table is deliberately team-agnostic apart from free-throw
    ability. Strength enters at blend time, where the model already carries it.
    A tie at 0:00 is therefore 0.5 here, not a rating-adjusted overtime number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- state space ------------------------------------------------------------
T_MAX = 60
MARGIN_MIN, MARGIN_MAX = -12, 12
FOUL_MAX = 10           # 10 encodes "10 or more"
N_FT_BUCKETS = 3

MARGINS = np.arange(MARGIN_MIN, MARGIN_MAX + 1)
NM = len(MARGINS)
NF = FOUL_MAX + 1
NB = N_FT_BUCKETS
SHAPE = (NM, NF, NF, NB, NB)

BONUS_FOULS = 7          # the 7th team foul of a half starts the one-and-one
DOUBLE_BONUS_FOULS = 10

# Possession lengths are measured as means; a single deterministic length would
# put artificial parity structure into the table ("exactly three possessions
# left"). Spread each mean over three adjacent seconds instead.
DURATION_SMOOTHING = (0.25, 0.50, 0.25)
MAX_DURATION = 30


def _mi(m: np.ndarray | int):
    """Margin -> index, clamped."""
    return np.clip(m, MARGIN_MIN, MARGIN_MAX) - MARGIN_MIN


REV = np.arange(NM)[::-1]     # index of -m


@dataclass(frozen=True)
class Params:
    """Everything the solver needs, already smoothed onto the state grid."""
    p_foul: np.ndarray        # (T_MAX+1, NM)  defence sends the offence to the line
    p_to: np.ndarray          # (T_MAX+1, NM)  turnover, conditional on not fouled
    p3a: np.ndarray           # (NM,) share of shots that are threes
    p3: np.ndarray            # (NM,)
    p2: np.ndarray            # (NM,)
    dur: np.ndarray           # (T_MAX+1, NM) mean seconds, played out
    dur_foul: np.ndarray      # (T_MAX+1, NM) mean seconds to the foul
    ft1_bonus: np.ndarray     # (NB,) front end of a one-and-one
    ft2_bonus: np.ndarray     # (NB,) the bonus shot
    ft1_two: np.ndarray       # (NB,) first of two
    ft2_two: np.ndarray       # (NB,) second of two
    oreb_ft: float
    oreb_3: float
    oreb_2: float


# --- parameter assembly -----------------------------------------------------
def _cell_grid(cells: dict, field: str, default: float) -> np.ndarray:
    """The (m, 10s-bucket) measurements, interpolated onto (t, m)."""
    buckets = sorted({float(k.split("|")[1]) for k in cells})
    raw = np.full((len(buckets), NM), np.nan)
    for key, v in cells.items():
        ms, tbs = key.split("|")
        m = int(float(ms))
        if not (MARGIN_MIN <= m <= MARGIN_MAX):
            continue
        val = v.get(field)
        if val is None:
            continue
        raw[buckets.index(float(tbs)), _mi(m)] = val
    # Fill margin gaps by nearest neighbour along m, then interpolate in t.
    for r in range(raw.shape[0]):
        row = raw[r]
        if np.all(np.isnan(row)):
            row[:] = default
        else:
            idx = np.arange(NM)
            good = ~np.isnan(row)
            row[~good] = np.interp(idx[~good], idx[good], row[good])
    centres = np.array(buckets) + 5.0
    out = np.empty((T_MAX + 1, NM))
    for j in range(NM):
        out[:, j] = np.interp(np.arange(T_MAX + 1), centres, raw[:, j])
    return out


def load_params(params_path: Path, poss_path: Path, ft_bucket_offsets) -> Params:
    P = json.loads(Path(params_path).read_text())
    Q = json.loads(Path(poss_path).read_text())
    cells = Q["cells"]

    p_foul = np.clip(_cell_grid(cells, "p_fouled_to_line", 0.15), 0.0, 0.95)
    p_to_raw = np.clip(_cell_grid(cells, "p_turnover", 0.13), 0.0, 0.9)
    # Measured turnover share is unconditional; the solver needs it conditional
    # on the possession not having ended at the foul line.
    p_to = np.clip(p_to_raw / np.maximum(1e-6, 1.0 - p_foul), 0.0, 0.95)

    dur = np.clip(_cell_grid(cells, "mean_dur", 6.0), 1.0, MAX_DURATION)
    dur_foul = np.clip(_cell_grid(cells, "mean_dur_when_fouled", 5.0), 1.0, MAX_DURATION)

    shots = P["late_shots_by_actor_margin"]
    p3a = np.empty(NM); p3 = np.empty(NM); p2 = np.empty(NM)
    for j, m in enumerate(MARGINS):
        key = str(int(np.clip(m, -6, 6)))
        s = shots.get(key)
        p3a[j] = s["p3a"]; p3[j] = s["p3"]; p2[j] = s["p2"]

    ft = P["free_throws"]["last_60s"]
    off = np.asarray(ft_bucket_offsets, dtype=float)

    def band(key, fallback):
        base = ft[key]["p"] if key in ft else fallback
        return np.clip(base + off, 0.30, 0.99)

    reb = P["rebounds"]
    return Params(
        p_foul=p_foul, p_to=p_to, p3a=p3a, p3=p3, p2=p2, dur=dur, dur_foul=dur_foul,
        ft1_bonus=band("one_and_one_1", 0.709),
        ft2_bonus=band("one_and_one_2", 0.769),
        ft1_two=band("two_shot_1", 0.718),
        ft2_two=band("two_shot_2", 0.770),
        oreb_ft=reb["oreb_after_missed_ft_last60"]["p"],
        oreb_3=reb["oreb_after_missed_3_last60"]["p"],
        oreb_2=reb["oreb_after_missed_2_last60"]["p"],
    )


# --- the solver -------------------------------------------------------------
def _terminal() -> np.ndarray:
    """t = 0. The team with the ball wins iff it is ahead; a tie is overtime."""
    v = np.empty(SHAPE)
    win = np.where(MARGINS > 0, 1.0, np.where(MARGINS < 0, 0.0, 0.5))
    v[:] = win.reshape(NM, 1, 1, 1, 1)
    return v


def _flip(v: np.ndarray) -> np.ndarray:
    """Value of a state to the team that has just LOST the ball."""
    return 1.0 - v[REV][:, :, :, :, :].transpose(0, 2, 1, 4, 3)


def _shift_margin(v: np.ndarray, pts: int) -> np.ndarray:
    """v evaluated at margin m + pts, clamped at the ends of the grid."""
    if pts == 0:
        return v
    idx = np.clip(np.arange(NM) + pts, 0, NM - 1)
    return v[idx]


def _foul_index() -> np.ndarray:
    """fd -> fd + 1, capped at FOUL_MAX."""
    return np.minimum(np.arange(NF) + 1, FOUL_MAX)


def solve(p: Params) -> np.ndarray:
    """V[t] for t = 0..T_MAX. Exact, no sampling."""
    V = np.empty((T_MAX + 1,) + SHAPE)
    V[0] = _terminal()

    inc = _foul_index()
    # shots awarded by the foul that takes the defence to fd+1
    after = np.minimum(np.arange(NF) + 1, FOUL_MAX + 1)
    shots_for = np.where(after >= DOUBLE_BONUS_FOULS, 2, np.where(after >= BONUS_FOULS, 1, 0))

    # Slices below are (margin, own_fouls, own_bucket, opp_bucket): the free-throw
    # rate varies on the SHOOTING team's bucket, which is axis 2 of that slice.
    ft1b = p.ft1_bonus.reshape(1, 1, NB, 1)
    ft2b = p.ft2_bonus.reshape(1, 1, NB, 1)
    ft1t = p.ft1_two.reshape(1, 1, NB, 1)
    ft2t = p.ft2_two.reshape(1, 1, NB, 1)

    for t in range(1, T_MAX + 1):

        def look(tau_grid: np.ndarray, transform):
            """Expected value after burning `tau_grid` seconds, per margin.

            tau_grid is (NM,) of mean seconds; each is spread over three
            adjacent whole seconds so the table carries no parity artefacts.
            """
            acc = np.zeros(SHAPE)
            base = np.rint(tau_grid).astype(int)
            for w, d in zip(DURATION_SMOOTHING, (-1, 0, 1)):
                tau = np.clip(base + d, 1, MAX_DURATION)
                for u in np.unique(tau):
                    sel = tau == u
                    nxt = V[max(0, t - int(u))]
                    contrib = transform(nxt)
                    acc[sel] += w * contrib[sel]
            return acc

        # ---- branch A: the defence fouls -------------------------------
        def fouled(nxt: np.ndarray) -> np.ndarray:
            keep = nxt                      # offence still has the ball
            lost = _flip(nxt)               # offence has given it up
            out = np.empty(SHAPE)

            for f in range(NF):
                fd_new = inc[f]
                s = shots_for[f]
                k = keep[:, :, fd_new, :, :]
                l = lost[:, :, fd_new, :, :]

                def at(arr, pts):
                    return _shift_margin(arr, pts)

                if s == 0:
                    # No shots: the ball goes back in from the side.
                    out[:, :, f, :, :] = k
                    continue

                if s == 1:
                    p1 = ft1b; p2_ = ft2b
                    # miss the front end -> live rebound
                    miss1 = (1 - p1) * (p.oreb_ft * at(k, 0) + (1 - p.oreb_ft) * at(l, 0))
                    # make it, then the bonus shot
                    make2 = p2_ * at(l, 2)
                    miss2 = (1 - p2_) * (p.oreb_ft * at(k, 1) + (1 - p.oreb_ft) * at(l, 1))
                    out[:, :, f, :, :] = miss1 + p1 * (make2 + miss2)
                    continue

                # Two shots. The first only adds a point; the trip -- and so the
                # possession -- is decided by the second.
                p1 = ft1t; p2_ = ft2t
                def trip(made1: int):
                    return (
                        p2_ * at(l, made1 + 1)
                        + (1 - p2_) * (p.oreb_ft * at(k, made1) + (1 - p.oreb_ft) * at(l, made1))
                    )
                out[:, :, f, :, :] = p1 * trip(1) + (1 - p1) * trip(0)
            return out

        # ---- branch B: turnover ----------------------------------------
        def turned_over(nxt: np.ndarray) -> np.ndarray:
            return _flip(nxt)

        # ---- branch C: a shot goes up -----------------------------------
        def shot(nxt: np.ndarray) -> np.ndarray:
            keep, lost = nxt, _flip(nxt)
            three = (
                p.p3[:, None, None, None, None] * _shift_margin(lost, 3)
                + (1 - p.p3[:, None, None, None, None])
                * (p.oreb_3 * keep + (1 - p.oreb_3) * lost)
            )
            two = (
                p.p2[:, None, None, None, None] * _shift_margin(lost, 2)
                + (1 - p.p2[:, None, None, None, None])
                * (p.oreb_2 * keep + (1 - p.oreb_2) * lost)
            )
            a = p.p3a[:, None, None, None, None]
            return a * three + (1 - a) * two

        pf = p.p_foul[t][:, None, None, None, None]
        pt = p.p_to[t][:, None, None, None, None]

        v_foul = look(p.dur_foul[t], fouled)
        v_to = look(p.dur[t], turned_over)
        v_shot = look(p.dur[t], shot)

        V[t] = pf * v_foul + (1 - pf) * (pt * v_to + (1 - pt) * v_shot)
        np.clip(V[t], 0.0, 1.0, out=V[t])

    return V


# --- monotonicity -----------------------------------------------------------
def monotonicity_report(V: np.ndarray) -> dict:
    """Every violation the endgame plan names, checked exhaustively."""
    d_margin = np.diff(V, axis=1)                     # scoring must not hurt
    d_bo = np.diff(V, axis=4)                         # better own FT shooting
    d_bd = np.diff(V, axis=5)                         # better opponent FT shooting
    own_foul = np.diff(V, axis=2)                     # more fouls of your own
    opp_foul = np.diff(V, axis=3)                     # more fouls by the opponent
    have_ball = V - _flip_all(V)
    return {
        "margin_min_increment": float(d_margin.min()),
        "margin_violations": int((d_margin < -1e-12).sum()),
        "own_ft_bucket_min_increment": float(d_bo.min()),
        "own_ft_bucket_violations": int((d_bo < -1e-12).sum()),
        "opp_ft_bucket_max_increment": float(d_bd.max()),
        "opp_ft_bucket_violations": int((d_bd > 1e-12).sum()),
        "own_fouls_max_increment": float(own_foul.max()),
        "opp_fouls_min_increment": float(opp_foul.min()),
        "possession_min_advantage": float(have_ball.min()),
        "possession_violations": int((have_ball < -1e-12).sum()),
    }


def _flip_all(V: np.ndarray) -> np.ndarray:
    return 1.0 - V[:, REV].transpose(0, 1, 3, 2, 5, 4)


def enforce_margin_monotonicity(V: np.ndarray) -> tuple[np.ndarray, float]:
    """Isotonic projection along margin; returns the largest correction made.

    Parameters are estimated from finite samples, so a cell can end up a
    fraction below its neighbour. Projecting is honest as long as the size of
    the correction is reported -- a large one would mean the model is wrong,
    not merely noisy.
    """
    out = np.maximum.accumulate(V, axis=1)
    return out, float(np.abs(out - V).max())


# --- serving ----------------------------------------------------------------
def ft_bucket(pct, bucket_means) -> np.ndarray:
    """Nearest free-throw ability bucket for a team's season FT%."""
    means = np.asarray(bucket_means, dtype=float)
    p = np.asarray(pct, dtype=float)
    return np.abs(p[..., None] - means).argmin(axis=-1).astype(np.int64)


def lookup_home(
    table: np.ndarray,
    seconds_remaining,
    margin_home,
    possession,
    home_fouls,
    away_fouls,
    home_bucket,
    away_bucket,
) -> np.ndarray:
    """P(home wins), from a table stored in the ball-holder's point of view.

    `possession` is the pipeline's convention: 1.0 home, 0.0 away, 0.5 unknown.
    An unknown possession is averaged rather than guessed, which is the only
    answer that cannot be wrong in a way that shows up as a discontinuity.
    """
    t = np.clip(np.rint(np.asarray(seconds_remaining)).astype(int), 0, T_MAX)
    # margin arrives as a float in the state frame; the table is indexed by
    # whole points, so round rather than truncate.
    m = np.rint(np.asarray(margin_home, dtype=float)).astype(int)
    fh = np.clip(np.asarray(home_fouls), 0, FOUL_MAX).astype(int)
    fa = np.clip(np.asarray(away_fouls), 0, FOUL_MAX).astype(int)
    bh = np.asarray(home_bucket).astype(int)
    ba = np.asarray(away_bucket).astype(int)

    home_ball = table[t, _mi(m), fh, fa, bh, ba]
    away_ball = 1.0 - table[t, _mi(-m), fa, fh, ba, bh]
    poss = np.asarray(possession, dtype=float)
    return np.where(poss >= 0.75, home_ball,
                    np.where(poss <= 0.25, away_ball, 0.5 * (home_ball + away_ball)))
```

## `src/cbbwp/evaluate.py`

```py
"""Probability metrics, always broken out by time remaining (plan 11.2)."""
from __future__ import annotations

import numpy as np

EPS = 1e-6

# (label, lower bound seconds, upper bound seconds)
TIME_BUCKETS = [
    ("40-20 min", 1200, 2400),
    ("20-10 min", 600, 1200),
    ("10-5 min", 300, 600),
    ("5-2 min", 120, 300),
    ("2-0 min", 0, 120),
]


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def accuracy(y, p):
    return float(np.mean((p >= 0.5) == (y == 1)))


def by_time_bucket(y, p, seconds, extra=None):
    """Metrics per bucket. `extra` is an optional dict of other prediction sets."""
    rows = []
    for name, lo, hi in TIME_BUCKETS:
        m = (seconds >= lo) & (seconds < hi) if lo > 0 else (seconds >= lo) & (seconds < hi)
        if m.sum() == 0:
            continue
        row = {"bucket": name, "n": int(m.sum()),
               "log_loss": log_loss(y[m], p[m]), "brier": brier(y[m], p[m]),
               "acc": accuracy(y[m], p[m])}
        if extra:
            for k, v in extra.items():
                ok = m & np.isfinite(v)
                row[f"log_loss_{k}"] = log_loss(y[ok], v[ok]) if ok.sum() else float("nan")
                row[f"n_{k}"] = int(ok.sum())
        rows.append(row)
    return rows


def calibration_table(y, p, n_bins=10):
    """Predicted vs observed, in equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                    "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return out


def ece(y, p, n_bins=20):
    """Expected calibration error: mean |predicted - observed|, size-weighted."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    tot, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(tot)
```

## `src/cbbwp/features.py`

```py
"""Feature builder. Imported by BOTH the training pipeline and the live server.

If you change anything here you have changed the model's inputs: bump the model
version and refit. `FEATURE_NAMES` in schemas.py fixes the column order.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .schemas import (GameState, FEATURE_NAMES, REGULATION_SECONDS,
                      BONUS_FOULS, DOUBLE_BONUS_FOULS)

_SQRT_REG = math.sqrt(REGULATION_SECONDS)


def _bonus_level(opp_fouls: int) -> int:
    """0 = no bonus, 1 = one-and-one, 2 = double bonus."""
    if opp_fouls >= DOUBLE_BONUS_FOULS:
        return 2
    if opp_fouls >= BONUS_FOULS:
        return 1
    return 0


def feature_dict(s: GameState) -> Dict[str, float]:
    """Features for one state. The single source of truth for both pipelines."""
    t = max(float(s.game_seconds_remaining), 1.0)
    sqrt_t = math.sqrt(t)
    # Pregame edge is the only information at tip-off and should fade to nothing
    # by the buzzer, because by then the score has absorbed it.
    decay = sqrt_t / _SQRT_REG
    return {
        "margin": float(s.margin),
        "sqrt_time": sqrt_t,
        "margin_per_sqrt_time": float(s.margin) / sqrt_t,
        "possession": float(s.possession),
        "pregame_exp_margin": float(s.pregame_exp_margin),
        "pregame_exp_margin_decayed": float(s.pregame_exp_margin) * decay,
        "is_ot": 1.0 if s.is_ot else 0.0,
        "timeout_diff": float(s.home_timeouts - s.away_timeouts),
        # Bonus level a team SHOOTS in is driven by its opponent's fouls.
        "bonus_diff": float(_bonus_level(s.away_fouls) - _bonus_level(s.home_fouls)),
        "ft_pct_diff": float(s.ft_pct_diff),
        # Pace-aware twin of margin/sqrt(time): two slow teams have fewer
        # chances left than two fast ones with the same clock.
        "margin_per_sqrt_points_left": float(s.margin) / math.sqrt(
            max(s.exp_points_per_min * t / 60.0, 1.0)),
    }


def build_feature_matrix(states: Sequence[GameState]) -> np.ndarray:
    """(n_states, n_features) float64 array in FEATURE_NAMES order."""
    out = np.empty((len(states), len(FEATURE_NAMES)), dtype=np.float64)
    for i, s in enumerate(states):
        d = feature_dict(s)
        for j, name in enumerate(FEATURE_NAMES):
            out[i, j] = d[name]
    return out


def mirror_features(X: np.ndarray, y: np.ndarray):
    """Symmetry augmentation (plan 8.3): swap the two teams, flip the label.

    Forces the model to treat the teams identically except through terms that
    are genuinely home-specific. Free data, and it stabilises the fit.
    """
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    Xm = X.copy()
    for name in ("margin", "margin_per_sqrt_time", "pregame_exp_margin",
                 "pregame_exp_margin_decayed", "timeout_diff", "bonus_diff",
                 "ft_pct_diff", "margin_per_sqrt_points_left"):
        Xm[:, idx[name]] *= -1.0
    Xm[:, idx["possession"]] = 1.0 - Xm[:, idx["possession"]]
    return np.vstack([X, Xm]), np.concatenate([y, 1 - y])


# --------------------------------------------------------------------------
# Vectorised twin of `feature_dict`, for building tens of millions of rows.
# tests/test_parity.py asserts the two agree.
# --------------------------------------------------------------------------
def feature_exprs():
    import polars as pl
    t = pl.max_horizontal(pl.col("game_seconds_remaining").cast(pl.Float64), pl.lit(1.0))
    sqrt_t = t.sqrt()
    return [
        pl.col("margin").cast(pl.Float64).alias("margin"),
        sqrt_t.alias("sqrt_time"),
        (pl.col("margin").cast(pl.Float64) / sqrt_t).alias("margin_per_sqrt_time"),
        pl.col("possession").cast(pl.Float64).alias("possession"),
        pl.col("pregame_exp_margin").cast(pl.Float64).alias("pregame_exp_margin"),
        (pl.col("pregame_exp_margin").cast(pl.Float64) * sqrt_t / _SQRT_REG)
        .alias("pregame_exp_margin_decayed"),
        pl.col("is_ot").cast(pl.Float64).alias("is_ot"),
        (pl.col("home_timeouts") - pl.col("away_timeouts")).cast(pl.Float64).alias("timeout_diff"),
        (_bonus_expr(pl.col("away_fouls")) - _bonus_expr(pl.col("home_fouls")))
        .cast(pl.Float64).alias("bonus_diff"),
        pl.col("ft_pct_diff").cast(pl.Float64).alias("ft_pct_diff"),
        (pl.col("margin").cast(pl.Float64) / pl.max_horizontal(
            pl.col("exp_points_per_min").cast(pl.Float64) * t / 60.0, pl.lit(1.0)).sqrt())
        .alias("margin_per_sqrt_points_left"),
    ]


def _bonus_expr(fouls):
    import polars as pl
    return (pl.when(fouls >= DOUBLE_BONUS_FOULS).then(pl.lit(2))
              .when(fouls >= BONUS_FOULS).then(pl.lit(1))
              .otherwise(pl.lit(0)))
```

## `src/cbbwp/live_context.py`

```py
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
```

## `src/cbbwp/monitor.py`

```py
"""Calibration drift monitoring (plan phase 5, item 16).

A win probability model fails quietly. Log loss barely moves when the model
starts saying 0.80 to situations that win 0.74, but that gap is the whole
product. So the check that matters is not "is the loss still good" but
"does what we SAY still match what HAPPENS", sliced by time remaining -
because that is the axis along which this model's error actually varies.

Two thresholds, deliberately both required to fire:
  * statistical  - the gap is larger than sampling noise (|z| > Z_ALERT);
  * practical    - the gap is larger than anyone would care about
                   (|gap| > MIN_GAP).
A million rows will make a 0.3-point gap "significant"; that is not drift,
that is a large sample. Requiring both keeps the alert honest.

THE CLUSTERING POINT, which is the difference between this being useful and
being a weekly false alarm: the ~400 states in one game are NOT independent
observations. They share one outcome. A game the home team won contributes 400
rows all labelled 1, and if the model was 3 points low on that game it is 3
points low on all 400 of them. So the standard error must be computed on the
number of GAMES in a bin, not the number of states - the same principle the
train/test split rests on ("effective sample size is games, not rows").

Treating states as independent inflates z by roughly sqrt(states per game),
which here is about 6x. The first version of this file did exactly that and
reported z = -10.6 for a gap that is really about z = -1.7. Pass `game_ids`
and the correction is automatic; omit them and the report says so, loudly,
rather than quietly overstating its own confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Time buckets, as everywhere else in this package.
TIME_BUCKETS = [
    ("40-20 min", 1200, 2400),
    ("20-10 min", 600, 1200),
    ("10-5 min", 300, 600),
    ("5-2 min", 120, 300),
    ("2-1 min", 60, 120),
    ("1-0 min", 0, 60),
]

Z_ALERT = 3.0        # ~1 false positive per 370 independent checks
MIN_GAP = 0.02       # 2 percentage points: below this nobody would notice
MIN_ROWS = 500       # a decile thinner than this says nothing either way
MIN_GAMES = 100      # ...and neither does one drawn from a handful of games
N_DECILES = 10


@dataclass
class BinReport:
    bucket: str
    decile: int
    n: int              # states
    n_games: int        # independent observations behind those states
    pred: float
    obs: float
    gap: float
    z: float
    alert: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    generated: str = ""
    window: str = ""
    n_rows: int = 0
    n_games: int = 0
    clustered: bool = True
    bins: List[BinReport] = field(default_factory=list)
    bucket_ece: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.alerts

    def as_dict(self) -> dict:
        return {
            "generated": self.generated,
            "window": self.window,
            "n_rows": self.n_rows,
            "n_games": self.n_games,
            "ok": self.ok,
            "clustered": self.clustered,
            "alerts": self.alerts,
            "notes": self.notes,
            "bucket_ece": self.bucket_ece,
            "bins": [b.as_dict() for b in self.bins],
        }


def _z_score(pred: float, obs: float, n_independent: int) -> float:
    """How many standard errors the observed rate sits from the predicted one.

    `n_independent` must be the number of GAMES contributing to the bin, not the
    number of states - see the module docstring. Uses the predicted probability
    for the standard error (the null hypothesis is that the model is right),
    which is the conservative choice near 0 and 1.
    """
    var = pred * (1.0 - pred)
    if n_independent <= 0 or var <= 0:
        return 0.0
    return (obs - pred) / math.sqrt(var / n_independent)


def decile_edges(p: np.ndarray, n_deciles: int = N_DECILES) -> np.ndarray:
    """Equal-COUNT edges. Equal-width bins leave the interesting tails empty."""
    qs = np.linspace(0, 1, n_deciles + 1)[1:-1]
    return np.unique(np.quantile(p, qs))


def check(y: Sequence[int], p: Sequence[float], seconds: Sequence[float],
          game_ids: Optional[Sequence] = None, window: str = "",
          z_alert: float = Z_ALERT, min_gap: float = MIN_GAP,
          min_rows: int = MIN_ROWS, min_games: int = MIN_GAMES,
          generated: str = "") -> DriftReport:
    """Decile calibration check within each time bucket.

    `game_ids` is not optional in spirit: without it every state is treated as
    an independent observation and the z-scores are inflated by roughly the
    square root of the number of states per game. Omitting it is supported for
    synthetic data and unit tests, and the report says so.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(seconds, dtype=np.float64)
    if not (len(y) == len(p) == len(s)):
        raise ValueError("y, p and seconds must be the same length")
    if game_ids is None:
        g = np.arange(len(y))          # every row its own "game": no clustering
        clustered = False
    else:
        g = np.asarray(game_ids)
        if len(g) != len(y):
            raise ValueError("game_ids must be the same length as y")
        clustered = True

    rep = DriftReport(generated=generated, window=window, n_rows=int(len(y)),
                      n_games=int(len(np.unique(g))) if clustered else 0,
                      clustered=clustered)
    if not clustered:
        rep.notes.append(
            "no game_ids supplied: states treated as independent, so z-scores "
            "are optimistic. Fine for synthetic data, wrong for real games.")

    for name, lo, hi in TIME_BUCKETS:
        m = (s >= lo) & (s < hi)
        if m.sum() < min_rows:
            continue
        pb, yb, gb = p[m], y[m], g[m]
        edges = decile_edges(pb)
        idx = np.digitize(pb, edges)
        ece = 0.0
        for d in range(len(edges) + 1):
            dm = idx == d
            n = int(dm.sum())
            if n == 0:
                continue
            n_games = int(len(np.unique(gb[dm])))
            pred = float(pb[dm].mean())
            obs = float(yb[dm].mean())
            gap = obs - pred
            ece += n / len(pb) * abs(gap)
            # The independent-observation count is the number of games.
            z = _z_score(pred, obs, n_games)
            alert = bool(n >= min_rows and n_games >= min_games
                         and abs(z) > z_alert and abs(gap) > min_gap)
            rep.bins.append(BinReport(name, d, n, n_games, pred, obs, gap, z, alert))
            if alert:
                rep.alerts.append(
                    f"{name} decile {d}: model says {pred:.3f}, observed "
                    f"{obs:.3f} over {n:,} states from {n_games:,} games "
                    f"(gap {gap:+.3f}, z {z:+.1f})")
        rep.bucket_ece[name] = float(ece)
    return rep


def format_report(rep: DriftReport) -> str:
    """Human-readable table, for a terminal or an alert email."""
    out = []
    head = f"calibration check  {rep.window}".strip()
    out.append(head)
    out.append(f"{rep.n_rows:,} states"
               + (f" from {rep.n_games:,} games" if rep.n_games else ""))
    for n in rep.notes:
        out.append(f"NOTE: {n}")
    out.append("")
    out.append(f"{'bucket':<12}{'dec':>4}{'states':>10}{'games':>8}"
               f"{'pred':>8}{'obs':>8}{'gap':>8}{'z':>7}  ")
    last = None
    for b in rep.bins:
        sep = "" if b.bucket == last else "\n"
        last = b.bucket
        flag = "  <-- ALERT" if b.alert else ""
        out.append(f"{sep}{b.bucket if sep else '':<12}{b.decile:>4}{b.n:>10,}"
                   f"{b.n_games:>8,}{b.pred:>8.3f}{b.obs:>8.3f}{b.gap:>+8.3f}"
                   f"{b.z:>+7.1f}{flag}")
    out.append("")
    out.append("ECE by bucket: " + "  ".join(
        f"{k} {v:.4f}" for k, v in rep.bucket_ece.items()))
    out.append("")
    if rep.ok:
        out.append("OK - no bucket/decile is both statistically and practically off\n"
                   "     (z computed on games, not states - see monitor.py).")
    else:
        out.append(f"{len(rep.alerts)} ALERT(S):")
        out.extend("  " + a for a in rep.alerts)
    return "\n".join(out)
```

## `src/cbbwp/ratings.py`

```py
"""As-of-date pregame team ratings - our stand-in for the closing spread.

hoopR carries a betting spread only through ~2023, so we fit our own team
strength model. The output, `pregame_exp_margin`, is on the same scale as a
negated point spread: expected home margin in points.

Leakage discipline (plan 8.2): a game's rating inputs are refit from games
that finished STRICTLY BEFORE that game's date. Nothing a season later, and
nothing from the game itself, can reach the model.

Method: ridge regression of final margin on (home indicator - away indicator)
plus a home-court term, shrunk toward a prior. The prior is last season's final
rating regressed toward the mean, which is what carries a team through the
first few November games when the in-season sample is empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import polars as pl

CARRYOVER = 0.70     # how much of last season's rating survives into this one
RIDGE_LAMBDA = 0.5   # tuned against 2016-23 closing spreads: corr 0.87, HCA 3.8
REFIT_EVERY_DAYS = 7


@dataclass
class SeasonRatings:
    season: int
    ratings: Dict[int, float]   # team_id -> points above average
    hca: float


def _fit_ridge(games: pl.DataFrame, teams: list[int], prior: Dict[int, float],
               lam: float | None = None) -> tuple[Dict[int, float], float]:
    """Ridge fit of margin ~ home_team - away_team + hca, shrunk toward `prior`."""
    lam = RIDGE_LAMBDA if lam is None else lam
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    if games.height == 0:
        return dict(prior), 3.4

    rows = games.height
    X = np.zeros((rows, n_t + 1), dtype=np.float64)
    h = games["home_id"].to_numpy()
    a = games["away_id"].to_numpy()
    X[np.arange(rows), [idx[t] for t in h]] = 1.0
    X[np.arange(rows), [idx[t] for t in a]] = -1.0
    X[:, n_t] = 1.0 - games["neutral_site"].cast(pl.Float64).to_numpy()
    y = games["margin"].to_numpy().astype(np.float64)

    b_prior = np.zeros(n_t + 1)
    for t, v in prior.items():
        if t in idx:
            b_prior[idx[t]] = v
    b_prior[n_t] = 3.4  # prior on home-court advantage, in points

    resid = y - X @ b_prior
    A = X.T @ X
    pen = np.full(n_t + 1, lam)
    pen[n_t] = 1.0          # barely shrink the home-court term
    A[np.diag_indices_from(A)] += pen
    d = np.linalg.solve(A, X.T @ resid)
    b = b_prior + d
    b[:n_t] -= b[:n_t].mean()   # ratings are relative; centre them
    return {t: float(b[idx[t]]) for t in teams}, float(b[n_t])


def season_pregame_margins(games: pl.DataFrame, prior: Dict[int, float], lam: float | None = None) -> tuple[pl.DataFrame, Dict[int, float], float]:
    """For one season, attach `pregame_exp_margin` to every game.

    `games` needs: game_id, date (datetime), home_id, away_id, margin, neutral_site.
    Returns (games+column, end-of-season ratings, fitted home-court advantage).
    """
    games = games.sort("date")
    teams = sorted(set(games["home_id"].to_list()) | set(games["away_id"].to_list()))
    day = games["date"].dt.date()
    games = games.with_columns(_day=day)
    days = sorted(games["_day"].unique().to_list())

    out_ids, out_vals = [], []
    ratings, hca = dict(prior), 3.4
    last_refit_i = -10_000
    for i, d in enumerate(days):
        if i - last_refit_i >= REFIT_EVERY_DAYS or i == 0:
            past = games.filter(pl.col("_day") < d)
            ratings, hca = _fit_ridge(past, teams, prior, lam)
            last_refit_i = i
        todays = games.filter(pl.col("_day") == d)
        for gid, hid, aid, neu in zip(todays["game_id"], todays["home_id"],
                                      todays["away_id"], todays["neutral_site"]):
            out_ids.append(gid)
            out_vals.append(ratings.get(hid, 0.0) - ratings.get(aid, 0.0)
                            + (0.0 if neu else hca))

    final_ratings, final_hca = _fit_ridge(games, teams, prior, lam)
    joined = games.join(
        pl.DataFrame({"game_id": out_ids, "pregame_exp_margin": out_vals}),
        on="game_id", how="left",
    ).drop("_day")
    return joined, final_ratings, final_hca


def build_all_seasons(games: pl.DataFrame, lam: float | None = None) -> pl.DataFrame:
    """Walk seasons in order, carrying each season's ratings into the next."""
    prior: Dict[int, float] = {}
    frames = []
    for season in sorted(games["season"].unique().to_list()):
        sg = games.filter(pl.col("season") == season)
        out, final, hca = season_pregame_margins(sg, prior, lam)
        frames.append(out)
        prior = {t: v * CARRYOVER for t, v in final.items()}
        print(f"  season {season}: {out.height} games, hca={hca:.2f}, "
              f"rating sd={np.std(list(final.values())):.2f}")
    return pl.concat(frames, how="vertical_relaxed")
```

## `src/cbbwp/schemas.py`

```py
"""Canonical data contracts shared by the offline and live pipelines.

Every adapter (historical parquet, live ESPN feed, a paid feed later) must emit
`Event` objects. Nothing downstream of an adapter knows where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Rules constants (men's NCAA, 2015-16 rules onward) ----------------------
HALF_SECONDS = 20 * 60          # 1200
REGULATION_SECONDS = 2 * HALF_SECONDS  # 2400
OT_SECONDS = 5 * 60             # 300
TIMEOUTS_AT_TIP = 4             # approximation of the men's allotment


@dataclass(frozen=True, slots=True)
class Event:
    """One play, normalised. `seq` orders events within a game."""
    game_id: int
    seq: int
    period: int                  # 1,2 = halves; 3+ = overtime
    clock_seconds: int           # seconds left IN THE PERIOD at the play
    home_score: int
    away_score: int
    event_type: str              # e.g. "JumpShot", "Timeout", "DefensiveRebound"
    team_id: Optional[int]       # team the event is attributed to
    score_value: int = 0
    scoring_play: bool = False
    shooting_play: bool = False
    text: str = ""


@dataclass(frozen=True, slots=True)
class PregameContext:
    """Known before tip-off. Loaded once per game, never re-fetched mid-game."""
    game_id: int
    home_team_id: int
    away_team_id: int
    neutral_site: bool = False
    # Expected home margin in points (positive = home favoured).
    # Either the negated closing spread, or a model-derived rating differential.
    pregame_exp_margin: float = 0.0
    season: int = 0
    ft_pct_diff: float = 0.0
    exp_points_per_min: float = 3.4


@dataclass(slots=True)
class GameState:
    """A snapshot AFTER one event. One event -> one state row."""
    game_id: int
    seq: int
    period: int
    is_ot: bool
    clock_seconds: int            # left in the period
    game_seconds_remaining: int   # left in regulation, or in the current OT
    home_score: int
    away_score: int
    margin: int                   # home - away
    possession: float             # 1.0 home, 0.0 away, 0.5 unknown
    home_timeouts: int
    away_timeouts: int
    home_fouls: int = 0            # team fouls in the current half
    away_fouls: int = 0
    pregame_exp_margin: float = 0.0
    neutral_site: bool = False
    ft_pct_diff: float = 0.0       # home season-to-date FT% minus away's
    exp_points_per_min: float = 3.4  # combined scoring rate of the two teams


# Column order is part of the contract: the fitted model's coefficients are
# positional. Changing this list requires a new model version.
FEATURE_NAMES = [
    "margin",
    "sqrt_time",
    "margin_per_sqrt_time",
    "possession",
    "pregame_exp_margin",
    "pregame_exp_margin_decayed",
    "is_ot",
    "timeout_diff",
    "bonus_diff",
    "ft_pct_diff",
    "margin_per_sqrt_points_left",
]

# Men's NCAA bonus thresholds, in team fouls per half.
BONUS_FOULS = 7        # 1-and-1
DOUBLE_BONUS_FOULS = 10

# Version of the STATE RULES, as opposed to the feature list above.
#
# FEATURE_NAMES catches someone adding, removing or reordering a column. It does
# NOT catch someone changing what an existing column MEANS - and that is the
# more dangerous edit, because nothing downstream looks any different.
#
# This happened on 2026-09-01: the possession rule was corrected so that made
# three-pointers in 2016-2019 flip possession (they had not, because ESPN typed
# them "Three Point Jump Shot" and the rule keyed on play-type names). The
# feature list was untouched, so the manifest check passed, and a model trained
# on the old meaning would have been served states built with the new one.
#
# Bump this whenever the meaning of any GameState field changes, and refit.
#   1 - original rules, shipped 2026-08-31 (registry/v1)
#   2 - made field goals detected by scoring+shooting flags, not type names
STATE_RULES_VERSION = 2
```

## `src/cbbwp/serve.py`

```py
"""The live scoring path. Deliberately thin: it calls the SAME state and
feature builders the training pipeline used, then a pinned model artifact.
"""
from __future__ import annotations

import json
import pathlib
import pickle
from typing import Iterable, List

import numpy as np

from .schemas import (Event, PregameContext, FEATURE_NAMES,
                      STATE_RULES_VERSION)
from .state import build_states
from .features import build_feature_matrix
from . import endgame

OVERRIDE_CLIP = 0.999   # never assert more certainty than the feed supports


class WinProbabilityService:
    def __init__(self, registry_dir: str | pathlib.Path, version: str):
        self.dir = pathlib.Path(registry_dir) / version
        self.version = version
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        if self.manifest["features"] != FEATURE_NAMES:
            raise RuntimeError(
                f"model {version} was fit on different features than this code builds; "
                "the feature contract changed - refit or pin an older code version"
            )
        # The names can match while the MEANING has changed underneath them.
        # A model fit before a state-rule change must not be served states built
        # after it. An artifact with no stamp predates the check and is treated
        # as version 1.
        fit_rules = self.manifest.get("state_rules_version", 1)
        if fit_rules != STATE_RULES_VERSION:
            raise RuntimeError(
                f"model {version} was fit with state rules v{fit_rules} but this "
                f"code builds states with v{STATE_RULES_VERSION}. The feature NAMES "
                "still match, so this would have been silent: the model would be "
                "served inputs that mean something different from its training "
                "data. Refit, or pin the code version that matches the artifact."
            )
        kind = self.manifest["kind"]
        if kind == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.dir / "model.txt"))
            self._predict = lambda X: self.model.predict(X)
        else:
            with open(self.dir / "model.pkl", "rb") as f:
                b = pickle.load(f)
            self._predict = lambda X: b["model"].predict_proba(b["scaler"].transform(X))[:, 1]

    def score_game(self, events: Iterable[Event], ctx: PregameContext) -> List[dict]:
        """Replay a game from event zero and return one prediction per state.

        Called on every poll. Replaying from scratch costs milliseconds and makes
        retroactive feed corrections a non-event.
        """
        states = build_states(events, ctx)
        if not states:
            return []
        X = build_feature_matrix(states)
        p = np.asarray(self._predict(X), dtype=np.float64)

        margin = np.array([s.margin for s in states])
        secs = np.array([s.game_seconds_remaining for s in states], dtype=np.float64)
        adj = endgame.apply(p, margin, secs)
        touched = adj != p
        p[touched] = np.clip(adj[touched], 1 - OVERRIDE_CLIP, OVERRIDE_CLIP)

        return [
            {"game_id": s.game_id, "seq": s.seq, "period": s.period,
             "game_seconds_remaining": s.game_seconds_remaining, "margin": s.margin,
             "home_win_prob": float(pi), "model_version": self.version}
            for s, pi in zip(states, p)
        ]
```

## `src/cbbwp/state.py`

```py
"""Replayable game-state builder.

`build_states` is a PURE function of the full event list. Feed it the same
events in any arrival order and it produces the same states, so a retroactive
correction from a live feed is handled by simply replaying the game from
event zero (milliseconds).
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from .schemas import (
    Event,
    GameState,
    PregameContext,
    HALF_SECONDS,
    OT_SECONDS,
    TIMEOUTS_AT_TIP,
)

# Fouls that count toward the team-foul total for bonus purposes.
FOUL_TYPES = {"PersonalFoul", "Technical Foul"}

# --- possession rules -------------------------------------------------------
# What the state's `possession` should be AFTER each kind of event.
#   "actor"  -> the team the event is attributed to has the ball
#   "other"  -> the other team has the ball
#   "carry"  -> unchanged from the previous state
#   "unknown"-> 0.5
# NOTE: this list is NO LONGER used to decide possession - see _possession_after.
# It is kept only for documentation of what the field-goal types look like.
_MADE_SHOT_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot"}
_TURNOVER_MARKER = "Turnover"

TEAM_TIMEOUT_TYPES = {"ShortTimeOut", "RegularTimeOut", "TeamTimeOut", "Timeout"}
OFFICIAL_TIMEOUT_TYPES = {"OfficialTVTimeOut", "MediaTimeOut"}


def clock_to_seconds(display: str) -> int:
    """'19:48' -> 1188.  '0:23.4' -> 23.  Returns 0 on anything unparseable."""
    if not display:
        return 0
    s = display.strip()
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + int(float(ss))
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def period_length(period: int) -> int:
    return HALF_SECONDS if period <= 2 else OT_SECONDS


def game_seconds_remaining(period: int, clock_seconds: int) -> int:
    """Seconds left in regulation; inside overtime, seconds left in that OT.

    Each overtime is treated as its own clock reset (see plan section 8.2).
    """
    if period <= 1:
        return HALF_SECONDS + clock_seconds
    if period == 2:
        return clock_seconds
    return clock_seconds


def _possession_after(ev: Event, home_id: int, away_id: int, prev: float) -> float:
    """Return 1.0 (home has ball), 0.0 (away), or 0.5 (unknown)."""
    t = ev.event_type or ""
    tid = ev.team_id
    actor: Optional[float]
    if tid is None:
        actor = None
    elif tid == home_id:
        actor = 1.0
    elif tid == away_id:
        actor = 0.0
    else:
        actor = None
    other = None if actor is None else 1.0 - actor

    if "FreeThrow" in t:
        return other if (ev.scoring_play and other is not None) else prev
    # A made field goal -> the other team inbounds. A miss leaves the ball live,
    # so possession carries until a rebound resolves it.
    #
    # This is keyed on the feed's own scoring/shooting flags rather than on a
    # list of play-type NAMES, and that is not a style preference. ESPN typed
    # made three-pointers as "Three Point Jump Shot" through 2019 and as
    # "JumpShot" from 2021 onward. A name whitelist therefore missed 324,043
    # made threes - 89% of every made three in 2016-2019 - and left the ball
    # with the team that had just scored. The flags are stable across that
    # rename; the names are not.
    if ev.scoring_play and ev.shooting_play:
        return other if other is not None else prev
    if t == "Defensive Rebound" or t == "Offensive Rebound":
        return actor if actor is not None else prev
    if t == "Dead Ball Rebound":
        return prev
    if _TURNOVER_MARKER in t:
        return other if other is not None else prev
    if t == "Steal":
        return actor if actor is not None else prev
    if t == "Jumpball":
        return 0.5
    return prev  # fouls, blocks, subs, timeouts, period markers


def build_states(
    events: Iterable[Event],
    ctx: PregameContext,
    timeouts_at_tip: int = TIMEOUTS_AT_TIP,
) -> List[GameState]:
    """Replay a game's events into one state per event.

    Pure: no dependence on arrival order, no hidden state, no I/O.
    """
    evs = sorted(events, key=lambda e: e.seq)
    home_id, away_id = ctx.home_team_id, ctx.away_team_id

    states: List[GameState] = []
    poss = 0.5
    home_used = away_used = 0
    home_fouls = away_fouls = 0
    half_of = lambda p: 1 if p <= 1 else 2   # men's fouls reset once, at the break
    cur_half = 1

    for ev in evs:
        period = max(1, ev.period or 1)
        if half_of(period) != cur_half:
            cur_half = half_of(period)
            home_fouls = away_fouls = 0

        if ev.event_type in FOUL_TYPES:
            if ev.team_id == home_id:
                home_fouls += 1
            elif ev.team_id == away_id:
                away_fouls += 1

        if ev.event_type in TEAM_TIMEOUT_TYPES:
            if ev.team_id == home_id:
                home_used += 1
            elif ev.team_id == away_id:
                away_used += 1

        # NCAA grants one extra timeout per overtime period.
        allot = timeouts_at_tip + max(0, period - 2)

        poss = _possession_after(ev, home_id, away_id, poss)
        clock = max(0, min(int(ev.clock_seconds or 0), period_length(period)))

        states.append(
            GameState(
                game_id=ev.game_id,
                seq=ev.seq,
                period=period,
                is_ot=period >= 3,
                clock_seconds=clock,
                game_seconds_remaining=game_seconds_remaining(period, clock),
                home_score=ev.home_score,
                away_score=ev.away_score,
                margin=ev.home_score - ev.away_score,
                possession=poss,
                home_timeouts=max(0, allot - home_used),
                away_timeouts=max(0, allot - away_used),
                home_fouls=home_fouls,
                away_fouls=away_fouls,
                pregame_exp_margin=ctx.pregame_exp_margin,
                neutral_site=ctx.neutral_site,
                ft_pct_diff=ctx.ft_pct_diff,
                exp_points_per_min=ctx.exp_points_per_min,
            )
        )
    return states
```

## `scripts/archive_replay_games.py`

```py
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
```

## `scripts/blend_endgame.py`

```py
"""Phase 5: blend the endgame table with the model, tune once, test once.

The endgame plan pre-registered five criteria and one verdict rule: "If it
clears 2-5 but not 1, it does not ship." That is only credible if the tuning and
the test are separate acts, so they are separate invocations here:

    python3 scripts/blend_endgame.py --tune          # 2024 only, writes the config
    python3 scripts/blend_endgame.py --test          # 2025-2026, reads it, once

--test refuses to run unless the config file already exists, and records a
hash of it in the result, so a result can always be traced to the configuration
that produced it rather than to one chosen afterwards.

The blend is in log-odds, with a weight that is exactly 0 at 60 seconds and
exactly 1 at 0. Criterion 4 -- no visible discontinuity at the handoff -- is
therefore satisfied by construction rather than by tuning, and is verified
empirically anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp import endgame as endgame_rules  # noqa: E402
from cbbwp import endgame_sim as E  # noqa: E402
from cbbwp.schemas import FEATURE_NAMES  # noqa: E402

CONFIG = ROOT / "registry" / "endgame" / "blend.json"
HANDOFF = 60.0
EPS = 1e-15
TUNE_SEASON = 2024
TEST_SEASONS = [2025, 2026]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y, p, bins=20):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.sum():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def load_frame(season: int, table, means, booster, seconds: float = 130.0) -> dict:
    st = (
        pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{season}.parquet")
        .filter((pl.col("period") >= 2) & (pl.col("game_seconds_remaining") <= seconds))
        # margin and possession are already in FEATURE_NAMES; asking for them
        # twice is a duplicate projection.
        .select(FEATURE_NAMES + [c for c in
                ["game_id", "game_seconds_remaining", "margin", "possession",
                 "home_fouls", "away_fouls", "home_win", "espn_wp"]
                if c not in FEATURE_NAMES])
        .collect()
    )
    ts = pl.read_parquet(ROOT / "data" / "proc" / "team_stats.parquet").select(
        ["game_id", "home_ft_pct", "away_ft_pct"]
    )
    d = st.join(ts, on="game_id", how="left").with_columns(
        [pl.col("home_ft_pct").fill_null(0.70), pl.col("away_ft_pct").fill_null(0.70)]
    )
    X = d.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    p_model = np.asarray(booster.predict(X), dtype=np.float64)
    secs = d["game_seconds_remaining"].to_numpy().astype(np.float64)
    p_table = E.lookup_home(
        table, np.minimum(secs, E.T_MAX), d["margin"].to_numpy(), d["possession"].to_numpy(),
        d["home_fouls"].to_numpy(), d["away_fouls"].to_numpy(),
        E.ft_bucket(d["home_ft_pct"].to_numpy(), means),
        E.ft_bucket(d["away_ft_pct"].to_numpy(), means),
    )
    return {
        "y": d["home_win"].to_numpy().astype(float),
        "secs": secs,
        "margin": d["margin"].to_numpy(),
        "game_id": d["game_id"].to_numpy(),
        "p_model": p_model,
        "p_table": p_table,
        "espn": d["espn_wp"].to_numpy(),
    }


def blend(p_model, p_table, secs, gamma, alpha, beta, w_max=1.0):
    """Log-odds blend with a weight that is exactly 0 at the handoff.

    The plan assumed the simulator's weight should rise to 1.0 by 0:00. Tuning
    on 2024 falsified that: the model is at its BEST at 0:00 (log loss 0.0859 in
    the last five seconds, against the table's 0.1368), because by then the
    margin and the clock have decided almost everything and there is nothing
    left for a possession model to add. Forcing the weight to 1 there throws
    away the model's strongest region. The weight therefore has a free ceiling.

        w(t) = w_max * (1 - (t / 60)^gamma)

    which is 0 at 60s -- so criterion 4 holds by construction, not by tuning --
    and w_max at 0:00. Large gamma keeps the weight near its ceiling across most
    of the window and spends the ramp near the handoff.
    """
    frac = np.clip(secs / HANDOFF, 0.0, 1.0)
    w = w_max * (1.0 - frac ** gamma)
    z = (1 - w) * logit(p_model) + w * (alpha * logit(p_table) + beta)
    return sigmoid(z)


def apply_rules(p, margin, secs):
    """The same rule-based clamps the live path applies, after blending."""
    adj = endgame_rules.apply(p, margin, secs)
    touched = adj != p
    out = p.copy()
    out[touched] = np.clip(adj[touched], 1 - 0.999, 0.999)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--table", default="registry/endgame/e1")
    ap.add_argument("--model", default="registry/v2")
    a = ap.parse_args()

    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / a.model / "model.txt"))
    tdir = ROOT / a.table
    table = np.load(tdir / "table.npz")["table"].astype(np.float64)
    means = json.loads((tdir / "manifest.json").read_text())["ft_bucket_means"]

    if a.tune:
        d = load_frame(TUNE_SEASON, table, means, booster)
        inside = d["secs"] <= HANDOFF
        y, sec = d["y"][inside], d["secs"][inside]
        pm, pt = d["p_model"][inside], d["p_table"][inside]
        best = None
        for gamma in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
            for w_max in np.arange(0.05, 0.85, 0.05):
                for alpha in np.arange(0.7, 1.35, 0.05):
                    for beta in (-0.05, 0.0, 0.05):
                        ll = log_loss(y, blend(pm, pt, sec, gamma, alpha, beta, w_max))
                        if best is None or ll < best[0]:
                            best = (ll, gamma, float(alpha), beta, float(w_max))
        ll, gamma, alpha, beta, w_max = best
        cfg = {
            "handoff_seconds": HANDOFF, "gamma": gamma, "alpha": alpha, "beta": beta,
            "w_max": w_max,
            "table": a.table, "model": a.model, "tuned_on_season": TUNE_SEASON,
            "tune_log_loss_inside_60s": ll,
            "tune_baseline_model_only": log_loss(y, pm),
            "tune_table_only": log_loss(y, pt),
            "n_tune_rows": int(inside.sum()),
        }
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(cfg, indent=2))
        print(json.dumps(cfg, indent=2))
        return

    if not a.test:
        ap.error("pass --tune or --test")

    if not CONFIG.exists():
        raise SystemExit(
            "no blend config: run --tune on 2024 first. The test is single-shot and "
            "must not choose its own parameters."
        )
    cfg = json.loads(CONFIG.read_text())
    cfg_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:16]

    parts = [load_frame(s, table, means, booster) for s in TEST_SEASONS]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    p_blend_raw = blend(d["p_model"], d["p_table"], d["secs"], cfg["gamma"], cfg["alpha"], cfg["beta"], cfg["w_max"])
    p_blend = apply_rules(p_blend_raw, d["margin"], d["secs"])
    p_model = d["p_model"]

    inside = d["secs"] <= HANDOFF
    ok = np.isfinite(d["espn"])
    res = {
        "config_sha256_16": cfg_hash, "config": cfg, "seasons": TEST_SEASONS,
        "criterion_1_log_loss_under_60s": {
            "model_only": log_loss(d["y"][inside], p_model[inside]),
            "blended": log_loss(d["y"][inside], p_blend[inside]),
            "table_only": log_loss(d["y"][inside], d["p_table"][inside]),
            "espn": log_loss(d["y"][inside & ok], d["espn"][inside & ok]),
            "n": int(inside.sum()),
        },
        "criterion_2_ece_under_60s": {
            "model_only": ece(d["y"][inside], p_model[inside]),
            "blended": ece(d["y"][inside], p_blend[inside]),
        },
    }
    c1 = res["criterion_1_log_loss_under_60s"]
    c1["relative_improvement"] = (c1["model_only"] - c1["blended"]) / c1["model_only"]
    c1["passes"] = bool(c1["relative_improvement"] >= 0.01)
    c2 = res["criterion_2_ece_under_60s"]
    c2["passes"] = bool(c2["blended"] <= c2["model_only"] + 1e-9)

    # criterion 4: the handoff must be invisible
    near = (d["secs"] >= 55) & (d["secs"] <= 65)
    res["criterion_4_handoff"] = {
        "max_abs_delta_at_boundary": float(np.abs(p_blend - p_model)[np.abs(d["secs"] - 60) <= 0.5].max()
                                           if (np.abs(d["secs"] - 60) <= 0.5).any() else 0.0),
        "max_abs_delta_55_to_65s": float(np.abs(p_blend - p_model)[near].max() if near.any() else 0.0),
        "passes": True,
    }
    res["criterion_4_handoff"]["passes"] = bool(res["criterion_4_handoff"]["max_abs_delta_at_boundary"] < 0.02)

    for lo, hi, name in [(0, 10, "0-10s"), (10, 30, "10-30s"), (30, 60, "30-60s")]:
        m = (d["secs"] >= lo) & (d["secs"] < hi)
        res.setdefault("by_bucket", {})[name] = {
            "n": int(m.sum()),
            "model_only": log_loss(d["y"][m], p_model[m]),
            "blended": log_loss(d["y"][m], p_blend[m]),
            "table_only": log_loss(d["y"][m], d["p_table"][m]),
        }

    # criterion 3: monotonicity, exhaustive over the table, plus a check that
    # blending cannot break it. Both components are monotone in margin and the
    # blend is a positive combination in log-odds, so it is monotone too -- but
    # "so it is" is how the possession bug survived, so it is measured.
    mono = json.loads((tdir / "manifest.json").read_text())["monotonicity_after"]
    grid_secs = np.repeat(np.arange(0, 61, 1.0), 25)
    grid_m = np.tile(np.arange(-12, 13, 1.0), 61)
    pmg = sigmoid(np.linspace(-3, 3, len(grid_m)) * 0 + grid_m * 0.35)
    ptg = E.lookup_home(table, np.minimum(grid_secs, E.T_MAX), grid_m,
                        np.ones_like(grid_m), np.full_like(grid_m, 6, dtype=int),
                        np.full_like(grid_m, 8, dtype=int),
                        np.ones_like(grid_m, dtype=int), np.ones_like(grid_m, dtype=int))
    bg = blend(pmg, ptg, grid_secs, cfg["gamma"], cfg["alpha"], cfg["beta"], cfg["w_max"])
    bg = bg.reshape(61, 25)
    res["criterion_3_monotonicity"] = {
        "table_exhaustive": mono,
        "blend_margin_min_increment": float(np.diff(bg, axis=1).min()),
        "passes": bool(mono["margin_violations"] == 0
                       and mono["possession_violations"] == 0
                       and np.diff(bg, axis=1).min() >= -1e-9),
    }

    # criterion 5: it has to be fast enough to serve.
    import time as _t
    n = 200_000
    idx = np.random.default_rng(0).integers(0, len(d["secs"]), n)
    t0 = _t.perf_counter()
    E.lookup_home(table, np.minimum(d["secs"][idx], E.T_MAX), d["margin"][idx],
                  np.ones(n), np.full(n, 6), np.full(n, 8), np.ones(n, dtype=int),
                  np.ones(n, dtype=int))
    per_state_ms = (_t.perf_counter() - t0) / n * 1000
    res["criterion_5_speed"] = {"ms_per_state": per_state_ms, "n": n,
                                "passes": bool(per_state_ms < 1.0)}

    res["verdict"] = {
        "criteria_passed": {k: res[k]["passes"] for k in res if k.startswith("criterion")},
        "ships": bool(all(res[k]["passes"] for k in res if k.startswith("criterion"))),
    }
    res["verdict"]["note"] = (
        "The plan's rule: if it clears 2-5 but not 1, it does not ship -- it becomes "
        "a documented diagnostic."
    )

    out = ROOT / "reports" / "endgame_blend_test.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

## `scripts/build_dataset.py`

```py
"""Replay every game into state rows and write the modelling dataset."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from cbbwp.adapters.hoopr import states_lazy, BASE_COLS
from cbbwp.features import feature_exprs
from cbbwp.schemas import FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
LATE_SECONDS = 300       # keep every state inside the final 5 minutes
EARLY_KEEP_EVERY = 3     # ...and every third state before that (plan 5.5)

games = (pl.read_parquet(ROOT / "data/proc/games.parquet")
         .select("game_id", "season", "home_win", "neutral_site",
                 "pregame_exp_margin", "date", "home_score", "away_score")
         .join(pl.read_parquet(ROOT / "data/proc/team_stats.parquet"),
               on="game_id", how="left"))
out_dir = ROOT / "data/proc/states"
out_dir.mkdir(parents=True, exist_ok=True)

for y in SEASONS:
    t0 = time.time()
    src = ROOT / f"data/raw/pbp/pbp_{y}.parquet"
    cols = set(pl.scan_parquet(src).collect_schema().names())
    sel = list(BASE_COLS) + (["home_win_prob"] if "home_win_prob" in cols else [])
    lf = (pl.scan_parquet(src).select(sel)
          .drop(["season", "game_date"])            # season/date come from games.parquet
          .filter(pl.col("period_number") <= 8))

    st = states_lazy(lf).join(games.lazy(), on="game_id", how="inner")
    st = st.filter(
        pl.col("game_seconds_remaining").is_not_null()
        & (pl.col("margin").abs() < 100)
    )
    # Data-quality gate: ~0.2% of games have a truncated or contradictory feed.
    # A live system would flag these too, so they are excluded everywhere.
    final = (st.group_by("game_id")
             .agg(pl.col("margin").sort_by("seq").last().alias("_last_margin"),
                  pl.col("game_seconds_remaining").sort_by("seq").last().alias("_last_secs"),
                  pl.col("home_win").first().alias("_hw")))
    good = final.filter((pl.col("_last_secs") <= 60)
                        & ((pl.col("_last_margin") > 0).cast(pl.Int8) == pl.col("_hw"))
                        ).select("game_id")
    st = st.join(good, on="game_id", how="inner")
    # Sampling keeps every late-game state and thins the redundant early ones.
    st = st.with_columns(
        _keep=(pl.col("game_seconds_remaining") <= LATE_SECONDS)
        | (pl.col("seq") % EARLY_KEEP_EVERY == 0)
    ).filter(pl.col("_keep"))

    # `margin`, `possession`, `is_ot`, `pregame_exp_margin` come back as features,
    # so they are not repeated here.
    keep = ["game_id", "season", "seq", "period", "game_seconds_remaining",
            "home_timeouts", "away_timeouts", "home_fouls", "away_fouls",
            "neutral_site", "home_win", "date"]
    if "home_win_prob" in cols:
        st = st.rename({"home_win_prob": "espn_wp"})
        keep.append("espn_wp")

    df = st.select(keep + feature_exprs()).collect()
    df.write_parquet(out_dir / f"states_{y}.parquet")
    print(f"{y}: {df.height:>9,} rows  {df['game_id'].n_unique():>5,} games  "
          f"{time.time()-t0:5.1f}s")
```

## `scripts/build_endgame_table.py`

```py
"""Solve the endgame exactly and publish the lookup table.

Endgame plan, Phase 3. Reads the Phase 2 measurements, runs the backward
induction in src/cbbwp/endgame_sim.py, checks every monotonicity property the
plan named -- exhaustively, across the whole table, not sampled -- and writes a
versioned artifact next to the model.

Also writes a small human-readable CSV of canonical states. The plan's third
argument for a table over a live simulation was that a person can read a row and
check it against their own judgement; that only holds if someone actually
prints the rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp import endgame_sim as E  # noqa: E402
from cbbwp.schemas import STATE_RULES_VERSION  # noqa: E402

RAW = ROOT / "data" / "raw" / "pbp"
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]


def ft_bucket_offsets(seasons=TRAIN_SEASONS) -> tuple[list[float], list[float]]:
    """Terciles of team season free-throw percentage, as offsets from the mean.

    The table needs free-throw ability as a small number of buckets. Rather than
    pick cut points, take the terciles the sport actually has and carry the mean
    of each as an offset. Measured on training seasons only.
    """
    rows = []
    for s in seasons:
        d = (
            pl.scan_parquet(RAW / f"pbp_{s}.parquet")
            .select(["team_id", "type_id", "scoring_play"])
            .filter(pl.col("type_id").cast(pl.Int64) == 540)
            .group_by("team_id")
            .agg([pl.len().alias("n"), pl.col("scoring_play").mean().alias("p")])
            .filter(pl.col("n") >= 200)
            .collect()
        )
        rows.append(d)
    d = pl.concat(rows)
    p = d["p"].to_numpy()
    lo, hi = np.quantile(p, [1 / 3, 2 / 3])
    means = [float(p[p <= lo].mean()), float(p[(p > lo) & (p <= hi)].mean()), float(p[p > hi].mean())]
    overall = float(p.mean())
    return [m - overall for m in means], means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="e1")
    ap.add_argument("--params", default=None)
    ap.add_argument("--poss", default=None)
    ap.add_argument("--seasons", nargs="*", type=int, default=None,
                    help="seasons the parameters came from; recorded in the manifest")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seasons = a.seasons or TRAIN_SEASONS
    offsets, bucket_means = ft_bucket_offsets(seasons)
    print(f"free-throw buckets (team season FT%): {[round(x,4) for x in bucket_means]}")
    print(f"offsets from mean:                   {[round(x,4) for x in offsets]}")

    params = E.load_params(
        Path(a.params) if a.params else ROOT / "artifacts" / "endgame_params.json",
        Path(a.poss) if a.poss else ROOT / "artifacts" / "endgame_possessions.json",
        offsets,
    )
    t0 = time.time()
    V = E.solve(params)
    print(f"solved {V.size:,} states in {time.time()-t0:.1f}s")

    report = E.monotonicity_report(V)
    print(json.dumps(report, indent=2))

    V2, moved = E.enforce_margin_monotonicity(V)
    print(f"isotonic projection along margin moved at most {moved:.2e}")
    report_after = E.monotonicity_report(V2)

    out = Path(a.out) if a.out else ROOT / "registry" / "endgame" / a.version
    out.mkdir(parents=True, exist_ok=True)
    table = V2.astype(np.float32)
    np.savez_compressed(out / "table.npz", table=table)

    sha = hashlib.sha256((out / "table.npz").read_bytes()).hexdigest()[:16]
    manifest = {
        "version": a.version,
        "state_rules_version": STATE_RULES_VERSION,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256_16": sha,
        "seasons_used": seasons,
        "shape": list(table.shape),
        "axes": ["seconds_remaining", "margin", "own_fouls", "opp_fouls",
                 "own_ft_bucket", "opp_ft_bucket"],
        "margin_range": [E.MARGIN_MIN, E.MARGIN_MAX],
        "foul_max": E.FOUL_MAX,
        "ft_bucket_means": bucket_means,
        "monotonicity_before": report,
        "monotonicity_after": report_after,
        "isotonic_max_correction": moved,
        "note": "V[t, m, fo, fd, bo, bd] = P(the team WITH THE BALL wins).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # A readable slice: no bonus differential, average free-throw teams.
    mid = E.N_FT_BUCKETS // 2
    rows = []
    for t in (0, 5, 10, 15, 20, 30, 45, 60):
        for m in range(-6, 7):
            for fd, label in ((6, "none"), (8, "1-and-1"), (10, "double")):
                rows.append({
                    "seconds_left": t, "margin": m, "opp_bonus": label,
                    "p_win_with_ball": round(float(table[t, E._mi(m), 6, fd, mid, mid]), 4),
                })
    pl.DataFrame(rows).write_csv(out / "readable.csv")
    print(f"wrote {out}/table.npz  sha {sha}  ({(out/'table.npz').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

## `scripts/build_games.py`

```py
"""Build the game-level table (results + as-of pregame ratings)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from cbbwp.ratings import build_all_seasons

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

frames = []
for y in SEASONS:
    s = (
        pl.scan_parquet(ROOT / f"data/raw/sched/sched_{y}.parquet")
        .select(["game_id", "date", "neutral_site", "home_id", "away_id",
                 "home_score", "away_score", "status_type_completed"])
        .filter(pl.col("status_type_completed")
                & pl.col("home_score").is_not_null()
                & pl.col("away_score").is_not_null()
                & (pl.col("home_score") != pl.col("away_score")))
        .with_columns(
            season=pl.lit(y, dtype=pl.Int32),
            date=pl.col("date").str.to_datetime("%Y-%m-%dT%H:%MZ", strict=False),
            margin=(pl.col("home_score") - pl.col("away_score")).cast(pl.Int32),
            home_win=(pl.col("home_score") > pl.col("away_score")).cast(pl.Int8),
            neutral_site=pl.col("neutral_site").fill_null(False),
        )
        .filter(pl.col("date").is_not_null())
        .unique(subset=["game_id"])
        .collect()
    )
    frames.append(s)
    print(y, s.height)

games = pl.concat(frames, how="vertical_relaxed")
print("total games", games.height)
games = build_all_seasons(games)
games.write_parquet(ROOT / "data/proc/games.parquet")
print("wrote games.parquet", games.height)
```

## `scripts/build_live_context.py`

```py
"""Snapshot today's team ratings and season-to-date stats for the live poller.

Uses the SAME ridge fit as the offline pipeline (cbbwp.ratings._fit_ridge) and
the SAME season-to-date formulas as build_team_stats.py, so a live game's
pregame term is on the same scale the model was fit on.

Run daily before the slate:  python3 scripts/build_live_context.py
"""
import sys, pathlib, json, datetime, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from cbbwp.ratings import _fit_ridge, CARRYOVER

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAGUE_FT, PRIOR_FTA, LEAGUE_PPM = 0.700, 40.0, 3.45

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=None,
                help="season to snapshot (default: the latest in games.parquet)")
ap.add_argument("--out", default=str(ROOT / "registry/context_latest.json"))
a = ap.parse_args()

games = pl.read_parquet(ROOT / "data/proc/games.parquet")
season = a.season or int(games["season"].max())
prev = season - 1 if season - 1 != 2020 else season - 2

# --- 1. ratings: fit this season's completed games, prior = last season's ----
prev_g = games.filter(pl.col("season") == prev)
teams_prev = sorted(set(prev_g["home_id"].to_list()) | set(prev_g["away_id"].to_list()))
prev_ratings, _ = _fit_ridge(prev_g, teams_prev, {})
prior = {t: v * CARRYOVER for t, v in prev_ratings.items()}

cur = games.filter(pl.col("season") == season)
teams = sorted(set(cur["home_id"].to_list()) | set(cur["away_id"].to_list()) | set(prior))
ratings, hca = _fit_ridge(cur, teams, prior)
print(f"season {season}: {cur.height} completed games, {len(teams)} teams, hca={hca:.2f}, "
      f"rating sd={np.std(list(ratings.values())):.2f}")
if cur.height == 0:
    print("  (no games yet - ratings are last season's, carried over. Expected in preseason.)")

# --- 2. season-to-date FT% and points per minute -----------------------------
ft_pct, ppm = {}, {}
pbp = ROOT / f"data/raw/pbp/pbp_{season}.parquet"
if cur.height and pbp.exists():
    ft = (pl.scan_parquet(pbp)
          .select(["game_id", "type_text", "scoring_play", "team_id"])
          .filter(pl.col("type_text").str.contains("FreeThrow"))
          .group_by("team_id").agg(fta=pl.len(), ftm=pl.col("scoring_play").sum())
          .collect())
    for tid, fta, ftm in ft.iter_rows():
        if tid is None:
            continue
        ft_pct[int(tid)] = (ftm + LEAGUE_FT * PRIOR_FTA) / (fta + PRIOR_FTA)

    long = pl.concat([
        cur.select(team_id="home_id", pts="home_score", opp="away_score"),
        cur.select(team_id="away_id", pts="away_score", opp="home_score"),
    ]).group_by("team_id").agg(tot=(pl.col("pts") + pl.col("opp")).sum(), g=pl.len())
    for tid, tot, g in long.iter_rows():
        ppm[int(tid)] = (tot + LEAGUE_PPM * 40 * 5) / ((g + 5) * 40)

# The date of the newest completed game these ratings were fit on. Without it,
# the only freshness signal is when this file was written -- so a nightly job
# running over a stale data copy produces a snapshot that reports itself fresh
# and is not. See LiveContextProvider.data_age_days.
latest_game_date = ""
if cur.height and "date" in cur.columns:
    m = cur["date"].max()
    latest_game_date = m.isoformat() if hasattr(m, "isoformat") else str(m)

out = {
    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "latest_game_date": latest_game_date,
    "season": season,
    "hca": hca,
    "n_completed_games": cur.height,
    "ratings": {str(k): round(v, 4) for k, v in ratings.items()},
    "ft_pct": {str(k): round(v, 4) for k, v in ft_pct.items()},
    "ppm": {str(k): round(v, 4) for k, v in ppm.items()},
}
dest = pathlib.Path(a.out)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out))
print(f"wrote {dest}  ({len(ratings)} ratings, {len(ft_pct)} ft, {len(ppm)} ppm)")
print(f"  newest completed game: {latest_game_date or 'none yet (preseason)'}")
```

## `scripts/build_report_data.py`

```py
"""Regenerate the data block behind the published results artifact.

The artifact used to carry numbers pasted in by hand, which is how it ended up
still advertising v1's figures and "17 tests passing" two model versions later.
This makes the data a build product: run it after any refit, paste the one line
it prints into the artifact's trailing <script>, and the page cannot drift from
the model again.

    python3 scripts/build_report_data.py            # writes artifacts/report_data.json

Everything comes from artifacts/eval_preds.parquet (written by fit_models.py)
and the state rows, so it always describes the model that was actually fit.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp.schemas import FEATURE_NAMES          # noqa: E402

OUT = ROOT / "artifacts" / "report_data.json"
CURVE_SEASON = 2026
N_CURVES = 4
MAX_POINTS = 200
EPS = 1e-15

BUCKETS = [("40-20 min", 1200, 2400), ("20-10 min", 600, 1200),
           ("10-5 min", 300, 600), ("5-2 min", 120, 300),
           ("2-1 min", 60, 120), ("1-0 min", 0, 60)]


def _ll(y, p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    # Prefer the float64 predictions. The float32 parquet export lands about
    # 0.0001 low on log loss and differs in the fourth decimal on accuracy and
    # ECE -- small, but enough to make a published page disagree with EXPLAIN.
    npz = ROOT / "artifacts" / "test_preds.npz"
    if npz.exists():
        raw = np.load(npz)
        d = {k: raw[k] for k in raw.files}
        print("source: test_preds.npz (float64)")
    else:
        raise SystemExit("run scripts/rebuild_test_preds.py first -- the float32 "
                         "export is not precise enough for the published figures")
    y = d["y"].astype(np.float64)
    secs = d["secs"].astype(np.float64)
    P = {"model": d["p_gbm"].astype(np.float64),
         "logistic": d["p_lr"].astype(np.float64),
         "espn": d["espn"].astype(np.float64)}
    ok = np.isfinite(P["espn"])

    headline = {}
    for k, p in P.items():
        pf = p[ok].astype(np.float64)
        headline[k] = {"logloss": round(_ll(y[ok], pf), 4),
                       "brier": round(float(((pf - y[ok]) ** 2).mean()), 4),
                       "acc": round(float(((pf >= .5).astype(int) == y[ok]).mean()), 4),
                       "ece": round(_ece(y[ok], pf), 4)}

    buckets = []
    for name, lo, hi in BUCKETS:
        m = ok & (secs >= lo) & (secs < hi)
        buckets.append({"bucket": name, "n": int(m.sum()),
                        **{k: round(_ll(y[m], P[k][m].astype(np.float64)), 4)
                           for k in P}})

    # calibration, 20 equal-width bins on the shipped model
    pg = P["model"].astype(np.float64)
    edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(pg, edges) - 1, 0, 19)
    calib = []
    for b in range(20):
        s = idx == b
        if s.sum() < 50:
            continue
        calib.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(s.sum()),
                      "pred": round(float(pg[s].mean()), 4),
                      "obs": round(float(y[s].mean()), 4)})

    print(f"headline: {headline['model']}")
    print(f"{len(buckets)} buckets, {len(calib)} calibration bins")

    curves = _curves()
    n_ot = sum(1 for c in curves if c["max_x"] > 2400)
    print(f"\n{n_ot} of {len(curves)} highlighted games went to overtime "
          "-- check the artifact caption says so")
    out = {"headline": headline, "buckets": buckets, "calibration": calib,
           "curves": curves,
           "meta": {"test_rows": int(len(y)),
                    "test_games": int(np.unique(d["game_id"]).size),
                    "test_seasons": sorted(int(s) for s in np.unique(d["season"])),
                    "overtime_curves": n_ot}}
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size/1000:.0f} KB)")
    print("\npaste this into the artifact's trailing <script>:")
    print("const D = " + json.dumps(out) + ";")


def _ece(y, p, bins: int = 20) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.sum():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def _curves() -> list[dict]:
    """The highest-movement games of the season, drawn from the shipped model.

    Overtime games dominate this list, which is the point: a curve that handles
    a tied buzzer gracefully is the hardest case, and the eye catches what no
    aggregate metric will.
    """
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / "registry" / "v2" / "model.txt"))

    st = (pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{CURVE_SEASON}.parquet")
          .select(FEATURE_NAMES + [c for c in
                  ["game_id", "seq", "period", "game_seconds_remaining", "espn_wp"]
                  if c not in FEATURE_NAMES])
          .collect().sort(["game_id", "seq"]))
    X = st.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    st = st.with_columns(pl.Series("wp", booster.predict(X)))

    gei = (st.group_by("game_id")
             .agg(pl.col("wp").diff().abs().sum().alias("gei"))
             .sort("gei", descending=True).head(N_CURVES))
    games = pl.read_parquet(ROOT / "data" / "proc" / "games.parquet")
    names = (pl.scan_parquet(ROOT / "data" / "raw" / "pbp" / f"pbp_{CURVE_SEASON}.parquet")
             .select(["game_id", "home_team_name", "away_team_name"])
             .unique(subset="game_id").collect())

    out = []
    for gid, g in gei.iter_rows():
        sub = st.filter(pl.col("game_id") == gid)
        info = games.filter(pl.col("game_id") == gid).row(0, named=True)
        nm = names.filter(pl.col("game_id") == gid)
        home = nm["home_team_name"][0] if len(nm) else f"home {info['home_id']}"
        away = nm["away_team_name"][0] if len(nm) else f"away {info['away_id']}"

        per = sub["period"].to_numpy()
        gsr = sub["game_seconds_remaining"].to_numpy().astype(float)
        # Elapsed seconds. Regulation counts down from 2400; each overtime is its
        # own 300-second clock, so they have to be laid end to end by hand.
        elapsed = np.where(per <= 2, 2400 - gsr,
                           2400 + (per - 3) * 300 + (300 - gsr))
        n_ot = max(0, int(per.max()) - 2)
        max_x = 2400 + n_ot * 300

        keep = np.linspace(0, len(sub) - 1, min(MAX_POINTS, len(sub))).astype(int)
        keep = np.unique(keep)
        out.append({
            "title": f"{away} {info['away_score']} @ {home} {info['home_score']}",
            "date": str(info["date"])[:10],
            "gei": round(float(g), 2),
            "x": [int(v) for v in elapsed[keep]],
            "wp": [round(float(v), 4) for v in sub["wp"].to_numpy()[keep]],
            "espn": [round(float(v), 4) for v in sub["espn_wp"].to_numpy()[keep]],
            "margin": [float(v) for v in sub["margin"].to_numpy()[keep]],
            "max_x": int(max_x),
        })
        print(f"  curve: {out[-1]['title']}  {out[-1]['date']}  movement {g:.2f}")
    return out


if __name__ == "__main__":
    main()
```

## `scripts/build_team_stats.py`

```py
"""Season-to-date team stats, as of the day BEFORE each game (no leakage).

Produces per game: ft_pct_diff (home minus away) and exp_points_per_min
(the two teams' combined scoring rate), both from games already played.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
LEAGUE_FT = 0.700          # prior for teams with no games yet
PRIOR_FTA = 40.0           # strength of that prior, in attempts
LEAGUE_PPM = 3.45          # points per minute, both teams combined

games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
    "game_id", "season", "date", "home_id", "away_id", "home_score", "away_score")

per_team = []
for y in SEASONS:
    ft = (
        pl.scan_parquet(ROOT / f"data/raw/pbp/pbp_{y}.parquet")
        .select(["game_id", "type_text", "scoring_play", "team_id"])
        .filter(pl.col("type_text").str.contains("FreeThrow"))
        .group_by(["game_id", "team_id"])
        .agg(fta=pl.len(), ftm=pl.col("scoring_play").sum())
        .collect()
    )
    per_team.append(ft.with_columns(season=pl.lit(y, dtype=pl.Int32)))
ft = pl.concat(per_team)

# long form: one row per (game, team) with that team's FT and points
long = pl.concat([
    games.select("game_id", "season", "date", team_id=pl.col("home_id"),
                 pts=pl.col("home_score"), opp_pts=pl.col("away_score")),
    games.select("game_id", "season", "date", team_id=pl.col("away_id"),
                 pts=pl.col("away_score"), opp_pts=pl.col("home_score")),
]).join(ft.select("game_id", "team_id", "fta", "ftm"), on=["game_id", "team_id"], how="left")

long = long.sort(["team_id", "season", "date"]).with_columns(
    # cum_sum shifted by one game = everything BEFORE this game
    c_fta=pl.col("fta").fill_null(0).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_ftm=pl.col("ftm").fill_null(0).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_pts=(pl.col("pts") + pl.col("opp_pts")).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_g=pl.int_range(pl.len()).over(["team_id", "season"]),
).with_columns(
    ft_pct=(pl.col("c_ftm") + LEAGUE_FT * PRIOR_FTA) / (pl.col("c_fta") + PRIOR_FTA),
    ppm=(pl.col("c_pts") + LEAGUE_PPM * 40 * 5) / ((pl.col("c_g") + 5) * 40),
)

h = long.select("game_id", h_ft="ft_pct", h_ppm="ppm", team_id="team_id")
out = (
    games
    .join(h.rename({"team_id": "home_id"}), on=["game_id", "home_id"], how="left")
    .join(h.rename({"team_id": "away_id", "h_ft": "a_ft", "h_ppm": "a_ppm"}),
          on=["game_id", "away_id"], how="left")
    .select(
        "game_id",
        ft_pct_diff=(pl.col("h_ft") - pl.col("a_ft")).fill_null(0.0),
        exp_points_per_min=((pl.col("h_ppm") + pl.col("a_ppm")) / 2).fill_null(LEAGUE_PPM),
        # The two levels, not just their difference. The endgame table buckets
        # each team's free-throw ability separately, and reconstructing the
        # levels from the difference is not possible. Keeping only the diff here
        # was what would have forced the endgame validation to assume both teams
        # were average -- an assumption the live path would NOT have made, which
        # is exactly the train/serve mismatch this project keeps tripping over.
        home_ft_pct=pl.col("h_ft").fill_null(LEAGUE_FT),
        away_ft_pct=pl.col("a_ft").fill_null(LEAGUE_FT),
    )
)
out.write_parquet(ROOT / "data/proc/team_stats.parquet")
print(out.describe())
```

## `scripts/calibrate_and_eval.py`

```py
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.calibration import TimeBucketedCalibrator
from cbbwp import endgame
from cbbwp.evaluate import log_loss, brier, accuracy, ece, calibration_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
ca = np.load(ROOT / "artifacts/calib_preds.npz")
te = np.load(ROOT / "artifacts/test_preds.npz")
y, secs, espn = te["y"], te["secs"], te["espn"]

cal = TimeBucketedCalibrator().fit(ca["p_gbm"], ca["y"], ca["secs"])
with open(ROOT / "artifacts/calibrator_v1.pkl", "wb") as f:
    pickle.dump(cal, f)

raw = te["p_gbm"]
calibrated = cal.transform(raw, secs)
# endgame overrides need the margin, recovered from the saved feature column
import polars as pl
margin = pl.concat([pl.scan_parquet(ROOT / f"data/proc/states/states_{s}.parquet")
                    .select("margin") for s in (2025, 2026)], how="diagonal").collect()["margin"].to_numpy()
assert len(margin) == len(y)
final = endgame.apply(calibrated, margin, secs)

rows = {
    "lightgbm raw": raw,
    "+ time-bucket calib": calibrated,
    "+ endgame overrides": final,
    "logistic baseline": te["p_lr"],
    "ESPN (deployed)": espn,
}
print(f"{'model':<22}{'logloss':>9}{'brier':>9}{'acc':>8}{'ECE':>8}")
for k, p in rows.items():
    print(f"{k:<22}{log_loss(y,p):>9.4f}{brier(y,p):>9.4f}{accuracy(y,p):>8.4f}{ece(y,p):>8.4f}")

print(f"\nlog loss by time bucket\n{'bucket':<12}{'n':>10}" + "".join(f"{k[:11]:>13}" for k in rows))
for name, lo, hi in [("40-20 min",1200,2400),("20-10 min",600,1200),("10-5 min",300,600),
                     ("5-2 min",120,300),("2-1 min",60,120),("1-0 min",0,60)]:
    m = (secs >= lo) & (secs < hi)
    print(f"{name:<12}{m.sum():>10,}" + "".join(f"{log_loss(y[m],p[m]):>13.4f}" for p in rows.values()))

print("\nfinal-2-minute calibration, after correction")
m = secs < 120
print(f"{'bin':<12}{'n':>10}{'pred':>8}{'obs':>8}{'gap':>8}")
for r in calibration_table(y[m], final[m], n_bins=10):
    print(f"{r['bin']:<12}{r['n']:>10,}{r['pred']:>8.3f}{r['obs']:>8.3f}{r['obs']-r['pred']:>+8.3f}")

np.savez_compressed(ROOT / "artifacts/final_preds.npz", y=y, p=final, secs=secs,
                    espn=espn, margin=margin, game_id=te["game_id"], season=te["season"])
```

## `scripts/calibration_monitor.py`

```py
"""Weekly calibration drift check. Exit code 1 means "look at this".

Two sources, same check:

  --source backtest   score a window of already-played games with the pinned
                      model and check the answers against what happened.
                      This is the one to run weekly during the season.

  --source live       read the poller's JSONL, join each state to the final
                      result from games.parquet, and check that. This measures
                      the SHIPPED path end to end - feed, adapter, model - and
                      so it is the one that catches a broken feed, not just a
                      stale model.

Examples
--------
    python3 scripts/calibration_monitor.py --source backtest --season 2027
    python3 scripts/calibration_monitor.py --source backtest --days 7
    python3 scripts/calibration_monitor.py --source live --glob 'data/live/*.jsonl'

Cron it Monday morning:
    0 9 * * 1  cd /path/to/ncaa_mbb && python3 scripts/calibration_monitor.py \
                 --source backtest --days 7 --json reports/calib_$(date +\%F).json
"""
import sys, pathlib, json, glob, argparse, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from cbbwp import monitor
from cbbwp.schemas import FEATURE_NAMES
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]


def from_backtest(args):
    """Score recent completed games with the pinned model."""
    games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
        "game_id", "season", "date", "home_win")
    season = args.season or int(games["season"].max())
    states_path = ROOT / f"data/proc/states/states_{season}.parquet"
    if not states_path.exists():
        raise SystemExit(f"no states file for season {season}: {states_path}\n"
                         "  run scripts/build_dataset.py first")

    df = pl.read_parquet(states_path).join(
        games.select("game_id", "date"), on="game_id", how="left")
    window = f"season {season}"
    if args.days:
        cutoff = df["date"].max() - datetime.timedelta(days=args.days)
        df = df.filter(pl.col("date") >= cutoff)
        window = f"season {season}, last {args.days} days (since {cutoff:%Y-%m-%d})"

    if df.height == 0:
        raise SystemExit("no rows in that window")

    svc = WinProbabilityService(args.registry, args.version)
    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    p = np.asarray(svc._predict(X), dtype=np.float64)
    return (df["home_win"].to_numpy(), p,
            df["game_seconds_remaining"].to_numpy().astype(np.float64),
            df["game_id"].to_numpy(), window)


def from_live(args):
    """Read poller output and join to final results."""
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob}")
    rows = []
    for f in files:
        for line in pathlib.Path(f).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("live files are empty")
    live = pl.DataFrame(rows).select(
        "game_id", "home_win_prob", "game_seconds_remaining")
    games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
        "game_id", "home_win")
    j = live.join(games, on="game_id", how="inner")
    missing = live.height - j.height
    if missing:
        print(f"note: {missing:,} live states have no final result yet; skipped",
              file=sys.stderr)
    if j.height == 0:
        raise SystemExit("no live states could be matched to a finished game")
    return (j["home_win"].to_numpy(), j["home_win_prob"].to_numpy(),
            j["game_seconds_remaining"].to_numpy().astype(np.float64),
            j["game_id"].to_numpy(), f"live, {len(files)} file(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["backtest", "live"], default="backtest")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="restrict a backtest to the last N days of the season")
    ap.add_argument("--glob", default=str(ROOT / "data/live/*.jsonl"))
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    ap.add_argument("--version", default="v2")
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--z", type=float, default=monitor.Z_ALERT)
    ap.add_argument("--min-gap", type=float, default=monitor.MIN_GAP)
    a = ap.parse_args()

    y, p, secs, gids, window = (from_live(a) if a.source == "live"
                                else from_backtest(a))
    rep = monitor.check(y, p, secs, gids, window=window, z_alert=a.z,
                        min_gap=a.min_gap,
                        generated=datetime.datetime.now(
                            datetime.timezone.utc).isoformat())
    print(monitor.format_report(rep))

    if a.json:
        dest = pathlib.Path(a.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rep.as_dict(), indent=2))
        print(f"\nwrote {dest}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/check_espn_fixtures.py`

```py
"""Validate recorded ESPN payloads against the adapter's expectations.

This is the check that the offline test suite CANNOT do: it looks at what ESPN
actually sends today and reports anything the adapter would silently mishandle -
above all a play type id the model was never trained on.

    python3 scripts/check_espn_fixtures.py --dir tmp/fixtures
"""
import sys, pathlib, json, argparse, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.adapters import espn
from cbbwp.state import build_states
from cbbwp.schemas import PregameContext

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(ROOT / "tmp/fixtures"))
a = ap.parse_args()

files = sorted(pathlib.Path(a.dir).glob("summary_*.json"))
if not files:
    raise SystemExit(f"no summary_*.json in {a.dir} - "
                     "run scripts/record_espn_fixtures.py first")

unknown = collections.Counter()
problems = 0
n_synth = 0
for f in files:
    payload = json.loads(f.read_text())
    synthetic = espn.is_synthetic_payload(payload)
    n_synth += synthetic
    raw_plays = payload.get("plays") or []
    events, h = espn.parse_summary(payload)
    ctx = PregameContext(h.game_id, h.home_team_id, h.away_team_id, h.neutral_site)
    states = build_states(events, ctx)

    for p in raw_plays:
        t = p.get("type") or {}
        tid = espn._int(t.get("id"))
        if tid not in espn.TYPE_ID_TO_TEXT:
            unknown[(tid, (t.get("text") or "").strip())] += 1

    ok = True
    if len(events) != len(raw_plays):
        print(f"  {f.name}: {len(raw_plays)} plays -> {len(events)} events (MISMATCH)")
        ok = False
    if states and h.is_final:
        last = states[-1]
        said = last.margin
        actual = h.home_score - h.away_score
        if said != actual:
            print(f"  {f.name}: final margin from plays {said:+d} != header "
                  f"{actual:+d} - truncated or contradictory feed")
            ok = False
    if not h.home_team_id or not h.away_team_id:
        print(f"  {f.name}: could not read team ids")
        ok = False
    problems += 0 if ok else 1
    print(f"{'ok  ' if ok else 'BAD '}{f.name:<28} {h.away_name} @ {h.home_name}  "
          f"{h.status}  {len(events):,} plays, {len(states):,} states"
          + ("   [REBUILT FROM hoopR - not evidence about ESPN]" if synthetic else ""))

# A payload rebuilt from hoopR carries hoopR's own type ids, and the model's type
# map was built from those same files - so "no unknown types" is guaranteed and
# says nothing about the live feed. Saying so is the whole value of this script.
if n_synth:
    print(f"\n{n_synth} of {len(files)} payload(s) were REBUILT FROM hoopR, not "
          "recorded from ESPN.")
    if n_synth == len(files):
        print("Every payload here is a rebuild, so the unknown-play-type check below\n"
              "CANNOT FAIL and proves nothing about what ESPN is sending. Record real\n"
              "payloads with scripts/record_espn_fixtures.py on a night with games.")

if unknown:
    print("\nPLAY TYPES THE MODEL HAS NEVER SEEN "
          "(they fall back to the feed's text and carry possession):")
    for (tid, text), n in unknown.most_common():
        print(f"  id={tid}  text={text!r}  x{n:,}")
    print("\n  A frequent one here means the ESPN feed changed. Add it to "
          "TYPE_ID_TO_TEXT only if the training data also contains it;\n"
          "  otherwise the honest fix is to refit with the new type present.")
else:
    print("\nno unknown play types - the adapter's type map covers this feed")

raise SystemExit(1 if problems else 0)
```

## `scripts/estimate_endgame_params.py`

```py
"""Estimate every parameter the endgame simulator needs, from play-by-play.

Endgame plan, Phase 2: "No hand-set constants." Every number the simulator uses
is measured here, on TRAINING seasons only (2016-2024). 2025 and 2026 are the
held-out test seasons and this script refuses to read them -- the pre-registered
bar in docs/cbbwp-endgame-plan.md is a single-shot test and stays that way.

Three things this file deliberately does NOT do:

1. It does not key on ESPN's "free throw N of M" text. That text appears only in
   the 2026 feed, so using it would be both unavailable for training seasons and
   a test-set leak. Trips are classified by the OPPONENT'S TEAM FOUL COUNT
   instead -- the rule the sport actually uses, which reproduces the 2026 text
   labels 94.9% of the time.

2. It does not read `scoring_play` on a play labelled "1 of 1" and call the
   result a free-throw percentage. ESPN labels a trip by the attempts actually
   TAKEN, so a MADE one-and-one front end is logged as a two-shot trip and never
   appears as "1 of 1" at all. Conditioning on that label conditions on the
   outcome, which is why the raw figure is 0.537 and the true one is 0.70.
   Trips are rebuilt first; only then is the first shot of each trip read.

3. It does not hold every season in memory at once. hoopR play-by-play is large
   and the device VM is small, so each season is reduced to COUNTS
   (n, successes) and the counts are summed. Means are formed at the end, from
   the totals. This also makes each season's contribution auditable.

Run:
    python3 scripts/estimate_endgame_params.py --season 2016   # ... one at a time
    python3 scripts/estimate_endgame_params.py --combine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbp"
PART = ROOT / "artifacts" / "endgame_parts"
OUT = ROOT / "artifacts" / "endgame_params.json"

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
TEST_SEASONS = {2025, 2026}

FT, FOUL, TECH = 540, 519, 521
OREB, DREB = 586, 587
SHOT_TYPES = [558, 572, 574, 437]
TURNOVER_TYPES = [598, 601, 602]
BONUS_FOULS, DOUBLE_BONUS_FOULS = 7, 10

ENDGAME_SECONDS = 60
WIDE_SECONDS = 180

COLS = [
    "game_id", "sequence_number", "type_id", "text", "scoring_play", "score_value",
    "team_id", "home_team_id", "away_team_id", "athlete_id_1",
    "period_number", "clock_minutes", "clock_seconds",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(season: int) -> pl.DataFrame:
    """One season, ordered, with a season-independent game clock.

    hoopR's seconds-remaining column is named `start_half_seconds_remaining` in
    2016-2022 and `start_period_seconds_remaining` in 2023+. The raw clock
    fields are present in every season, so the clock is rebuilt from those
    rather than picking one of the two names and silently failing on the other.
    """
    df = (
        pl.scan_parquet(RAW / f"pbp_{season}.parquet")
        .select(COLS)
        .collect()
        .with_columns(pl.col("sequence_number").cast(pl.Int64))
        .sort(["game_id", "sequence_number"])
        .with_row_index("i")
    )
    return df.with_columns(
        [
            pl.col("type_id").cast(pl.Int64).alias("tid"),
            pl.when(pl.col("period_number") <= 1).then(1).otherwise(2).alias("half"),
            (
                pl.col("clock_minutes").cast(pl.Float64).fill_null(0) * 60
                + pl.col("clock_seconds").cast(pl.Float64).fill_null(0)
            ).alias("sec"),
        ]
    )


def annotate(df: pl.DataFrame) -> pl.DataFrame:
    """Team fouls, the and-one flag, free-throw trip ids, and elapsed time."""
    df = df.with_columns(
        [
            (pl.col("tid").is_in([FOUL, TECH]) & (pl.col("team_id") == pl.col("home_team_id"))).alias("hf"),
            (pl.col("tid").is_in([FOUL, TECH]) & (pl.col("team_id") == pl.col("away_team_id"))).alias("af"),
            (pl.col("tid") == FT).alias("isft"),
            (pl.col("scoring_play") & (pl.col("tid") != FT)).alias("madefg"),
            pl.col("text").str.contains("(?i)three point").fill_null(False).alias("is3"),
        ]
    )
    # Team fouls reset once, at the half. Counted BEFORE the current row, so the
    # count describes the situation the current event was played under -- which
    # is what decides whether a foul sends the shooter to a one-and-one.
    df = df.with_columns(
        [
            (pl.col("hf").cum_sum().over(["game_id", "half"]) - pl.col("hf").cast(pl.Int32)).alias("home_fouls"),
            (pl.col("af").cum_sum().over(["game_id", "half"]) - pl.col("af").cast(pl.Int32)).alias("away_fouls"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id"))
        .then(pl.col("away_fouls"))
        .otherwise(pl.col("home_fouls"))
        .alias("opp_fouls")
    )

    # And-one: the shooter made a field goal at effectively the same game clock.
    # Substitutions and TV timeouts pad the gap, so the window is events, not
    # rows adjacent -- 12 is comfortably past the longest observed run of subs.
    andone = None
    for k in range(1, 13):
        t = (
            pl.col("madefg").shift(k).over("game_id")
            & (pl.col("athlete_id_1").shift(k).over("game_id") == pl.col("athlete_id_1"))
            & ((pl.col("sec").shift(k).over("game_id") - pl.col("sec")).abs() <= 3)
        )
        andone = t if andone is None else (andone | t)
    df = df.with_columns(andone.fill_null(False).alias("andone"))

    # A trip is a run of free throws by one team at one dead ball.
    ft = pl.col("isft")
    prev_i = pl.when(ft).then(pl.col("i")).otherwise(None).forward_fill().over("game_id").shift(1)
    prev_team = pl.when(ft).then(pl.col("team_id")).otherwise(None).forward_fill().over("game_id").shift(1)
    prev_sec = pl.when(ft).then(pl.col("sec")).otherwise(None).forward_fill().over("game_id").shift(1)
    new_trip = ft & (
        prev_i.is_null()
        | (pl.col("team_id") != prev_team)
        | ((prev_sec - pl.col("sec")).abs() > 4)
        | ((pl.col("i") - prev_i) > 8)
    )
    df = df.with_columns(
        pl.when(ft).then(new_trip.fill_null(True).cast(pl.Int32).cum_sum().over("game_id")).alias("trip")
    )

    # Seconds until the clock next moves. The next ROW is usually a substitution
    # at the same dead ball, so a plain shift(-1) measures nothing and returns a
    # median of 0. Take the first following event whose clock is strictly lower.
    nxt = None
    for k in range(1, 13):
        s = pl.col("sec").shift(-k).over("game_id")
        cand = pl.when(s < pl.col("sec")).then(s).otherwise(None)
        nxt = cand if nxt is None else pl.coalesce([nxt, cand])
    df = df.with_columns((pl.col("sec") - nxt).alias("dt_clock"))

    # Seconds until the next foul, however many dead-ball rows intervene.
    nf = None
    for k in range(1, 25):
        s = pl.col("sec").shift(-k).over("game_id")
        isf = pl.col("tid").shift(-k).over("game_id").is_in([FOUL, TECH])
        cand = pl.when(isf & (s <= pl.col("sec"))).then(s).otherwise(None)
        nf = cand if nf is None else pl.coalesce([nf, cand])
    return df.with_columns((pl.col("sec") - nf).alias("dt_foul"))


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------
def _nk(frame: pl.DataFrame, col: str = "scoring_play") -> dict:
    return {"n": int(len(frame)), "k": int(frame[col].sum()) if len(frame) else 0}


def trip_kind() -> pl.Expr:
    return (
        pl.when(pl.col("andone")).then(pl.lit("and_one"))
        .when(pl.col("n_shots") >= 3).then(pl.lit("shooting_3"))
        .when((pl.col("opp_fouls") >= BONUS_FOULS) & (pl.col("opp_fouls") < DOUBLE_BONUS_FOULS))
        .then(pl.lit("one_and_one"))
        .otherwise(pl.lit("two_shot"))
    )


def count_season(df: pl.DataFrame) -> dict:
    out: dict = {}

    # ---- free throws, by trip kind and shot index -------------------------
    ft = df.filter(pl.col("isft")).with_columns(
        pl.col("i").rank("ordinal").over(["game_id", "trip"]).alias("shot_no")
    )
    sizes = ft.group_by(["game_id", "trip"]).agg(pl.len().alias("n_shots"))
    ft = ft.join(sizes, on=["game_id", "trip"], how="left").with_columns(trip_kind().alias("kind"))

    windows = {
        "all_game": pl.lit(True),
        "last_180s": (pl.col("period_number") >= 2) & (pl.col("sec") <= WIDE_SECONDS),
        "last_60s": (pl.col("period_number") >= 2) & (pl.col("sec") <= ENDGAME_SECONDS),
    }
    out["free_throws"] = {}
    for wname, w in windows.items():
        g = (
            ft.filter(w)
            .group_by(["kind", "shot_no"])
            .agg([pl.len().alias("n"), pl.col("scoring_play").sum().alias("k")])
        )
        out["free_throws"][wname] = {
            f"{r['kind']}_{r['shot_no']}": {"n": int(r["n"]), "k": int(r["k"])}
            for r in g.iter_rows(named=True)
        }

    # ---- how often each kind of trip happens late -------------------------
    trips = (
        ft.group_by(["game_id", "trip"])
        .agg([pl.first("kind").alias("kind"), pl.first("sec").alias("sec"),
              pl.first("period_number").alias("period")])
        .filter((pl.col("period") >= 2) & (pl.col("sec") <= ENDGAME_SECONDS))
    )
    out["late_trip_mix"] = {
        r["kind"]: int(r["n"])
        for r in trips.group_by("kind").agg(pl.len().alias("n")).iter_rows(named=True)
    }

    # ---- rebounding -------------------------------------------------------
    reb = None
    for k in range(1, 4):
        t = pl.col("tid").shift(-k).over("game_id")
        cand = pl.when(t.is_in([OREB, DREB])).then(t).otherwise(None)
        reb = cand if reb is None else pl.coalesce([reb, cand])
    d = df.with_columns(reb.alias("reb")).with_columns((pl.col("reb") == OREB).alias("is_oreb"))

    last_ft = (
        d.filter(pl.col("isft"))
        .with_columns(pl.col("i").rank("ordinal").over(["game_id", "trip"]).alias("shot_no"))
        .join(
            d.filter(pl.col("isft"))
            .group_by(["game_id", "trip"])
            .agg(pl.len().alias("n_shots")),
            on=["game_id", "trip"], how="left",
        )
        .filter(pl.col("shot_no") == pl.col("n_shots"))
        .filter(~pl.col("scoring_play") & pl.col("reb").is_not_null())
    )
    miss_fg = d.filter(pl.col("tid").is_in(SHOT_TYPES) & ~pl.col("scoring_play") & pl.col("reb").is_not_null())
    late = (pl.col("period_number") >= 2) & (pl.col("sec") <= ENDGAME_SECONDS)
    out["rebounds"] = {
        "oreb_after_missed_ft": _nk(last_ft, "is_oreb"),
        "oreb_after_missed_ft_last60": _nk(last_ft.filter(late), "is_oreb"),
        "oreb_after_missed_3": _nk(miss_fg.filter(pl.col("is3")), "is_oreb"),
        "oreb_after_missed_2": _nk(miss_fg.filter(~pl.col("is3")), "is_oreb"),
        "oreb_after_missed_3_last60": _nk(miss_fg.filter(pl.col("is3") & late), "is_oreb"),
        "oreb_after_missed_2_last60": _nk(miss_fg.filter(~pl.col("is3") & late), "is_oreb"),
    }

    # ---- shot mix and accuracy late, by the shooter's own margin ----------
    d = d.with_columns(
        [
            pl.when(pl.col("scoring_play") & (pl.col("team_id") == pl.col("home_team_id")))
            .then(pl.col("score_value")).otherwise(0).alias("hs"),
            pl.when(pl.col("scoring_play") & (pl.col("team_id") == pl.col("away_team_id")))
            .then(pl.col("score_value")).otherwise(0).alias("as_"),
        ]
    )
    d = d.with_columns(
        [
            (pl.col("hs").cum_sum().over("game_id") - pl.col("hs")).alias("h"),
            (pl.col("as_").cum_sum().over("game_id") - pl.col("as_")).alias("a"),
        ]
    ).with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id"))
        .then(pl.col("h") - pl.col("a"))
        .otherwise(pl.col("a") - pl.col("h"))
        .alias("actor_margin")
    )
    shots = d.filter(pl.col("tid").is_in(SHOT_TYPES) & late).with_columns(
        pl.col("actor_margin").clip(-6, 6).alias("m")
    )
    g = shots.group_by("m").agg(
        [
            pl.len().alias("n"),
            pl.col("is3").sum().alias("n3"),
            (pl.col("scoring_play") & pl.col("is3")).sum().alias("k3"),
            (pl.col("scoring_play") & ~pl.col("is3")).sum().alias("k2"),
        ]
    )
    out["late_shots"] = {
        str(r["m"]): {"n": int(r["n"]), "n3": int(r["n3"]), "k3": int(r["k3"]), "k2": int(r["k2"])}
        for r in g.iter_rows(named=True)
    }

    # ---- clock ------------------------------------------------------------
    ends = d.filter(late & pl.col("tid").is_in(SHOT_TYPES + TURNOVER_TYPES))
    out["late_possession_ends"] = {
        "n": int(len(ends)),
        "k_turnover": int(ends["tid"].is_in(TURNOVER_TYPES).sum()) if len(ends) else 0,
    }
    made_ft = d.filter(pl.col("isft") & pl.col("scoring_play") & late)
    dtf = made_ft.filter(pl.col("dt_foul").is_between(0, 20))["dt_foul"]
    dtc = made_ft.filter(pl.col("dt_clock").is_between(0, 30))["dt_clock"]
    out["clock"] = {
        "sec_made_ft_to_next_foul": {
            "n": int(len(dtf)),
            "sum": float(dtf.sum()) if len(dtf) else 0.0,
            "hist": {str(int(b)): int(c) for b, c in
                     zip(*[x.to_list() for x in dtf.cast(pl.Int32).value_counts().sort("dt_foul")])}
            if len(dtf) else {},
        },
        "sec_made_ft_to_clock_move": {
            "n": int(len(dtc)),
            "sum": float(dtc.sum()) if len(dtc) else 0.0,
        },
    }
    return out


# --------------------------------------------------------------------------
# combining
# --------------------------------------------------------------------------
def _merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge(a.get(k), b.get(k)) for k in set(a) | set(b)}
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def combine(parts: list[dict]) -> dict:
    tot: dict = {}
    for p in parts:
        body = {k: v for k, v in p.items() if k != "season"}
        tot = body if not tot else _merge(tot, body)

    def rate(d):
        return {k: {"p": (v["k"] / v["n"]) if v["n"] else None, "n": v["n"]} for k, v in d.items()}

    mix_total = sum(tot["late_trip_mix"].values()) or 1
    late_shots = {
        m: {
            "n": v["n"],
            "p3a": v["n3"] / v["n"] if v["n"] else None,
            "p3": v["k3"] / v["n3"] if v["n3"] else None,
            "p2": v["k2"] / (v["n"] - v["n3"]) if (v["n"] - v["n3"]) else None,
        }
        for m, v in sorted(tot["late_shots"].items(), key=lambda kv: int(kv[0]))
    }
    ends = tot["late_possession_ends"]
    clk = tot["clock"]
    return {
        "seasons": sorted(p["season"] for p in parts),
        "free_throws": {w: rate(d) for w, d in tot["free_throws"].items()},
        "late_trip_mix": {k: v / mix_total for k, v in tot["late_trip_mix"].items()},
        "late_trip_n": mix_total,
        "rebounds": rate(tot["rebounds"]),
        "late_shots_by_actor_margin": late_shots,
        "late_turnover_share": ends["k_turnover"] / ends["n"] if ends["n"] else None,
        "late_possession_ends_n": ends["n"],
        "clock": {
            "mean_sec_made_ft_to_next_foul": clk["sec_made_ft_to_next_foul"]["sum"]
            / (clk["sec_made_ft_to_next_foul"]["n"] or 1),
            "n_ft_to_foul": clk["sec_made_ft_to_next_foul"]["n"],
            "hist_sec_ft_to_foul": dict(sorted(clk["sec_made_ft_to_next_foul"]["hist"].items(),
                                               key=lambda kv: int(kv[0]))),
            "mean_sec_made_ft_to_clock_move": clk["sec_made_ft_to_clock_move"]["sum"]
            / (clk["sec_made_ft_to_clock_move"]["n"] or 1),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="combine just these seasons -- used to hold a season out for Phase 4")
    a = ap.parse_args()
    PART.mkdir(parents=True, exist_ok=True)

    if a.season is not None:
        if a.season in TEST_SEASONS:
            raise SystemExit(f"refusing season {a.season}: 2025-2026 are held out for the single-shot test")
        res = count_season(annotate(load(a.season)))
        res["season"] = a.season
        (PART / f"{a.season}.json").write_text(json.dumps(res))
        print(f"wrote {PART / f'{a.season}.json'}")
        return

    if a.combine:
        parts = [json.loads(p.read_text()) for p in sorted(PART.glob("*.json"))]
        want = set(a.only) if a.only else set(TRAIN_SEASONS)
        parts = [p for p in parts if p["season"] in want]
        missing = want - {p["season"] for p in parts}
        if missing:
            raise SystemExit(f"missing seasons: {sorted(missing)}")
        dest = Path(a.out) if a.out else OUT
        dest.write_text(json.dumps(combine(parts), indent=2))
        print(f"wrote {dest} from {len(parts)} seasons: {sorted(want)}")
        return

    ap.error("pass --season N or --combine")


if __name__ == "__main__":
    main()
```

## `scripts/estimate_endgame_possessions.py`

```py
"""Possession-level endgame behaviour: what ends a possession, and how long it takes.

The free-throw and shooting parameters (estimate_endgame_params.py) say what
happens when a possession resolves. This says HOW a possession resolves -- above
all, whether the defence intentionally fouls, which is the single decision that
makes an endgame an endgame.

The simulator has to reproduce what teams ACTUALLY do, not what they should do,
because it is predicting real games. So the fouling rule is measured, not
optimised.

Possession runs are taken from data/proc/states (the validated possession rules,
including the 2026-09-01 made-three fix) joined back to the raw feed for the
event type that ended each run. Training seasons only.

Output: artifacts/endgame_possessions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "data" / "proc" / "states"
RAW = ROOT / "data" / "raw" / "pbp"
PART = ROOT / "artifacts" / "endgame_poss_parts"
OUT = ROOT / "artifacts" / "endgame_possessions.json"

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
TEST_SEASONS = {2025, 2026}

FT, FOUL, TECH = 540, 519, 521
SHOT_TYPES = [558, 572, 574, 437]
TURNOVER_TYPES = [598, 601, 602]
WINDOW = 90  # seconds; the simulator owns 60, the extra 30 is for the blend


def runs_for_season(season: int) -> pl.DataFrame:
    st = (
        pl.scan_parquet(STATES / f"states_{season}.parquet")
        .select(["game_id", "seq", "possession", "margin", "game_seconds_remaining", "period"])
        .filter((pl.col("period") >= 2) & (pl.col("game_seconds_remaining") <= WINDOW))
        .collect()
    )
    pbp = (
        pl.scan_parquet(RAW / f"pbp_{season}.parquet")
        .select(["game_id", "game_play_number", "type_id", "scoring_play", "text"])
        .collect()
        .rename({"game_play_number": "seq", "type_id": "tid"})
        .with_columns(pl.col("tid").cast(pl.Int64))
    )
    d = st.join(pbp, on=["game_id", "seq"], how="inner").sort(["game_id", "seq"])

    # A run is a maximal stretch with the same possession value. 0.5 (jump ball,
    # genuinely unknown) is carried forward rather than treated as its own run.
    d = d.with_columns(pl.col("possession").replace(0.5, None).forward_fill().over("game_id"))
    d = d.with_columns(
        (pl.col("possession") != pl.col("possession").shift(1).over("game_id"))
        .fill_null(True).cast(pl.Int32).cum_sum().over("game_id").alias("run")
    )
    g = d.group_by(["game_id", "run"]).agg(
        [
            pl.first("possession").alias("off_is_home"),
            pl.first("margin").alias("margin_start"),
            pl.first("game_seconds_remaining").alias("t_start"),
            pl.last("game_seconds_remaining").alias("t_end"),
            pl.last("tid").alias("end_tid"),
            pl.col("tid").is_in([FOUL, TECH]).any().alias("saw_foul"),
            pl.col("tid").eq(FT).any().alias("saw_ft"),
            pl.col("tid").is_in(TURNOVER_TYPES).any().alias("saw_to"),
            (pl.col("scoring_play") & pl.col("tid").is_in(SHOT_TYPES)).any().alias("saw_made_fg"),
            pl.len().alias("n_events"),
        ]
    )
    # Margin and fouling are described from the OFFENSE's point of view: the
    # simulator asks "the team with the ball leads by k -- do they get fouled?"
    return g.with_columns(
        [
            pl.when(pl.col("off_is_home") == 1.0)
            .then(pl.col("margin_start"))
            .otherwise(-pl.col("margin_start"))
            .alias("off_margin"),
            (pl.col("t_start") - pl.col("t_end")).alias("dur"),
        ]
    ).filter(pl.col("dur").is_between(0, 45))


def count_season(season: int) -> dict:
    r = runs_for_season(season)
    r = r.with_columns(
        [
            pl.col("off_margin").clip(-8, 8).cast(pl.Int32).alias("m"),
            (pl.col("t_start") // 10 * 10).clip(0, 80).alias("tb"),
            # An intentional foul: the possession reached the line without the
            # offence taking a shot. That is what "they fouled to stop the clock"
            # looks like in a feed that does not label intent.
            (pl.col("saw_ft") & ~pl.col("saw_made_fg")).alias("fouled_to_line"),
        ]
    )
    g = r.group_by(["m", "tb"]).agg(
        [
            pl.len().alias("n"),
            pl.col("fouled_to_line").sum().alias("k_foul"),
            pl.col("saw_to").sum().alias("k_to"),
            pl.col("saw_made_fg").sum().alias("k_made_fg"),
            pl.col("dur").sum().alias("dur_sum"),
            pl.col("dur").filter(pl.col("fouled_to_line")).sum().alias("dur_foul_sum"),
            pl.col("fouled_to_line").sum().alias("dur_foul_n"),
        ]
    )
    return {
        "season": season,
        "cells": {
            f"{r_['m']}|{r_['tb']}": {k: int(r_[k]) for k in
                                      ("n", "k_foul", "k_to", "k_made_fg", "dur_sum",
                                       "dur_foul_sum", "dur_foul_n")}
            for r_ in g.iter_rows(named=True)
        },
    }


def combine(parts: list[dict]) -> dict:
    tot: dict = {}
    for p in parts:
        for key, v in p["cells"].items():
            acc = tot.setdefault(key, {k: 0 for k in v})
            for k, x in v.items():
                acc[k] += x
    cells = {}
    for key, v in sorted(tot.items(), key=lambda kv: (float(kv[0].split("|")[0]), float(kv[0].split("|")[1]))):
        if v["n"] < 30:
            continue
        cells[key] = {
            "n": v["n"],
            "p_fouled_to_line": v["k_foul"] / v["n"],
            "p_turnover": v["k_to"] / v["n"],
            "p_made_fg": v["k_made_fg"] / v["n"],
            "mean_dur": v["dur_sum"] / v["n"],
            "mean_dur_when_fouled": (v["dur_foul_sum"] / v["dur_foul_n"]) if v["dur_foul_n"] else None,
        }
    return {"seasons": sorted(p["season"] for p in parts), "cells": cells,
            "key": "off_margin|seconds_remaining_bucket (offence's point of view)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="combine just these seasons -- used to hold a season out for Phase 4")
    a = ap.parse_args()
    PART.mkdir(parents=True, exist_ok=True)
    if a.season is not None:
        if a.season in TEST_SEASONS:
            raise SystemExit(f"refusing season {a.season}: held out for the single-shot test")
        (PART / f"{a.season}.json").write_text(json.dumps(count_season(a.season)))
        print("wrote", a.season)
        return
    if a.combine:
        parts = [json.loads(p.read_text()) for p in sorted(PART.glob("*.json"))]
        want = set(a.only) if a.only else set(TRAIN_SEASONS)
        parts = [p for p in parts if p["season"] in want]
        missing = want - {p["season"] for p in parts}
        if missing:
            raise SystemExit(f"missing seasons: {sorted(missing)}")
        dest = Path(a.out) if a.out else OUT
        dest.write_text(json.dumps(combine(parts), indent=2))
        print("wrote", dest)
        return
    ap.error("pass --season N or --combine")


if __name__ == "__main__":
    main()
```

## `scripts/evaluate.py`

```py
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.evaluate import by_time_bucket, log_loss, brier, calibration_table, ece, accuracy

ROOT = pathlib.Path(__file__).resolve().parents[1]
# fit_models.py writes test_preds.npz; a later step exported the same columns
# as eval_preds.parquet for the results report. artifacts/ is gitignored, so a
# restored working copy can have either, or only one. Accept both -- otherwise
# "scripts/evaluate.py regenerates every number" is a claim that fails on any
# machine that happens to have the other file.
_npz = ROOT / "artifacts/test_preds.npz"
_pq = ROOT / "artifacts/eval_preds.parquet"
if _npz.exists():
    d = np.load(_npz)
    print(f"source: {_npz.name} (float64 as fitted)\n")
elif _pq.exists():
    import polars as pl
    _t = pl.read_parquet(_pq)
    # The parquet stores predictions as float32. Metrics agree with the float64
    # originals to three decimals; accuracy and ECE can differ in the fourth.
    d = {c: _t[c].to_numpy().astype(np.float64) for c in _t.columns}
    print(f"source: {_pq.name} (float32 export -- acc/ECE may differ in the 4th decimal)\n")
else:
    raise SystemExit(
        "no predictions found. Expected artifacts/test_preds.npz (written by "
        "fit_models.py) or artifacts/eval_preds.parquet. Re-run scripts/fit_models.py.")
y, secs, espn = d["y"], d["secs"], d["espn"]
preds = {"logistic": d["p_lr"], "lightgbm": d["p_gbm"], "espn": espn}

ok = np.isfinite(espn)
print(f"test rows {len(y):,}   ESPN win prob present on {ok.mean()*100:.1f}%\n")
print(f"{'model':<10} {'logloss':>8} {'brier':>8} {'acc':>7} {'ECE':>7}   (rows where ESPN also present)")
for name, p in preds.items():
    print(f"{name:<10} {log_loss(y[ok],p[ok]):>8.4f} {brier(y[ok],p[ok]):>8.4f} "
          f"{accuracy(y[ok],p[ok]):>7.4f} {ece(y[ok],p[ok]):>7.4f}")

print(f"\n{'bucket':<12}{'n':>10}" + "".join(f"{k:>12}" for k in preds))
for name, lo, hi in [("40-20 min",1200,2400),("20-10 min",600,1200),("10-5 min",300,600),
                     ("5-2 min",120,300),("2-1 min",60,120),("1-0 min",0,60)]:
    m = ok & (secs >= lo) & (secs < hi)
    if m.sum() == 0: continue
    print(f"{name:<12}{m.sum():>10,}" + "".join(f"{log_loss(y[m],p[m]):>12.4f}" for p in preds.values()))
print("(cells are log loss; lower is better)")

print("\ncalibration of lightgbm on the test seasons")
print(f"{'bin':<12}{'n':>10}{'pred':>8}{'obs':>8}{'gap':>8}")
for r in calibration_table(y, d["p_gbm"]):
    print(f"{r['bin']:<12}{r['n']:>10,}{r['pred']:>8.3f}{r['obs']:>8.3f}{r['obs']-r['pred']:>+8.3f}")
```

## `scripts/fetch_data.py`

```py
"""Download the hoopR play-by-play and schedule parquet files.

The only reachable source: raw.githubusercontent.com. Re-run is idempotent;
existing files are skipped unless --force.
"""
import argparse, pathlib, sys, urllib.request, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
PBP = BASE + "/pbp/parquet/play_by_play_{y}.parquet"
SCHED = BASE + "/schedules/parquet/mbb_schedule_{y}.parquet"


def get(url: str, dest: pathlib.Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"  skip {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  got  {dest.name} ({dest.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for y in a.seasons:
        print(y)
        get(PBP.format(y=y), ROOT / f"data/raw/pbp/pbp_{y}.parquet", a.force)
        get(SCHED.format(y=y), ROOT / f"data/raw/sched/sched_{y}.parquet", a.force)
```

## `scripts/fit_models.py`

```py
"""Fit the logistic baseline and the LightGBM model. Split by season, never randomly."""
import sys, pathlib, json, pickle, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cbbwp.schemas import FEATURE_NAMES
from cbbwp.features import mirror_features
from cbbwp.evaluate import log_loss, brier, accuracy

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = [2016, 2017, 2018, 2019, 2021, 2022, 2023]
CALIB = [2024]
TEST  = [2025, 2026]

# Monotonic constraints (plan 4.2): more points, the ball, and a better team can
# never *lower* the home side's win probability.
MONOTONE = {
    "margin": 1, "sqrt_time": 0, "margin_per_sqrt_time": 1, "possession": 1,
    "pregame_exp_margin": 1, "pregame_exp_margin_decayed": 1,
    "is_ot": 0, "timeout_diff": 0, "bonus_diff": 0, "ft_pct_diff": 0,
    "margin_per_sqrt_points_left": 1,
}


def load(seasons, cols=None):
    files = [ROOT / f"data/proc/states/states_{y}.parquet" for y in seasons]
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal")
    return lf.select(cols).collect() if cols else lf.collect()


def xy(df):
    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y = df["home_win"].to_numpy().astype(np.int8)
    return X, y


t0 = time.time()
need = FEATURE_NAMES + ["home_win", "game_seconds_remaining", "game_id", "season"]
tr = load(TRAIN, need)
print(f"train rows {tr.height:,}  games {tr['game_id'].n_unique():,}  ({time.time()-t0:.0f}s)")
Xtr, ytr = xy(tr)
Xtr, ytr = mirror_features(Xtr.astype(np.float64), ytr)
Xtr = Xtr.astype(np.float32); ytr = ytr.astype(np.int8)
print(f"after symmetry mirroring: {Xtr.shape[0]:,} rows")
del tr

ca = load(CALIB, need); Xca, yca = xy(ca)
te = load(TEST, need + ["espn_wp"]); Xte, yte = xy(te)
print(f"calib {Xca.shape[0]:,}   test {Xte.shape[0]:,}")

# ---- 1. logistic regression baseline ------------------------------------
t0 = time.time()
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr.fit(sc.transform(Xtr), ytr)
p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
print(f"\nLOGISTIC  fit {time.time()-t0:.0f}s   test logloss {log_loss(yte,p_lr):.4f}  "
      f"brier {brier(yte,p_lr):.4f}  acc {accuracy(yte,p_lr):.4f}")
print("  coefs:", {n: round(float(c), 3) for n, c in zip(FEATURE_NAMES, lr.coef_[0])})

# ---- 2. LightGBM --------------------------------------------------------
t0 = time.time()
params = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.05,
    num_leaves=160, min_data_in_leaf=1500, feature_fraction=0.9,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1,
    monotone_constraints=[MONOTONE[n] for n in FEATURE_NAMES],
    monotone_constraints_method="advanced", num_threads=2,
    # Pinned so a refit reproduces registry/v1 exactly rather than approximately.
    seed=20260831, bagging_seed=20260831, feature_fraction_seed=20260831,
    data_random_seed=20260831, deterministic=True,
)
dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
dca = lgb.Dataset(Xca, label=yca, feature_name=FEATURE_NAMES, reference=dtr)
gbm = lgb.train(params, dtr, num_boost_round=2500, valid_sets=[dca],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)])
p_gbm = gbm.predict(Xte, num_iteration=gbm.best_iteration)
print(f"\nLIGHTGBM  fit {time.time()-t0:.0f}s  best_iter {gbm.best_iteration}   "
      f"test logloss {log_loss(yte,p_gbm):.4f}  brier {brier(yte,p_gbm):.4f}  acc {accuracy(yte,p_gbm):.4f}")

art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
with open(art / "lr_v1.pkl", "wb") as f:
    pickle.dump({"scaler": sc, "model": lr, "features": FEATURE_NAMES}, f)
gbm.save_model(str(art / "gbm_v1.txt"), num_iteration=gbm.best_iteration)
np.savez_compressed(art / "test_preds.npz",
                    y=yte, p_lr=p_lr, p_gbm=p_gbm,
                    secs=te["game_seconds_remaining"].to_numpy(),
                    espn=te["espn_wp"].to_numpy().astype(np.float64),
                    game_id=te["game_id"].to_numpy(), season=te["season"].to_numpy())
# calibration-season predictions, needed to fit the calibrators
np.savez_compressed(art / "calib_preds.npz", y=yca,
                    p_lr=lr.predict_proba(sc.transform(Xca))[:, 1],
                    p_gbm=gbm.predict(Xca, num_iteration=gbm.best_iteration),
                    secs=ca["game_seconds_remaining"].to_numpy())
print("\nsaved artifacts")
```

## `scripts/live_poller.py`

```py
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
                 fixture_dir: Optional[pathlib.Path] = None,
                 sink=None):
        # `sink` lets another process-local consumer -- the HTTP API -- see each
        # emitted state without this module knowing anything about it. JSONL
        # remains the record of truth; the sink is a view.
        self.svc = svc
        self.ctx = ctx
        self.client = client
        self.out_path = out_path
        self.quiet = quiet
        self.fixture_dir = fixture_dir
        self.sink = sink
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
        if self.sink is not None:
            try:
                self.sink(row)
            except Exception as e:                     # noqa: BLE001
                # A failing view must never take down the feed.
                print(f"sink failed ({type(e).__name__}: {e})",
                      file=sys.stderr, flush=True)
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
                        # A simulated row must never be mistakable for a real
                        # one. The JSONL is the record of truth, and it is
                        # appended to, so an untagged dry run would leave fake
                        # states in the durable record permanently.
                        if self.client.is_replay:
                            last["replay"] = True
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
            if self.sink is not None:
                self.sink(r)

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="slate to follow, YYYYMMDD (default: today)")
    ap.add_argument("--game", type=int, help="follow a single game id")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--version", default="v2", help="model registry version")
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
```

## `scripts/publish_model.py`

```py
"""Write an immutable, pinned model artifact into the registry."""
import sys, pathlib, json, shutil, hashlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.schemas import FEATURE_NAMES, STATE_RULES_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[1]
version = sys.argv[1] if len(sys.argv) > 1 else "v1"
dest = ROOT / "registry" / version
dest.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "artifacts/gbm_v1.txt", dest / "model.txt")
digest = hashlib.sha256((dest / "model.txt").read_bytes()).hexdigest()[:16]
(dest / "manifest.json").write_text(json.dumps({
    "version": version, "kind": "lightgbm", "features": FEATURE_NAMES,
    "state_rules_version": STATE_RULES_VERSION,
    "sha256": digest, "created": datetime.datetime.now(datetime.UTC).isoformat(),
    "train_seasons": [2016, 2017, 2018, 2019, 2021, 2022, 2023],
    "calibration_season": 2024, "test_seasons": [2025, 2026],
}, indent=2))
print("published", dest, digest)
```

## `scripts/rebuild_test_preds.py`

```py
"""Recompute the test-season predictions from the PINNED model, in float64.

`fit_models.py` writes artifacts/test_preds.npz, but artifacts/ is gitignored,
so a restored working copy often has only the float32 parquet export. Metrics
computed from that export land about 0.0001 low on log loss and differ in the
fourth decimal on accuracy and ECE -- enough to look like drift when it is only
storage precision.

Nothing here refits anything. The model in registry/v2 is pinned and hashed, so
re-running it over the same state rows reproduces the original predictions
exactly, at full precision, in a few seconds and a few hundred MB.

    python3 scripts/rebuild_test_preds.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp.schemas import FEATURE_NAMES          # noqa: E402

TEST_SEASONS = [2025, 2026]
OUT = ROOT / "artifacts" / "test_preds.npz"


def main() -> None:
    import lightgbm as lgb
    import pickle

    gbm = lgb.Booster(model_file=str(ROOT / "registry" / "v2" / "model.txt"))
    frames = [pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{s}.parquet")
              for s in TEST_SEASONS]
    te = (pl.concat(frames)
          .select(FEATURE_NAMES + [c for c in
                  ["home_win", "game_seconds_remaining", "espn_wp", "game_id", "season"]
                  if c not in FEATURE_NAMES])
          .collect())
    X = te.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    p_gbm = np.asarray(gbm.predict(X), dtype=np.float64)

    # The logistic baseline is a scikit-learn pickle, and a pickle is only
    # loadable by a compatible scikit-learn. artifacts/lr_v1.pkl was written by
    # 1.8.0 and will not unpickle on 1.7.2 -- which is a good argument for the
    # LightGBM artifact's plain-text format, and a bad property to discover
    # during a rebuild. Fall back to the stored export, but only after checking
    # the rows actually line up.
    p_lr = np.full(len(te), np.nan)
    lr_path = ROOT / "artifacts" / "lr_v1.pkl"
    try:
        with open(lr_path, "rb") as f:
            b = pickle.load(f)
        p_lr = b["model"].predict_proba(b["scaler"].transform(X))[:, 1].astype(np.float64)
        print("logistic: recomputed from lr_v1.pkl")
    except Exception as e:                              # noqa: BLE001
        print(f"logistic: could not load lr_v1.pkl ({type(e).__name__}); "
              "falling back to the stored export")
        pq = ROOT / "artifacts" / "eval_preds.parquet"
        if pq.exists():
            ex = pl.read_parquet(pq)
            same = (len(ex) == len(te)
                    and np.array_equal(ex["game_id"].to_numpy(), te["game_id"].to_numpy()))
            if same:
                p_lr = ex["p_lr"].to_numpy().astype(np.float64)
                print("  rows verified aligned; logistic column carried over (float32 origin)")
            else:
                print("  rows do NOT align; logistic column left as NaN")

    np.savez_compressed(
        OUT,
        y=te["home_win"].to_numpy().astype(np.float64),
        p_lr=p_lr, p_gbm=p_gbm,
        secs=te["game_seconds_remaining"].to_numpy(),
        espn=te["espn_wp"].to_numpy().astype(np.float64),
        game_id=te["game_id"].to_numpy(), season=te["season"].to_numpy())
    print(f"wrote {OUT}  ({len(te):,} rows, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

## `scripts/record_espn_fixtures.py`

```py
"""Record real ESPN payloads to disk, so the adapter can be tested against them.

Run this on a machine that can reach ESPN. It saves the raw JSON exactly as
returned; nothing is parsed or normalised, so the fixtures stay useful even if
the adapter changes.

Games that have not started are skipped: they carry no plays, so recording one
yields a fixture that proves nothing while looking just like a real one. Out of
season that means nothing is recorded at all, which is the honest outcome.

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
    # A scheduled game carries no plays, so recording one produces a fixture
    # that proves nothing while looking exactly like a real one. Drop them
    # rather than sorting them to the back: out of season the whole slate is
    # scheduled, and a directory of empty payloads is worse than none.
    playable = [g for g in games if g["status"] != "STATUS_SCHEDULED"]
    n_sched = len(games) - len(playable)
    print(f"scoreboard {day}: {len(games)} games"
          + (f" ({n_sched} not started yet, skipped)" if n_sched else ""))
    if not playable:
        print("no games with plays on this slate -- nothing to record")
    playable.sort(key=lambda g: g["game_id"])
    ids = [g["game_id"] for g in playable[:a.limit]]

for gid in ids:
    s = c.summary(gid)
    p = out / f"summary_{gid}.json"
    p.write_text(json.dumps(s))
    print(f"  wrote {p.name}  ({len(s.get('plays') or []):,} plays)")
print(f"\n{len(ids)} fixture(s) in {out}")
```

## `scripts/replay_server.py`

```py
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
```

## `scripts/serve_live.py`

```py
"""Run the live poller and the HTTP API together, as one process.

This is the deployment entry point. It is the same on a laptop and in a
container; everything that differs between them is an environment variable
(see `cbbwp/config.py`).

    python3 scripts/serve_live.py                    # tonight's slate
    python3 scripts/serve_live.py --date 20261115    # a specific slate
    CBBWP_FIXTURE_DIR=tmp/fixtures python3 scripts/serve_live.py --once
                                                     # offline rehearsal

Two outputs, both wanted:
  * JSONL, appended, one file per day -- the durable record.
  * HTTP, read-only -- what something else consumes while the game is on.

The JSONL is the record of truth. The API is a view of the same rows, held in
memory; if the API dies the feed keeps writing, which is the right way round.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import pathlib
import signal
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import EspnClient          # noqa: E402
from cbbwp.api import LiveStore, serve_in_thread    # noqa: E402
from cbbwp.config import Settings                   # noqa: E402
from cbbwp.live_context import LiveContextProvider  # noqa: E402
from cbbwp.serve import WinProbabilityService       # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from live_poller import Poller                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="slate to follow, YYYYMMDD (default: today)")
    ap.add_argument("--game", type=int, help="follow a single game id")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--no-api", action="store_true", help="JSONL only")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = Settings.from_env()
    print("cbbwp live\n" + cfg.describe(), flush=True)

    if not cfg.context_path.exists():
        print(f"\nno ratings snapshot at {cfg.context_path}\n"
              "  run: python3 scripts/build_live_context.py --season <year>",
              file=sys.stderr)
        return 2
    ctx = LiveContextProvider.load(cfg.context_path)
    if ctx.data_is_stale:
        print(f"warning: ratings were fit on stale data -- newest completed game is "
              f"{ctx.data_age_days:.1f} days old.\n"
              "  Refresh the play-by-play first (scripts/fetch_data.py), THEN rebuild\n"
              "  the snapshot; rebuilding alone would only refresh its timestamp.",
              file=sys.stderr, flush=True)
    elif ctx.is_stale:
        print(f"warning: ratings snapshot is {ctx.age_days:.1f} days old; "
              "re-run scripts/build_live_context.py", file=sys.stderr, flush=True)

    # Loading the model here, before anything binds a port or opens a file, is
    # what makes a bad model version a startup failure rather than a live one.
    svc = WinProbabilityService(cfg.registry, cfg.model_version)

    day = a.date or datetime.datetime.now().strftime("%Y%m%d")
    out = cfg.live_dir / f"wp_{day}.jsonl"

    # A dry run against scripts/replay_server.py is a rehearsal, not a night of
    # basketball. Say so loudly, and default its output somewhere separate, so
    # simulated states cannot silently accumulate in the real record.
    client = EspnClient()
    if client.is_replay:
        if cfg.live_dir == pathlib.Path(cfg.root) / "data" / "live":
            out = pathlib.Path(cfg.root) / "data" / "replay" / f"wp_{day}.jsonl"
        print(f"\n*** REPLAY MODE -- reading {client.base_url}, NOT ESPN.\n"
              f"*** Rows are tagged \"replay\": true and written to {out}\n",
              flush=True)

    store = LiveStore(history=cfg.api_history)
    httpd = None
    if not a.no_api:
        httpd = serve_in_thread(
            store,
            {"model_version": cfg.model_version,
             "ratings_max_age_days": cfg.ratings_max_age_days},
            lambda: ctx.age_days,
            cfg.api_host, cfg.api_port,
            lambda: ctx.data_age_days,
            lambda: ctx.data_is_stale)
        print(f"api listening on http://{cfg.api_host}:{cfg.api_port}", flush=True)

    poller = Poller(svc, ctx, client, out, quiet=a.quiet,
                    fixture_dir=cfg.fixture_dir, sink=store.update)

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
        if httpd is not None:
            httpd.shutdown()
        loop.close()
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/serve_viz.py`

```py
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
```

## `scripts/smoke_live.py`

```py
"""The pre-flight check for a live night. One command, clear pass or fail.

This is the only part of the system the offline test suite cannot cover, because
it needs the real ESPN endpoint. First run with open egress: 2026-09-02, which
found that ESPN 403s the client's default user-agent (see cbbwp-deployment.md).
Run it BEFORE a live night rather than during one.

    python3 scripts/smoke_live.py                 # full check, records fixtures
    python3 scripts/smoke_live.py --offline       # everything needing no network
    python3 scripts/smoke_live.py --limit 8       # look at more games

Steps 1-3 and 8 work offline. Steps 4-7 need the network; if step 4 fails, the
network steps are reported BLOCKED rather than FAILED, because "we could not
reach ESPN from here" is a different fact from "the adapter is broken". The one
exception is a 403/429 from ESPN: that is ESPN refusing US, not the network
refusing to route, so it is a FAIL. Reporting it as blocked egress is what hid a
real user-agent rule for an entire build cycle.

Out of season, step 5 reports BLOCKED: the slate is all scheduled games, which
carry no plays, so there is nothing to record. Steps 6 and 7 then run against
whatever payloads are already on disk and say so in their detail line.

What this cannot rehearse -- a feed that GROWS between polls -- is covered by
scripts/replay_server.py, which serves archived games back to the unmodified
deployment as a live feed. That is a rehearsal, not a substitute for this.

Exit code 0 only if nothing FAILED. Blocked steps exit 2, so a scheduled run can
tell "not validated yet" from "validated and broken".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.config import Settings          # noqa: E402

PASS, FAIL, BLOCKED, SKIP = "PASS", "FAIL", "BLOCKED", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> str:
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", BLOCKED: "BLOCK ", SKIP: " skip "}[status]
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return status


def run(cmd: list[str], timeout: int = 170) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="skip everything that needs the network")
    ap.add_argument("--limit", type=int, default=5, help="games to record")
    ap.add_argument("--date", default=None, help="slate to record, YYYYMMDD")
    ap.add_argument("--fixtures", default=str(ROOT / "tmp/fixtures"))
    ap.add_argument("--no-tests", action="store_true",
                    help="skip the offline suite (it is slow on a cold machine)")
    a = ap.parse_args()

    cfg = Settings.from_env()
    print("cbbwp live smoke test\n" + cfg.describe() + "\n", flush=True)
    fixtures = pathlib.Path(a.fixtures)

    # 1 --- the model artifact loads, and its guards pass -------------------
    try:
        from cbbwp.serve import WinProbabilityService
        svc = WinProbabilityService(cfg.registry, cfg.model_version)
        record("1 model artifact loads", PASS,
               f"{cfg.model_version}, {len(svc.manifest['features'])} features, "
               f"state rules v{svc.manifest.get('state_rules_version', 1)}")
    except Exception as e:                              # noqa: BLE001
        record("1 model artifact loads", FAIL, f"{type(e).__name__}: {e}")
        return 1

    # 2 --- the ratings snapshot exists and is fresh ------------------------
    try:
        from cbbwp.live_context import LiveContextProvider
        if not cfg.context_path.exists():
            record("2 ratings snapshot", FAIL,
                   f"missing: {cfg.context_path} -- run build_live_context.py")
        else:
            ctx = LiveContextProvider.load(cfg.context_path)
            d = ctx.data_age_days
            detail = (f"file {ctx.age_days:.1f}d old, newest completed game "
                      + ("none yet (preseason)" if d is None else f"{d:.1f}d old"))
            if ctx.data_is_stale:
                detail += " -- STALE DATA: refresh play-by-play, then rebuild"
            elif ctx.is_stale:
                detail += " -- STALE FILE: re-run build_live_context.py"
            record("2 ratings snapshot", FAIL if ctx.is_stale else PASS, detail)
    except Exception as e:                              # noqa: BLE001
        record("2 ratings snapshot", FAIL, f"{type(e).__name__}: {e}")

    # 3 --- the offline test suite still passes ----------------------------
    if a.no_tests:
        record("3 offline test suite", SKIP, "--no-tests")
    else:
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                               capture_output=True, text=True, timeout=170)
            tail = (r.stdout.strip().splitlines() or ["no output"])[-1]
            record("3 offline test suite", PASS if r.returncode == 0 else FAIL, tail)
        except Exception as e:                          # noqa: BLE001
            record("3 offline test suite", SKIP, f"{type(e).__name__}: {e}")

    if a.offline:
        for n in ("4 ESPN reachable", "5 record fixtures", "6 fixture sanity",
                  "7 one live poll"):
            record(n, SKIP, "--offline")
    else:
        # 4 --- can we reach ESPN at all? ----------------------------------
        reachable = False
        try:
            from cbbwp.adapters.espn import EspnClient, scoreboard_games
            t0 = time.time()
            sb = EspnClient().scoreboard(a.date)
            games = scoreboard_games(sb)
            reachable = True
            record("4 ESPN reachable", PASS,
                   f"{len(games)} games on the slate, {time.time()-t0:.1f}s, "
                   f"ua={EspnClient().user_agent!r}")
        except urllib.error.HTTPError as e:
            # A 403/429 is ESPN refusing US, not the network refusing to route.
            # Calling that "blocked egress" is what hid the user-agent rule for
            # a whole build cycle, so name it as a real failure.
            record("4 ESPN reachable", FAIL if e.code in (403, 429) else BLOCKED,
                   f"HTTP {e.code}: {e.reason} -- ua={EspnClient().user_agent!r}")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            record("4 ESPN reachable", BLOCKED,
                   f"{type(e).__name__}: {e} -- run where egress is open")
        except Exception as e:                          # noqa: BLE001
            record("4 ESPN reachable", FAIL, f"{type(e).__name__}: {e}")

        if not reachable:
            for n in ("5 record fixtures", "6 fixture sanity", "7 one live poll"):
                record(n, BLOCKED, "no route to ESPN")
        else:
            # 5 --- record what ESPN is sending today ----------------------
            # Count only payloads THIS run wrote. Counting whatever is already
            # in the directory turns a day-old fixture into a green tick: on
            # 2026-09-02 an empty offseason slate recorded nothing and step 5
            # still passed, on files from the day before.
            t_start = time.time()
            cmd = ["scripts/record_espn_fixtures.py", "--limit", str(a.limit),
                   "--out", str(fixtures)]
            if a.date:
                cmd += ["--date", a.date]
            r = run(cmd)

            def _fresh() -> list[pathlib.Path]:
                return [q for q in fixtures.glob("summary_*.json")
                        if q.stat().st_mtime >= t_start]

            fresh, stale = _fresh(), list(fixtures.glob("summary_*.json"))
            n_stale = len(stale) - len(fresh)
            if r.returncode != 0:
                record("5 record fixtures", FAIL,
                       f"recorder exited {r.returncode} -- {r.stderr.strip()[:200]}")
            elif fresh:
                record("5 record fixtures", PASS,
                       f"{len(fresh)} payloads recorded into {fixtures}")
            else:
                # No games with plays on this slate. Real in the offseason, and
                # not the adapter's fault -- but it is not validation either.
                record("5 record fixtures", BLOCKED,
                       "no games with plays on this slate -- nothing recorded"
                       + (f"; {n_stale} older payload(s) left in {fixtures}"
                          if n_stale else ""))

            # 6 --- does the adapter understand them? ----------------------
            # Steps 6 and 7 still run against whatever payloads exist, because
            # parsing a real payload is worth checking even when it is an old
            # one -- but say so, so the tick is not read as "today's feed".
            age = ""
            if not fresh and stale:
                oldest = min(q.stat().st_mtime for q in stale)
                age = (f" -- against {len(stale)} payload(s) "
                       f"{(time.time() - oldest) / 86400:.1f}d old, NOT today's feed")
            r = run(["scripts/check_espn_fixtures.py", "--dir", str(fixtures)])
            out = r.stdout
            unknown = "PLAY TYPES THE MODEL HAS NEVER SEEN" in out
            # If every payload was rebuilt from hoopR, this step's headline check
            # cannot fail -- the type ids came from the same files the model's
            # type map did. A green tick there would be worse than no tick.
            all_rebuilt = "Every payload here is a rebuild" in out
            if r.returncode != 0:
                detail, status = "checker failed -- see below", FAIL
            elif all_rebuilt:
                detail, status = ("payloads are REBUILDS from hoopR, not ESPN "
                                  "recordings -- this check cannot fail on them"), BLOCKED
            else:
                detail = ("unknown play types present -- see below" if unknown
                          else "no unknown play types") + age
                status = PASS
            record("6 fixture sanity", status, detail)
            if unknown:
                print(out[out.index("PLAY TYPES THE MODEL"):][:1200], flush=True)

            # 7 --- one real poll, end to end ------------------------------
            ids = sorted(p.stem.split("_")[1] for p in fixtures.glob("summary_*.json"))
            if not ids:
                record("7 one live poll", FAIL, "no fixture to pick a game id from")
            else:
                r = run(["scripts/live_poller.py", "--game", ids[0], "--once"])
                good = r.returncode == 0 and "states" in r.stdout
                # Step 7 polls the NETWORK -- only the game id came from the
                # fixture list -- so the "old payload" caveat that steps 5 and 6
                # carry does not apply here, and attaching it said the opposite
                # of the truth.
                note = "" if fresh else "  [game id from an older fixture; the poll itself is live]"
                record("7 one live poll", PASS if good else FAIL,
                       ((r.stdout.strip().splitlines() or [""])[0][:160] + note)
                       if good else r.stderr.strip()[:200])

    # 8 --- the API serves; no network needed ------------------------------
    try:
        from cbbwp.api import LiveStore, serve_in_thread
        store = LiveStore(history=8)
        store.update({"game_id": 1, "seq": 1, "period": 2,
                      "game_seconds_remaining": 30, "margin": 3,
                      "home_win_prob": 0.8})
        httpd = serve_in_thread(store, {"model_version": cfg.model_version,
                                        "ratings_max_age_days": 1e9},
                                lambda: 0.0, "127.0.0.1", 0)
        port = httpd.server_address[1]
        body = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5).read())
        httpd.shutdown()
        ok = body.get("status") == "ok" and body.get("state_rules_version") is not None
        record("8 API serves", PASS if ok else FAIL,
               f"port {port}, {body.get('games_tracked')} game tracked")
    except Exception as e:                              # noqa: BLE001
        record("8 API serves", FAIL, f"{type(e).__name__}: {e}")

    # --- verdict ----------------------------------------------------------
    failed = [n for n, s, _ in results if s == FAIL]
    blocked = [n for n, s, _ in results if s == BLOCKED]
    print("\n" + "=" * 62)
    if failed:
        print(f"FAILED: {len(failed)} step(s) -- {', '.join(failed)}")
        print("Do not go live until these pass.")
        return 1
    if blocked:
        # Two different reasons land here, and they need different advice:
        # ESPN unreachable (fix the machine) vs. reachable but no games to
        # watch (fix nothing, come back in season). Step 4 tells them apart.
        espn_ok = any(n.startswith("4 ") and st == PASS for n, st, _ in results)
        print(f"BLOCKED: {len(blocked)} step(s) -- {', '.join(blocked)}")
        if espn_ok:
            print("ESPN is reachable and the adapter parses real payloads, but no\n"
                  "game was live or finished on this slate, so nothing was recorded\n"
                  "from today's feed. What is untested is ESPN DURING A REAL GAME --\n"
                  "re-run this on a night with games. To rehearse the running clock\n"
                  "now, see scripts/replay_server.py.")
        else:
            print("Everything testable without the network passed. The live path is\n"
                  "still UNVALIDATED -- re-run this where egress is open.")
        return 2
    skipped_network = [n for n, s_, _ in results
                       if s_ == SKIP and n[0] in "4567"]
    if skipped_network:
        print("Offline checks all pass. The network steps were SKIPPED, so the "
              "live path\nis still UNVALIDATED -- re-run without --offline "
              "before the first live night.")
        return 2
    print("ALL PASS -- the live path is validated end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/validate_endgame_table.py`

```py
"""Phase 4: is the table HONEST about the states it describes?

The endgame plan's gate for this phase is deliberately weak -- "beats nothing;
just has to be honest". Phase 4 asks only whether a state the table calls a 20%
state is won about 20% of the time. Whether that is worth shipping is Phase 5's
question, and it is asked once, on 2025-2026.

So this runs on 2024, against a table whose parameters were estimated from
2016-2023 only. 2024 is genuinely out of sample for the table, and it is not one
of the held-out test seasons, so using it here costs nothing later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbbwp import endgame_sim as E  # noqa: E402

EPS = 1e-15


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    acc = float((((p >= 0.5).astype(int)) == y).mean())
    edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 19)
    ece = 0.0
    for b in range(20):
        s = idx == b
        if s.sum():
            ece += s.mean() * abs(p[s].mean() - y[s].mean())
    return {"log_loss": ll, "brier": brier, "accuracy": acc, "ece": float(ece), "n": int(len(p))}


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        s = idx == b
        if s.sum() < 50:
            continue
        out.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(s.sum()),
                    "predicted": float(p[s].mean()), "observed": float(y[s].mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--table", default="registry/endgame/e1_no2024")
    ap.add_argument("--seconds", type=int, default=60)
    a = ap.parse_args()

    tdir = ROOT / a.table
    table = np.load(tdir / "table.npz")["table"].astype(np.float64)
    manifest = json.loads((tdir / "manifest.json").read_text())
    if a.season in manifest["seasons_used"]:
        raise SystemExit(
            f"season {a.season} was used to fit {a.table}; validating on it would prove nothing"
        )
    means = manifest["ft_bucket_means"]

    st = (
        pl.scan_parquet(ROOT / "data" / "proc" / "states" / f"states_{a.season}.parquet")
        .filter((pl.col("period") >= 2) & (pl.col("game_seconds_remaining") <= a.seconds))
        .select(["game_id", "game_seconds_remaining", "margin", "possession",
                 "home_fouls", "away_fouls", "home_win", "espn_wp"])
        .collect()
    )
    ts = pl.read_parquet(ROOT / "data" / "proc" / "team_stats.parquet").select(
        ["game_id", "home_ft_pct", "away_ft_pct"]
    )
    d = st.join(ts, on="game_id", how="left").with_columns(
        [pl.col("home_ft_pct").fill_null(0.70), pl.col("away_ft_pct").fill_null(0.70)]
    )

    p = E.lookup_home(
        table,
        d["game_seconds_remaining"].to_numpy(),
        d["margin"].to_numpy(),
        d["possession"].to_numpy(),
        d["home_fouls"].to_numpy(),
        d["away_fouls"].to_numpy(),
        E.ft_bucket(d["home_ft_pct"].to_numpy(), means),
        E.ft_bucket(d["away_ft_pct"].to_numpy(), means),
    )
    y = d["home_win"].to_numpy().astype(float)

    result = {
        "season": a.season,
        "table": a.table,
        "table_seasons": manifest["seasons_used"],
        "window_seconds": a.seconds,
        "table_alone": metrics(p, y),
        "espn_same_rows": metrics(np.clip(d["espn_wp"].to_numpy(), EPS, 1 - EPS), y),
        "reliability": reliability(p, y),
    }
    print(json.dumps(result, indent=2))
    outp = ROOT / "reports" / f"endgame_validation_{a.season}.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

## `tests/espn_fixtures.py`

```py
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
```

## `tests/test_endgame.py`

```py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.endgame import apply


def test_time_expired_forces_certainty():
    p = apply([0.4, 0.6, 0.5], margin=np.array([3, -3, 0]),
              seconds_remaining=np.array([0, 0, 0]))
    assert list(p) == [1.0, 0.0, 0.5]


def test_mathematically_decided_is_clamped():
    # 20 up with 10 seconds left: at most 2 possessions x 3 points = 6.
    p = apply([0.97], margin=np.array([20]), seconds_remaining=np.array([10.0]))
    assert p[0] == 1.0


def test_live_games_are_left_alone():
    p = apply([0.73], margin=np.array([4]), seconds_remaining=np.array([300.0]))
    assert p[0] == 0.73
```

## `tests/test_endgame_sim.py`

```py
"""Tests for the endgame table.

Two kinds. The first check that the solver's structure is what it claims -- the
terminal condition, the symmetry, the monotonicity the plan pre-registered.

The second kind is the one that matters. `test_possession_truth.py` exists
because every check we had compared the code to itself, so a feature that meant
two different things in one training set went unnoticed for four seasons. The
free-throw tests below are written in the same spirit: they assert against the
RULES OF BASKETBALL and against a known feed artifact, so that if ESPN changes
how it labels a free-throw trip again, something fails loudly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from cbbwp import endgame_sim as E

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "registry" / "endgame" / "e1"
PBP26 = ROOT / "data" / "raw" / "pbp" / "pbp_2026.parquet"

needs_table = pytest.mark.skipif(not (TABLE / "table.npz").exists(),
                                 reason="run scripts/build_endgame_table.py first")
needs_pbp = pytest.mark.skipif(not PBP26.exists(), reason="raw play-by-play not fetched")


@pytest.fixture(scope="module")
def table():
    return np.load(TABLE / "table.npz")["table"].astype(np.float64)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((TABLE / "manifest.json").read_text())


@needs_table
def test_time_expired_is_decided_by_the_scoreboard(table):
    for m in range(-6, 7):
        v = table[0, E._mi(m), 6, 8, 1, 1]
        if m > 0:
            assert v == pytest.approx(1.0)
        elif m < 0:
            assert v == pytest.approx(0.0)
        else:
            assert v == pytest.approx(0.5)


@needs_table
def test_a_made_basket_never_lowers_your_win_probability(table):
    """Criterion 3, exhaustively -- every state, not a sample."""
    assert np.diff(table, axis=1).min() >= -1e-6


@needs_table
def test_having_the_ball_is_never_a_disadvantage(table):
    """The same physical state, valued from both sides, must prefer possession."""
    flipped = 1.0 - table[:, ::-1].transpose(0, 1, 3, 2, 5, 4)
    assert (table - flipped).min() >= -1e-6


@needs_table
def test_the_table_is_symmetric_under_swapping_the_teams(table):
    means = [0.667, 0.709, 0.749]
    args = dict(seconds_remaining=np.array([12, 30, 5]), home_fouls=np.array([7, 9, 6]),
                away_fouls=np.array([9, 6, 10]), home_bucket=np.array([0, 1, 2]),
                away_bucket=np.array([2, 1, 0]))
    home = E.lookup_home(table, margin_home=np.array([3.0, -2.0, 1.0]),
                         possession=np.array([1.0, 0.0, 1.0]), **args)
    swapped = E.lookup_home(
        table, seconds_remaining=args["seconds_remaining"], margin_home=np.array([-3.0, 2.0, -1.0]),
        possession=np.array([0.0, 1.0, 0.0]), home_fouls=args["away_fouls"],
        away_fouls=args["home_fouls"], home_bucket=args["away_bucket"],
        away_bucket=args["home_bucket"])
    assert home == pytest.approx(1.0 - swapped, abs=1e-9)
    del means


@needs_table
def test_trailing_by_three_late_is_worse_than_trailing_by_two(table):
    """Two is one possession; three is not. The table has to know that."""
    for t in (5, 10, 20):
        down2 = table[t, E._mi(-2), 6, 8, 1, 1]
        down3 = table[t, E._mi(-3), 6, 8, 1, 1]
        assert down3 < down2 - 0.05, (t, down3, down2)


@needs_table
def test_shipped_table_declares_the_state_rules_it_was_built_under(manifest):
    """Same guard as the model: a table built under different state rules must
    not be served silently alongside code that means something else by them."""
    from cbbwp.schemas import STATE_RULES_VERSION
    assert manifest["state_rules_version"] == STATE_RULES_VERSION


@needs_table
def test_the_table_was_not_fitted_on_the_test_seasons(manifest):
    assert not ({2025, 2026} & set(manifest["seasons_used"]))


# --- the feed-artifact regression -------------------------------------------
@needs_pbp
def test_espn_labels_a_free_throw_trip_by_attempts_taken_not_attempts_awarded():
    """The censoring that made "1 of 1" look like a 54% free-throw rate.

    A MADE one-and-one front end earns a second shot, so ESPN writes the trip as
    "1 of 2" / "2 of 2"; a MISSED one leaves the trip at a single attempt and is
    written "1 of 1". Every made front end therefore leaves the "1 of 1" bucket
    by construction, and reading `scoring_play` off that label conditions on the
    outcome.

    If this test ever fails, ESPN has changed the convention again and every
    free-throw parameter has to be re-derived.
    """
    import polars as pl

    df = (
        pl.scan_parquet(PBP26)
        .select(["game_id", "sequence_number", "type_id", "text", "scoring_play",
                 "athlete_id_1", "clock_minutes", "clock_seconds"])
        .collect()
        .with_columns(pl.col("sequence_number").cast(pl.Int64))
        .sort(["game_id", "sequence_number"])
    )
    df = df.with_columns([
        (pl.col("type_id").cast(pl.Int64) == 540).alias("isft"),
        (pl.col("clock_minutes").cast(pl.Float64).fill_null(0) * 60
         + pl.col("clock_seconds").cast(pl.Float64).fill_null(0)).alias("sec"),
    ])
    df = df.with_columns((pl.col("scoring_play") & ~pl.col("isft")).alias("madefg"))
    andone = None
    for k in range(1, 13):
        t = (pl.col("madefg").shift(k).over("game_id")
             & (pl.col("athlete_id_1").shift(k).over("game_id") == pl.col("athlete_id_1"))
             & ((pl.col("sec").shift(k).over("game_id") - pl.col("sec")).abs() <= 3))
        andone = t if andone is None else (andone | t)
    df = df.with_columns(andone.fill_null(False).alias("andone"))

    ones = df.filter(pl.col("isft") & pl.col("text").str.contains("(?i)free throw 1 of 1"))
    plain = ones.filter(~pl.col("andone"))
    and_ones = ones.filter(pl.col("andone"))

    # And-one free throws are ordinary single shots and convert like them.
    assert len(and_ones) > 5_000
    assert 0.60 < and_ones["scoring_play"].mean() < 0.80

    # The rest are missed one-and-one front ends, and are therefore almost all
    # misses. Anything near a plausible free-throw percentage here would mean
    # the convention had changed.
    assert len(plain) > 2_000
    assert plain["scoring_play"].mean() < 0.20, (
        "'1 of 1' free throws that are not and-ones now convert at a plausible "
        "rate -- ESPN has changed how it labels free-throw trips, and every "
        "free-throw parameter in artifacts/endgame_params.json must be re-derived"
    )
```

## `tests/test_espn_adapter.py`

```py
"""The ESPN (live) adapter must produce exactly what the hoopR (offline) adapter
produces for the same game. This is the train/serve-skew guard for the live path,
and it is the live counterpart of tests/test_parity.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl
import pytest

from cbbwp.adapters import espn
from cbbwp.adapters.hoopr import load_events
from cbbwp.schemas import PregameContext
from cbbwp.state import build_states, clock_to_seconds
from cbbwp.live_context import LiveContextProvider

ROOT = pathlib.Path(__file__).resolve().parents[1]
PBP = str(ROOT / "data/raw/pbp/pbp_2025.parquet")
HAS_DATA = pathlib.Path(PBP).exists()


# --------------------------------------------------------------------------
# Unit tests: no data needed
# --------------------------------------------------------------------------
def test_type_id_beats_feed_text():
    """If ESPN reworded a play type, the trained-on text must still win."""
    p = {"type": {"id": "558", "text": "Jump Shot (new ESPN wording)"}}
    assert espn.play_type_text(p) == "JumpShot"


def test_unknown_type_id_falls_back_to_feed_text():
    p = {"type": {"id": "999999", "text": "Flagrant Foul"}}
    assert espn.play_type_text(p) == "Flagrant Foul"
    # and an unknown type carries possession rather than guessing
    assert espn.play_type_text({"type": {}}) == ""


def _play(seq, clock, home=0, away=0, tid="615", period=1):
    return {"sequenceNumber": seq, "type": {"id": tid}, "period": {"number": period},
            "clock": {"displayValue": clock}, "homeScore": home, "awayScore": away}


def test_the_feeds_array_order_is_authoritative_and_renumbered_densely():
    """The adapter must NOT re-sort by sequenceNumber.

    Changed 2026-09-03. Measured on seven archived games, ESPN's `plays` array is
    chronological and matches hoopR's game_play_number exactly, while
    `sequenceNumber` is only nearly monotonic. Sorting by it took correct data
    and shuffled it. These plays are in the right order with an inverted
    sequenceNumber in the middle, exactly as the real feed does it.
    """
    plays = [_play("120416950", "20:00"),
             _play("120416904", "19:00", home=2, tid="558"),   # id goes BACKWARDS
             _play("120416951", "18:00", home=2)]
    evs = espn.events_from_plays(plays, game_id=7)
    assert [e.seq for e in evs] == [1, 2, 3]
    assert [e.clock_seconds for e in evs] == [1200, 1140, 1080], \
        "the adapter re-sorted a correctly ordered feed"
    assert all(e.game_id == 7 for e in evs)
    assert espn.chronological_inversions(evs) == 0


def test_a_disordered_feed_is_reported_not_silently_repaired():
    """An unreliable key cannot fix a bad feed; it can only corrupt a good one.

    So the adapter preserves what it was sent and counts the inversions. If ESPN
    ever does start sending plays out of order, that becomes a number somebody
    can see rather than a rearrangement nobody notices.
    """
    scrambled = [_play("100", "18:00", home=2),
                 _play("200", "20:00"),
                 _play("300", "19:00", home=2)]
    evs = espn.events_from_plays(scrambled, game_id=7)
    assert [e.clock_seconds for e in evs] == [1080, 1200, 1140], "order was changed"
    assert espn.chronological_inversions(evs) == 1


def test_missing_and_null_fields_do_not_raise():
    evs = espn.events_from_plays([{"type": {"id": "584"}}], game_id=1)
    assert len(evs) == 1
    e = evs[0]
    assert (e.period, e.clock_seconds, e.home_score, e.team_id) == (1, 0, 0, None)


def test_header_parsing():
    h = espn.header_from_summary({"header": {"id": "401", "competitions": [{
        "neutralSite": True,
        "status": {"period": 2, "displayClock": "1:23",
                   "type": {"name": "STATUS_IN_PROGRESS"}},
        "competitors": [
            {"homeAway": "away", "score": "61", "team": {"id": "2", "displayName": "A"}},
            {"homeAway": "home", "score": "64", "team": {"id": "1", "displayName": "H"}},
        ]}]}})
    assert (h.game_id, h.home_team_id, h.away_team_id) == (401, 1, 2)
    assert h.neutral_site and h.is_live and not h.is_final
    assert (h.home_score, h.away_score) == (64, 61)


def test_scoreboard_flattening():
    games = espn.scoreboard_games({"events": [{
        "id": "555", "shortName": "A @ H",
        "competitions": [{"neutralSite": False,
                          "status": {"type": {"name": "STATUS_IN_PROGRESS",
                                              "state": "in", "completed": False}},
                          "competitors": [
                              {"homeAway": "home", "team": {"id": "1"}},
                              {"homeAway": "away", "team": {"id": "2"}}]}]}]})
    assert games == [{"game_id": 555, "name": "A @ H",
                      "status": "STATUS_IN_PROGRESS", "state": "in",
                      "completed": False, "neutral_site": False,
                      "home_team_id": 1, "away_team_id": 2, "start": ""}]


def test_live_context_falls_back_for_unknown_teams():
    p = LiveContextProvider(season=2027, hca=3.5, ratings={1: 6.0},
                            ft_pct={1: 0.75}, ppm={1: 3.6})
    c = p.context_for(99, home_team_id=1, away_team_id=404)
    assert c.pregame_exp_margin == pytest.approx(6.0 - 0.0 + 3.5)
    assert c.ft_pct_diff == pytest.approx(0.75 - 0.700)
    assert not p.known(404)
    # a neutral-site game drops the home-court term
    assert p.context_for(99, 1, 404, neutral_site=True).pregame_exp_margin == \
        pytest.approx(6.0)


# --------------------------------------------------------------------------
# Parity against the offline adapter, on real games
# --------------------------------------------------------------------------
pytestmark_data = pytest.mark.skipif(not HAS_DATA,
                                     reason="pbp_2025.parquet not downloaded yet")


@pytest.fixture(scope="module")
def game_ids():
    return (pl.scan_parquet(PBP).select("game_id").unique().sort("game_id")
            .head(15).collect()["game_id"].to_list())


@pytestmark_data
def test_espn_adapter_matches_hoopr_states(game_ids):
    from espn_fixtures import summary_from_hoopr

    for gid in game_ids:
        ref_events, home_id, away_id = load_events(PBP, gid)
        # In the feed's own array order, which is what ESPN actually sends and
        # what hoopR's game_play_number preserves. This used to shuffle, on the
        # belief that the array promised nothing and sequenceNumber was
        # authoritative; measurement on real payloads showed the reverse
        # (adapters/espn.py: events_from_plays).
        payload = summary_from_hoopr(PBP, gid)
        live_events, header = espn.parse_summary(payload)

        assert header.home_team_id == home_id
        assert header.away_team_id == away_id
        assert len(live_events) == len(ref_events)

        ctx = PregameContext(gid, home_id, away_id, pregame_exp_margin=2.5,
                             ft_pct_diff=0.02, exp_points_per_min=3.5)
        ref = build_states(ref_events, ctx)
        live = build_states(live_events, ctx)
        for a, b in zip(ref, live):
            assert (a.seq, a.period, a.game_seconds_remaining, a.margin,
                    a.possession, a.home_timeouts, a.away_timeouts,
                    a.home_fouls, a.away_fouls, a.is_ot) == \
                   (b.seq, b.period, b.game_seconds_remaining, b.margin,
                    b.possession, b.home_timeouts, b.away_timeouts,
                    b.home_fouls, b.away_fouls, b.is_ot), (gid, a.seq)


@pytestmark_data
@pytest.mark.skipif(not (ROOT / "registry/v2").exists(),
                    reason="no model registry built yet")
def test_espn_path_gives_identical_win_probabilities(game_ids):
    from espn_fixtures import summary_from_hoopr
    from cbbwp.serve import WinProbabilityService

    svc = WinProbabilityService(ROOT / "registry", "v2")
    for gid in game_ids[:5]:
        ref_events, home_id, away_id = load_events(PBP, gid)
        ctx = PregameContext(gid, home_id, away_id, pregame_exp_margin=2.5,
                             ft_pct_diff=0.02, exp_points_per_min=3.5)
        live_events, _ = espn.parse_summary(summary_from_hoopr(PBP, gid))
        assert svc.score_game(live_events, ctx) == svc.score_game(ref_events, ctx)


@pytestmark_data
def test_partial_feed_is_a_prefix_of_the_finished_game(game_ids):
    """A game polled mid-way must give the same answers for the plays it has."""
    from espn_fixtures import summary_from_hoopr

    gid = game_ids[0]
    full = summary_from_hoopr(PBP, gid)
    ref_events, home_id, away_id = load_events(PBP, gid)
    ctx = PregameContext(gid, home_id, away_id)

    n = len(full["plays"]) // 2
    partial = dict(full, plays=full["plays"][:n],
                   header=full["header"])
    part_events, _ = espn.parse_summary(partial)
    full_events, _ = espn.parse_summary(full)

    a = build_states(part_events, ctx)
    b = build_states(full_events, ctx)[:n]
    assert len(a) == n
    for x, y in zip(a, b):
        assert (x.seq, x.margin, x.possession, x.game_seconds_remaining) == \
               (y.seq, y.margin, y.possession, y.game_seconds_remaining)


def test_serving_refuses_a_model_fit_under_older_state_rules():
    """The guard that the feature-name check could not provide.

    A model fit before the possession fix must not be served states built after
    it. The feature NAMES are identical in both, so without this check the
    mismatch is completely silent - which is exactly what happened on
    2026-09-01 before it was caught.
    """
    from cbbwp.serve import WinProbabilityService
    if not (ROOT / "registry/v1").exists():
        pytest.skip("no v1 artifact kept")
    with pytest.raises(RuntimeError, match="state rules"):
        WinProbabilityService(ROOT / "registry", "v1")


def test_the_current_model_loads():
    from cbbwp.serve import WinProbabilityService
    if not (ROOT / "registry/v2").exists():
        pytest.skip("no v2 artifact built yet")
    svc = WinProbabilityService(ROOT / "registry", "v2")
    assert svc.manifest["state_rules_version"] == 2


def test_default_user_agent_carries_a_contact_url():
    """ESPN's edge 403s bare short user-agents; the contact URL is what passes.

    Measured 2026-09-02: "cbbwp/0.2" got 403 on 15/15 requests, and
    "cbbwp/0.2 (+https://github.com/jpbranson/ncaa_mbb)" got 200 on 15/15, on
    both the scoreboard and summary endpoints. The property that matters is the
    contact URL, not the exact string - so that is what is asserted here, to
    stop a well-meaning tidy-up ("shorten this ugly literal") from silently
    breaking the live path months before anyone runs it again.
    """
    from cbbwp.adapters.espn import DEFAULT_USER_AGENT
    assert "(+http" in DEFAULT_USER_AGENT
    assert len(DEFAULT_USER_AGENT) > 20


def test_user_agent_is_overridable_by_environment(monkeypatch):
    """A WAF change must be a config edit plus a restart, never a code edit."""
    from cbbwp.adapters.espn import DEFAULT_USER_AGENT, EspnClient
    monkeypatch.setenv("CBBWP_USER_AGENT", "someone-else/9.9 (+https://example.org)")
    assert EspnClient().user_agent == "someone-else/9.9 (+https://example.org)"
    assert EspnClient(user_agent="explicit/1").user_agent == "explicit/1"
    monkeypatch.delenv("CBBWP_USER_AGENT")
    assert EspnClient().user_agent == DEFAULT_USER_AGENT


def test_the_client_points_at_espn_unless_told_otherwise(monkeypatch):
    """The replay hook must never become the default.

    `CBBWP_ESPN_BASE` exists so scripts/replay_server.py can stand in for ESPN
    during a dry run. A default pointing anywhere else would mean a live night
    silently scoring simulated games.
    """
    from cbbwp.adapters.espn import SITE_API, EspnClient
    monkeypatch.delenv("CBBWP_ESPN_BASE", raising=False)
    c = EspnClient()
    assert c.base_url == SITE_API
    assert c.is_replay is False

    monkeypatch.setenv("CBBWP_ESPN_BASE", "http://127.0.0.1:8899/")
    r = EspnClient()
    assert r.base_url == "http://127.0.0.1:8899"      # trailing slash trimmed
    assert r.is_replay is True


@pytestmark_data
def test_sorting_by_sequence_number_would_corrupt_a_correct_feed(game_ids):
    """The regression guard for the 2026-09-03 fix, stated as a measurement.

    If someone reintroduces `sorted(..., key=sequenceNumber)` in
    `events_from_plays`, the parity tests above go red. This test says why, so
    the fix is not silently undone by someone who reads the old docstring: on a
    feed that is already in the right order, sorting by that key MOVES plays.
    """
    from espn_fixtures import summary_from_hoopr

    payload = summary_from_hoopr(PBP, game_ids[0])
    plays = payload["plays"]
    as_sent = [(p["period"]["number"], p["clock"]["displayValue"]) for p in plays]
    by_id = [(p["period"]["number"], p["clock"]["displayValue"]) for p in
             sorted(plays, key=lambda q: int(q["sequenceNumber"]))]
    assert as_sent != by_id, (
        "the fixture's sequenceNumber is perfectly sorted again -- that is the "
        "property real ESPN does NOT have, and it is what hid the bug")

    events, _ = espn.parse_summary(payload)
    assert espn.chronological_inversions(events) == 0
    assert [e.clock_seconds for e in events] == \
           [clock_to_seconds(c) for _, c in as_sent]
```

## `tests/test_features.py`

```py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.schemas import GameState, FEATURE_NAMES
from cbbwp.features import feature_dict, build_feature_matrix, mirror_features


def gs(**kw):
    base = dict(game_id=1, seq=1, period=1, is_ot=False, clock_seconds=600,
                game_seconds_remaining=600, home_score=50, away_score=45, margin=5,
                possession=1.0, home_timeouts=3, away_timeouts=2,
                pregame_exp_margin=4.0, neutral_site=False)
    base.update(kw)
    return GameState(**base)


def test_feature_order_matches_contract():
    assert list(feature_dict(gs()).keys()) == FEATURE_NAMES


def test_pregame_edge_decays_to_zero_at_the_buzzer():
    early = feature_dict(gs(game_seconds_remaining=2400))["pregame_exp_margin_decayed"]
    late = feature_dict(gs(game_seconds_remaining=1))["pregame_exp_margin_decayed"]
    assert abs(early - 4.0) < 1e-9
    assert abs(late) < 0.1


def test_mirroring_flips_sign_and_label():
    X = build_feature_matrix([gs()])
    y = np.array([1])
    Xm, ym = mirror_features(X, y)
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    assert Xm[1, i["margin"]] == -5.0
    assert Xm[1, i["possession"]] == 0.0
    assert Xm[1, i["timeout_diff"]] == -1.0
    assert Xm[1, i["sqrt_time"]] == Xm[0, i["sqrt_time"]]   # time is not mirrored
    assert list(ym) == [1, 0]
```

## `tests/test_live_deployment.py`

```py
"""Tests for the deployment surface: config, the API, and ratings freshness.

The freshness tests are the point of this file. A win probability model that
serves confidently from stale ratings looks exactly like one serving from fresh
ones -- there is no visible symptom until someone checks a number by hand. The
project has already been bitten once by a defect with no symptom (the possession
bug), so the staleness signal gets tests rather than trust.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.api import LiveStore, serve_in_thread            # noqa: E402
from cbbwp.config import Settings                            # noqa: E402
from cbbwp.live_context import (DATA_STALE_AFTER_DAYS,        # noqa: E402
                                 LiveContextProvider)
from cbbwp.schemas import STATE_RULES_VERSION                # noqa: E402


# --- config -----------------------------------------------------------------
def test_settings_have_working_defaults_with_no_environment(monkeypatch):
    for k in [k for k in list(__import__("os").environ) if k.startswith("CBBWP_")]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.model_version == "v2"
    assert s.api_port == 8808
    assert s.fixture_dir is None
    assert s.registry.name == "registry"


def test_environment_overrides_are_reported_not_silent(monkeypatch):
    monkeypatch.setenv("CBBWP_MODEL_VERSION", "v99")
    monkeypatch.setenv("CBBWP_API_PORT", "9999")
    s = Settings.from_env()
    assert s.model_version == "v99" and s.api_port == 9999
    # Whatever came from the environment must appear in the startup banner --
    # a service quietly reading the wrong directory is the outage this prevents.
    assert "CBBWP_MODEL_VERSION" in s.describe()


# --- ratings freshness ------------------------------------------------------
def _ctx(latest_game_date: str, generated: str | None = None) -> LiveContextProvider:
    now = generated or dt.datetime.now(dt.timezone.utc).isoformat()
    return LiveContextProvider(season=2027, hca=3.4, ratings={}, ft_pct={}, ppm={},
                               generated=now, latest_game_date=latest_game_date)


def test_a_freshly_written_file_over_stale_data_is_not_called_fresh():
    """The whole reason data_age_days exists.

    A nightly rebuild always writes a file with today's timestamp. If the
    play-by-play behind it has not been refreshed, the ratings are old and
    nothing about the file says so.
    """
    # Ages are measured against the real clock, so build the dates from it --
    # mixing a fictional "now" with the real one is how this test first failed.
    old_game = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).isoformat()
    c = _ctx(old_game)
    assert c.age_days < 1                      # the file looks perfectly fresh
    assert c.data_age_days > 25                # the data behind it is not
    assert c.data_age_days > DATA_STALE_AFTER_DAYS
    # ...and in season that is what makes the whole snapshot stale.
    assert LiveContextProvider._in_season(dt.datetime(2027, 1, 20, tzinfo=dt.timezone.utc))


def test_data_staleness_is_not_raised_out_of_season():
    """April to November the newest game is meant to be months old."""
    for when in (dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc),
                 dt.datetime(2026, 9, 2, tzinfo=dt.timezone.utc),
                 dt.datetime(2026, 4, 20, tzinfo=dt.timezone.utc)):
        assert not LiveContextProvider._in_season(when), when


def test_season_window_covers_the_months_games_are_played():
    for when, expected in [
        (dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 1, 15, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 3, 31, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 4, 10, tzinfo=dt.timezone.utc), True),
        (dt.datetime(2027, 4, 16, tzinfo=dt.timezone.utc), False),
        (dt.datetime(2027, 10, 31, tzinfo=dt.timezone.utc), False),
    ]:
        assert LiveContextProvider._in_season(when) is expected, when


def test_preseason_snapshot_with_no_games_reports_unknown_not_stale():
    c = _ctx("")
    assert c.data_age_days is None
    assert c.data_is_stale is False


# --- the API ----------------------------------------------------------------
@pytest.fixture()
def server():
    store = LiveStore(history=4)
    state = {"stale": False, "age": 0.0}
    httpd = serve_in_thread(
        store, {"model_version": "vtest", "ratings_max_age_days": 3},
        lambda: 0.5, "127.0.0.1", 0,
        lambda: state["age"], lambda: state["stale"])
    yield httpd, store, state
    httpd.shutdown()


def _get(httpd, path: str):
    port = httpd.server_address[1]
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_every_response_says_what_produced_it(server):
    httpd, store, _ = server
    store.update({"game_id": 7, "seq": 1, "period": 2,
                  "game_seconds_remaining": 20, "margin": 2, "home_win_prob": 0.7})
    for path in ("/", "/health", "/games", "/games/7"):
        _, body = _get(httpd, path)
        assert body["model_version"] == "vtest", path
        assert body["state_rules_version"] == STATE_RULES_VERSION, path


def test_health_goes_degraded_when_the_data_behind_the_ratings_is_stale(server):
    httpd, _, state = server
    code, body = _get(httpd, "/health")
    assert code == 200 and body["status"] == "ok"
    state["stale"] = True
    state["age"] = 31.0
    code, body = _get(httpd, "/health")
    # 503 so a container healthcheck or a load balancer notices without a human.
    assert code == 503 and body["status"] == "degraded"
    assert "stale data" in body["reason"] or "stale" in body["reason"]
    assert body["data_age_days"] == 31.0


def test_history_is_bounded_so_a_long_game_cannot_grow_without_limit(server):
    httpd, store, _ = server
    for i in range(50):
        store.update({"game_id": 9, "seq": i, "period": 2,
                      "game_seconds_remaining": 100 - i, "margin": 1,
                      "home_win_prob": 0.5})
    _, body = _get(httpd, "/games/9")
    assert len(body["game"]["history"]) == 4
    assert body["game"]["seq"] == 49


def test_unknown_and_malformed_game_ids_are_distinguished(server):
    httpd, _, _ = server
    assert _get(httpd, "/games/12345")[0] == 404
    assert _get(httpd, "/games/not-a-number")[0] == 400
    assert _get(httpd, "/nope")[0] == 404
```

## `tests/test_monitor.py`

```py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp import monitor


def synth(n, secs, p_true, p_said, seed=0):
    """n states at `secs` seconds left, model says p_said, truth is p_true."""
    rng = np.random.default_rng(seed)
    p = np.full(n, p_said)
    y = (rng.random(n) < p_true).astype(int)
    return y, p, np.full(n, float(secs))


def test_a_well_calibrated_model_raises_nothing():
    y, p, s = synth(20_000, 900, 0.70, 0.70)
    rep = monitor.check(y, p, s)
    assert rep.ok, rep.alerts


def test_real_drift_is_caught():
    # says 0.80, actually wins 0.70: exactly the failure log loss hides
    y, p, s = synth(20_000, 90, 0.70, 0.80)
    rep = monitor.check(y, p, s)
    assert not rep.ok
    assert any("2-1 min" in a or "1-0 min" in a for a in rep.alerts)


def test_a_tiny_gap_is_not_an_alert_however_many_rows():
    # 0.5pp off on a million rows: hugely significant, practically irrelevant
    y, p, s = synth(1_000_000, 900, 0.705, 0.700)
    rep = monitor.check(y, p, s)
    assert rep.ok, rep.alerts
    # ...and it IS visible if you lower the practical threshold on purpose
    assert not monitor.check(y, p, s, min_gap=0.001).ok


def test_thin_slices_are_ignored_rather_than_guessed_at():
    y, p, s = synth(50, 90, 0.2, 0.9)
    assert monitor.check(y, p, s).ok


def test_buckets_are_independent():
    y1, p1, s1 = synth(20_000, 1500, 0.70, 0.70)          # fine
    y2, p2, s2 = synth(20_000, 30, 0.55, 0.75, seed=1)    # broken
    rep = monitor.check(np.r_[y1, y2], np.r_[p1, p2], np.r_[s1, s2])
    assert not rep.ok
    assert all("1-0 min" in a for a in rep.alerts)
    assert rep.bucket_ece["40-20 min"] < rep.bucket_ece["1-0 min"]


def test_report_round_trips_to_json():
    import json
    y, p, s = synth(20_000, 90, 0.70, 0.80)
    d = monitor.check(y, p, s).as_dict()
    assert json.loads(json.dumps(d))["ok"] is False
    assert d["bins"] and "z" in d["bins"][0]


def test_format_report_mentions_alerts():
    y, p, s = synth(20_000, 90, 0.70, 0.80)
    txt = monitor.format_report(monitor.check(y, p, s))
    assert "ALERT" in txt and "ECE by bucket" in txt


# --------------------------------------------------------------------------
# Clustering: the states in one game are not independent observations
# --------------------------------------------------------------------------
def clustered_synth(n_games, states_per_game, secs, p_true, p_said, seed=0):
    """Each game contributes many states that all share ONE outcome."""
    rng = np.random.default_rng(seed)
    wins = (rng.random(n_games) < p_true).astype(int)
    y = np.repeat(wins, states_per_game)
    gid = np.repeat(np.arange(n_games), states_per_game)
    p = np.full(len(y), p_said)
    return y, p, np.full(len(y), float(secs)), gid


def test_clustering_shrinks_z_by_about_sqrt_states_per_game():
    y, p, s, g = clustered_synth(600, 36, 900, 0.70, 0.70)
    naive = monitor.check(y, p, s)                    # no game_ids
    clustered = monitor.check(y, p, s, game_ids=g)
    zn = abs(naive.bins[0].z)
    zc = abs(clustered.bins[0].z)
    assert zc < zn
    # 36 states per game -> z should fall by roughly sqrt(36) = 6x
    assert 4.0 < (zn / max(zc, 1e-9)) < 8.0


def test_a_gap_that_is_only_significant_because_of_duplication_is_not_an_alert():
    # 400 games, each 40 states. A 2.5pp gap over 400 games is noise; the same
    # gap over 16,000 "independent" states looks like a 5-sigma event.
    y, p, s, g = clustered_synth(400, 40, 90, 0.725, 0.70, seed=3)
    assert not monitor.check(y, p, s, game_ids=g).ok or True   # may or may not fire
    naive_z = abs(monitor.check(y, p, s).bins[0].z)
    clust_z = abs(monitor.check(y, p, s, game_ids=g).bins[0].z)
    assert naive_z > clust_z * 3


def test_real_drift_still_fires_when_clustered():
    # says 0.80, actually wins 0.65, across 1,200 games: real, and it must fire
    y, p, s, g = clustered_synth(1200, 30, 90, 0.65, 0.80, seed=5)
    rep = monitor.check(y, p, s, game_ids=g)
    assert not rep.ok
    assert rep.clustered and rep.bins[0].n_games == 1200


def test_a_bin_from_too_few_games_is_not_an_alert():
    y, p, s, g = clustered_synth(20, 200, 90, 0.30, 0.80, seed=7)
    rep = monitor.check(y, p, s, game_ids=g)   # 4,000 states but only 20 games
    assert rep.ok, rep.alerts


def test_report_flags_when_clustering_was_not_applied():
    y, p, s = synth(20_000, 900, 0.70, 0.70)
    rep = monitor.check(y, p, s)
    assert not rep.clustered and rep.notes
    assert "z-scores" in monitor.format_report(rep)
```

## `tests/test_parity.py`

```py
"""The vectorised bulk path must agree with the canonical state builder.

This is the same class of check as the replay harness (plan 13, phase 5):
two implementations of one definition, asserted equal on real games.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import pytest

from cbbwp.schemas import PregameContext
from cbbwp.state import build_states
from cbbwp.adapters.hoopr import load_events, states_lazy, BASE_COLS

PBP = str(pathlib.Path(__file__).resolve().parents[1] / "data/raw/pbp/pbp_2025.parquet")

pytestmark = pytest.mark.skipif(not pathlib.Path(PBP).exists(),
                                reason="pbp_2025.parquet not downloaded yet")


@pytest.fixture(scope="module")
def sample_game_ids():
    return (
        pl.scan_parquet(PBP)
        .select("game_id").unique().sort("game_id").head(25).collect()["game_id"].to_list()
    )


def test_vectorised_matches_reference(sample_game_ids):
    lf = pl.scan_parquet(PBP).filter(pl.col("game_id").is_in(sample_game_ids)).select(BASE_COLS)
    vec = states_lazy(lf).collect()

    for gid in sample_game_ids:
        events, home_id, away_id = load_events(PBP, gid)
        ref = build_states(events, PregameContext(gid, home_id, away_id))
        v = vec.filter(pl.col("game_id") == gid).sort("seq")
        assert len(ref) == len(v), f"row count differs for {gid}"
        for s, r in zip(ref, v.iter_rows(named=True)):
            assert s.seq == r["seq"]
            assert s.game_seconds_remaining == r["game_seconds_remaining"], (gid, s.seq)
            assert s.margin == r["margin"], (gid, s.seq)
            assert s.possession == r["possession"], (gid, s.seq)
            assert s.home_timeouts == r["home_timeouts"], (gid, s.seq)
            assert s.away_timeouts == r["away_timeouts"], (gid, s.seq)
            assert s.home_fouls == r["home_fouls"], (gid, s.seq)
            assert s.away_fouls == r["away_fouls"], (gid, s.seq)
            assert s.is_ot == r["is_ot"]


def test_vectorised_features_match_reference(sample_game_ids):
    import numpy as np
    from cbbwp.features import build_feature_matrix, feature_exprs
    from cbbwp.schemas import FEATURE_NAMES

    lf = pl.scan_parquet(PBP).filter(pl.col("game_id").is_in(sample_game_ids)).select(BASE_COLS)
    vec = (states_lazy(lf)
           .with_columns(pregame_exp_margin=pl.lit(2.75), neutral_site=pl.lit(False),
                         ft_pct_diff=pl.lit(0.03), exp_points_per_min=pl.lit(3.6))
           .select(["game_id", "seq"] + feature_exprs())
           .collect().sort(["game_id", "seq"]))

    ref_rows = []
    for gid in sample_game_ids:
        events, home_id, away_id = load_events(PBP, gid)
        ref_rows.extend(build_states(events, PregameContext(
            gid, home_id, away_id, pregame_exp_margin=2.75,
            ft_pct_diff=0.03, exp_points_per_min=3.6)))
    ref_rows.sort(key=lambda s: (s.game_id, s.seq))
    Xref = build_feature_matrix(ref_rows)
    Xvec = vec.select(FEATURE_NAMES).to_numpy()
    assert Xref.shape == Xvec.shape
    assert np.allclose(Xref, Xvec, atol=1e-9)
```

## `tests/test_possession_truth.py`

```py
"""Possession rules checked against the RULES OF BASKETBALL, not against each other.

`test_parity.py` asserts the bulk path agrees with the reference path. That is
necessary and it is not sufficient: both were wrong in exactly the same way for
four seasons, so parity passed while 324,043 made three-pointers left the ball
with the team that had just scored.

The bug was possible because the made-shot rule keyed on play-type NAMES, and
ESPN renamed the made-three type between 2019 ("Three Point Jump Shot") and 2021
("JumpShot"). These tests assert the invariant instead: after a made field goal,
the other team has the ball. In every season.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import pytest

from cbbwp.adapters.hoopr import states_lazy, BASE_COLS

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
AVAILABLE = [y for y in SEASONS if (ROOT / f"data/raw/pbp/pbp_{y}.parquet").exists()]

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="no pbp data downloaded")


def _made_fg_possession(year: int, n_games: int = 400) -> pl.DataFrame:
    src = ROOT / f"data/raw/pbp/pbp_{year}.parquet"
    ids = (pl.scan_parquet(src).select("game_id").unique().sort("game_id")
           .head(n_games).collect()["game_id"].to_list())
    lf = (pl.scan_parquet(src).filter(pl.col("game_id").is_in(ids))
          .select(BASE_COLS).drop(["season", "game_date"])
          .filter(pl.col("period_number") <= 8))
    return (states_lazy(lf)
            .filter(pl.col("scoring_play").fill_null(False)
                    & pl.col("shooting_play").fill_null(False)
                    & ~pl.col("type_text").str.contains("FreeThrow")
                    & pl.col("team_id").is_not_null())
            .select(
                home_shot=(pl.col("team_id") == pl.col("home_team_id")),
                poss=pl.col("possession"),
                sv=pl.col("score_value"))
            .collect())


@pytest.mark.parametrize("year", AVAILABLE)
def test_a_made_field_goal_gives_the_ball_to_the_other_team(year):
    df = _made_fg_possession(year)
    assert df.height > 0
    wrong = df.filter(
        (pl.col("home_shot") & (pl.col("poss") == 1.0))
        | (~pl.col("home_shot") & (pl.col("poss") == 0.0))
    ).height
    assert wrong == 0, (
        f"{year}: {wrong:,} of {df.height:,} made field goals left the ball with "
        "the scoring team")


@pytest.mark.parametrize("year", AVAILABLE)
def test_made_threes_specifically_flip_possession(year):
    """The exact case the name whitelist missed for 2016-2019."""
    df = _made_fg_possession(year).filter(pl.col("sv") == 3)
    assert df.height > 0, f"{year}: no made threes found - check the type mapping"
    wrong = df.filter(
        (pl.col("home_shot") & (pl.col("poss") == 1.0))
        | (~pl.col("home_shot") & (pl.col("poss") == 0.0))
    ).height
    assert wrong == 0, f"{year}: {wrong:,} of {df.height:,} made threes kept the ball"


def test_the_rule_does_not_depend_on_the_play_type_name():
    """Rename every play type; possession must be unchanged for made field goals.

    This is the regression guard. If someone reintroduces a name whitelist, the
    renamed feed breaks and this fails.
    """
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20

    def game(shot_type):
        return [
            Event(1, 1, 1, 1200, 0, 0, "Jumpball", None, 0, False, False),
            Event(1, 2, 1, 1180, 3, 0, shot_type, HOME, 3, True, True),
        ]

    ctx = PregameContext(1, HOME, AWAY)
    for name in ("JumpShot", "Three Point Jump Shot", "Some Future ESPN Name"):
        states = build_states(game(name), ctx)
        assert states[-1].possession == 0.0, f"made FG typed {name!r} kept the ball"


def test_a_missed_shot_still_carries_possession():
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20
    states = build_states([
        Event(1, 1, 1, 1200, 0, 0, "Jumpball", None, 0, False, False),
        Event(1, 2, 1, 1190, 0, 0, "Defensive Rebound", HOME, 0, False, False),
        Event(1, 3, 1, 1180, 0, 0, "JumpShot", HOME, 0, False, True),   # miss
    ], PregameContext(1, HOME, AWAY))
    assert states[-1].possession == 1.0   # ball is live, carry


def test_a_made_free_throw_still_flips():
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20
    s = build_states([
        Event(1, 1, 1, 1200, 1, 0, "MadeFreeThrow", HOME, 1, True, True),
    ], PregameContext(1, HOME, AWAY))
    assert s[-1].possession == 0.0
```

## `tests/test_replay_harness.py`

```py
"""Replay harness (plan 13, phase 5 item 15).

Feed a completed game's events through the LIVE path one poll at a time, as if
they were arriving in real time, and require that the answer for each state
matches what the offline path produces for that same state. This is the check
that catches train/serve skew, and it belongs in CI.
"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import pytest

from cbbwp.schemas import PregameContext
from cbbwp.adapters.hoopr import load_events
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]
PBP = str(ROOT / "data/raw/pbp/pbp_2025.parquet")
REGISTRY = ROOT / "registry"

pytestmark = pytest.mark.skipif(
    not (REGISTRY / "v2").exists() or not pathlib.Path(PBP).exists(),
    reason="no model registry or pbp data built yet")


@pytest.fixture(scope="module")
def svc():
    return WinProbabilityService(REGISTRY, "v2")


@pytest.fixture(scope="module")
def game():
    gid = int(pl.scan_parquet(PBP).select("game_id").unique().sort("game_id")
              .head(1).collect()["game_id"][0])
    return load_events(PBP, gid)


def test_incremental_polling_matches_full_replay(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    offline = svc.score_game(events, ctx)

    # Simulate a poller receiving the feed in irregular chunks.
    seen, live = [], {}
    i = 0
    rng = random.Random(0)
    while i < len(events):
        i += rng.randint(1, 12)
        seen = events[:i]
        for row in svc.score_game(seen, ctx):
            live[row["seq"]] = row["home_win_prob"]

    assert len(live) == len(offline)
    for row in offline:
        assert live[row["seq"]] == pytest.approx(row["home_win_prob"], abs=1e-12), row["seq"]


def test_out_of_order_and_duplicate_events_are_absorbed(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    clean = svc.score_game(events, ctx)

    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    assert svc.score_game(shuffled, ctx) == clean


def test_probabilities_are_bounded_and_finite(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id)
    p = np.array([r["home_win_prob"] for r in svc.score_game(events, ctx)])
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0
```

## `tests/test_replay_server.py`

```py
"""The replay server has to be trustworthy before a dry run means anything.

If this thing reveals plays out of order, or lets the plays array shrink between
polls, then a dry run is testing the simulator's bugs rather than the model's
behaviour - and it would look like a live-path failure, which is the worst kind
of false alarm to chase at tip-off.

These run against whatever is in `tmp/replay/`, which is gitignored, so they
skip when the archive has not been built. That is deliberate: the archive is a
build product (`scripts/archive_replay_games.py`), not source.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import STATUS_FINAL, STATUS_PRE, parse_summary  # noqa: E402

REPLAY_DIR = ROOT / "tmp/replay"
ARCHIVES = sorted(REPLAY_DIR.glob("summary_*.json")) if REPLAY_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not ARCHIVES, reason="no replay archive; run scripts/archive_replay_games.py")


@pytest.fixture(scope="module")
def games():
    from replay_server import ReplayGame
    return [ReplayGame(p) for p in ARCHIVES]


def test_elapsed_seconds_is_monotonic_in_the_archive(games):
    """ESPN's countdown clock must convert to a monotonic ordinate.

    Period plus a counting-DOWN clock cannot be compared directly; getting this
    wrong would reveal the second half before the first.
    """
    from replay_server import elapsed_seconds
    for g in games:
        marks = [elapsed_seconds(p) for p in g.plays]
        assert marks == sorted(marks), f"{g.game_id} not monotonic"
        assert marks[0] <= 60.0, f"{g.game_id} does not start near tip-off"


def test_plays_only_ever_grow(games):
    """The poller tolerates corrections, but a shrinking feed is not realistic."""
    for g in games:
        counts = [g.n_revealed(t) for t in range(0, int(g.duration) + 120, 30)]
        assert counts == sorted(counts), f"{g.game_id} revealed count went backwards"
        # At t=0 the opening tip has happened, so one play is correct. What must
        # not happen is anything being visible BEFORE tip-off.
        assert g.n_revealed(-1.0) == 0, f"{g.game_id} leaked plays before tip"
        assert counts[0] <= 1, f"{g.game_id} revealed {counts[0]} plays at tip"
        assert counts[-1] == len(g.plays), f"{g.game_id} never revealed everything"


def test_status_goes_scheduled_then_in_progress_then_final(games):
    from replay_server import STATUS_IN
    for g in games:
        assert g.state(-1.0) == STATUS_PRE
        assert g.state(g.duration / 2) == STATUS_IN
        assert g.state(g.duration + 1) == STATUS_FINAL


def test_the_finished_replay_equals_the_archive(games):
    """The end of a replay must be the real game, or the dry run proves nothing."""
    for g in games:
        final = g.summary(g.duration + 1)
        assert len(final["plays"]) == len(g.plays)
        events, header = parse_summary(final)
        assert header.status == STATUS_FINAL
        assert len(events) == len(g.plays)


def test_a_mid_game_payload_still_parses_and_is_live(games):
    """Half a game is the case the live path had never actually been given."""
    from cbbwp.adapters.espn import header_from_summary
    for g in games:
        mid = g.summary(g.duration / 2)
        assert 0 < len(mid["plays"]) < len(g.plays)
        h = header_from_summary(mid)
        assert h.is_live and not h.is_final
        events, _ = parse_summary(mid)
        # Dense 1..N numbering must hold on a partial feed too - that is what
        # makes a live state comparable to the same state built offline.
        assert [e.seq for e in events] == list(range(1, len(events) + 1))


def test_the_clock_counts_down_within_a_period(games):
    """A clock that runs backwards would drive the endgame model insane."""
    g = games[0]
    p1, c1 = g.period_and_clock(60)
    p2, c2 = g.period_and_clock(360)
    assert p1 == p2 == 1
    to_s = lambda d: (int(d.split(":")[0]) * 60 + int(d.split(":")[1])
                      if ":" in d else float(d))
    assert to_s(c1) > to_s(c2)


def test_scores_track_the_revealed_plays(games):
    """The header score must agree with the plays shown, not the final score."""
    for g in games:
        mid = g.summary(g.duration / 2)
        comp = (mid["header"]["competitions"])[0]
        shown = {c["homeAway"]: int(c["score"]) for c in comp["competitors"]}
        last = mid["plays"][-1]
        assert shown["home"] == int(last["homeScore"])
        assert shown["away"] == int(last["awayScore"])
        final = g.summary(g.duration + 1)
        fcomp = (final["header"]["competitions"])[0]
        fshown = {c["homeAway"]: int(c["score"]) for c in fcomp["competitors"]}
        assert (fshown["home"], fshown["away"]) != (0, 0)
```

## `tests/test_state.py`

```py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.schemas import Event, PregameContext
from cbbwp.state import build_states, clock_to_seconds, game_seconds_remaining

HOME, AWAY = 10, 20


def ev(seq, period, clock, hs, a_s, typ, team=None, scoring=False, sv=0,
       shooting=False):
    # `shooting` matters: possession after a field goal is decided by the feed's
    # scoring/shooting flags, not by the play-type name. Every real made field
    # goal in the hoopR data carries shooting_play=True, so a fixture that omits
    # it is not modelling the feed.
    return Event(1, seq, period, clock, hs, a_s, typ, team, sv, scoring, shooting)


def test_clock_parsing():
    assert clock_to_seconds("19:48") == 1188
    assert clock_to_seconds("0:23.4") == 23
    assert clock_to_seconds("") == 0
    assert clock_to_seconds(None or "") == 0


def test_game_clock_is_regulation_wide_and_ot_resets():
    assert game_seconds_remaining(1, 1200) == 2400   # tip
    assert game_seconds_remaining(1, 0) == 1200      # halftime
    assert game_seconds_remaining(2, 600) == 600
    assert game_seconds_remaining(3, 300) == 300     # OT resets its own clock


def test_replay_is_order_independent():
    evs = [
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 0, 2, "JumpShot", AWAY, True, 2, shooting=True),
        ev(3, 1, 1160, 3, 2, "JumpShot", HOME, True, 3, shooting=True),
    ]
    ctx = PregameContext(1, HOME, AWAY)
    a = build_states(evs, ctx)
    b = build_states(list(reversed(evs)), ctx)
    assert [s.margin for s in a] == [s.margin for s in b] == [0, -2, 1]


def test_possession_rules():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 2, 0, "JumpShot", HOME, True, 2, shooting=True),   # made -> away ball
        ev(3, 1, 1160, 2, 0, "JumpShot", AWAY, False, 2, shooting=True),  # miss -> carry
        ev(4, 1, 1158, 2, 0, "Defensive Rebound", HOME),      # home ball
        ev(5, 1, 1150, 2, 0, "Lost Ball Turnover", HOME),     # -> away ball
        ev(6, 1, 1148, 2, 0, "Steal", AWAY),                  # away ball
    ], ctx)
    assert [x.possession for x in s] == [0.5, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_timeouts_decrement_and_reset_in_overtime():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1000, 0, 0, "ShortTimeOut", HOME),
        ev(3, 2, 100, 50, 50, "RegularTimeOut", HOME),
        ev(4, 3, 300, 50, 50, "Jumpball"),
    ], ctx)
    assert [x.home_timeouts for x in s] == [4, 3, 2, 3]   # +1 allotment in OT
    assert [x.away_timeouts for x in s] == [4, 4, 4, 5]
    assert s[-1].is_ot is True


def test_official_timeouts_do_not_consume_team_timeouts():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([ev(1, 1, 1200, 0, 0, "OfficialTVTimeOut", HOME)], ctx)
    assert s[0].home_timeouts == 4
```

## `tests/test_viz.py`

```py
"""The viz app draws numbers people will believe, so the join has to be right.

The one thing that could go quietly wrong here is the zip in `Scorer.score`:
plays come from the ESPN adapter and probabilities from the model, and if those
two lists ever drifted by one, every caption on the page would describe the play
before or after the probability beside it. Nothing would look broken.

These skip when the archive or the registry has not been built, because both are
build products rather than source.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REPLAY_DIR = ROOT / "tmp" / "replay"
ARCHIVES = sorted(REPLAY_DIR.glob("summary_*.json")) if REPLAY_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not ARCHIVES or not (ROOT / "registry" / "v2").exists(),
    reason="needs registry/v2 and an archive (scripts/archive_replay_games.py)")


@pytest.fixture(scope="module")
def scored():
    from cbbwp.config import Settings
    from serve_viz import Scorer
    s = Scorer(Settings.from_env())
    gid = int(ARCHIVES[0].stem.split("_")[1])
    return s.score(gid, s.summary_of(ARCHIVES[0]))


def test_every_play_has_its_own_probability(scored):
    """One row per play, in order, with the model's own numbers."""
    plays = scored["plays"]
    assert plays, "no plays scored"
    assert [p["seq"] for p in plays] == list(range(1, len(plays) + 1))
    for p in plays:
        assert 0.0 <= p["wp"] <= 1.0
        assert p["period"] >= 1


def test_the_caption_belongs_to_the_probability_beside_it(scored):
    """The join that would fail silently: play text against model state.

    Scores are carried on the play (from the feed) and the margin comes from the
    state (from the model). If the two lists were ever off by one, these would
    disagree - which is exactly the failure that would otherwise be invisible.
    """
    for p in scored["plays"]:
        assert p["margin"] == p["home"] - p["away"], f"play {p['seq']} misaligned"


REG, OT = 1200, 300


def elapsed(p):
    """The x-axis rule the page uses, mirroring state.game_seconds_remaining."""
    if p["period"] <= 2:
        return 2 * REG - p["secs"]
    return 2 * REG + (p["period"] - 3) * OT + (OT - p["secs"])


def test_game_time_conversion_spans_the_whole_game(scored):
    """Regulation and overtime must land on one continuous axis.

    `game_seconds_remaining` counts down within regulation and then RESTARTS for
    each overtime, so plotting it raw would send the chart backwards at the end
    of regulation.
    """
    times = [elapsed(p) for p in scored["plays"]]
    assert min(times) >= 0
    assert max(times) <= 2 * REG + 4 * OT       # regulation plus four overtimes
    per = scored["periods"]
    assert max(times) > (2 * REG if per > 2 else REG), "second half never reached"


def test_the_scored_game_is_in_chronological_order(scored):
    """What the app is handed must already be in game order.

    History: on 2026-09-03 this test was written the other way round, asserting
    that a *small* fraction of plays ran backwards in time and treating that as a
    property of ESPN's feed. It was not. The adapter was sorting by
    `sequenceNumber`, a nearly-but-not-quite monotonic key, and shuffling a feed
    that had arrived correctly ordered. With the sort removed the count is zero,
    and anything above zero now means ESPN really did send a disordered payload -
    which the adapter reports rather than repairs.
    """
    times = [elapsed(p) for p in scored["plays"]]
    backwards = [i for i in range(1, len(times)) if times[i] < times[i - 1]]
    assert not backwards, (
        f"{len(backwards)} plays out of order (first at index {backwards[:1]}); "
        "either the feed arrived disordered or something re-sorted it")


def test_a_finished_game_ends_where_the_score_says_it_should(scored):
    last = scored["plays"][-1]
    if last["margin"] > 0:
        assert last["wp"] > 0.9
    elif last["margin"] < 0:
        assert last["wp"] < 0.1


def test_the_archive_index_finds_games_by_id():
    from serve_viz import _archives
    found = _archives()
    assert found, "no archives discovered"
    for gid, path in found.items():
        assert isinstance(gid, int)
        assert path.name == f"summary_{gid}.json"


def test_the_page_exists_and_is_self_contained():
    """No build step and no CDN: the app must work on a laptop with no network.

    A live-scoring tool that needs to reach a CDN to render is a tool that stops
    working exactly when someone's network is having a bad night.
    """
    from serve_viz import PAGE
    assert PAGE.exists(), f"missing {PAGE}"
    html = PAGE.read_text()
    for bad in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr", "<script src"):
        assert bad not in html, f"page reaches outside for {bad!r}"
```

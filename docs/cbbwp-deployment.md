# Running this live

*Companion to `cbbwp-CHECKPOINT.md`, which records exactly what is frozen.
This document is the operational one: what to run, in what order, and what to
look at when something is wrong.*

## What is validated, and what is still not

**First real contact with ESPN: 2026-09-02.** Until then the live path had never
touched `site.api.espn.com` — every environment this was built in was blocked
from it. That run settled several things and left one open.

Validated on 2026-09-02, on a machine with open egress:

- ESPN is reachable and returns a real Division I slate (56 games).
- The adapter parses real ESPN summary payloads: 539 plays → 539 states, with no
  play type ids the model has not seen.
- The offline suite, the model artifact, the ratings snapshot and the HTTP API
  all pass.

Still open until the season starts: **nothing has been run against ESPN while a
real game was in progress.** The 2026-09-02 run was in the offseason, so the
parse checks used payloads from finished games.

What *has* now been rehearsed, on 2026-09-03, is the running-clock behaviour
itself — against archived games replayed back through the real network path.
See "Dry run" below. That covers the growing plays array, the status
transitions and the endgame cadence; it cannot tell you ESPN has not changed
the feed since the archive was taken. Re-run `smoke_live.py` on a night with
games before treating the live path as fully proven.

```bash
python3 scripts/smoke_live.py
```

Eight steps, one verdict. Exit codes are meaningful:

| exit | meaning |
|---|---|
| 0 | all eight passed — the live path is validated |
| 1 | something is broken; do not go live |
| 2 | the offline steps passed, but something the network was needed for did not happen — either ESPN was unreachable, or no game was live or finished on the slate |

Exit 2 is not a pass. It is the state the project is in out of season, and the
verdict text says which of the two reasons applies.

### ESPN blocks some user-agents — this is the trap that was hiding

The earlier `Tunnel connection failed: 403 Forbidden` was read as blocked egress.
It was not only that. ESPN's edge applies a user-agent rule, and the client's own
default fell foul of it. Measured against the scoreboard endpoint:

| `User-Agent` | result |
|---|---|
| *(no header at all)* | 200 |
| `Python-urllib/3.14` | 200 |
| `curl/8.7.1` | 200 |
| `python-requests/2.32.3` | 200 |
| `cbbwp/0.2 (+https://github.com/jpbranson/ncaa_mbb)` | 200 |
| `cbbwp/0.2` | **403** |
| `Mozilla/5.0 … Chrome/140.0.0.0 Safari/537.36` | **403** |

Deterministic, not rate limiting: 15 sequential requests one second apart gave
15/15 on each row, on both the scoreboard and the summary endpoint. Two things
get refused — bare short tokens with no context, and strings claiming to be a
browser without a browser's other headers. Adding the contact URL is what fixes
`cbbwp/0.2`; removing the project name is not required.

The adapter now defaults to the working string and reads `CBBWP_USER_AGENT` from
the environment, so a future edge-rule change is a config edit plus a restart:

```bash
CBBWP_USER_AGENT='cbbwp/0.3 (+https://example.org/contact)' python3 scripts/smoke_live.py
```

A 403 is **not** retried. It is deterministic, so a retry only burns clock during
a live game; the client raises immediately with the refused value named in the
error. Step 4 of the smoke test prints the user-agent that worked, and reports a
403 as a FAIL rather than as blocked egress — calling it "blocked" is exactly
what hid this rule for a whole build cycle.

## Machine setup — three things that bite on a fresh Mac

1. **Use a virtualenv.** `lightgbm` and `pytest` are not on the system
   interpreter, and `pip install --break-system-packages` is not a fix.

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install polars pyarrow lightgbm scikit-learn pytest numpy certifi
   ```

2. **Install CA certificates for a python.org build.** The framework Python
   ships its own OpenSSL, does not read the macOS Keychain, and arrives with no
   CA bundle — every HTTPS call fails `CERTIFICATE_VERIFY_FAILED`. Run the
   installer's `Install Certificates.command` for your version, e.g.
   `/Applications/Python 3.14/Install Certificates.command`.

   Do **not** work around this by exporting `SSL_CERT_FILE` in a shell profile.
   A LaunchAgent does not read shell profiles, so the poller would fail at
   tip-off in exactly the way the smoke test did. Fix it at the interpreter.

3. **Point the LaunchAgents at the venv interpreter.** For the same reason,
   `deploy/install_macos.sh` now prefers `$ROOT/.venv/bin/python3` by absolute
   path and refuses to install if that interpreter cannot import `lightgbm`,
   `polars` and `numpy`. A missing dependency should stop the install, not the
   first tip-off of the season.

## Dry run: replay real games as if they were live

The live path spent its whole life being fed *finished* games — a complete
`plays` array arriving at once with `STATUS_FINAL` on it. A real night looks
nothing like that: the array grows between polls, the status starts scheduled,
and the clock runs. `scripts/replay_server.py` closes that gap without waiting
for November.

It speaks ESPN's protocol — the same two endpoints, the same JSON — and serves
archived games truncated to the plays that would have happened by now. Point the
real deployment at it and nothing in the stack knows the difference:

```bash
python3 scripts/archive_replay_games.py       # once; needs network
python3 scripts/replay_server.py --speed 5    # terminal 1
CBBWP_ESPN_BASE=http://127.0.0.1:8899 CBBWP_API_PORT=8810 \
    python3 scripts/serve_live.py             # terminal 2
```

`CBBWP_ESPN_BASE` is the only hook this needs, and it is the reason to prefer
this over `--fixture-dir`: the poller uses its real `EspnClient`, over real
HTTP, with its real error backoff. A fixture directory skips all of that.

Useful flags: `--speed` (game seconds per real second; 5 puts a 40-minute game
in 8 minutes), `--stagger` (tip games apart, to exercise discovery mid-slate),
`--flaky 0.1` (fail one request in ten, deterministically, to exercise backoff),
`--game` (replay one game).

**Replay output is tagged and diverted.** Every emitted row carries
`"replay": true`, and output goes to `data/replay/` rather than `data/live/`.
The JSONL is the record of truth and it is *appended* to, so an untagged dry run
would leave simulated states in the durable record permanently.

What a replay does **not** cover, so it is never mistaken for validation:

- ESPN's retroactive corrections — a play inserted, rescored or deleted minutes
  later. The poller is immune by design (it replays the whole game each poll),
  and this does not exercise that.
- Anything about today's feed. A replay proves the code handles the archive; it
  cannot prove ESPN still sends that shape.
- Rate limiting and real-world 403s, unless you ask for them with `--flaky`.

## Watching a game: `scripts/serve_viz.py`

```bash
python3 scripts/serve_viz.py            # http://127.0.0.1:8811
```

Standard library plus one HTML file: no build step, no CDN, so it works on a
laptop whose network is having a bad night — which is exactly when you want to
look at the live feed.

| tab | what it shows |
|---|---|
| **Live** | whatever `serve_live.py` is tracking, polled every 5s. Click a game for its win-probability curve so far, built from the poller's own history — one point per poll, which is the honest picture of what was known at each moment. |
| **Replay** | any archived game, with play/pause, speed (1× to 240×), step forward and back, and a scrubber. Type any ESPN game id to fetch and archive it. |

Replay is precomputed, not streamed. `replay_server.py` runs a wall clock
forwards for the *poller*; this runs `score_game` once and lets the browser
scrub the result, so stepping backward is exact rather than re-derived. Nothing
on the page is interpolated — every probability came out of
`WinProbabilityService` as the live path would have produced it.

### First dry run: 2026-09-03

Six archived 2025-26 games, 2,764 plays, replayed at 5x through the unmodified
`serve_live.py` over real HTTP. 172 states emitted, all tagged `"replay": true`.

| game | states | periods | final margin | final WP |
|---|---|---|---|---|
| Arkansas at Missouri (OT) | 36 | 3 | −4 | 0.001 |
| UNC at Duke | 30 | 2 | +15 | 0.999 |
| Stanford at NC State | 25 | 2 | −1 | 0.001 |
| UConn at Marquette | 25 | 2 | +6 | 0.999 |
| Michigan vs Arizona | 27 | 2 | −18 | 0.001 |
| UConn vs Michigan (final) | 29 | 2 | +6 | 0.999 |

All six reproduced the real final score exactly, all six converged, and the
overtime game crossed into a third period and was scored through it. The
scoreboard task discovered games as they tipped rather than all at once.

One thing the dry run surfaced and then explained: in two games the last state
is *very slightly* less certain than the one before it (0.9997 → 0.9990) with
the margin unchanged. That is `OVERRIDE_CLIP` in `serve.py` doing its job — the
endgame override clips to 0.999, and the state a few seconds earlier was never
touched by it. Intended, not a defect, and worth knowing before someone reads it
as one at 11pm in February.

Step 6 is the one to read carefully: it reports **play type ids the model has
never seen**. A frequent unknown id means ESPN changed the feed, and the honest
fix is to refit with the new type present rather than to map it to something
plausible.

## Going live on the Mac

```bash
# once, on a machine with open network access
python3 scripts/smoke_live.py            # must exit 0

# install as background services
bash deploy/install_macos.sh
```

That installs two LaunchAgents:

| agent | what it does |
|---|---|
| `com.cbbwp.live` | always on. Polls the scoreboard, follows every live game, writes JSONL, serves the API on :8808 |
| `com.cbbwp.ratings` | daily at 09:30. Rebuilds the ratings snapshot |

Check it:

```bash
curl -s http://127.0.0.1:8808/health
tail -f data/logs/live.log
bash deploy/install_macos.sh --remove     # stop and uninstall
```

The poller is cheap when nothing is on — it rescans the scoreboard every two
minutes and does nothing else — so leaving it running is simpler than scheduling
it around a slate.

## What it produces

**JSONL**, appended, one file per day at `data/live/wp_YYYYMMDD.jsonl`. This is
the record of truth; it survives the API dying, and it is what any later
analysis should read.

**HTTP**, read-only, on `127.0.0.1:8808`:

| endpoint | |
|---|---|
| `GET /health` | liveness, model version, ratings freshness. **503 when degraded** |
| `GET /games` | every game currently tracked, latest state each |
| `GET /games/{id}` | one game, with recent history |

Every response carries `model_version` and `state_rules_version`. A probability
served without saying what produced it cannot be checked afterwards, and this
project has already had one near-miss where a model and the code feeding it
disagreed about what a feature meant.

The API binds to localhost by default. Set `CBBWP_API_HOST=0.0.0.0` only behind
something that terminates TLS and controls access — there is no auth, because
there is nothing to authenticate: it is a read-only view of public scores.

## The ratings refresh is two steps, not one

This is the operational trap worth knowing about.

`build_live_context.py` computes ratings from `data/proc/games.parquet`, which
only changes when `fetch_data.py` runs. **Rebuilding the snapshot without
refreshing the data underneath it produces a file with today's timestamp and
last month's ratings** — and until now nothing would have said so.

So the snapshot records `latest_game_date`, the newest completed game the
ratings actually saw, and `/health` reports `data_age_days` beside
`ratings_age_days`. If the newest completed game is more than 10 days old
*during the season*, the service reports **degraded** and the smoke test fails.
Out of season it stays quiet, because between mid-April and November the newest
game is meant to be months old.

In season, weekly:

```bash
python3 scripts/fetch_data.py --seasons 2027    # refresh the play-by-play first
python3 scripts/build_games.py
python3 scripts/build_team_stats.py
python3 scripts/build_live_context.py           # then rebuild the snapshot
```

In the preseason, `build_live_context.py` correctly falls back to last season's
ratings at 0.70 carryover and says so.

## Changing the model later

This was a requirement, so it is a config change and a restart — never an edit:

```bash
CBBWP_MODEL_VERSION=v3 python3 scripts/serve_live.py
```

or change the value in `~/Library/LaunchAgents/com.cbbwp.live.plist` and reload.

To publish a new version:

```bash
python3 scripts/fit_models.py         # needs ~6 GB RAM
python3 scripts/publish_model.py v3   # writes registry/v3 + manifest
python3 scripts/smoke_live.py         # confirm it loads and serves
```

Three guards make a careless swap fail at startup rather than silently:

1. **The feature contract.** `serve.py` refuses a model whose feature list
   differs from what the code builds.
2. **`STATE_RULES_VERSION`.** It refuses a model fit under different state
   rules even when the feature *names* still match — the case that nearly
   shipped train/serve skew on 2026-09-01. **Bump it whenever the meaning of a
   `GameState` field changes, and refit.**
3. **Startup ordering.** `serve_live.py` loads the model before it binds a port
   or opens a file, so a bad version is a failure to start, not a failure at
   tip-off.

Old versions stay in `registry/`. `registry/v1` is deliberately refused at load
by current code and kept only for provenance — that refusal is itself tested.

## The container, when you want it

The same entry point; only the environment differs.

```bash
docker compose -f deploy/docker-compose.yml up --build
curl -s http://127.0.0.1:8808/health
```

The image contains the code and **not** the 527 MB of training data: it scores
games, it does not fit models. The registry is mounted read-only, because a
serving process has no business rewriting the model it serves. Rebuild the
ratings snapshot wherever the training data lives, and restart the container to
pick it up.

## Configuration

Every setting is an environment variable with a working default, so a bare
`python3 scripts/serve_live.py` does the right thing on a laptop.

| variable | default | |
|---|---|---|
| `CBBWP_MODEL_VERSION` | `v2` | which model serves |
| `CBBWP_REGISTRY` | `<root>/registry` | where artifacts live |
| `CBBWP_CONTEXT` | `<registry>/context_latest.json` | ratings snapshot |
| `CBBWP_LIVE_DIR` | `<root>/data/live` | JSONL output |
| `CBBWP_API_HOST` / `CBBWP_API_PORT` | `127.0.0.1` / `8808` | API bind |
| `CBBWP_API_HISTORY` | `240` | states kept per game in memory |
| `CBBWP_RATINGS_MAX_AGE` | `3` | days before the snapshot file is called stale |
| `CBBWP_FIXTURE_DIR` | unset | replay from disk instead of the network |

Every entry point prints its resolved settings at startup, and names which came
from the environment. A service quietly reading the wrong directory is the
outage that costs the most to diagnose.

## When something looks wrong

**A probability that looks absurd.** Pull the game and look at the states:
`python3 scripts/live_poller.py --game <id> --once`. It prints the last ten
states with margin and clock, and warns if either team is missing from the
ratings snapshot.

**`/health` says degraded.** Read `reason`. It distinguishes "the snapshot file
is old" from "the data behind the ratings is old" — different fixes.

**Suspect the feed changed.** Record and inspect:
`python3 scripts/record_espn_fixtures.py --limit 10` then
`python3 scripts/check_espn_fixtures.py`. Unknown play type ids are the signal.

**Reproduce a night offline.** Fixtures replay with no network at all:
`CBBWP_FIXTURE_DIR=tmp/fixtures python3 scripts/serve_live.py --once`.

**The poller stops following a game.** It gives up after 8 consecutive fetch
failures and logs it. Discovery rescans every 120s, so a game usually comes
back on its own; the JSONL will show the gap.

## Known limits, carried into production deliberately

- **The live path has never run against ESPN during a real game** (top of this
  file). It reaches the endpoint, parses real payloads, and has been rehearsed
  against a running clock via the replay server — but a replay cannot prove
  ESPN still sends that shape tonight.
- **ESPN's play order differs from the training data's** (EXPLAIN 8.8b, found
  2026-09-03, unresolved). Same plays, different order; hoopR's is
  chronological and ESPN's `sequenceNumber` is not. Effect on served
  probabilities is unmeasured.
- **No licensed spread.** The pregame term uses ratings built from scratch;
  a closing spread would close roughly 0.8 points of RMSE and is the single
  largest available gain.
- **The 40–20 minute bucket is 2–3× worse calibrated** than any other. Probably
  the shape of the pregame decay rather than its strength.
- **Late-game home advantage** is zeroed by the symmetry mirroring rather than
  flipped (EXPLAIN §8.1) — the largest single accuracy gain available.
- **The endgame table is built, tested and not wired in** (EXPLAIN §7.10). It
  missed its pre-registered bar at 0.40% against 1%.
- **No auth on the API**, by design. Do not expose it without a proxy.

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

Exit 0 = validated. Exit 1 = broken, do not go live. **Exit 2 = the offline
steps passed but ESPN was unreachable, so the live path is still unvalidated** —
which is the state the project is in today, because no environment available
during development could reach `site.api.espn.com`.

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
scripts/           the pipeline, the poller, the smoke test, the monitor
deploy/            macOS LaunchAgents, Dockerfile, compose
tests/             82 tests
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

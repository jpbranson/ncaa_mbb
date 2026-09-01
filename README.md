# cbbwp — college basketball live win probability

Live win probability for NCAA men's basketball. Trained on 2016–2023,
calibrated on 2024, tested on 2025–2026 (2.23M states, 12,398 games).
Beats ESPN's deployed model in every time bucket.

| Model | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|
| **LightGBM v1 (shipped)** | **0.3104** | 0.1008 | 85.17% | 0.0028 |
| Logistic baseline | 0.3109 | 0.1009 | 85.19% | 0.0043 |
| ESPN (deployed, same rows) | 0.3295 | 0.1061 | 84.58% | 0.0069 |

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
python3 scripts/publish_model.py v1    # pinned registry artifact
python3 scripts/evaluate.py            # metrics by time bucket vs ESPN
pytest tests -q
```

Seeds are pinned (`seed=20260831`, `deterministic=True`), so a refit on the
same machine reproduces `registry/v1` exactly.

**Memory note:** `fit_models.py` peaks around 4–6 GB — the symmetry mirroring
doubles 5.4M rows and briefly holds them as float64. It will be OOM-killed in a
3 GB container.

## Run it live

```bash
python3 scripts/build_live_context.py     # daily, before the slate
python3 scripts/live_poller.py            # follow tonight's games
python3 scripts/live_poller.py --date 20261115
python3 scripts/live_poller.py --game 401585555 --once     # smoke test
```

Output goes to stdout and to `data/live/wp_YYYYMMDD.jsonl`.

Before the first live night, on a machine that can reach ESPN:

```bash
python3 scripts/record_espn_fixtures.py --limit 5   # save real payloads
python3 scripts/check_espn_fixtures.py              # flag unknown play types
```

`check_espn_fixtures.py` is the one check the offline suite cannot do: it
reports play-type ids the model was never trained on. **A frequent unknown type
means the ESPN feed has changed and the model needs a refit, not a patched
adapter.**

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
  endgame.py       rule-based clamps the data cannot teach efficiently
  calibration.py   time-bucketed isotonic (diagnostic only — see EXPLAIN)
  monitor.py       calibration drift statistics
  adapters/
    hoopr.py       historical parquet -> Events   (offline)
    espn.py        live ESPN feed     -> Events   (live)
scripts/           the pipeline, the poller, the monitor
tests/             37 tests
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

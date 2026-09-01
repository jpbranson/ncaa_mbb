# CBB Win Probability — build log and re-entry point

**Last worked: 2026-09-01.** Phases 1–5 complete except a live smoke test against
the real ESPN endpoint, which is blocked from every sandbox available here.

**This folder is now the source of truth.** `~/Downloads/ncaa_mbb` on
cas-w7r21674vv holds the package, the data, the fitted model and these docs. The
project docs are a mirror; the folder is where the work happens.

Published results report (artifact, updatable by passing this URL as `url`):
https://claude.ai/code/artifact/e474963f-e992-4ed3-a15c-edd8c95ee6dd

## Headline result

Trained on 2016–2023, calibrated on 2024, tested on 2025–2026 (2.23M states, 12,398 games).

| Model | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|
| **LightGBM v1 (shipped)** | **0.3103** | 0.1008 | 85.19% | 0.0024 |
| Logistic baseline | 0.3109 | 0.1009 | 85.19% | 0.0043 |
| ESPN (deployed, same rows) | 0.3295 | 0.1061 | 84.58% | 0.0069 |

Beats ESPN in every time bucket; the gap is widest in the final minute
(0.1248 vs 0.1551).

**On the 2026-09-01 refit.** The model was rebuilt from scratch to verify the
seed pinning. It reproduced: 0.3103 vs the documented 0.3104, identical logistic
coefficients, identical bucket ordering, `best_iter` 149 vs 165. The residual
difference is thread scheduling on a different machine, not a code change.
`registry/v1` now carries sha `aaddca0d81606bc0`.

## What is where

```
~/Downloads/ncaa_mbb/
  README.md              setup, run order, layout, the two parity tests
  docs/                  these docs
  src/cbbwp/             the package
  scripts/               pipeline, live poller, fixture tools, monitor
  tests/                 39 tests, ~2s, all passing
  data/raw/              541 MB of hoopR parquet, 10 seasons
  data/proc/             games, team stats, 8.5M state rows
  data/live/             poller output, one JSONL per day
  registry/v1/           the pinned model + manifest
  registry/context_latest.json   today's ratings, for live games
  artifacts/             fitted model, eval predictions
  reports/               calibration monitor output
```

Two project docs are **not** mirrored here: `cbbwp-report-page.md` and
`cbbwp-report-data.md`. They are the source of the published results artifact,
not working files — they live in the project, which is where the artifact gets
updated from. `scripts/evaluate.py` regenerates every number in them.

## Restoring on another machine

```bash
pip3 install --break-system-packages polars pyarrow lightgbm scikit-learn pytest numpy
python3 scripts/fetch_data.py     # ~540 MB, about a minute
# then the run order in README.md
```

**`fit_models.py` needs ~6 GB of RAM.** The symmetry mirroring doubles 5.4M rows
and briefly holds them as float64. It was OOM-killed in the 3 GB local VM and had
to be run in a 7 GB container; the fitted artifact was copied back. Everything
else in the pipeline runs comfortably in 3 GB.

## Done this session (2026-09-01)

1. **Package restored** from `cbbwp-source.md` into the connected folder, all 10
   seasons downloaded, full pipeline re-run, `registry/v1` rebuilt and verified
   against the documented numbers.
2. **`src/cbbwp/adapters/espn.py`** — the live adapter. Maps play types by
   numeric `type.id` rather than display text (the text is what ESPN rewords),
   using a 29-entry table extracted from all ten training seasons. Sorts by
   `sequenceNumber` and renumbers 1..N to match hoopR's `game_play_number`.
   Deliberately bug-compatible: id 30558 stays `"Three Point Jump Shot"`, which
   the state builder does not treat as a possession flip, because that is how the
   model was fit.
3. **`src/cbbwp/live_context.py`** + `scripts/build_live_context.py` — pregame
   context for a game that has not been played. Same ridge fit, last season's
   ratings carried at 0.70. Verified: the 2027 preseason snapshot has rating sd
   13.8 (= 20.6 × 0.70), HCA falls back to the 3.4 prior, and Michigan hosting
   Duke comes out a 5.0-point favourite.
4. **`scripts/live_poller.py`** — asyncio, one task per live game, full replay
   every poll, cadence 20s → 10s → 5s as the clock runs, JSONL output. Tested
   end to end against fixtures: three concurrent games, correct final
   probabilities including an overtime game, clean exit.
5. **`tests/test_espn_adapter.py`** — the live adapter must produce identical
   states *and* identical win probabilities to the offline adapter on real games,
   with the payload shuffled on purpose. This is the live counterpart of the
   replay harness and the reason the poller can be trusted without ever having
   seen the live feed.
6. **`src/cbbwp/monitor.py`** + `scripts/calibration_monitor.py` — Phase 5 item
   16. Decile calibration by time bucket, alerting only when a gap is both
   statistically and practically significant.
7. **`scripts/record_espn_fixtures.py`** and **`check_espn_fixtures.py`** — the
   two commands to run on a networked machine before the first live night.

## Findings from this session

**The monitor's first honest bug was its own.** The first version computed
z-scores treating every state as an independent observation. The ~400 states in
one game share one outcome, so that inflates z by roughly √(states per game) —
about 6× here. It reported four alerts on the 2026 season, three of which were
artefacts of that. Corrected to cluster on games (the same "effective sample size
is games, not rows" principle the train/test split rests on), and the count drops
to one.

**The surviving alert is new and is not in EXPLAIN.md.** In the 40–20 minute
bucket the model says 0.783 where the observed rate is 0.760 — 2.4 points
over-confident on home favourites in the first half, over 2,995 games, z −3.1.
That bucket also has by far the worst ECE (0.0113 vs 0.0036–0.0059 elsewhere).
Worth investigating: it is the *opposite* direction to the documented §8.1
late-game flaw, which suggests the decaying pregame term may be mis-shaped rather
than simply too weak late.

**The documented §8.1 weakness reappeared, at the right size.** Late-game
deciles show +3.0 and +2.6 points under-predicted on the home side — the
direction and magnitude EXPLAIN.md records. On one season clustered by game it is
z ≈ +2.3, i.e. visible but not conclusive; the EXPLAIN evidence pools both test
seasons and cuts on tied games specifically, which is the sharper test.

## Findings that contradict the plan — do not redo these

1. **Post-hoc calibration makes it worse.** Time-bucketed isotonic (plan §9.3)
   cost 0.0008 log loss and raised ECE. Per-bucket Platt also failed. The raw
   model's residual bias (~+0.003, roughly constant across buckets, not
   time-dependent) is smaller than season-to-season noise. Shipped **without** a
   calibrator; the diagnostic stays in `scripts/calibrate_and_eval.py`.
2. **Monotone constraints improve accuracy.** Constrained 0.3104 vs
   unconstrained 0.3113 — they regularise. No trust-vs-accuracy tradeoff.
3. **LightGBM barely beats logistic** (0.0006). The hand-built
   `margin/sqrt(time)` term does almost all the work; the trees mainly help in
   the last minute. Logistic is a viable fallback.
4. **Endgame overrides slightly hurt log loss** — not because the rules are wrong
   but because ~0.2% of games have a self-contradictory feed. Those games are
   dropped at build time and overrides clip at 0.999 rather than 1.0.
5. **Ratings sanity check passed again on the refit:** HCA 3.2–3.8 in normal
   seasons, **2.53 in 2021**, the no-crowds season.

## Next up

1. **The live smoke test.** On a machine that can reach `site.api.espn.com`:
   `python3 scripts/record_espn_fixtures.py --limit 5` then
   `python3 scripts/check_espn_fixtures.py`, then
   `python3 scripts/live_poller.py --game <id> --once`. This is the only
   unvalidated link in the chain and it must happen before the first live night,
   not during it. Everything else is tested.
2. **Refresh the ratings for the 2027 season** once games start:
   `python3 scripts/build_live_context.py --season 2027`, daily. In the
   preseason it correctly falls back to 2026 ratings at 0.70 carryover.
3. **Investigate the 40–20 minute over-confidence** found above. Likely the
   shape of the pregame decay rather than its strength.
4. **Fix the late-game home advantage** (EXPLAIN §8.1) — the largest single
   accuracy gain available. Mirroring should flip a home-court term, not zero it.
5. **Possession-level endgame simulator** (plan §4.4 option 2) for the final 60
   seconds.
6. **Foul trouble / lineup state** — reachable from the same feed, unmodelled.
7. **A licensed spread** would close a ~0.8 point RMSE gap on the pregame term.

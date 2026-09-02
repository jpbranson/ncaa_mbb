# CBB Win Probability — build log and re-entry point

**Last worked: 2026-09-02 (session 4).** Phases 1–5 of the main model are
complete except a live smoke test against the real ESPN endpoint. The **endgame
simulator is finished and has been tested once. It does not ship** — see
`cbbwp-endgame-results.md`. All five pre-registered criteria were evaluated;
it cleared four.

Shipped model is **v2** (`registry/v2`, sha `2d4bf58134fa2e64`). v1 is kept for
provenance where it exists and is deliberately refused at load by current code.

**Two working copies exist and they have diverged.**

- `~/Downloads/ncaa_mbb` on **cas-w7r21674vv** (this file's copy) — has
  `registry/v1`, so all 72 tests run with none skipped. Session 4's endgame work
  was done here.
- `C:\Users\jpbra\Downloads\mbb_prob_claude` on **jpbranson-desk** — session 3's
  bit-identical rebuild. Lacks `registry/v1`, so one test skips there.

Neither is stale in a way that matters for the model: session 3 proved the
pipeline reproduces exactly across machines. Session 4's endgame work exists
only in the Mac copy and in the project docs. Copy `registry/v1/` between them
and they agree.

Published results report (artifact, updatable by passing this URL as `url`):
https://claude.ai/code/artifact/e474963f-e992-4ed3-a15c-edd8c95ee6dd

## Headline result

Trained on 2016–2023, calibrated on 2024, tested on 2025–2026 (2.23M states, 12,398 games).

| Model | Log loss | Brier | Accuracy | ECE |
|---|---|---|---|---|
| **LightGBM v2 (shipped)** | **0.3103** | 0.1008 | 85.20% | 0.0026 |
| Logistic baseline | 0.3108 | 0.1009 | 85.19% | 0.0042 |
| ESPN (deployed, same rows) | 0.3295 | 0.1061 | 84.58% | 0.0069 |

Beats ESPN in every time bucket; the gap is widest in the final minute
(0.1246 vs 0.1551).

## What is where

```
  README.md              setup, run order, layout, the parity tests
  docs/                  README-docs.md points at the project docs
  src/cbbwp/             the package
    endgame.py           rule-based clamps applied on the live path
    endgame_sim.py       the endgame solver and lookup (NOT wired into serve.py)
  scripts/               pipeline, live poller, fixture tools, monitor
    estimate_endgame_params.py       Phase 2, free throws / shots / rebounds
    estimate_endgame_possessions.py  Phase 2, possession outcomes and fouling
    build_endgame_table.py           Phase 3, backward induction + monotonicity
    validate_endgame_table.py        Phase 4, honesty check on 2024
    blend_endgame.py                 Phase 5, --tune then --test, once
  tests/                 72 tests, ~4s
  data/raw/              527 MB of hoopR parquet, 10 seasons
  data/proc/             games, team stats, 8.56M state rows
  data/live/             poller output, one JSONL per day
  registry/v2/           the pinned model + manifest
  registry/endgame/e1/   the endgame table, manifest, readable.csv (unused in serving)
  registry/context_latest.json   ratings snapshot, for live games
  artifacts/             fitted model, eval predictions, endgame parameters
  reports/               calibration monitor and endgame test output
```

Two project docs are **not** mirrored to the folder: `cbbwp-report-page.md` and
`cbbwp-report-data.md`. They are the source of the published results artifact,
not working files. `scripts/evaluate.py` regenerates every number in them.

## Restoring on another machine

The folder is rebuilt from `cbbwp-source.md` (the full source bundle) plus:

```bash
pip3 install --break-system-packages polars pyarrow lightgbm scikit-learn pytest numpy
python3 scripts/fetch_data.py     # ~527 MB, about a minute
python3 scripts/build_games.py
python3 scripts/build_team_stats.py
python3 scripts/build_dataset.py  # accepts optional season args
python3 scripts/fit_models.py     # needs ~6 GB RAM
python3 scripts/publish_model.py v2
```

**`fit_models.py` needs ~6 GB of RAM.** The symmetry mirroring doubles 5.4M rows
and briefly holds them as float64. It is OOM-killed in a 3–4 GB VM; run that one
step in a larger container and copy the fitted artifact back.

**Working inside the Cowork device VM:** each `device_bash` call gets its own PID
namespace, so a `nohup`-backgrounded job does **not** survive the call that
started it, and `pgrep` matches the harness's own command line (false positives).
Run every step synchronously inside one call, in season-sized chunks. Python
packages are not persistent between sessions — reinstall with pip at the start.
The endgame estimators are already written to work one season at a time for this
reason; concatenating all eight seasons of play-by-play is OOM-killed.

## Session 4 (2026-09-02) — the endgame simulator, built and declined

Full detail in `cbbwp-endgame-phase2.md` (the measurements) and
`cbbwp-endgame-results.md` (the table, the test, the verdict).

**The Phase 2 blocker was resolved and it was not a rule change.** The NCAA's
2025-26 men's rules changes touch nothing about team fouls, the bonus or the
one-and-one, so `BONUS_FOULS = 7` / `DOUBLE_BONUS_FOULS = 10` and `bonus_diff`
are correct. The 0.537 conversion on "1 of 1" free throws was **outcome-conditioned
selection**: ESPN labels a trip by the attempts actually TAKEN, so a *made*
one-and-one front end becomes a two-shot trip and never appears as "1 of 1" at
all. Splitting by and-one gives 0.694 for genuine single shots and 0.086 for the
rest — the rest being, by construction, missed front ends. Decensoring gives a
true first-shot rate of 0.700 (0.724 late), confirmed independently by
classifying trips on foul count in a season that has no such text.

**The table works and beats ESPN alone.** On 2024, out of sample, knowing the
score, clock, possession, fouls and free-throw ability but **nothing about how
good either team is**: log loss 0.1384 against ESPN's 0.1518 on the same rows.

**The blend does not clear the bar.** Tested once on 2025–2026: 0.124624 →
0.124130, a **0.40%** relative gain against a pre-registered 1%. Criteria 2–5 all
pass (ECE improves, monotonicity exact, handoff invisible at 0.0007, 0.000068 ms
per state). Per the plan's own rule, it does not ship.

**The plan's blend shape was also wrong.** It assumed the simulator's weight
should rise to 1.0 by 0:00. The model is at its *best* at 0:00 (0.0859 in the
last five seconds against the table's 0.1368) — the margin and clock have
already decided nearly everything. The tuned ceiling was 0.20, not 1.0.

**Why it failed:** the model already has every feature the table has, *and* team
strength, which the table deliberately lacks. LightGBM with monotone constraints
on 5.4M states has already learned most of the endgame's shape. Two things might
change that, neither a tweak: give the table team strength, or feed the table's
output to the model as a feature and refit.

**Incidental fixes from this session:**

- `build_team_stats.py` now emits `home_ft_pct` and `away_ft_pct` alongside
  `ft_pct_diff`. Only the difference was stored, which would have forced the
  endgame validation to assume both teams were average while the live path would
  not have. Backwards compatible: `ft_pct_diff` is unchanged, no refit needed.
- Both endgame estimators rebuild the game clock from `clock_minutes` /
  `clock_seconds` rather than a seconds-remaining column, because hoopR renames
  it (`start_half_seconds_remaining` in 2016–2022,
  `start_period_seconds_remaining` in 2023+). Picking one name fails silently on
  half the seasons.
- `tests/test_endgame_sim.py` (8 tests) pins the free-throw labelling convention
  and the table's structural properties. 72 tests, none skipped, ~4s.

## Findings that contradict the plan — do not redo these

1. **Post-hoc calibration makes it worse.** Time-bucketed isotonic (plan §9.3)
   cost 0.0008 log loss and raised ECE. Per-bucket Platt also failed. Shipped
   **without** a calibrator; the diagnostic stays in `scripts/calibrate_and_eval.py`.
2. **Monotone constraints improve accuracy.** Constrained 0.3104 vs
   unconstrained 0.3113 — they regularise. No trust-vs-accuracy tradeoff.
3. **LightGBM barely beats logistic** (0.0005). The hand-built
   `margin/sqrt(time)` term does almost all the work.
4. **Endgame overrides slightly hurt log loss** — ~0.2% of games have a
   self-contradictory feed. Those games are dropped at build time and overrides
   clip at 0.999.
5. **Ratings sanity check passes on every rebuild:** HCA 3.2–3.8 in normal
   seasons, **2.53 in 2021**, the no-crowds season.
6. **The endgame simulator does not clear its bar** (session 4, above). Do not
   rebuild it expecting a different answer without changing one of the two
   things named in `cbbwp-endgame-results.md`.
7. **Late free-throw shooting is better than the game average**, not worse:
   0.681 overall against 0.718 in the last minute, consistently across eight
   seasons. The folk belief is backwards.

## The possession bug (fixed in v2, kept here for the lesson)

ESPN typed made three-pointers as `"Three Point Jump Shot"` (id 30558) through
2019 and as `"JumpShot"` from 2021. The possession rule keyed on a whitelist of
play-type **names**, which had "JumpShot" but not the older spelling. Result:
**89% of every made three in 2016–2019 left the ball with the scoring team** —
324,043 plays, 4.8% of the feed.

Fixed by keying made field goals on the feed's `scoring_play` + `shooting_play`
flags instead of names. Verified across all ten seasons, and **changes zero rows
in 2024–2026**, so v1 and v2 are directly comparable on identical test data.

**It was worth almost nothing.** Overall log loss moved +0.00001; the gain is
confined to under 60s (−0.0002). Fixed anyway: a feature with two meanings inside
one training set is a defect regardless.

**Why nothing caught it.** The parity test compares the bulk path to the
reference path — both were wrong identically. The manifest guard compares feature
*names* — none changed. Every check compared the code to itself.
`tests/test_possession_truth.py` now asserts the invariant against the rules of
basketball instead, in every season.

**The near-miss this created.** After the fix, the folder briefly held code at
state-rules v2 and a model trained under v1. The feature names matched, so
`serve.py` loaded it happily. Fixed by `STATE_RULES_VERSION` in `schemas.py`,
stamped into every artifact at publish and checked at load. **Bump it whenever
the meaning of a GameState field changes.** The endgame table carries the same
stamp and the same test.

Session 4 found the same *shape* of bug twice more and both times it was benign:
foul type names (519/521) are stable across all ten seasons, and the free-throw
label change was a text-format change, not a rule change. The habit of checking
the sport rather than the code is what settled both.

## Next up

1. **The live smoke test.** On a machine that can reach `site.api.espn.com`:
   `python3 scripts/record_espn_fixtures.py --limit 5`, then
   `python3 scripts/check_espn_fixtures.py`, then
   `python3 scripts/live_poller.py --game <id> --once`. This is the only
   unvalidated link in the chain and it must happen before the first live night.
2. **Refresh the ratings for the 2027 season** once games start:
   `python3 scripts/build_live_context.py --season 2027`, daily. In the preseason
   it correctly falls back to 2026 ratings at 0.70 carryover.
3. **Investigate the 40–20 minute over-confidence.** That bucket's ECE is 2–3x
   every other bucket's, consistently. Likely the shape of the pregame decay
   rather than its strength.
4. **Fix the late-game home advantage** (EXPLAIN §8.1) — the largest single
   accuracy gain available. Mirroring should flip a home-court term, not zero it.
5. **Foul trouble / lineup state** — reachable from the same feed, unmodelled.
6. **A licensed spread** would close a ~0.8 point RMSE gap on the pregame term.
7. *(Optional, and only with a clear hypothesis)* the two routes that might make
   the endgame table earn its place, in `cbbwp-endgame-results.md`.

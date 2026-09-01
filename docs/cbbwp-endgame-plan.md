# Endgame simulator — plan and pre-registered success criteria

*Written 2026-09-01, BEFORE any simulator existed and before any result was
seen. The point of writing it first is that "we built it and did not ship it"
has to be a possible outcome, and it is only credible if the bar was set in
advance. This project has already reached that verdict twice — the calibrator
(EXPLAIN §7.7) and, in part, the endgame overrides (§7.8).*

## What is being built

Plan §4.4 option 2: for the final 60 seconds, explicitly simulate the remaining
possessions — intentional fouling, free throws, three-point attempts — and count
how often each team wins. Blend with the model, weighting the simulator to 1.0
by 0:00.

## The bar it has to clear

The shipped model is **already very good here**. In the 1–0 minute bucket it
scores 0.1248 log loss against ESPN's 0.1551 — 19.5% better, its widest margin
anywhere in the game. The simulator is not competing with nothing.

**Ship only if, on the 2025–2026 test seasons, touched once:**

1. **Log loss improves in the <60s bucket.** Target ≥ 1% relative
   (0.1248 → ≤ 0.1236). Below that, the added machinery is not worth its
   failure modes.
2. **Calibration does not get worse.** ECE in the <60s bucket must not rise, and
   `monitor.check` (clustered on games) must not gain an alert.
3. **Monotonicity holds.** No state where a made basket, gaining possession, or
   a better pregame rating lowers that team's win probability. Checked
   exhaustively across the lookup table, not sampled.
4. **The blend is invisible.** No discontinuity at the 60s handoff large enough
   to see on a chart — max |Δp| across the boundary < 0.02 at matched states.
5. **It stays fast enough to serve.** < 1 ms per state.

**If it clears 2–5 but not 1, it does not ship.** It becomes a documented
diagnostic, like `calibration.py`, and EXPLAIN gets a section saying so.

## Why it might fail, stated in advance

- The model may already have learned the endgame well enough that explicit rules
  add nothing but variance.
- The simulator depends on `timeout` and `possession` state that the pipeline
  only approximates (EXPLAIN §8.2, §8.3). Phase 1 addresses this.
- ~0.2% of feeds contradict themselves (§7.8). The overrides already lose
  ~0.0002 log loss to this; a simulator has far more surface area.
- Every parameter is estimated, so each one adds sampling error to states that
  are already sparse.

## Phases

| Phase | What | Gate |
|---|---|---|
| 0 | This document | — |
| 1 | Fix the possession and timeout inputs the simulator depends on | Inputs measurably correct |
| 2 | Estimate every simulator parameter from the same play-by-play | No hand-set constants |
| 3 | Pure-function simulator, then an exhaustive precomputed lookup table | Monotone by construction |
| 4 | Validate the simulator ALONE against observed frequencies | Beats nothing; just has to be honest |
| 5 | Blend, tune on 2024, test once on 2025–26 | The five criteria above |

## Design decision: a table, not a live simulation

The endgame state space is small and discrete — margin × seconds × possession ×
both bonus levels × timeouts × bucketed free-throw ability. So it is simulated
exhaustively **offline, once**, and shipped as a table.

Three reasons this beats simulating at serve time:

1. Serving becomes a lookup: microseconds, and no Monte Carlo noise in the
   output, so two identical states can never disagree.
2. Monotonicity can be **enforced across the whole table** rather than hoped for.
3. A person can read it. "Down 3, 12 seconds, ball, double bonus" is a row
   someone can check against their own judgement. A simulator is not reviewable
   that way.

The table is versioned in `registry/` alongside the model, for the same reason
the model is pinned.

# Endgame simulator — Phases 3–5, and the verdict

*The bar was written on 2026-09-01, before any simulator existed
(`cbbwp-endgame-plan.md`). The test below was run once, on 2025–2026, after the
blend was tuned on 2024. This document reports what happened.*

## Verdict: it does not ship

| # | Criterion | Result | |
|---|---|---|---|
| 1 | Log loss improves ≥1% relative in the <60s bucket | **0.40%** (0.124624 → 0.124130) | **fail** |
| 2 | Calibration does not get worse | ECE 0.004855 → **0.003663**; `monitor.check` alerts 0 → 0 | pass |
| 3 | Monotonicity holds, checked exhaustively | 0 margin violations, 0 possession violations across 1,660,725 states | pass |
| 4 | The blend is invisible at the 60s handoff | max \|Δp\| = **0.00069** (bar: 0.02) | pass |
| 5 | Fast enough to serve | **0.000068 ms**/state (bar: 1 ms) | pass |

The plan's rule was explicit: *"If it clears 2–5 but not 1, it does not ship. It
becomes a documented diagnostic, like `calibration.py`, and EXPLAIN gets a
section saying so."* That is what happens. The table stays in `registry/endgame/`
and is not wired into `serve.py`.

This is the third time this project has built something good and declined to
ship it — after the calibrator (EXPLAIN §7.7) and the endgame overrides (§7.8).
The bar existing in advance is the only reason that verdict is worth anything.

## Phase 3 — the table

Built by **backward induction**, not Monte Carlo. The plan asked for a
precomputed table; going one step further and solving exactly removes simulation
noise entirely, which matters for two of the criteria. Two identical states
cannot disagree, and a monotonicity violation becomes a statement about the
model rather than about how many samples were drawn.

State, from the point of view of the team **with the ball**: seconds remaining
(0–60), that team's margin (−12…+12), each team's team fouls (0–10), each team's
free-throw ability bucket (3 levels, at the terciles of team season FT%:
0.669 / 0.711 / 0.750). 27,225 states per second, **1,660,725 in all, solved in
1.1 seconds**, shipped as a 3.2 MB compressed array plus a readable CSV of
canonical rows.

Symmetry is structural rather than fitted. When possession changes, the value to
the team that just lost the ball is `1 − V[t′][−m, fd, fo, bd, bo]`, so the table
cannot disagree with itself about which side of a game it is describing.

What it says, for average free-throw teams with the opponent in the one-and-one:

| secs | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.011 | 0.014 | 0.228 | 0.337 | 0.595 | 0.817 | 0.888 | 0.994 |
| 10 | 0.083 | 0.145 | 0.296 | 0.388 | 0.583 | 0.744 | 0.836 | 0.938 |
| 30 | 0.160 | 0.218 | 0.333 | 0.416 | 0.565 | 0.717 | 0.791 | 0.882 |
| 60 | 0.180 | 0.241 | 0.351 | 0.444 | 0.566 | 0.694 | 0.772 | 0.860 |

Down three with the ball and ten seconds left is 0.145; down two is 0.296. The
gap between −2 and −3 at three seconds (0.228 against 0.014) is the whole
endgame in two numbers, and it is there because the measured shot mix says teams
down exactly three shoot a three 60% of the time while teams down two shoot one
32% of the time.

### What is deliberately absent

- **Timeouts**, which the plan listed in the state space. No separable effect was
  measured; the possession-length and foul-rate cells already average over how
  teams actually used them. Adding a dimension with an invented coefficient
  would put sampling error into sparse states for no measured gain.
- **Optimal play.** The fouling rule is observed behaviour, because the model
  predicts real games.
- **Team strength**, apart from free-throw ability. This is the design decision
  that ultimately explains the verdict — see below.

### Monotonicity, and two violations that are not bugs

Margin and possession monotonicity hold exactly. Two other orderings do not, and
both turn out to be correct basketball rather than solver error: raising the
opponent's foul count can *lower* the ball-holder's win probability by up to
0.021. Being fouled while leading late ends your possession — you shoot, and
give the ball back — whereas a foul with the opponent under seven team fouls
returns the ball to you with the clock stopped. More opponent fouls is not
monotonically good, and the table is right to say so. The plan's criterion named
margin, possession and pregame rating, all of which hold.

## Phase 4 — is the table honest?

Parameters re-estimated on **2016–2023 only**, then evaluated on 2024, which is
genuinely out of sample and is not one of the held-out test seasons.

| Last 60s of 2H/OT, 2024, 103,554 states | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|
| **Endgame table alone** | **0.1384** | 0.0387 | 94.71% | 0.0173 |
| ESPN (deployed), same rows | 0.1518 | 0.0429 | 94.40% | 0.0309 |

A table that knows the score, the clock, possession, the foul counts and how
well the two teams shoot free throws — and **nothing whatever about how good
either team is** — beats a deployed commercial model in the last minute. Phase 4
asked only for honesty; it got that and more.

Its one systematic flaw is under-confidence: states it calls 0.75 are won 0.85 of
the time, states it calls 0.145 are won 0.109 of the time. The table is too
generous to comebacks, most likely because possession lengths enter as means and
so allow slightly too many possessions to be squeezed in.

## Phase 5 — the blend

### The plan's blend shape was wrong

The plan said to weight the simulator "to 1.0 by 0:00". Tuning on 2024 falsified
that immediately. Log loss by bucket on 2024:

| window | model | table | best constant weight on the table |
|---|---|---|---|
| 0–5s | **0.0859** | 0.1368 | 0.10 |
| 5–10s | 0.1272 | 0.1390 | 0.20 |
| 10–20s | 0.1370 | 0.1458 | 0.35 |
| 20–30s | 0.1385 | 0.1460 | 0.35 |
| 30–45s | 0.1248 | 0.1314 | 0.25 |
| 45–60s | 0.1290 | 0.1341 | 0.25 |

The model is at its **best** at 0:00, not its worst: by then the margin and the
clock have decided nearly everything and there is nothing left for a possession
model to add. Forcing the simulator's weight to 1.0 there throws away the
model's strongest region. The weight schedule was therefore given a free
ceiling — `w(t) = w_max · (1 − (t/60)^γ)`, which is exactly 0 at the handoff, so
criterion 4 holds by construction rather than by tuning.

Tuned on 2024: γ = 6, **w_max = 0.20**, table log-odds scaled by α = 0.75,
β = 0.05. On the tuning season itself that gives 0.12084 against the model's
0.12199 — a 0.94% improvement **on the data it was tuned on**, already under the
1% bar.

### The single-shot test

2025–2026, 263,619 states in the last 60 seconds, run once from a config file
written before the test and hashed into the result.

| | log loss |
|---|---|
| Model alone | 0.124624 |
| **Blended** | **0.124130** (+0.40%) |
| Table alone | 0.140429 |
| ESPN, same rows | 0.154912 |

| bucket | n | model | blended | table alone |
|---|---|---|---|---|
| 0–10s | 74,374 | 0.11219 | **0.11126** | 0.14319 |
| 10–30s | 85,572 | 0.13713 | **0.13624** | 0.14901 |
| 30–60s | 101,586 | **0.12313** | 0.12328 | 0.13126 |

The gain is real but small, and it is not uniform: 30–60s gets very slightly
worse. 0.40% against a 1% bar is not close enough to argue about.

## Why it failed, and what would change it

The model already knows everything the table knows — margin, clock, possession,
bonus, free-throw ability are all features — **and also knows how good the two
teams are**, which the table does not. So the table's only possible contribution
is a better functional form for the endgame, and LightGBM with monotone
constraints, trained on 5.4 million states, has already learned most of that
shape. The 0.40% is what remains.

Two things would plausibly change the answer, neither of them a tweak:

1. **Give the table team strength.** It would stop being team-agnostic and the
   state space would grow, but it would remove the one piece of information the
   model has and the table does not.
2. **Feed the table's output to the model as a feature and refit**, instead of
   blending after the fact. That lets the trees use the structural estimate
   where it helps and ignore it where it does not, rather than applying one
   global weight — but it means a new model version and a new state-rules
   question, so it is not a Phase 5 change.

## What was worth doing anyway

- The free-throw censoring artifact was found and corrected
  (`cbbwp-endgame-phase2.md`), and is now pinned by a regression test that fails
  loudly if ESPN changes the convention again.
- `build_team_stats.py` now carries each team's free-throw percentage, not only
  the difference. Keeping only the difference would have forced the validation
  to assume both teams were average while the live path would not have — the
  same class of train/serve mismatch this project keeps finding.
- The table stands on its own as a diagnostic: it beats ESPN in the last minute
  knowing nothing about either team, which is a sharper statement about how much
  of endgame win probability is pure structure than any number in the model.

## Reproducing

```bash
for s in 2016 2017 2018 2019 2021 2022 2023 2024; do
  python3 scripts/estimate_endgame_params.py --season $s
  python3 scripts/estimate_endgame_possessions.py --season $s
done
python3 scripts/estimate_endgame_params.py --combine
python3 scripts/estimate_endgame_possessions.py --combine
python3 scripts/build_endgame_table.py                 # -> registry/endgame/e1
python3 scripts/validate_endgame_table.py              # Phase 4, on 2024
python3 scripts/blend_endgame.py --tune                # 2024
python3 scripts/blend_endgame.py --test                # 2025-2026, once
```

Both estimators refuse to read 2025 or 2026, and `validate_endgame_table.py`
refuses to validate on a season the table was fitted on.

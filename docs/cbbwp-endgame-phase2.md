# Endgame simulator — Phase 2: measured parameters

*Phase 2 of `cbbwp-endgame-plan.md`. The gate was "no hand-set constants."
Every number below is measured from hoopR play-by-play on the TRAINING seasons
only — 2016–2024. 2025 and 2026 are held out; both estimation scripts refuse to
read them, because the bar in the endgame plan is a single-shot test and stays
that way.*

Scripts: `scripts/estimate_endgame_params.py`, `scripts/estimate_endgame_possessions.py`
Outputs: `artifacts/endgame_params.json`, `artifacts/endgame_possessions.json`

## 1. The blocker: "1 of 1" free throws convert at 0.537

This was recorded as an open question and flagged as possibly a second instance
of the possession bug — a rule encoded as a constant while the sport changed
underneath it. **It is not.** It is a feed artifact, and the correct reading of
it produces a real correction to the free-throw parameters.

### It is not a rule change

The NCAA's 2025-26 and 2026-27 men's rules changes document lists ten approved
changes — shot clock, uniform accessories, continuous motion, flagrant fouls,
out-of-bounds, basket interference, technical fouls, fighting suspensions, coach's
challenge. **None touches team fouls, the bonus, the one-and-one, or how free
throws are awarded.** `BONUS_FOULS = 7` and `DOUBLE_BONUS_FOULS = 10` are correct
for 2026 and `bonus_diff` is not broken.

### It is a feed text change plus outcome-conditioned selection

Two separate things happened, and conflating them is what produced 0.537.

**First**, ESPN only began emitting `"free throw N of M"` text in the 2026 feed.
2024 and 2025 say plainly "makes/misses free throw" with no position. So "1 of 1
appears only in 2026" was a fact about the text format, not about the sport.

**Second — the part that matters — ESPN labels a trip by the attempts actually
TAKEN.** A one-and-one front end that is *missed* is one attempt, so it is
written `"1 of 1"`. A front end that is *made* earns a second shot, so the trip
becomes two attempts and is written `"1 of 2"` / `"2 of 2"`. **Every made front
end leaves the `1 of 1` bucket by construction.** Reading `scoring_play` on a row
labelled `1 of 1` conditions on the outcome.

Splitting the bucket by whether the shooter had just made a field goal at the
same game clock (an and-one) makes it unmistakable:

| `1 of 1` subtype | n | conversion |
|---|---|---|
| and-one | 13,958 | 0.694 |
| everything else — i.e. missed front ends | 4,859 | **0.086** |

An 8.6% free-throw percentage is not a shooting statistic; it is a censored
sample. Inside the last 30 seconds it falls to 0.039, because late fouling is
deliberate and almost every late one-and-one in the bucket is a missed front end.

### Decensoring recovers the true rates

Let `q` be the true first-shot conversion, `M` the missed front ends, `B` the
one-and-one trips and `T` the genuine two-shot trips. Then
`n(1of2) = T + qB` and `makes(1of2) = q(T + B)`, and the `B` terms cancel:

```
q = makes(1of2) / ( n(1of2) + M )
```

| Window (2026) | raw `1 of 2` | **true 1st shot** | 2nd shot | one-and-one trips | two-shot trips |
|---|---|---|---|---|---|
| Full season | 0.744 | **0.700** | 0.757 | 14,805 | 59,800 |
| Last 2:00 | 0.775 | 0.722 | 0.773 | 3,055 | 9,253 |
| Last 1:00 | 0.776 | **0.724** | 0.775 | 2,043 | 6,418 |
| Last 0:30 | 0.775 | 0.727 | 0.779 | 1,170 | 3,928 |

The solution is self-checking: predicted second-shot volume `T + qB` reproduces
the observed `2 of 2` count to within 0.02% in every window.

### Two independent confirmations

1. **The foul count agrees.** 94.9% of missed front ends occur at exactly 7–9
   opponent team fouls, which is the rule. The tails are feed noise.
2. **A training season with no such text agrees.** Classifying trips by foul
   count on 2023 — which has no `N of M` text at all — gives a last-minute front
   end of 0.720 and a bonus second shot of 0.775, against 0.724 / 0.775 from the
   2026 algebra. Two unrelated methods, agreeing to 0.004.

That second point is also the method the estimator uses, so nothing downstream
depends on the 2026 text.

### What to do with it

- The recorded 0.744 / 0.537 figures are selection artifacts. **Do not use
  either as a simulator parameter.** The corrected table below replaces them.
- `bonus_diff`, `BONUS_FOULS` and `DOUBLE_BONUS_FOULS` need no change.
- Free-throw trips are classified by **opponent team foul count**, never by the
  ESPN label, in every season.

## 2. Free throws (training seasons, last 60s of 2H/OT)

| Trip kind | shot 1 | shot 2 | shot 3 | n (shot 1) |
|---|---|---|---|---|
| One-and-one (opp fouls 7–9) | **0.7088** | 0.7692 | — | 21,834 |
| Two shots (double bonus or shooting foul) | **0.7182** | 0.7705 | — | 59,479 |
| And-one | 0.6794 | — | — | 4,491 |
| Fouled on a three | 0.7050 | 0.6999 | 0.7525 | 3,173 |

Full-game rates run 3–4 points lower (one-and-one first shot 0.681, two-shot
first 0.681), so **late free-throw shooting is better, not worse** — the opposite
of the folk belief, and consistent across all eight training seasons.

Late trip mix: 66.8% two-shot, **24.5% one-and-one**, 5.0% and-one, 3.6% fouled
on a three (n = 88,977 trips). A quarter of late trips being one-and-ones is why
the front-end rate has to be right.

## 3. Rebounding

| After | all game | last 60s |
|---|---|---|
| Missed final free throw | 0.1117 | **0.0869** |
| Missed three | 0.2739 | 0.3189 |
| Missed two | 0.3256 | 0.3764 |

Offensive rebounding rises late on field goals and falls on free throws — both
are what deliberate endgame play predicts.

## 4. Shot selection, by the shooter's own margin (last 60s)

| Offence's margin | 3PA share | 3P% | 2P% | n |
|---|---|---|---|---|
| −5 | 0.578 | 0.231 | 0.561 | 3,959 |
| −4 | 0.534 | 0.227 | 0.537 | 3,868 |
| **−3** | **0.604** | 0.217 | 0.546 | 3,891 |
| −2 | 0.377 | 0.257 | 0.465 | 3,367 |
| −1 | 0.319 | 0.226 | 0.417 | 3,185 |
| 0 | 0.361 | 0.234 | 0.428 | 3,392 |
| +1 | 0.297 | 0.302 | 0.458 | 1,736 |
| +3 | 0.275 | 0.312 | 1,402 | 1,402 |

Down exactly three, teams shoot a three 60% of the time; down one or two, 32–38%.
This is the behavioural structure the simulator needs, and it is measured rather
than assumed. Note the accuracy asymmetry: a trailing team's late threes go in at
0.22, a leading team's at 0.30 — late shots are harder because they are taken
under duress, so a simulator using season-average 3P% would be optimistic about
comebacks.

## 5. Possession outcomes and the intentional foul

`artifacts/endgame_possessions.json`, keyed by the **offence's** margin and
seconds remaining. `p_fouled_to_line` = the possession reached the line without
the offence taking a shot, which is what "they fouled to stop the clock" looks
like in a feed that does not label intent.

| Offence's margin | t = 0–9s | 10–19s | 20–29s | 30–39s | 50–59s |
|---|---|---|---|---|---|
| +1 | 0.241 | 0.314 | 0.350 | 0.312 | 0.245 |
| +2 | 0.329 | 0.357 | 0.367 | 0.333 | 0.234 |
| +3 | 0.180 | 0.343 | 0.374 | 0.334 | 0.294 |
| +5 | 0.140 | 0.300 | 0.341 | 0.359 | 0.328 |
| +8 or more | 0.082 | 0.166 | 0.198 | 0.236 | 0.250 |
| **−3** | **0.714** | **0.692** | 0.555 | 0.387 | 0.250 |

Two coaching strategies fall straight out of the feed without being told to:

- **Foul when trailing.** With the ball and up 1–5 inside 30 seconds, a team is
  sent to the line on a third of its possessions; up 8 or more, half that.
- **Foul up three.** A team *trailing by three* with the ball inside 10 seconds
  is fouled 71% of the time — the defence trading a free-throw trip for the
  chance to deny a tying three.

Mean possession length collapses from 13.8s (50–59s left) to 1.3s (0–9s left):
the endgame is a rapid foul/inbound cycle, not a sequence of normal possessions.
Turnovers end 28.5% of late possession-ending events. Mean gap from a made free
throw to the next foul is 8.7s, modal 7–9s.

## 6. Notes for Phase 3

- Simulate by **backward induction over the discrete state space**, not Monte
  Carlo: exact, so two identical states can never disagree, and monotonicity can
  be enforced rather than hoped for (endgame plan, "a table, not a live
  simulation").
- The fouling rule is **observed behaviour**, not optimal play. The model is
  predicting real games.
- `hoopR` column names drift between seasons — 2016–2022 carry
  `start_half_seconds_remaining`, 2023+ `start_period_seconds_remaining`. Both
  scripts rebuild the clock from `clock_minutes`/`clock_seconds`, which exist in
  every season, rather than picking a name and silently failing on the other.
- Free-throw ability enters as a team bucket; `ft_pct_diff` already exists in the
  feature set and is drawn from team box scores, unaffected by any of the above.

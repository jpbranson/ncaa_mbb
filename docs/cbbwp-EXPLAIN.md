# EXPLAIN.md — how the win probability model works, and why every choice was made

*Written for someone who has to stand up and explain this, and answer questions about it.*
Last updated: 2026-09-01 (evening) · Model version `v2` (`registry/v2`, sha `2d4bf58134fa2e64`)

---

## TL;DR

**What it does.** At any moment in a men's college basketball game, it outputs one number:
the probability the home team wins. It updates after every play.

**How it works, in one sentence.** We replayed ten seasons of play-by-play into 8.6 million
snapshots of "here is the score, the clock, who has the ball, and how good each team is,"
labelled each snapshot with who eventually won, and let a model learn the relationship.

**How good it is.** On two seasons it had never seen (2025 and 2026 — 12,398 games,
2.23 million snapshots), it scores **0.3103 log loss** against ESPN's own deployed model at
**0.3295** on exactly the same plays. It is better in every phase of the game, and the gap is
widest in the last minute (0.125 vs 0.155), which is where people are actually watching.

**The four claims to be ready to defend.**

1. *We are not using betting spreads.* The data source stops carrying them in 2024, so we
   build our own team ratings from scratch. They correlate 0.87 with the real spreads we can
   check against.
2. *We did not calibrate the output.* The standard recipe says to fit a correction curve on a
   held-out season. We tried it. It made the model worse, twice, in two different ways. We
   documented that and shipped without one.
3. *We forced the model to be monotonic* — more points can never lower your win probability.
   This usually costs accuracy. Here it **improved** accuracy.
4. *The fancy model barely beat the simple one.* Gradient boosting beat plain logistic
   regression by 0.0005. Almost all the signal is in one hand-built term, `margin ÷ √time`.

**The honest weakness.** In tied games in the last minute, the model under-predicts the home
team by about 3–5 percentage points. We know why (see §8.1) and we know the fix.

---

## Table of contents

1. [What the model actually predicts](#1-what-the-model-actually-predicts)
2. [Plain-language glossary](#2-plain-language-glossary)
3. [The pipeline, stage by stage](#3-the-pipeline-stage-by-stage)
4. [The features, one at a time](#4-the-features-one-at-a-time)
5. [How the model behaves — numbers you can quote](#5-how-the-model-behaves--numbers-you-can-quote)
6. [Results](#6-results)
7. [Every decision, and the alternative we rejected](#7-every-decision-and-the-alternative-we-rejected)
8. [Known weaknesses](#8-known-weaknesses)
9. [Questions you will get, and answers](#9-questions-you-will-get-and-answers)
10. [What is not built yet](#10-what-is-not-built-yet)

---

## 1. What the model actually predicts

### One row of data = one moment in a game

Every play in the feed produces one **game state**: a snapshot of the world right after that
play. Score 54–51, 4:12 left in the second half, home team has the ball, each team has three
timeouts left, and before tip-off we expected the home team to win by 6.

### The label is the same for the whole game

Here is the part that confuses people, so lead with it:

> A game the home team won produces ~400 rows, and **every one of them is labelled "home
> won"** — including the rows from when they were down 15.

The model never learns "this specific team came back." It learns from the *distribution*
across many games: of all the thousands of times a team was down 15 with 8 minutes left,
what fraction eventually won? That fraction is the probability.

This is why you cannot judge the model by looking at one game and saying "but they lost."
A 30% prediction is supposed to be wrong 70% of the time.

### The output is a curve, not a verdict

The deliverable is the whole path of the probability across the game, plus a live number for
a scoreboard. Nothing about it is a prediction of the final score.

---

## 2. Plain-language glossary

Everything mathematical in this document, defined once. No prior knowledge assumed.

### Basic terms

**Probability** — a number between 0 and 1. 0.70 means "70 times out of 100."

**Feature** — one input column. "Seconds remaining" is a feature. The model has 11.

**Label** — the answer we are teaching it. Here: did the home team win, 1 or 0.

**Model** — a formula, fit to data, that turns features into a probability.

**Training** — showing the model millions of examples and letting it adjust itself so its
answers match what actually happened.

**Overfitting** — when a model memorises the specific examples it was shown instead of
learning the general pattern. It looks brilliant on data it has seen and fails on new data.
The whole design of the train/test split (§3.6) exists to detect this.

### The two model types

**Logistic regression** — the simplest possible tool for a yes/no prediction. It multiplies
each feature by a weight, adds them up, and squashes the total into the 0–1 range. Think of
it as a weighted scorecard. Its output is smooth and every weight is readable, which is why
we use it as the benchmark: *anything more complicated has to beat it clearly to earn its
place.*

**Gradient boosting** (we use the LightGBM implementation) — builds hundreds of small
decision trees, where each new tree is trained to fix the errors the previous ones made.
A "decision tree" is just a chain of yes/no questions: *is the margin above 6? then is the
clock under 90 seconds? then...* Each tree is weak; hundreds of them stacked together are
strong. More accurate than logistic regression, much harder to read.

**Monotone constraint** — an instruction to the model that a relationship can only go one
direction. We tell it: *holding everything else fixed, increasing the home team's margin can
never decrease the home team's win probability.* Without this, random noise in the data
produces chart segments where a made basket lowers the scoring team's probability. That is
statistically defensible and completely fatal to anyone's trust in the chart.

**Ridge regression** — ordinary "fit a line through the points," plus a penalty that pulls
all the fitted values toward zero. The penalty is a defence against small samples: a team
that has played two games shouldn't get an extreme rating on that evidence. The strength of
the pull is a knob called **lambda (λ)**. Large λ = "I don't trust the data much, keep
everything near average." Small λ = "trust the data." We use this for team ratings (§3.3).

### Scoring the predictions

**Log loss** — the primary metric. For each prediction, take the probability you assigned to
what *actually happened*, and take its natural logarithm, negated. Average that over every
row. Lower is better; 0 would be perfect.

Why the logarithm? Because it punishes confident wrongness brutally. Saying 99% about
something that doesn't happen costs 4.6. Saying 60% about it costs 0.9. A metric that just
counted right-vs-wrong would let a model be recklessly overconfident for free.

For reference: always guessing 50% scores 0.693. Our model scores 0.310.

**Brier score** — the average of (predicted probability − outcome)². Same idea, gentler
punishment, easier to explain to non-technical people. Ours: 0.101.

**Accuracy** — the share of moments where the side we favoured actually won. Reported for
familiarity, but it is a *weak* metric here: it only asks whether we were on the right side
of 50%, not whether "80%" meant 80%.

**Calibration** — whether the probabilities are honest. Of all the moments where the model
said 70%, did the home team win about 70% of the time? This is a separate question from
accuracy. A model can rank every game perfectly and still be systematically overconfident
about all of them.

**Expected calibration error (ECE)** — calibration as a single number. Sort predictions into
20 buckets, compare each bucket's average prediction to what actually happened, average the
gaps (weighted by bucket size). Ours: 0.0028, i.e. about a quarter of a percentage point off
on average. ESPN's: 0.0069.

**Isotonic regression** — a way to *fix* calibration after the fact. It fits a flexible,
never-decreasing correction curve: "when the model says 0.62, the truth is really 0.65."
Flexible enough to fix odd-shaped errors, which also means flexible enough to fit noise.

**Platt scaling** — a simpler alternative: fit a tiny logistic regression to correct the
output. Only one or two parameters, so much less able to chase noise, and correspondingly
less able to fix complicated errors.

We tried both. See §7.7 — neither helped.

### The two ways these projects fail

**Leakage** — when information about the future accidentally reaches the model's inputs. The
classic version here: joining *end-of-season* team ratings onto a November game tells the
model who turned out to be good. Your test results look wonderful and production doesn't
match. Our defence is in §3.3.

**Train/serve skew** — when the code that builds features during training differs, even
slightly, from the code that builds them live. The model then receives subtly different
inputs in production than it was trained on, degrades silently, and no backtest will ever
show it. This is the single most common way sports models break. Our defence is in §3.9.

### Miscellaneous

**Parquet** — a file format for tables. Compressed, columnar, fast to query.

**Pure function** — a piece of code whose output depends only on its inputs. Same input,
same output, every time, with no hidden memory of what it did before. The state builder is
written as one, deliberately (§3.2).

**Monte Carlo simulation** — simulate a random process thousands of times and count how
often each outcome happens. Not currently used; it is the proposed approach for the last 60
seconds (§10).

---

## 3. The pipeline, stage by stage

```
     hoopR play-by-play parquet (10 seasons)
                    │
        ┌───────────┴────────────┐
        │                        │
  build_games.py           build_team_stats.py
  (as-of team ratings)     (as-of FT%, pace)
        │                        │
        └───────────┬────────────┘
                    │
             build_dataset.py
       state builder → feature builder
        (the SHARED code, §3.2, §3.5)
                    │
             8.6M labelled rows
                    │
              fit_models.py
     logistic baseline + LightGBM, split by season
                    │
            publish_model.py
        pinned artifact in registry/v1
                    │
                serve.py
       SAME state + feature builders, live
                    │
            adapters/espn.py
       live ESPN feed → the same Event objects
                    │
           scripts/live_poller.py
        one asyncio task per in-progress game
```

### 3.1 Data

**Source:** the hoopR / sportsdataverse mirror of ESPN play-by-play, read directly as parquet
files from `raw.githubusercontent.com`.

**Seasons:** 2016–2019 and 2021–2026. Ten seasons, 59,229 games.

- *Why not earlier than 2016?* The shot clock changed from 35 to 30 seconds in 2015. Games
  before that are a structurally different sport — more possessions, different comeback
  arithmetic. Including them would teach the model rules that no longer apply.
- *Why is 2020 missing?* The season was cut short by COVID mid-March, so it has no
  postseason and a truncated tail. 2021 *is* included and is genuinely useful — see §5.4.

**What's in it:** every made and missed shot, rebound, foul, turnover, timeout and period
marker, each with a clock, a score, and the team it belongs to. Also, usefully, **ESPN's own
win probability on every row from 2022 onward** — a deployed industry model's answer on the
exact games we score, for free.

**Licensing caveat to be aware of:** this data is scraped from ESPN, not licensed. Fine for
research and internal analysis. If this ever becomes public-facing or commercial, it needs
to move to a paid feed (Sportradar, SportsDataIO).

### 3.2 The state builder — turning plays into snapshots

`src/cbbwp/state.py`. It takes a game's full list of events and returns one snapshot per
event. It computes:

- **Time remaining**, converted to one number: seconds left in regulation. First half
  gets 1200 added to the clock; second half is the clock as-is. Overtime resets — each OT
  period is its own 300-second clock. (Rationale in §7.3.)
- **Margin** — home score minus away score. Always from the home team's point of view,
  everywhere in the system, without exception.
- **Possession** — who has the ball. The feed does not state this, so we derive it from
  a rules table: a made field goal or made free throw gives the ball to the other team; a
  rebound or a steal gives it to whoever is credited; a turnover gives it to the other team;
  a missed shot leaves it unresolved and we carry the previous value forward. It is unknown
  for only 8,096 of 2.2 million test rows (0.4%), all at tip-off before the first jump ball
  resolves.
- **Timeouts remaining** — start at 4 each, subtract team-charged timeouts, add one per
  overtime period. Official/TV timeouts do not count. This is an approximation of a fiddly
  rule (see §8.3).
- **Team fouls in the current half** — for bonus free-throw state. Resets once, at halftime;
  overtime continues the second-half count, which matches the men's rule.

**The design decision that matters: it is a pure function of the full event list.**

Live feeds retroactively correct themselves — a three becomes a two, a foul gets reassigned,
a duplicate event arrives. Rather than write code that tries to patch a running state, we
made the builder stateless: hand it every event, get every snapshot. Handling a correction is
then just *re-run the game from event zero*, which takes milliseconds.

This is tested directly: `test_replay_is_order_independent` shuffles the events and requires
identical output, and the replay harness (§3.9) feeds a game in irregular chunks and requires
bit-identical agreement.

### 3.3 Pregame team ratings — our replacement for the betting spread

**The problem.** The model needs to know how good each team is, especially at tip-off when
the score tells you nothing. The standard cheap answer is the closing betting spread. But
hoopR's spread column is populated for 2016–2023 and **empty from 2024 onward** — which is
exactly our test period. We checked: 0 of 6,136 games in 2025 have it.

**Options considered:**

| Option | Why we rejected it |
|---|---|
| Use the spread where it exists, our own rating elsewhere | The feature would mean different things in training and test. Unacceptable — that *is* train/serve skew. |
| Buy KenPom (~$25/yr) | Would work, but introduces a paid dependency and a scraping step for a project that doesn't otherwise need one. Still an option later. |
| Use NCAA NET / Barttorvik | Both blocked by this sandbox's egress policy; also need careful as-of snapshotting to avoid leakage. |
| **Build our own** ✓ | Self-contained, free, no leakage by construction, and checkable against the real spreads we *do* have for 2016–23. |

**How ours works.** Ridge regression predicting final score margin. Each game contributes one
equation:

```
margin  =  (rating of home team)  −  (rating of away team)  +  home-court advantage
```

with the home-court term switched off for neutral-site games. Solve across all games at once
and you get a rating per team, in points. A team rated +8 beats an average team by 8 on a
neutral floor.

**The three choices inside it:**

1. **λ = 0.5** (the shrinkage strength). Tuned by comparing against the real closing spreads
   on 2016–2023. Results:

   | λ | Home-court advantage | Correlation w/ Vegas | RMSE vs actual margin |
   |---|---|---|---|
   | 0.5 ✓ | 3.78 | **0.865** | **12.04** |
   | 1.0 | 4.13 | 0.862 | 12.06 |
   | 4.0 | 5.39 | 0.827 | 12.45 |
   | 16.0 | 6.83 | 0.720 | 13.33 |
   | 40.0 | 7.52 | 0.587 | 14.02 |

   Note the diagnostic in the second column: over-shrinking the ratings doesn't just make them
   worse, it *pushes team strength into the home-court term* until home advantage reads as an
   absurd 7.5 points. That's how we caught the first version being badly mis-tuned.
   For reference, Vegas closing spreads get RMSE 11.19 against actual margins. We get 12.04.

2. **Refit every 7 days**, using only games that finished *strictly before* the game date.
   This is the anti-leakage guarantee: on 15 January the model knows only what happened
   through 14 January. Weekly rather than daily is purely a compute choice — the ratings barely
   move in a day, and it cuts the number of fits by 7×.

3. **Carry 70% of last season's final rating into the next season as the starting prior.**
   In November a team has played two games; the data alone can't rate anyone. The prior is
   what carries them. 0.70 is the conventional year-over-year regression-to-the-mean factor
   for college basketball, reflecting roster turnover.

**The sanity check that convinced us it works.** The fitted home-court advantage by season:

| 2016 | 2017 | 2018 | 2019 | **2021** | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| 3.79 | 3.23 | 3.71 | 3.44 | **2.53** | 3.19 | 3.78 | 3.35 | 3.49 | 3.20 |

Every normal season lands between 3.2 and 3.8, which matches the published literature. **2021
— the season played in empty arenas — comes in at 2.53.** Nothing in the code knows about the
pandemic. The model found the crowd effect on its own. That is the single best piece of
evidence that the ratings are measuring something real.

### 3.4 Team shooting and pace

`build_team_stats.py`. Two more as-of-date numbers per game:

- **Free-throw percentage**, season-to-date, for each team. Blended with a league-average
  prior of 70% worth 40 attempts, so a team that has taken six free throws isn't rated at
  100%.
- **Combined scoring rate** of the two teams (points per minute), same as-of and
  prior-blended treatment. This is a cheap proxy for pace: two slow teams with four minutes
  left have fewer chances to change the score than two fast ones.

Both use a cumulative sum *shifted by one game*, so a game never contributes to its own
inputs. Same anti-leakage discipline as the ratings.

### 3.5 The feature builder

`src/cbbwp/features.py` turns a snapshot into 11 numbers. Full detail in §4.

Two things about this file specifically:

- **It is imported by both the training script and the live server.** There is exactly one
  definition of every feature in the system. This is the answer to train/serve skew, and it
  is the most important structural decision in the whole project.
- It has a **vectorised twin** written in Polars, because building 8.6 million rows one at a
  time in Python is too slow. Two implementations of one definition is a risk, so
  `tests/test_parity.py` asserts they produce identical numbers on 25 real games, to within
  1e-9. If someone edits one and not the other, CI fails.

### 3.6 Training

**The split — by season, never randomly:**

```
train        2016–2023   5,440,218 rows   37,828 games
calibration  2024          882,142 rows    6,141 games   (held out)
test         2025–2026   2,233,937 rows   12,398 games   (touched once, at the end)
```

*Why this matters more than it sounds.* Two consecutive snapshots from the same game are
nearly identical rows with identical labels. If you split rows at random, near-duplicates land
in both training and test, and your test score measures memorisation, not generalisation.
It is the most common way one of these projects quietly produces a fake result.

A corollary worth internalising: **the effective sample size is games, not rows.** 8.6 million
sounds enormous. It is really about 56,000 independent observations. Any confidence interval
must be computed on that basis.

**Symmetry mirroring.** Every training row is duplicated with the two teams swapped: flip the
sign of margin, expected margin and timeout difference, flip who has the ball, flip the label.
This doubles the training set to 10.9 million rows for free. It forces the model to treat the
two sides identically except through terms that are genuinely home-specific. It costs nothing
and stabilises the fit. (It also has a side effect we pay for — see §8.1.)

**Sampling.** We keep every snapshot inside the final five minutes, and every third one before
that. Early-game states are enormously redundant — nothing changes between play 40 and play 41
of a 20-point game. This cuts the dataset roughly in half with no measurable cost, and
concentrates it where the action is.

**Data-quality gate.** About 0.23% of games have a play-by-play feed that contradicts itself:
the last play's score doesn't match the final result, or the feed is truncated. Those games
are dropped from training *and* test. This isn't cherry-picking — a live system would flag
these games too — and §7.8 explains what they were breaking.

**The model.** LightGBM, 11 features, 165 trees, learning rate 0.05, 160 leaves, minimum 1,500
rows per leaf, with monotone constraints on the five features where direction is not
negotiable (margin, both margin-over-time terms, possession, and both pregame terms).
Trained on log loss. **No class rebalancing and no sample weighting** — those distort
probabilities, and honest probabilities are the entire product.

All random seeds are pinned (`seed=20260831`, `deterministic=True`), so refitting reproduces
the shipped artifact exactly rather than approximately. The fitted model itself is not stored
in the project docs — it is 2.7 MB of text — so the seed pinning is what makes `registry/v1`
recoverable. It now also lives on disk in this folder.

### 3.7 Calibration — and why there isn't any

The plan called for isotonic calibrators fit separately per time bucket on the held-out 2024
season, on the theory that miscalibration in these models is mostly time-dependent.

We built it, measured it, and did not ship it. Full detail in §7.7.

### 3.8 Endgame overrides

The model learns statistics; it does not know the rules. Two rules are imposed on top:

1. **Clock at zero.** If time has expired in the final period and someone is ahead, the answer
   is 1 or 0 — that isn't a probability, it's a fact. Tied at 0:00 goes to 0.5, because
   overtime is coming.
2. **Mathematically decided.** If the trailing team cannot score enough points in the
   possessions that physically remain, clamp to certainty. We assume a maximum of one
   possession per 6 seconds and 3 points per possession — deliberately generous to the
   trailing team, so the clamp only fires when a comeback is genuinely impossible.

Both clip at 0.999 rather than 1.000. §7.8 explains why that last detail matters.

### 3.9 Serving, and the defence against train/serve skew

`src/cbbwp/serve.py` is deliberately thin. It:

- loads a **pinned** model artifact from `registry/v1` — an immutable directory with a
  manifest recording the version, the exact feature list, a content hash, and the seasons it
  was trained on;
- **refuses to start** if the feature list in the manifest doesn't match the feature list the
  current code produces. If someone edits `features.py` without refitting, the service stops
  rather than silently feeding the model the wrong inputs;
- calls the *same* `build_states` and `build_feature_matrix` the training pipeline used;
- re-scores the whole game from event zero on every poll.

**Three tests tie it together.**

- **The replay harness**, `tests/test_replay_harness.py`. It takes a completed game, feeds it
  through the live path in random-sized chunks as though it were arriving in real time, and
  requires that every state's answer matches the offline path exactly. It also shuffles the
  events and requires the same result.
- **Bulk parity**, `tests/test_parity.py`. The fast vectorised Polars path must agree
  row-for-row with the canonical state builder.
- **Adapter parity**, `tests/test_espn_adapter.py`. The *live* ESPN adapter must produce
  identical states — and identical win probabilities — to the *offline* hoopR adapter for the
  same game, including when the feed arrives shuffled. See §3.10.

Current status: 92 tests, all passing, none skipped, about 6 seconds.

### 3.10 The live path (added 2026-09-01)

**`src/cbbwp/adapters/espn.py`** turns ESPN's live `summary` payload into the same `Event`
objects the historical adapter emits. Three details carry the weight:

- **Play type is mapped by numeric id, not by text.** hoopR stores ESPN's `type.text`
  verbatim, and the possession rules in `state.py` are written against those exact strings.
  ESPN occasionally rewords a play type's display text but almost never renumbers its
  `type.id`. So the adapter maps id → *the text the model was trained on*, and falls back to
  the feed's own text only for an id that has never appeared in training. `TYPE_ID_TO_TEXT`
  was extracted from all ten seasons: 29 distinct (id, text) pairs.
- **It no longer decides possession.** The map originally preserved a bug: id 30558,
  `"Three Point Jump Shot"`, was not in the made-shot whitelist. That whitelist has been
  removed entirely — possession now keys on the feed's scoring/shooting flags (§8.2), so a
  future ESPN rename cannot reintroduce the failure. The map survives for every *other*
  rule, where the trained-on text is still what matters.
- **Plays are ordered by `sequenceNumber` and renumbered 1..N**, exactly as hoopR's
  `game_play_number` is dense and 1-based. The live feed does not promise ordered plays; the
  adapter sorts, and the parity test shuffles the payload on purpose to prove it.

**`src/cbbwp/live_context.py`** solves the problem that a game which has not been played has
no row in `games.parquet`. `scripts/build_live_context.py` snapshots today's team ratings
(the same ridge fit, with last season's ratings carried over at 0.70) plus season-to-date FT%
and pace into `registry/context_latest.json`; the provider turns that into a `PregameContext`
for any pair of team ids. An unknown team id is not an error — it is a first-time opponent or
a non-D1 side, and rating 0.0 (league average) is the right prior for a team we know nothing
about.

**`scripts/live_poller.py`** is an asyncio loop, one task per in-progress game. It re-scores
the whole game from event zero on every poll, which is what makes ESPN's retroactive
corrections a non-event. Poll cadence tightens as the clock runs: 20s early, 10s inside five
minutes, 5s inside two. A scoreboard task rediscovers the slate every two minutes. Output is
one JSONL line per changed state.

**What is validated, and what is not.** On 2026-09-02 the adapter reached the real endpoint
for the first time and parsed real ESPN summary payloads correctly — 539 plays to 539 states,
no unknown play-type ids. On 2026-09-03 the whole deployment was rehearsed against a *running
clock*: six archived games replayed through `scripts/replay_server.py`, which speaks ESPN's
protocol back to an unmodified `serve_live.py` over real HTTP. 172 states, all six reproducing
the real final score, an overtime game scored through a third period. That covers the growing
plays array, the status transitions and the endgame cadence — everything except ESPN itself,
which only a real night can supply. What *is*
proven, offline and in CI, is that ESPN-shaped payloads rebuilt from hoopR rows produce
byte-identical states and win probabilities to the offline path on real games. Since hoopR is
itself a scrape of this same ESPN feed, that covers ordering, numbering, type mapping and
field coercion. It does **not** cover a field ESPN sends today that the rebuild omits. That
gap is closed by `scripts/record_espn_fixtures.py` (save real payloads on a machine that can
reach ESPN) and `scripts/check_espn_fixtures.py`, which reports any play-type id the model was
never trained on. **Run those two before the first live night.**

### 3.11 The calibration monitor (added 2026-09-01)

`src/cbbwp/monitor.py` and `scripts/calibration_monitor.py`. A win probability model fails
quietly: log loss barely moves when the model starts saying 0.80 to situations that win 0.74,
and that gap is the whole product. So the weekly check is not "is the loss still good" but
"does what we say still match what happens", in deciles, within each time bucket.

An alert requires **both** conditions:

- *statistical* — the gap is bigger than sampling noise (|z| > 3);
- *practical* — the gap is bigger than anyone would care about (> 2 percentage points).

Requiring both is the point. A million rows will make a 0.3-point gap "significant"; that is
a large sample, not drift. The monitor runs against a backtest window (`--source backtest`)
or against the poller's own output joined to final results (`--source live`), and exits 1 on
an alert so cron can mail it.

---

## 4. The features, one at a time

Eleven inputs, in the order the model sees them. The "importance" column is *gain* — how much
of the model's total predictive power comes from splits on that feature.

| # | Feature | Plain meaning | Importance |
|---|---|---|---|
| 1 | `margin` | Home score minus away score | 32.8% |
| 2 | `sqrt_time` | √(seconds remaining) | 0.1% |
| 3 | `margin_per_sqrt_time` | margin ÷ √time | 28.2% |
| 4 | `possession` | 1 home, 0 away, 0.5 unknown | 0.1% |
| 5 | `pregame_exp_margin` | Expected home margin from ratings | 6.7% |
| 6 | `pregame_exp_margin_decayed` | The same, faded out as the clock runs | 6.2% |
| 7 | `is_ot` | In overtime? | 0.0% |
| 8 | `timeout_diff` | Home timeouts minus away | 0.0% |
| 9 | `bonus_diff` | Bonus free-throw advantage, −2 to +2 | 0.0% |
| 10 | `ft_pct_diff` | Home season FT% minus away | 0.6% |
| 11 | `margin_per_sqrt_points_left` | margin ÷ √(expected points still to be scored) | 25.2% |

### The three that do all the work

**`margin`, `margin_per_sqrt_time`, and `margin_per_sqrt_points_left` account for 86% of the
model.** The other eight share the remaining 14%.

**Why divide by the square root of time?** This is the one piece of real mathematics in the
model and it's worth being able to explain.

A basketball game's remaining scoring is roughly a random walk — a sequence of small
independent swings. For a random walk, the *uncertainty* in where you end up grows with the
**square root** of how many steps are left, not with the number of steps. Twice as much time
left means only about 1.4× as much uncertainty, not 2×.

So `margin ÷ √time` is, in effect, *how many standard deviations ahead you are.* A 6-point
lead with 30 minutes left and a 6-point lead with 30 seconds left are numerically identical
and practically opposite; dividing by √time is what tells them apart. Feed the model margin
and time as separate columns and it cannot easily discover this. Hand it the ratio and even
plain logistic regression works well.

`margin_per_sqrt_points_left` is the same idea with a better denominator: instead of raw
seconds, it uses how many points the two teams are expected to score in the time remaining,
based on their season pace. Four minutes between two fast teams contains more basketball than
four minutes between two slow ones. The model uses both versions; together they beat either
alone.

### Why the pregame rating appears twice

Feature 5 is the raw expected margin; feature 6 is the same number multiplied by
`√time ÷ √2400`, so it starts at full strength and fades to zero at the buzzer.

The reasoning: **at tip-off the rating is the only information you have; by the final minute
the score has already absorbed everything the rating was telling you.** A model that keeps
weighting the rating late becomes stubborn about upsets — it keeps insisting the favourite
will win after the game has demonstrated otherwise.

Including both the decaying and the non-decaying version lets the model choose how fast to
fade rather than having us impose it. It chose to keep a modest permanent effect, and §5.3
shows that decision is supported by the data.

### The features that are nearly inert

`possession`, `timeout_diff`, `bonus_diff` and `is_ot` contribute almost nothing to the gain
number. Be ready for this question, because it looks bad.

The explanation: gain is a *whole-dataset average*, and these features only matter in the last
two minutes, which is a small slice of the rows. Possession is worth 2.8 percentage points in
a tied game with 20 minutes left and **12 points with 10 seconds left** (§5.1). That is a
large effect confined to a small region, which is exactly what a whole-dataset average hides.

They stay in for three reasons: they cost nothing, they are needed for the endgame simulator
that comes next, and removing them would make the final-minute numbers worse in precisely the
place people scrutinise hardest.

---

## 5. How the model behaves — numbers you can quote

These are the model's actual outputs, and they are the most persuasive thing in this
document, because they match what any basketball person already believes.

### 5.1 What is possession worth?

Tied game, evenly matched teams:

| Time left | Home has ball | Away has ball | Swing |
|---|---|---|---|
| 20:00 | 52.0% | 49.2% | 2.8 pts |
| 10:00 | 52.2% | 49.2% | 3.0 pts |
| 5:00 | 52.4% | 48.1% | 4.3 pts |
| 2:00 | 54.0% | 47.0% | 7.0 pts |
| 0:30 | 56.1% | 44.3% | 11.7 pts |
| 0:10 | 56.1% | 44.1% | 12.1 pts |

Having the ball is nearly irrelevant early and decisive late. Nobody taught it that.

### 5.2 What is a lead worth?

Home win probability, evenly matched teams, possession unknown:

| Time left | +1 | +3 | +5 | +10 |
|---|---|---|---|---|
| 40:00 (tip) | 54.6% | 58.4% | 63.2% | 74.9% |
| 20:00 | 55.8% | 62.0% | 67.5% | 81.7% |
| 10:00 | 57.0% | 64.5% | 74.3% | 90.2% |
| 5:00 | 57.1% | 69.8% | 80.8% | 96.7% |
| 2:00 | 61.3% | 79.5% | 91.4% | 99.4% |
| 1:00 | 64.9% | 86.3% | 96.2% | 99.9% |
| 0:30 | 69.9% | 93.4% | 98.5% | ~100% |

Read across any row and the lead gets more valuable; read down any column and it gets more
valuable as the clock runs. A 10-point lead is worth 75% at tip and 97% with five minutes left.

### 5.3 Does the pregame rating fade correctly?

Tied game, home team a 10-point favourite:

| Time left | Model says | Even matchup for comparison |
|---|---|---|
| 40:00 | 81.3% | 50.6% |
| 20:00 | 74.4% | 50.9% |
| 10:00 | 68.7% | 51.0% |
| 2:00 | 64.1% | 50.6% |
| 0:30 | 62.5% | 50.1% |

The pregame edge shrinks from a 31-point advantage to a 12-point one — but it does not
vanish.

**The obvious objection: "62.5% for a tied game with 30 seconds left is too stubborn."**
We checked it directly against reality. Tied games under one minute in the test seasons,
grouped by pregame edge:

| Home pregame edge | Games | **Actual** home win rate |
|---|---|---|
| worse by 8+ | 513 | 31.0% |
| worse by 3–8 | 982 | 53.0% |
| even (±3) | 2,059 | 58.4% |
| better by 3–8 | 1,961 | 55.2% |
| better by 8+ | 2,161 | 65.6% |

Big favourites really do win 66% of tied endgames, and big underdogs really do win 31%. The
model's persistence is empirically correct, not stubbornness.

### 5.4 Sanity checks the model passed

- Home-court advantage of 2.53 points in the 2021 no-crowds season versus 3.2–3.8 in every
  other season (§3.3).
- Only **0.06%** of home scoring plays produce a visible drop in home win probability, and
  those are cases where the clock legitimately outran the comeback. (Roughly 3% show a
  technical drop, but the median size is 0.0001 — invisible.)
- Curves for four real overtime thrillers were plotted and inspected by eye. They track ESPN
  closely through regulation and separate late. The vertical move at the regulation buzzer is
  real: each of those games was tied at 0:00, so probability correctly snaps back to a coin
  flip.
- The ratings snapshot built for live use puts Michigan, Duke, Arizona, Florida and Indiana
  at the top of the 2026 season — a plausible list produced by a model that knows nothing
  about team names.

---

## 6. Results

Test set: 2025 and 2026, 12,398 games, 2,233,937 snapshots, none of which the model saw.

| Model | Log loss | Brier | Accuracy | Calibration error |
|---|---|---|---|---|
| **LightGBM v2 (shipped)** | **0.3103** | **0.1008** | 85.20% | **0.0026** |
| Logistic baseline | 0.3109 | 0.1009 | 85.19% | 0.0043 |
| ESPN (deployed) | 0.3295 | 0.1061 | 84.58% | 0.0069 |

Broken out by time remaining — never report the average alone, because it hides the failure
mode:

| Time remaining | Snapshots | Ours | Logistic | ESPN | Ours vs ESPN |
|---|---|---|---|---|---|
| 40–20 min | 823,955 | 0.4416 | 0.4418 | 0.4708 | −6.2% |
| 20–10 min | 393,654 | 0.3399 | 0.3405 | 0.3538 | −3.9% |
| 10–5 min | 212,904 | 0.2642 | 0.2648 | 0.2715 | −2.7% |
| 5–2 min | 393,658 | 0.2125 | 0.2128 | 0.2190 | −3.0% |
| 2–1 min | 148,020 | 0.1553 | 0.1553 | 0.1638 | −5.2% |
| **1–0 min** | 261,532 | **0.1246** | 0.1267 | 0.1551 | **−19.7%** |

Log loss is highest early. That is correct and expected — early in a game you genuinely know
less, and a model that claimed otherwise would be lying.

**Calibration.** Across 20 buckets on 2.23 million test rows, predicted and observed sit
within half a percentage point of each other almost everywhere.

---

## 7. Every decision, and the alternative we rejected

### 7.1 Build our own team ratings instead of using a spread
**Because** the data source stops carrying spreads in 2024, which is the test period.
**Rejected alternative:** spread-where-available, ratings-elsewhere — that makes one feature
mean two different things, which is train/serve skew by another name.
**Cost:** our ratings get RMSE 12.04 against actual margins; real closing spreads get 11.19.
A licensed line source is a free upgrade if one ever becomes available.

### 7.2 Split by season, not randomly
**Because** consecutive snapshots from one game are near-duplicates with the same label.
Random splitting puts near-duplicates on both sides and measures memorisation.
**Consequence to state out loud:** effective sample size is ~56,000 games, not 8.6M rows.

### 7.3 Overtime resets its own clock
**Because** the alternative — counting overtime as "negative time remaining" or extending
regulation — makes `margin ÷ √time` meaningless exactly where it matters most.
**Consequence:** a tied game at 0:00 in regulation outputs ~50% rather than 0 or 1, which is
correct, and the model gets an `is_ot` flag so it can distinguish "5 minutes left in a 40-minute
game" from "5 minutes left in overtime."

### 7.4 Mirror every row (symmetry augmentation)
**Because** it doubles the data for free and forces the model to treat both teams identically
except through genuinely home-specific terms.
**Cost:** it also suppresses a real late-game home advantage. See §8.1 — this is the model's
main known flaw, and it traces directly to this decision.

### 7.5 Monotone constraints on margin, possession and the rating terms
**Because** a chart where a made basket lowers the scoring team's probability destroys trust
instantly, however defensible it is statistically.
**The surprise:** we expected to pay for this. We measured it and the constrained model is
*better* — 0.3104 versus 0.3113 unconstrained (both measured at v1, the fit where
the ablation was run; the comparison is between the two, not against today's headline). The constraints act as regularisation: they
stop the trees fitting noise in directions we know are wrong. **There is no
trust-versus-accuracy tradeoff here.** This is worth mentioning; most people assume there is.

### 7.6 Ship LightGBM even though logistic regression nearly matches it
**The numbers:** 0.3103 vs 0.3109 overall. In the final minute, 0.1246 vs 0.1267 — a 1.7%
improvement in the place people scrutinise most.
**Why LightGBM anyway:** it wins where it counts, it is better calibrated (0.0026 vs 0.0043),
and it is the foundation for the extra features still to come.
**Be honest about it:** the hand-built `margin ÷ √time` term is doing almost all the work.
If the gradient boosting ever becomes an operational burden, the logistic model is a
defensible fallback and you would lose very little.

### 7.7 Ship *without* a calibrator
This one contradicts the plan and the textbook, so here is the full evidence.

| Variant | Log loss | Calibration error |
|---|---|---|
| **Raw model** | **0.3104** | **0.0028** |
| + time-bucketed isotonic | 0.3116 | 0.0043 |
| + per-bucket Platt scaling | 0.3109 | 0.0041 |

*(This ablation was run on the v1 fit; all three rows come from it, so the
comparison between them stands. The shipped v2 raw model scores 0.3103 / 0.0026
— the possession fix moved the headline, not the conclusion.)*

Both corrections made *both* metrics worse. The diagnosis: the theory says miscalibration
here is time-dependent — overconfident early, underconfident late — so per-bucket correction
should help. We measured the bias per bucket and it isn't time-dependent at all. It's a
near-constant +0.003 in every bucket:

| Bucket | Bias on calibration season | Bias on test seasons |
|---|---|---|
| 40–20 min | +0.0047 | +0.0010 |
| 20–10 min | +0.0039 | +0.0008 |
| 10–5 min | +0.0026 | +0.0015 |
| 5–2 min | +0.0028 | +0.0009 |
| 2–1 min | +0.0036 | +0.0033 |
| 1–0 min | +0.0032 | +0.0017 |

The bias is smaller than the season-to-season variation in it, so a curve fit on 2024 fits
2024's noise and transfers that noise onto 2025–26.

**Why the raw model is already calibrated:** it was trained on log loss (which is directly a
calibration-sensitive objective), with symmetry mirroring (which cancels systematic
directional bias), with monotone constraints (which suppress noise-fitting), and with no
class rebalancing or sample weighting (both of which would have distorted probabilities). The
recipe that usually creates the need for a calibrator was avoided upstream.

**What we did instead of deleting the code:** `src/cbbwp/calibration.py` and
`scripts/calibrate_and_eval.py` remain in the repo as a standing diagnostic. If a future
refit does drift, the measurement is one command away — and the weekly monitor (§3.11) is
what will tell you to run it.

### 7.8 Endgame overrides clip at 0.999, not 1.0
Applying the overrides at full certainty made log loss slightly *worse*. Investigating:

- 31,784 test rows have the clock at zero. **50 of them have a label that contradicts the
  final play's score** — a broken or truncated feed.
- 53,251 rows are mathematically decided. **11 of them are contradicted by the outcome.**

The rules aren't wrong; the data is, about 0.2% of the time. And asserting *certainty* on a
broken row is catastrophically expensive under log loss — one wrong 1.000 costs as much as
about 30,000 ordinary rows.

Two fixes, both applied: those games are now dropped at dataset build time, and the overrides
clip at 0.999 so a bad row costs half as much.

**Why keep the overrides at all, given they cost ~0.0002 log loss?** Because a scoreboard
showing 99.4% when the game is arithmetically over looks broken to every viewer. That is a
fair price for never displaying an impossible number.

### 7.9 Skip 2020, keep 2021
2020 was truncated by COVID with no postseason. 2021 was played, and its anomalous
home-court advantage turns out to be a *feature* — it is the cleanest evidence the ratings
model is measuring something real.

### 7.10 Keep every snapshot in the last 5 minutes, sample every third one before
Early-game states are enormously redundant. Halves the dataset, concentrates it where the
action is, no measurable cost.

### 7.11 Map ESPN play types by numeric id, not by display text
**Because** the rules for fouls, timeouts, rebounds and turnovers are written against the
exact strings hoopR stored, and those strings are ESPN's display text — the thing most likely
to be reworded. The numeric `type.id` is the stable key.
**Since corrected:** possession no longer depends on play-type names at all (§8.2). The rename
this decision anticipated had already happened *inside the training data* before anyone looked.
**Rejected alternative:** normalising both sides (strip spaces, lowercase) so "Jump Shot" and
"JumpShot" match. That silently changes what the model receives the moment ESPN introduces a
type whose normalised form collides with an existing one, and it hides feed changes instead
of surfacing them.
**Consequence:** an unrecognised id falls through to the feed's own text, which the state
builder treats as "carry possession" — the safe default — and
`scripts/check_espn_fixtures.py` exists specifically to make that situation loud.

### 7.12 Require both statistical *and* practical significance in the monitor
**Because** at 2.2M rows almost any gap is statistically significant, and an alert that fires
every week is an alert nobody reads.
**Rejected alternative:** alert on log loss crossing a threshold. Log loss is exactly the
metric that does *not* move when calibration drifts — which is the failure we are watching
for.

### 7.13 Version the MEANING of state, not just the list of features
**Because** `serve.py` compared feature names and nothing else, so the possession fix — which
changed what `possession` means without touching any name — would have been served silently
against a model trained on the old meaning. That is the exact failure the manifest exists to
prevent, walking straight past it.
**The fix:** `STATE_RULES_VERSION` in `schemas.py`, stamped into every artifact at publish time
and checked at load. An unstamped artifact is treated as version 1. `registry/v1` is stamped 1
and is now *refused* by current code, which is correct, and a test asserts it.
**The general lesson:** every check in this system compared the code to itself — two
implementations, a manifest, a replay harness. All passed. The bug was visible only by
comparing the code to the rules of basketball. **Parity proves consistency, not correctness**,
and it is easy to mistake one for the other.

---

### 7.10 Build the endgame simulator, then decline to ship it

The plan (§4.4 option 2) called for explicitly simulating the last 60 seconds and blending
that with the model. It was built, in full, and it is **not in the serving path**.

The bar was written first, in `cbbwp-endgame-plan.md`, before any code existed, because
"we built it and did not ship it" is only a credible outcome if the threshold was set in
advance. Five criteria; the first was a 1% relative log-loss improvement inside 60 seconds.
Tested once on 2025-2026: **0.40%**. Criteria 2-5 all passed — calibration improved (ECE
0.00485 to 0.00366, no new monitor alert), monotonicity is exact across all 1,660,725
states, the 60-second handoff moves probabilities by at most 0.0007, and a lookup costs
0.000068 ms. The plan's rule was that clearing 2-5 but not 1 means it does not ship, so it
does not ship.

Three things came out of it that are worth more than the blend would have been.

**The table is a sharp measurement in its own right.** On 2024, out of sample, knowing the
score, the clock, possession, both foul counts and how well the two teams shoot free
throws — and *nothing at all* about how good either team is — it scores 0.1384 against
ESPN's 0.1518 on the same rows. That is a statement about how much of endgame win
probability is pure structure, and it is worth more as a diagnostic than as a 0.4% blend.

**A free-throw statistic that looked like a rule change was a censoring artifact.** "1 of 1"
free throws convert at 0.537, which was flagged as a possible second instance of the
possession bug — a rule encoded as a constant while the sport moved. It was not. ESPN labels
a free-throw trip by the attempts actually *taken*, so a **made** one-and-one front end earns
a second shot and is written "1 of 2"/"2 of 2", while a **missed** one stays a single attempt
and is written "1 of 1". Every made front end leaves the bucket by construction. Reading
`scoring_play` off that label conditions on the outcome. Decensoring gives a true first-shot
rate of 0.700 — confirmed independently by classifying trips on team foul count in a season
whose feed has no such text at all. `tests/test_endgame_sim.py` now fails loudly if the
convention changes again. Full working in `cbbwp-endgame-phase2.md`.

**The plan's blend shape was wrong, and the data said so immediately.** The plan assumed the
simulator's weight should rise to 1.0 by 0:00. The model is at its *best* at 0:00 — 0.0859 in
the last five seconds against the table's 0.1368 — because by then the margin and the clock
have decided nearly everything and there is nothing left for a possession model to add. The
tuned ceiling was 0.20.

The reason the blend cannot do better is structural rather than fixable by tuning: the model
already has every feature the table has, **and** team strength, which the table deliberately
lacks. Monotone-constrained LightGBM on 5.4M states has already learned most of the endgame's
shape. Two changes might alter that — giving the table team strength, or feeding the table's
output to the model as a feature and refitting — and both are new model versions, not tweaks.

## 8. Known weaknesses

Be forthright about these. Knowing them is the difference between defending the model and
being caught out by it.

### 8.1 Late-game home advantage is under-predicted — the biggest one

In tied games in the last minute, the model is consistently 3–5 percentage points low on
the home team:

| Situation | Snapshots | Model says | Actually happened |
|---|---|---|---|
| Tied, 0–15s left, home ball | 2,552 | 57.5% | 62.5% |
| Tied, 0–15s left, away ball | 2,460 | 49.7% | 53.1% |
| Tied, 0–30s left, home ball | 3,320 | 58.1% | 63.1% |
| Tied, 30–60s left, home ball | 1,398 | 59.0% | 62.7% |

**Cause:** symmetry mirroring (§7.4) forces the model to be symmetric between the teams, and
the only home-specific channel left is the pregame rating — which we deliberately fade toward
zero as the clock runs. So by the final minute the model has no way to express a home
advantage. But one exists: home teams shoot free throws better in front of their own crowd,
and foul calls skew slightly.

**Scale of the problem:** it is confined to close, late states. Global calibration error is
still 0.0028. But it is precisely the situation people watch most closely, and the direction is
consistent across every slice we checked.

**Fix:** add a non-decaying home-court term that mirroring leaves alone (mirroring should flip
it, not zero it), or drop mirroring in favour of an explicit home indicator. Not yet done.

### 8.2 Possession is inferred, not observed
The feed doesn't state who has the ball. Our rules table is good but not perfect — a made
free throw that is 1-of-2 briefly flips possession to the wrong team until the next event
corrects it. We deliberately do **not** look ahead at future events to resolve this, because
at live time the future doesn't exist and using it would create train/serve skew. Affects a
small number of rows; more relevant late than early.

**This section previously called the next item a rare gap. It was not rare, and it is now
fixed — the story is worth keeping because of how well it hid.**

ESPN typed made three-pointers as `"Three Point Jump Shot"` (id 30558) through 2019 and as
`"JumpShot"` (id 558) from 2021 onward. The possession rule keyed on a whitelist of play-type
NAMES containing "JumpShot" but not "Three Point Jump Shot". So for four of the seven training
seasons, **89% of every made three left the ball with the team that had just scored** — 324,043
plays, 4.8% of the entire feed.

The consequence was subtler than a wrong value: `possession` *meant something different* in
2016–2019 than from 2021 on. The model trained on four seasons of one definition and three of
another, and was tested on a period using the second.

**Why nothing caught it.** `test_parity.py` asserts the bulk path agrees with the reference
builder — it passed, because both were wrong identically. The manifest guard compares feature
*names*, and no name changed. Every check in the system compared the code to itself. Nothing
compared it to the rules of basketball.

**The fix** (state rules v2) keys made field goals on the feed's own `scoring_play` and
`shooting_play` flags. Across all ten seasons that rule drops nothing the whitelist caught and
adds exactly the missing threes. `tests/test_possession_truth.py` now asserts the invariant
directly — after a made field goal the other team has the ball, in every season — plus a
regression test that renames every play type and requires possession to be unchanged.

**What it was worth: almost nothing, and that is the honest headline.** The test seasons
contain no 30558 rows, so v1 and v2 are comparable on identical test data:

| Bucket | v1 (buggy) | v2 (fixed) | Δ |
|---|---|---|---|
| overall | 0.31031 | 0.31032 | +0.00001 |
| 2–1 min | 0.1553 | 0.1553 | −0.0000 |
| 1–0 min | 0.1248 | 0.1246 | **−0.0002** |
| under 30s | 0.1258 | 0.1255 | **−0.0003** |
| under 10s | 0.1124 | 0.1122 | **−0.0002** |

The gain is confined to exactly where possession matters, is consistent across all three late
buckets, and appears in the logistic baseline too (0.12668 → 0.12655 under 60s). The logistic
`possession` coefficient rose from 0.068 to 0.077 — with a cleaner feature the model leans on
it more. But the magnitude is two ten-thousandths of a nat: **the model had largely learned to
work around the corrupted feature.**

It was still right to fix, for reasons that are not about log loss. The endgame simulator
depends on possession being correct in a way the model does not, and a feature that means two
different things inside one training set is a defect whatever the metric says.

### 8.3 Timeouts are approximate
The men's rule is fiddly: a team that doesn't use its 60-second timeout in the first half
loses it, media timeouts absorb team timeouts under some conditions. We model it as
"4 each, minus charged timeouts, plus one per overtime." Given that `timeout_diff` contributes
essentially nothing to the model, this approximation is not currently costing anything — but
it would need tightening before an endgame simulator relies on it.

### 8.4 The pregame ratings are weaker than a real betting line
12.04 RMSE versus 11.19 for closing spreads. Roughly 0.8 points of expected margin worse. It
matters most at tip-off and decays from there.

### 8.5 Non-Division-I teams are in the ratings pool
Early-season "buy games" against non-D1 opponents are included, and those opponents get their
own (very negative) ratings. This is the right treatment — it stops the home-court term
absorbing the mismatch — but it means the rating distribution has a long low tail and its
standard deviation (~20 points) is not comparable to a D1-only rating system like KenPom.

### 8.6 Data provenance
ESPN-scraped, not licensed. Fine for research; a legal problem for a public or commercial
product. Coverage is also uneven — low-major and early-season non-conference games sometimes
have no play-by-play at all. A graceful fallback that runs on score and clock alone is not yet
built.

### 8.7 The comparison to ESPN is fair, but state its limits
ESPN's numbers come from the same feed and are correctly aligned (we verified: at the final
play ESPN's favourite is the eventual winner 99.6% of the time, and its probabilities track
margin sensibly). But ESPN's model is optimised under live production constraints we don't
face, and it visibly hedges on blowouts — where it says 3.4% the true rate is 1.7%. Part of
our edge is simply being willing to be more confident where the data supports it. That is a
legitimate win, not a trick, but it is the honest characterisation.

### 8.8b The adapter was re-sorting a correctly ordered feed (found and fixed 2026-09-03)

Found while building `scripts/serve_viz.py`: a chart makes an ordering problem
visible in a way a test of final scores does not.

**The bug.** `events_from_plays` sorted ESPN's plays by `sequenceNumber`, on the
stated assumption that the array promised nothing about order and the id was
authoritative. Measured on seven archived games, the reverse is true on all
seven:

| | result |
|---|---|
| raw `plays` array is chronological | **7 / 7** |
| raw array order equals hoopR's `game_play_number` order | **7 / 7** |
| `sequenceNumber` order is chronological | 0 / 7 |

`sequenceNumber` is *nearly* monotonic and not reliably so — the 2026
championship payload has 12 inversions in 482 plays, e.g. `120416951` followed
by `120416904` while the clock runs correctly forwards. Sorting on it took
correct data and shuffled it.

**What it cost.** Displacements as large as 986 seconds, one of them stepping
from period 2 back into period 1, and mid-game win probability moving by up to
**26.7 points**. Final probabilities were identical in every game, which is
exactly why nothing ever complained: a displaced play perturbs the path and
washes out by the buzzer.

**Why every existing test missed it.** The parity tests run on payloads rebuilt
from hoopR by `tests/espn_fixtures.py`, which synthesised `sequenceNumber` as
`game_play_number * 10` — perfectly monotonic. The fixture handed the key a
property the real feed does not have, so sorting by it was harmless there and
destructive in production. This is the second time a rebuilt fixture has quietly
guaranteed a green tick (see the run book on step 6); the pattern is worth
remembering — **a fixture that is cleaner than reality tests the fixture.**

**The fix.** The feed's array order is authoritative; the sort is gone.
Disorder is now *reported* rather than repaired, via
`espn.chronological_inversions()`: an unreliable key cannot fix an out-of-order
feed, it can only corrupt an ordered one. The rebuilt fixtures now carry an
occasional inverted `sequenceNumber` of their own, so reintroducing the sort
fails four tests immediately.

**Verified after the fix.** On the seven archived games, states built from real
ESPN payloads are now identical to states built from hoopR — seq, period,
`game_seconds_remaining`, margin *and possession* — with zero chronological
inversions. That is the train/serve invariant this project rests on, holding on
real feed data rather than on reconstructions of it.

### 8.8 The live adapter has not seen ESPN during a real game
Corrected twice, and the caveat is now much narrower than it started.

2026-09-02: the adapter read the real endpoint and parsed real payloads (§3.10), retiring
"never touched ESPN". 2026-09-03: the growing-feed case — a partial plays array between polls,
a status that is neither scheduled nor final — was rehearsed by replaying archived games back
to the unmodified deployment over real HTTP (`scripts/replay_server.py`). Six games, 172
states, every final score reproduced, one of them through overtime.

What is left is the part a rehearsal cannot reach: **ESPN itself, tonight.** A replay proves
the code handles the archive. It cannot prove ESPN has not changed the feed since, and it does
not exercise ESPN's retroactive corrections — a play inserted, rescored or deleted minutes
later. The poller is immune to those by construction (it re-scores the whole game every poll),
which is the argument for why that gap is small, not evidence that it is closed.

That first run also found something the old caveat was hiding. The 403 everyone read as
blocked egress was partly ESPN's edge refusing the client's own user-agent, which would have
hit the Mac in November with no sandbox to blame. See `cbbwp-deployment.md`. The general
lesson is the one worth keeping: an environmental excuse for a failure is a hypothesis, not
a diagnosis, and it stops being tested the moment it sounds sufficient.

---

## 9. Questions you will get, and answers

**"How is this different from just looking at the score and the clock?"**
It mostly *is* that — and that's the point. 86% of the model's power comes from margin, time,
and their ratio. The remaining 14% is knowing how good the teams are, who has the ball, and
the free-throw arithmetic. Anyone who claims a win probability model is doing something
mysterious is overselling it.

**"Why is it only 85% accurate?"**
Accuracy is the wrong question. It counts every moment of every game, including tip-off,
where the correct answer is close to a coin flip. A model that scored 99% "accuracy" would be
lying about how much is knowable. The right question is calibration: when it says 70%, does
it happen 70% of the time? Ours is off by about a quarter of a percentage point.

**"It said 94% and they lost."**
It should. If everything the model calls 94% happened, the model would be broken — it would
mean it was systematically under-confident. Roughly 1 in 17 of those should lose.

**"Why should I trust this over ESPN's?"**
On 2.23 million plays neither model was fit on, ours scores better on every metric and in
every phase of the game, and its probabilities are more honest. That said — see §8.7 for the
fair characterisation of *why*.

**"You beat ESPN — did you accidentally leak the future?"**
The three places leakage could enter, and what blocks each:
(a) team ratings — refit weekly using only games completed before each game's date;
(b) team shooting/pace stats — cumulative sums shifted one game, so a game never feeds itself;
(c) the train/test split — by season, so no game appears on both sides, and 2025–26 were
scored exactly once, at the end.

**"Why not use the betting spread? It's better and it's free."**
We would, if we had it. The data source stops carrying it in 2024. Mixing a real spread into
training and a substitute into serving would be worse than using the substitute consistently.

**"You didn't calibrate it. Isn't that the standard step?"**
It is, and we did it, and it made things worse — measurably, in both directions, with two
different methods. §7.7 has the table. The model is already calibrated because of how it was
trained; adding a correction fit on one season just imports that season's noise.

**"Why constrain the model? Doesn't that hurt accuracy?"**
That's the usual assumption and it is wrong here. Constrained scores 0.3104, unconstrained
0.3113 — both from the v1 fit where the ablation was run. The constraints stop the trees
fitting noise in directions we already know are wrong.

**"Why does possession barely matter?"**
Because that number is a whole-game average, and possession only matters late. In a tied
game it's worth 2.8 percentage points with 20 minutes left and 12 points with 10 seconds left.

**"Would more data help?"**
Probably not much. We have 56,000 games and the simple model nearly matches the complex one,
which is the signature of being near the information ceiling of these *features*. Better
*features* would help — a real spread, foul trouble, lineup state — more than more rows.

**"What happens if ESPN changes its feed mid-season?"**
The adapter maps plays by numeric id, so a reworded label changes nothing. A genuinely new
play type falls through to the feed's own text and is treated as "carry possession", which is
the safe default rather than a wrong guess — and `check_espn_fixtures.py` reports it. The
honest response to a frequent new type is a refit with that type present, not a patched
adapter, because the model has never seen it.

**"How would you know if the model stopped working?"**
The weekly monitor (§3.11), which checks whether stated probabilities still match observed
outcomes by decile within each time bucket. Log loss would not tell you — that is the point.

**"What would make it meaningfully better?"**
In order: fixing the late-game home advantage (§8.1), a possession-level endgame simulator for
the last 60 seconds, foul trouble and lineup state, and a licensed betting line.

**"Can this be used for betting?"**
Nothing here was built or validated for that. It is not benchmarked against closing lines for
profitability, it has no vig model, and its pregame input is measurably worse than the market's.
Beating a *broadcast* win probability model and beating a *market* are different problems.

---

## 10. What is not built yet

1. **A live smoke test against the real ESPN endpoint.** The adapter, the poller and the
   context provider are written and tested offline; nothing has touched the live feed,
   because both sandboxes are blocked from it. Two commands on a networked machine close
   this (§3.10).
2. **Possession-level endgame simulator.** For the final 60 seconds, explicitly simulate the
   remaining possessions — intentional fouling, free-throw percentages, three-point attempts
   — thousands of times and count how often each team wins. Encoding the rules beats learning
   them from sparse data. Blend it with the model output, weighting the simulator to 100% by
   0:00.
3. **The late-game home advantage fix** (§8.1). The largest single accuracy gain available.
4. **Foul trouble and lineup state.** Whether a team's best player is on the bench with four
   fouls. Reachable from the same feed.
5. **A graceful fallback** for games with no play-by-play — score and clock only.
6. **A licensed feed**, if this ever leaves internal use (§8.6).

---

## Appendix: file map

| File | What it holds |
|---|---|
| `src/cbbwp/schemas.py` | The data contracts. `Event`, `GameState`, `PregameContext`, the feature list, the rules constants. |
| `src/cbbwp/state.py` | The state builder. Pure function, events → snapshots. |
| `src/cbbwp/features.py` | The feature builder. One definition, used by training and serving. Plus its vectorised twin. |
| `src/cbbwp/ratings.py` | As-of-date team ratings (ridge regression). |
| `src/cbbwp/live_context.py` | Pregame context for a game that has not been played yet. |
| `src/cbbwp/adapters/hoopr.py` | Historical parquet → `Event` objects, and the bulk path. |
| `src/cbbwp/adapters/espn.py` | Live ESPN feed → the same `Event` objects. Type mapping by numeric id. |
| `src/cbbwp/calibration.py` | Time-bucketed calibrators. Built, measured, not shipped. Kept as a diagnostic. |
| `src/cbbwp/endgame.py` | The two rule-based overrides. |
| `src/cbbwp/evaluate.py` | Metrics, always broken out by time bucket. |
| `src/cbbwp/monitor.py` | Calibration drift statistics. |
| `src/cbbwp/serve.py` | The live scoring path and the version-pinning guard. |
| `scripts/` | The pipeline, the live poller, the fixture tools, the replay server, the monitor. |
| `src/cbbwp/config.py` | Every deployment setting, from the environment, with working defaults. |
| `src/cbbwp/api.py` | The read-only HTTP view of the live feed. Standard library only. |
| `scripts/serve_live.py` | The deployment entry point: poller plus API, one process. |
| `scripts/smoke_live.py` | The eight-step pre-flight check for a live night. |
| `scripts/replay_server.py` | Serves archived ESPN games back as a live feed, so the deployment can be rehearsed out of season. See §3.10. |
| `scripts/archive_replay_games.py` | Archives the real ESPN payloads that the replay server serves. |
| `deploy/` | macOS LaunchAgents, Dockerfile, compose. Same entry point either way. |
| `src/cbbwp/endgame_sim.py` | The endgame solver and lookup table. A documented diagnostic; **not** wired into `serve.py` — see §7.10. |
| `registry/endgame/e1/` | The solved table, its manifest and a readable CSV of canonical states. |
| `tests/` | 92 tests: state rules, feature contract, bulk parity, ESPN-adapter parity, endgame rules, replay harness, the dry-run replay server, monitor statistics, endgame-table structure and the ESPN free-throw labelling convention. |
| `registry/v2/` | The pinned model artifact and manifest, stamped with the state-rules version. |
| `registry/v1/` | The pre-fix model, kept for provenance. Refused at load by current code. |
| `registry/context_latest.json` | Today's team ratings and season-to-date stats, for live games. |

---

*Keep this file current. When a number in it changes, change it here too — this document is
the one people will read.*

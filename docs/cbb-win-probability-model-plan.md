# Real-Time College Basketball Win Probability Model
## Design and Production Plan

*The original plan, kept verbatim as the reference document. Where the build
departed from it, `cbbwp-EXPLAIN.md` §7 records the departure and the evidence.*

---

## 1. What we are building

A system that, at any moment during a live college basketball game, outputs a number between 0 and 1: **the probability that the home team wins**.

It updates after every play. It needs to know the score, how much time is left, who has the ball, how many timeouts each team has, and how good each team is.

The output is a curve — win probability over the course of the game — plus a live number you can put on a scoreboard or a dashboard.

---

## 2. Glossary

Terms used throughout this document, defined once here.

| Term | Meaning |
|---|---|
| **Play-by-play (PBP)** | A structured log of every event in a game: made shots, fouls, rebounds, timeouts, each with a timestamp and score. This is the raw material for everything below. |
| **Game state** | A snapshot of the game at one moment — score, clock, possession, timeouts, etc. One play-by-play event produces one game state. |
| **Feature** | A single input column to the model (e.g. "seconds remaining"). |
| **Label** | The answer we are training the model to predict. Here: did the home team win, 1 or 0. |
| **Logistic regression** | A simple statistical model that takes numeric inputs and outputs a probability between 0 and 1. The standard starting point for yes/no predictions. |
| **Gradient boosting** | A machine learning method that builds many small decision trees, each correcting the errors of the last. Common implementations: XGBoost, LightGBM. More accurate than logistic regression, harder to interpret. |
| **Calibration** | Whether the probabilities are *honest*. A calibrated model that says "70%" is right about 70% of the time. Separate from accuracy. |
| **Log loss** | A score measuring how good probability predictions are. Lower is better. Punishes confident wrong answers severely. |
| **Brier score** | Another probability score — the average squared difference between predicted probability and actual outcome. Lower is better, more forgiving than log loss. |
| **Leakage** | When information about the future accidentally leaks into the model's inputs. Makes test results look great and production results terrible. |
| **Train/serve skew** | When the code that builds features during training differs from the code that builds them live. The model receives subtly different inputs in production than it was trained on. |
| **Adjusted efficiency margin** | A team rating: points scored minus points allowed per 100 possessions, adjusted for opponent quality. KenPom's headline metric. |
| **Closing spread** | The betting point spread just before tip-off. A very good, cheap proxy for the difference in team quality. |
| **Monte Carlo simulation** | Running a random simulation thousands of times and counting how often each outcome happens. |
| **Isotonic regression** | A calibration method that fits a flexible, never-decreasing correction curve to raw model outputs. |
| **Platt scaling** | A simpler calibration method: fit a small logistic regression that maps raw outputs to corrected ones. |

---

## 3. The model

### 3.1 What it predicts

One row of data = one game state. The label is whether the home team eventually won.

Critically, **every row from the same game shares the same label.** A game where the home team won produces ~400 rows, all labeled 1, even the rows from moments when they were losing badly. The model learns from the *distribution* of outcomes across many similar states, not from any individual game.

### 3.2 Core features

These are the required inputs.

**Score margin** — home team points minus away team points. Always from the home team's perspective.

**Seconds remaining** — in the game, not the half. Simplifies the math.

**The interaction between those two.** This is where the real signal lives, and it deserves its own explanation. A 6-point lead means almost nothing with 30 minutes left and almost everything with 30 seconds left. The model cannot learn this from margin and time as separate columns — you have to give it the combination.

Two ways to do this:
- Hand-build the feature: `margin / sqrt(seconds_remaining)`. The square root reflects that uncertainty in the final score grows with the square root of time remaining, which is how random processes behave.
- Let a gradient boosting model discover it, by giving it both columns and enough data.

Do both. The hand-built version makes the simple baseline work; the tree model refines it.

**Possession indicator** — 1 if the home team has the ball, 0 otherwise. Worth roughly one point of margin, more as time gets short.

**Timeouts remaining**, for each team. Near-zero effect until the final two minutes, then meaningful: timeouts stop the clock and let a team advance the ball to half court.

**Pregame rating differential** — how much better one team is than the other, plus home court advantage. Two good sources:
- KenPom or Barttorvik adjusted efficiency margin, subtracted
- The closing betting spread (simpler, arguably better, free)

Important nuance: **this feature's influence must decay as the game progresses.** At tip-off it is the only information you have. By the final minute the score itself has absorbed everything the rating told you, and continuing to weight it makes the model stubborn about upsets.

### 3.3 Additional features worth adding

**Bonus / double-bonus foul state.** In the final minute, trailing teams foul deliberately to stop the clock. Whether the leading team shoots one free throw or two changes the arithmetic completely.

**Free throw shooting ability of the leading team.** A team shooting 62% from the line protects a 3-point lead far worse than a team shooting 82%. This is one of the largest late-game effects and most public models ignore it.

**Foul trouble.** Whether a team's best player is on the bench with four fouls.

**Period and overtime flag.**

**Expected possessions remaining**, derived from each team's pace. Two slow teams with 4 minutes left have fewer chances to change the score than two fast ones.

### 3.4 College-specific rules to encode

Do not reuse an NBA-shaped model. The structural differences matter:

- **Men's:** two 20-minute halves. Team fouls reset at halftime. 30-second shot clock.
- **Women's:** four 10-minute quarters. Team fouls reset each quarter.
- Overtime periods are 5 minutes in both.
- The shot clock changed to 30 seconds in 2015 — if you train on data older than that, either adjust or exclude it.

---

## 4. Model choice

### 4.1 Start with logistic regression

Inputs: `margin`, `sqrt(time_remaining)`, `margin / sqrt(time_remaining)`, `spread`, `possession`.

It fits in seconds, produces smooth sensible curves, and is easy to debug. **This is your benchmark.** Everything more complex has to beat it clearly to justify itself.

### 4.2 Then gradient boosting

LightGBM or XGBoost. Two settings that are not obvious:

**Monotonic constraints** — tell the model that more points can never decrease your win probability:

```python
params = {
    "objective": "binary",
    "monotone_constraints": [1, 0, -1, 0],  # margin↑, time, spread↓, possession
}
```

Without this you will produce chart segments where a made basket lowers the scoring team's win probability. It is technically within statistical noise. It will also be the first thing any viewer notices, and it destroys trust in the whole model.

**Transform time before feeding it in.** Trees split on raw numeric values. With raw `seconds_remaining`, the model wastes most of its splits carving up the first 30 minutes where almost nothing matters. Pass `sqrt(seconds_remaining)` instead so split points concentrate where the action is.

**Do not rebalance classes or reweight samples.** You want honest probabilities, not classifications. Reweighting distorts them.

### 4.3 A published alternative: time-varying coefficients

Luke Benz's approach fits a *separate* set of logistic regression coefficients for each point in time, with a random effect capturing game-level noise. Instead of one model that knows about time, it's many small models — one per time slice.

This handles the margin-and-time interaction elegantly and stays interpretable. It's a legitimate alternative to gradient boosting rather than a stepping stone toward it.

### 4.4 The endgame is a different problem

Tree models behave badly in the final 60 seconds, where outcomes become discrete and rule-driven rather than statistical. Two options:

1. Train a separate model specifically on endgame states.
2. Run a **possession-level Monte Carlo simulator**: simulate the remaining possessions thousands of times, explicitly modeling intentional fouling, free throw percentages, and three-point attempts. Count how often the home team wins.

Option 2 is more work but far more correct, because it encodes the rules rather than trying to learn them from sparse data.

---

## 5. Data sources

### 5.1 Free and open

**`sportsdataverse-py`** (Python) — the primary recommendation. Built explicitly as a companion to the R packages hoopR and wehoop, with the design goal that the function you know in R is the call you make in Python.

```bash
pip install sportsdataverse
```

Two functions matter:

| Function | Use |
|---|---|
| `sportsdataverse.mbb.load_mbb_pbp(seasons=range(2010, 2026))` | Bulk historical play-by-play back to 2002. Your training set. Returns a Polars or pandas dataframe. |
| `sportsdataverse.mbb.espn_mbb_pbp(game_id=...)` | One game's live feed — plays, boxscore, timeouts, betting lines, and ESPN's own win probability. |

That last field is a free benchmark: a deployed industry model's output on the exact games you are scoring.

**`CBBpy`** (Python) — narrower but cleaner for prototyping. Its play-by-play output already includes `secs_left_half`, `secs_left_reg`, `half`, `play_team`, and `play_type` — most of what your state builder needs, pre-computed.

**`hoopR`** (R) — the reference implementation. Also scrapes KenPom for subscribers, which `sportsdataverse-py` does not.

**`wehoop`** (R) — the women's equivalent.

**`ncaa-api`** — a free wrapper on ncaa.com with endpoints for play-by-play, scoring summaries, team stats, and live brackets.

**Ratings:** Barttorvik (free, downloadable), KenPom (~$25/yr), EvanMiya, NCAA NET.

**Betting lines:** closing spreads are the cheapest high-quality team rating available. `oddsapiR` wraps The Odds API.

### 5.2 Commercial

| Provider | Notes |
|---|---|
| **Sportradar NCAAMB** | Real-time play-by-play for most conference matchups, with optional push feeds for lower latency. The serious option. |
| **SportsDataIO** | Scores, odds, projections, stats over authenticated HTTP endpoints, plus a historical database for modeling. |
| **Genius Sports** | Official NCAA data partner for some feeds. |

### 5.3 The licensing caveat

ESPN-derived data (which is what hoopR, `sportsdataverse-py`, and CBBpy all use) is **scraped, not licensed.** Fine for personal projects, research, and internal analysis. Risky for anything commercial or public-facing. That is the main reason to pay for Sportradar.

### 5.4 Coverage reality

Coverage is uneven. Low-major and early-season non-conference games frequently have no play-by-play at all. Build a graceful fallback that runs on score and clock alone when possession and timeout data are missing.

### 5.5 Data volume

Roughly 5,500 Division I men's games per season × ~350–400 events each ≈ **2 million rows per season.** Twenty seasons is about 40 million rows — more than you need. Early-game states are highly redundant; consider keeping every state inside the final five minutes and sampling every third state before that.

---

## 6. Live APIs — do they exist?

Yes, at three tiers:

1. **Free, scraped:** ESPN's undocumented endpoints, accessed through `espn_mbb_pbp()`. Poll every 5–15 seconds. No SLA, no support, subject to change without notice.
2. **Free, official-ish:** the ncaa.com wrapper. Similar reliability profile.
3. **Paid, contractual:** Sportradar and SportsDataIO offer real-time feeds with latency guarantees and push delivery.

For building and validating the model, the free tier is entirely sufficient — you're training on history, not live data. The paid tier becomes necessary only when you deploy something people depend on.

---

## 7. Architecture

Two pipelines that share one piece of code. That shared piece is the whole design.

```
OFFLINE (training)
  Historical PBP  ──┐
                    │
              ┌─────▼───────────────────────┐
              │  SHARED                     │
              │  State builder              │
              │  Feature builder            │
              └─────┬───────────────────────┘
                    │
  Train + calibrate ◄┘  ──►  Model registry
                                   │
ONLINE (live)                      │
  Live feed poller ──► [same shared code] ──► Model service ──► State store ──► API / chart
                                                    ▲
                                                    └── loads pinned version
```

### 7.1 The five decisions that matter

**1. One feature-building library, imported by both pipelines.**

This is the single most important structural choice. If your training code computes `seconds_remaining` from a clock string one way and your live code does it another, the model silently degrades in production and you will never see it in a backtest.

Package the state builder and feature builder as a real importable module. Unit-test it against fixed event sequences. This failure — *train/serve skew* — accounts for most broken sports models.

**2. The state builder must be replayable.**

Live feeds retroactively correct plays: a three becomes a two, a foul gets reassigned, a duplicate event arrives. So the builder should be a pure function — given the full event list for a game, produce the full state list — with no dependence on the order events arrived in.

Then handling a correction is simply "re-run the game from event zero," which takes milliseconds.

**3. Store game states, not just probabilities.**

If you persist the feature rows alongside the output, you can re-score all of history with a new model version without re-ingesting anything. If you store only the final probability, every model change requires a full backfill.

**4. Pin the model version.**

The registry holds immutable artifacts. The serving process loads exactly one and tags every prediction it makes with that version. When someone asks why a chart looked strange in January, you need to know which model produced it.

**5. Load pregame context once, at tip-off.**

Ratings, closing spread, and rosters do not change during a game. Fetch them when the game starts and hold them in memory. Only the play-by-play stream needs polling.

### 7.2 Scale

A busy Saturday is roughly 350 Division I games at ~400 events each — about **140,000 inferences per day.** Gradient-boosted tree inference takes microseconds.

Your bottleneck is entirely feed latency and I/O, not compute. This runs comfortably on one small VM. Do not build for a scale you do not have.

### 7.3 A concrete stack

| Layer | Choice |
|---|---|
| Historical store | Parquet files on disk, queried with DuckDB |
| Offline transforms | Polars (`sportsdataverse-py` returns Polars frames natively) |
| Model | LightGBM + isotonic calibration, pickled |
| Registry | A versioned directory with a JSON manifest — MLflow is overkill here |
| Poller | A plain asyncio loop, one task per live game |
| State store | Postgres, one row per game state |
| API | FastAPI, plus websockets if you want live-pushing charts |

---

## 8. Training, step by step

### 8.1 Build the training set

```
for each game:
    load events
    replay events → list of game states
    attach pregame context (rating differential, spread, home flag)
    attach label = did_home_win
    emit rows
```

### 8.2 Two leakage traps

**Ratings as-of date.** If you join KenPom's *end-of-season* efficiency ratings onto a November game, you have told the model who turned out to be good. Your backtest will look excellent and production will not match it.

Fix: snapshot ratings weekly and join on date, or use the closing point spread, which is by definition known before tip-off.

**Overtime.** Decide up front. Recommended: label by the final result including overtime, and treat each OT period as its own clock reset. A tied game at 0:00 in regulation should output roughly 50% plus the rating edge — not 0 or 1.

### 8.3 Free data doubling through symmetry

Mirror every row: flip the sign of margin, spread, and possession; flip the home indicator; flip the label.

This forces the model to treat the two teams identically except through the home-court term. It costs nothing and meaningfully improves stability.

### 8.4 Split by season, never randomly

**This is the mistake that quietly ruins these models.**

Two consecutive game states are nearly identical rows with identical labels. Split rows at random and near-duplicates land in both the training and test sets, so your test score measures memorization rather than generalization.

```
train:       seasons 2010–2022
calibration: season 2023          (held out, used only for calibration)
test:        seasons 2024–2025    (touched once, at the end)
```

Your **effective sample size is games, not rows** — about 5,500 per season. Compute confidence intervals on that basis, and cluster standard errors by game.

---

## 9. Calibration

### 9.1 What it is

Calibration asks: when the model says 70%, does it happen 70% of the time?

This is separate from accuracy. A model can rank games perfectly — always giving the eventual winner a higher number — while being systematically overconfident about every one of them.

### 9.2 How to do it

Fit the calibrator on the **held-out calibration season**, never on training data.

| Method | Description | When to use |
|---|---|---|
| **Platt scaling** | Fit a small logistic regression mapping raw output to corrected output. One parameter. | Stable, but only fixes uniform over/underconfidence. |
| **Isotonic regression** | Fit a flexible never-decreasing step function. | Much more expressive, needs lots of data (you have it). Can misbehave in the sparse tails — clip output to [0.001, 0.999]. |

Gradient boosting trained on log loss usually emerges reasonably calibrated already, so expect the global correction to be small.

### 9.3 The refinement that actually matters

**Calibrate separately within time buckets.**

Miscalibration in these models is almost entirely time-dependent — typically overconfident early, underconfident late. A single global calibrator averages those two errors together and fixes neither.

Fit separate calibrators for roughly:

- 40–20 minutes remaining
- 20–5 minutes
- 5–2 minutes
- under 2 minutes

Then interpolate across the bucket boundaries so the curve does not visibly jump.

---

## 10. Endgame overrides

The model does not know the rules. You do. Override it:

- **At `time_remaining == 0`:** force 1.0 or 0.0 (or route to overtime logic if tied).
- **Mathematically decided flag:** if the trailing team cannot possibly score enough points in the possessions remaining, clamp to 1.0.
- **Final ~60 seconds:** blend the model output with the possession-level simulator, weighting the simulator to 1.0 by 0:00.

These are not hacks. They are constraints that the training data cannot teach efficiently, because those specific states are rare.

---

## 11. Evaluation

### 11.1 Metrics

**Log loss** is the primary metric: the average of `-log(probability you assigned to what actually happened)`. It punishes confident wrong answers harshly, which is exactly right for a probability model.

**Brier score** — mean squared error on probabilities. More forgiving, easier to explain to non-technical stakeholders.

**Calibration curve** — bucket predictions into deciles, plot predicted probability against observed win rate. A perfect model produces a 45-degree line.

### 11.2 The most useful thing you can do

**Break every metric out by minutes remaining.**

A single average number hides the failure mode entirely. Expect log loss to be highest early in the game (correctly — you genuinely know less) and calibration to be worst in the final two minutes (incorrectly — and that is exactly where viewers are paying attention).

### 11.3 Benchmarks, cheapest first

1. **Your own logistic regression baseline.** If gradient boosting does not beat it clearly, something is wrong with your setup.
2. **ESPN's win probability**, which arrives free inside the `espn_mbb_pbp` response. A deployed industry model on the exact same games. The most honest grading available for zero extra work.
3. **Published academic numbers** from Benz or the JQAS paper below, if you want a research-grade target.

### 11.4 The check that is not a metric

Pull up five games you personally watched and look at the curves.

Numbers will not tell you that your model has a team at 94% while they are inbounding down two with fifteen seconds left. Your eyes will.

---

## 12. Prior art

### Directly on college basketball

**Luke Benz — ncaahoopR win probability model.** The closest public analogue to this plan.
- Methodology writeup: `lukebenz.com/post/ncaahoopr_win_prob` — the time-varying coefficient framework, developed as his undergraduate thesis.
- Ratings model: `lukebenz.com/post/hoops_methodology/` — weighted least squares to predict score differential, then logistic regression on top. This is the team-rating input, built from scratch.
- The `ncaahoopR` GitHub repo — the model ships inside the package, so the fitting code and commit history are readable.
- His **Game Excitement Index** posts derive a metric from win probability curves. Useful as a sanity check on whether your curves behave plausibly.

**Maddox, Sides & Harvill (2022)**, "Bayesian estimation of in-game home team win probability for college basketball," *Journal of Quantitative Analysis in Sports*. Peer-reviewed, exactly this problem, cites Benz as prior art. The journal version is paywalled, but the same authors published near-identical companion papers on arXiv for the NBA (2207.05114) and college football (2207.13747) using the same method — free, and enough to reconstruct the approach.

### Transferable, with the development process visible

**`tonyelhabr/nba_wp`** — a self-described one-day project, unusually well documented. Includes a short literature review of existing public models, a season-based train/test split, and error broken out by game time. Also benchmarks its log loss against Benz's college numbers. The best single example of *how to evaluate* one of these.

**`colekev/nba_win_prob_calc`** — R, and structurally identical to the baseline described here: joins play-by-play to historical closing spreads, then fits logistic regression on spread, time, margin, and possession. Documents hitting and solving the endpoint problem (forcing 1.0/0.0 at zero time) and diagnosing by eye that the curve moved too dramatically early and not enough late. That failure mode will be yours too.

**`doganjr/LWPNBA`** — the deep learning ceiling. Fuses an embedding of the play-by-play event sequence with a game-state embedding, then benchmarks against ESPN's deployed model on log loss, Brier score, and expected calibration error. Their evaluation protocol is worth copying even if you never build a neural network.

### Engineering references

**nflfastR** and **cfbfastR** — different sport, same architecture. The most mature open-source win probability pipelines in the ecosystem, with public training scripts and versioned model artifacts.

---

## 13. Build order

A sequence that produces something working at every stage.

**Phase 1 — Baseline (a weekend)**
1. `pip install sportsdataverse`, pull three seasons of play-by-play.
2. Build the state builder and feature builder as an importable module. Test it.
3. Fit logistic regression on margin, time, and their interaction.
4. Plot a curve for one game you remember. Does it look right?

**Phase 2 — Real model (a week)**
5. Add spread or ratings, joined correctly by date.
6. Expand to fifteen seasons; add symmetry mirroring.
7. Fit LightGBM with monotonic constraints.
8. Split by season; measure log loss against the baseline and against ESPN.

**Phase 3 — Honest probabilities**
9. Fit time-bucketed calibrators on a held-out season.
10. Add endgame overrides and the mathematically-decided clamp.
11. Build calibration curves broken out by minutes remaining.

**Phase 4 — Production**
12. Model registry with pinned versions.
13. Live poller against `espn_mbb_pbp`, running through the *same* feature module.
14. State store, then the API and chart.

**Phase 5 — Trust**
15. **Replay harness:** feed a completed game's events through the live path as if arriving in real time, and confirm the output matches the offline path exactly. Run it in CI. This is your defense against train/serve skew.
16. **Calibration monitor:** weekly, bucket predictions by decile and check the observed win rate, broken out by time remaining. Alert on drift.

Build items 15 and 16 earlier than feels necessary. They are what separates a model that works from a model you can trust.

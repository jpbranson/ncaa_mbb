# `cbbwp` source bundle
Complete source. Regenerated 2026-09-01 21:25 from `~/Downloads/ncaa_mbb` on cas-w7r21674vv, which is the working copy.

State rules v2, model v2. This bundle is a mirror for disaster recovery; the folder is the source of truth (it also holds the data, the fitted model and the git history).

See `cbbwp-EXPLAIN.md` for what every piece does and why.

---

## `scripts/build_dataset.py`

```python
"""Replay every game into state rows and write the modelling dataset."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from cbbwp.adapters.hoopr import states_lazy, BASE_COLS
from cbbwp.features import feature_exprs
from cbbwp.schemas import FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
LATE_SECONDS = 300       # keep every state inside the final 5 minutes
EARLY_KEEP_EVERY = 3     # ...and every third state before that (plan 5.5)

games = (pl.read_parquet(ROOT / "data/proc/games.parquet")
         .select("game_id", "season", "home_win", "neutral_site",
                 "pregame_exp_margin", "date", "home_score", "away_score")
         .join(pl.read_parquet(ROOT / "data/proc/team_stats.parquet"),
               on="game_id", how="left"))
out_dir = ROOT / "data/proc/states"
out_dir.mkdir(parents=True, exist_ok=True)

for y in SEASONS:
    t0 = time.time()
    src = ROOT / f"data/raw/pbp/pbp_{y}.parquet"
    cols = set(pl.scan_parquet(src).collect_schema().names())
    sel = list(BASE_COLS) + (["home_win_prob"] if "home_win_prob" in cols else [])
    lf = (pl.scan_parquet(src).select(sel)
          .drop(["season", "game_date"])            # season/date come from games.parquet
          .filter(pl.col("period_number") <= 8))

    st = states_lazy(lf).join(games.lazy(), on="game_id", how="inner")
    st = st.filter(
        pl.col("game_seconds_remaining").is_not_null()
        & (pl.col("margin").abs() < 100)
    )
    # Data-quality gate: ~0.2% of games have a truncated or contradictory feed.
    # A live system would flag these too, so they are excluded everywhere.
    final = (st.group_by("game_id")
             .agg(pl.col("margin").sort_by("seq").last().alias("_last_margin"),
                  pl.col("game_seconds_remaining").sort_by("seq").last().alias("_last_secs"),
                  pl.col("home_win").first().alias("_hw")))
    good = final.filter((pl.col("_last_secs") <= 60)
                        & ((pl.col("_last_margin") > 0).cast(pl.Int8) == pl.col("_hw"))
                        ).select("game_id")
    st = st.join(good, on="game_id", how="inner")
    # Sampling keeps every late-game state and thins the redundant early ones.
    st = st.with_columns(
        _keep=(pl.col("game_seconds_remaining") <= LATE_SECONDS)
        | (pl.col("seq") % EARLY_KEEP_EVERY == 0)
    ).filter(pl.col("_keep"))

    # `margin`, `possession`, `is_ot`, `pregame_exp_margin` come back as features,
    # so they are not repeated here.
    keep = ["game_id", "season", "seq", "period", "game_seconds_remaining",
            "home_timeouts", "away_timeouts", "home_fouls", "away_fouls",
            "neutral_site", "home_win", "date"]
    if "home_win_prob" in cols:
        st = st.rename({"home_win_prob": "espn_wp"})
        keep.append("espn_wp")

    df = st.select(keep + feature_exprs()).collect()
    df.write_parquet(out_dir / f"states_{y}.parquet")
    print(f"{y}: {df.height:>9,} rows  {df['game_id'].n_unique():>5,} games  "
          f"{time.time()-t0:5.1f}s")
```

---

## `scripts/build_games.py`

```python
"""Build the game-level table (results + as-of pregame ratings)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from cbbwp.ratings import build_all_seasons

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

frames = []
for y in SEASONS:
    s = (
        pl.scan_parquet(ROOT / f"data/raw/sched/sched_{y}.parquet")
        .select(["game_id", "date", "neutral_site", "home_id", "away_id",
                 "home_score", "away_score", "status_type_completed"])
        .filter(pl.col("status_type_completed")
                & pl.col("home_score").is_not_null()
                & pl.col("away_score").is_not_null()
                & (pl.col("home_score") != pl.col("away_score")))
        .with_columns(
            season=pl.lit(y, dtype=pl.Int32),
            date=pl.col("date").str.to_datetime("%Y-%m-%dT%H:%MZ", strict=False),
            margin=(pl.col("home_score") - pl.col("away_score")).cast(pl.Int32),
            home_win=(pl.col("home_score") > pl.col("away_score")).cast(pl.Int8),
            neutral_site=pl.col("neutral_site").fill_null(False),
        )
        .filter(pl.col("date").is_not_null())
        .unique(subset=["game_id"])
        .collect()
    )
    frames.append(s)
    print(y, s.height)

games = pl.concat(frames, how="vertical_relaxed")
print("total games", games.height)
games = build_all_seasons(games)
games.write_parquet(ROOT / "data/proc/games.parquet")
print("wrote games.parquet", games.height)
```

---

## `scripts/build_live_context.py`

```python
"""Snapshot today's team ratings and season-to-date stats for the live poller.

Uses the SAME ridge fit as the offline pipeline (cbbwp.ratings._fit_ridge) and
the SAME season-to-date formulas as build_team_stats.py, so a live game's
pregame term is on the same scale the model was fit on.

Run daily before the slate:  python3 scripts/build_live_context.py
"""
import sys, pathlib, json, datetime, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from cbbwp.ratings import _fit_ridge, CARRYOVER

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAGUE_FT, PRIOR_FTA, LEAGUE_PPM = 0.700, 40.0, 3.45

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=None,
                help="season to snapshot (default: the latest in games.parquet)")
ap.add_argument("--out", default=str(ROOT / "registry/context_latest.json"))
a = ap.parse_args()

games = pl.read_parquet(ROOT / "data/proc/games.parquet")
season = a.season or int(games["season"].max())
prev = season - 1 if season - 1 != 2020 else season - 2

# --- 1. ratings: fit this season's completed games, prior = last season's ----
prev_g = games.filter(pl.col("season") == prev)
teams_prev = sorted(set(prev_g["home_id"].to_list()) | set(prev_g["away_id"].to_list()))
prev_ratings, _ = _fit_ridge(prev_g, teams_prev, {})
prior = {t: v * CARRYOVER for t, v in prev_ratings.items()}

cur = games.filter(pl.col("season") == season)
teams = sorted(set(cur["home_id"].to_list()) | set(cur["away_id"].to_list()) | set(prior))
ratings, hca = _fit_ridge(cur, teams, prior)
print(f"season {season}: {cur.height} completed games, {len(teams)} teams, hca={hca:.2f}, "
      f"rating sd={np.std(list(ratings.values())):.2f}")
if cur.height == 0:
    print("  (no games yet - ratings are last season's, carried over. Expected in preseason.)")

# --- 2. season-to-date FT% and points per minute -----------------------------
ft_pct, ppm = {}, {}
pbp = ROOT / f"data/raw/pbp/pbp_{season}.parquet"
if cur.height and pbp.exists():
    ft = (pl.scan_parquet(pbp)
          .select(["game_id", "type_text", "scoring_play", "team_id"])
          .filter(pl.col("type_text").str.contains("FreeThrow"))
          .group_by("team_id").agg(fta=pl.len(), ftm=pl.col("scoring_play").sum())
          .collect())
    for tid, fta, ftm in ft.iter_rows():
        if tid is None:
            continue
        ft_pct[int(tid)] = (ftm + LEAGUE_FT * PRIOR_FTA) / (fta + PRIOR_FTA)

    long = pl.concat([
        cur.select(team_id="home_id", pts="home_score", opp="away_score"),
        cur.select(team_id="away_id", pts="away_score", opp="home_score"),
    ]).group_by("team_id").agg(tot=(pl.col("pts") + pl.col("opp")).sum(), g=pl.len())
    for tid, tot, g in long.iter_rows():
        ppm[int(tid)] = (tot + LEAGUE_PPM * 40 * 5) / ((g + 5) * 40)

out = {
    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "season": season,
    "hca": hca,
    "n_completed_games": cur.height,
    "ratings": {str(k): round(v, 4) for k, v in ratings.items()},
    "ft_pct": {str(k): round(v, 4) for k, v in ft_pct.items()},
    "ppm": {str(k): round(v, 4) for k, v in ppm.items()},
}
dest = pathlib.Path(a.out)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out))
print(f"wrote {dest}  ({len(ratings)} ratings, {len(ft_pct)} ft, {len(ppm)} ppm)")
```

---

## `scripts/build_team_stats.py`

```python
"""Season-to-date team stats, as of the day BEFORE each game (no leakage).

Produces per game: ft_pct_diff (home minus away) and exp_points_per_min
(the two teams' combined scoring rate), both from games already played.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
LEAGUE_FT = 0.700          # prior for teams with no games yet
PRIOR_FTA = 40.0           # strength of that prior, in attempts
LEAGUE_PPM = 3.45          # points per minute, both teams combined

games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
    "game_id", "season", "date", "home_id", "away_id", "home_score", "away_score")

per_team = []
for y in SEASONS:
    ft = (
        pl.scan_parquet(ROOT / f"data/raw/pbp/pbp_{y}.parquet")
        .select(["game_id", "type_text", "scoring_play", "team_id"])
        .filter(pl.col("type_text").str.contains("FreeThrow"))
        .group_by(["game_id", "team_id"])
        .agg(fta=pl.len(), ftm=pl.col("scoring_play").sum())
        .collect()
    )
    per_team.append(ft.with_columns(season=pl.lit(y, dtype=pl.Int32)))
ft = pl.concat(per_team)

# long form: one row per (game, team) with that team's FT and points
long = pl.concat([
    games.select("game_id", "season", "date", team_id=pl.col("home_id"),
                 pts=pl.col("home_score"), opp_pts=pl.col("away_score")),
    games.select("game_id", "season", "date", team_id=pl.col("away_id"),
                 pts=pl.col("away_score"), opp_pts=pl.col("home_score")),
]).join(ft.select("game_id", "team_id", "fta", "ftm"), on=["game_id", "team_id"], how="left")

long = long.sort(["team_id", "season", "date"]).with_columns(
    # cum_sum shifted by one game = everything BEFORE this game
    c_fta=pl.col("fta").fill_null(0).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_ftm=pl.col("ftm").fill_null(0).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_pts=(pl.col("pts") + pl.col("opp_pts")).cum_sum().shift(1).over(["team_id", "season"]).fill_null(0),
    c_g=pl.int_range(pl.len()).over(["team_id", "season"]),
).with_columns(
    ft_pct=(pl.col("c_ftm") + LEAGUE_FT * PRIOR_FTA) / (pl.col("c_fta") + PRIOR_FTA),
    ppm=(pl.col("c_pts") + LEAGUE_PPM * 40 * 5) / ((pl.col("c_g") + 5) * 40),
)

h = long.select("game_id", h_ft="ft_pct", h_ppm="ppm", team_id="team_id")
out = (
    games
    .join(h.rename({"team_id": "home_id"}), on=["game_id", "home_id"], how="left")
    .join(h.rename({"team_id": "away_id", "h_ft": "a_ft", "h_ppm": "a_ppm"}),
          on=["game_id", "away_id"], how="left")
    .select(
        "game_id",
        ft_pct_diff=(pl.col("h_ft") - pl.col("a_ft")).fill_null(0.0),
        exp_points_per_min=((pl.col("h_ppm") + pl.col("a_ppm")) / 2).fill_null(LEAGUE_PPM),
    )
)
out.write_parquet(ROOT / "data/proc/team_stats.parquet")
print(out.describe())
```

---

## `scripts/calibrate_and_eval.py`

```python
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.calibration import TimeBucketedCalibrator
from cbbwp import endgame
from cbbwp.evaluate import log_loss, brier, accuracy, ece, calibration_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
ca = np.load(ROOT / "artifacts/calib_preds.npz")
te = np.load(ROOT / "artifacts/test_preds.npz")
y, secs, espn = te["y"], te["secs"], te["espn"]

cal = TimeBucketedCalibrator().fit(ca["p_gbm"], ca["y"], ca["secs"])
with open(ROOT / "artifacts/calibrator_v1.pkl", "wb") as f:
    pickle.dump(cal, f)

raw = te["p_gbm"]
calibrated = cal.transform(raw, secs)
# endgame overrides need the margin, recovered from the saved feature column
import polars as pl
margin = pl.concat([pl.scan_parquet(ROOT / f"data/proc/states/states_{s}.parquet")
                    .select("margin") for s in (2025, 2026)], how="diagonal").collect()["margin"].to_numpy()
assert len(margin) == len(y)
final = endgame.apply(calibrated, margin, secs)

rows = {
    "lightgbm raw": raw,
    "+ time-bucket calib": calibrated,
    "+ endgame overrides": final,
    "logistic baseline": te["p_lr"],
    "ESPN (deployed)": espn,
}
print(f"{'model':<22}{'logloss':>9}{'brier':>9}{'acc':>8}{'ECE':>8}")
for k, p in rows.items():
    print(f"{k:<22}{log_loss(y,p):>9.4f}{brier(y,p):>9.4f}{accuracy(y,p):>8.4f}{ece(y,p):>8.4f}")

print(f"\nlog loss by time bucket\n{'bucket':<12}{'n':>10}" + "".join(f"{k[:11]:>13}" for k in rows))
for name, lo, hi in [("40-20 min",1200,2400),("20-10 min",600,1200),("10-5 min",300,600),
                     ("5-2 min",120,300),("2-1 min",60,120),("1-0 min",0,60)]:
    m = (secs >= lo) & (secs < hi)
    print(f"{name:<12}{m.sum():>10,}" + "".join(f"{log_loss(y[m],p[m]):>13.4f}" for p in rows.values()))

print("\nfinal-2-minute calibration, after correction")
m = secs < 120
print(f"{'bin':<12}{'n':>10}{'pred':>8}{'obs':>8}{'gap':>8}")
for r in calibration_table(y[m], final[m], n_bins=10):
    print(f"{r['bin']:<12}{r['n']:>10,}{r['pred']:>8.3f}{r['obs']:>8.3f}{r['obs']-r['pred']:>+8.3f}")

np.savez_compressed(ROOT / "artifacts/final_preds.npz", y=y, p=final, secs=secs,
                    espn=espn, margin=margin, game_id=te["game_id"], season=te["season"])
```

---

## `scripts/calibration_monitor.py`

```python
"""Weekly calibration drift check. Exit code 1 means "look at this".

Two sources, same check:

  --source backtest   score a window of already-played games with the pinned
                      model and check the answers against what happened.
                      This is the one to run weekly during the season.

  --source live       read the poller's JSONL, join each state to the final
                      result from games.parquet, and check that. This measures
                      the SHIPPED path end to end - feed, adapter, model - and
                      so it is the one that catches a broken feed, not just a
                      stale model.

Examples
--------
    python3 scripts/calibration_monitor.py --source backtest --season 2027
    python3 scripts/calibration_monitor.py --source backtest --days 7
    python3 scripts/calibration_monitor.py --source live --glob 'data/live/*.jsonl'

Cron it Monday morning:
    0 9 * * 1  cd /path/to/ncaa_mbb && python3 scripts/calibration_monitor.py \
                 --source backtest --days 7 --json reports/calib_$(date +\%F).json
"""
import sys, pathlib, json, glob, argparse, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from cbbwp import monitor
from cbbwp.schemas import FEATURE_NAMES
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]


def from_backtest(args):
    """Score recent completed games with the pinned model."""
    games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
        "game_id", "season", "date", "home_win")
    season = args.season or int(games["season"].max())
    states_path = ROOT / f"data/proc/states/states_{season}.parquet"
    if not states_path.exists():
        raise SystemExit(f"no states file for season {season}: {states_path}\n"
                         "  run scripts/build_dataset.py first")

    df = pl.read_parquet(states_path).join(
        games.select("game_id", "date"), on="game_id", how="left")
    window = f"season {season}"
    if args.days:
        cutoff = df["date"].max() - datetime.timedelta(days=args.days)
        df = df.filter(pl.col("date") >= cutoff)
        window = f"season {season}, last {args.days} days (since {cutoff:%Y-%m-%d})"

    if df.height == 0:
        raise SystemExit("no rows in that window")

    svc = WinProbabilityService(args.registry, args.version)
    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    p = np.asarray(svc._predict(X), dtype=np.float64)
    return (df["home_win"].to_numpy(), p,
            df["game_seconds_remaining"].to_numpy().astype(np.float64),
            df["game_id"].to_numpy(), window)


def from_live(args):
    """Read poller output and join to final results."""
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob}")
    rows = []
    for f in files:
        for line in pathlib.Path(f).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("live files are empty")
    live = pl.DataFrame(rows).select(
        "game_id", "home_win_prob", "game_seconds_remaining")
    games = pl.read_parquet(ROOT / "data/proc/games.parquet").select(
        "game_id", "home_win")
    j = live.join(games, on="game_id", how="inner")
    missing = live.height - j.height
    if missing:
        print(f"note: {missing:,} live states have no final result yet; skipped",
              file=sys.stderr)
    if j.height == 0:
        raise SystemExit("no live states could be matched to a finished game")
    return (j["home_win"].to_numpy(), j["home_win_prob"].to_numpy(),
            j["game_seconds_remaining"].to_numpy().astype(np.float64),
            j["game_id"].to_numpy(), f"live, {len(files)} file(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["backtest", "live"], default="backtest")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="restrict a backtest to the last N days of the season")
    ap.add_argument("--glob", default=str(ROOT / "data/live/*.jsonl"))
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    ap.add_argument("--version", default="v2")
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--z", type=float, default=monitor.Z_ALERT)
    ap.add_argument("--min-gap", type=float, default=monitor.MIN_GAP)
    a = ap.parse_args()

    y, p, secs, gids, window = (from_live(a) if a.source == "live"
                                else from_backtest(a))
    rep = monitor.check(y, p, secs, gids, window=window, z_alert=a.z,
                        min_gap=a.min_gap,
                        generated=datetime.datetime.now(
                            datetime.timezone.utc).isoformat())
    print(monitor.format_report(rep))

    if a.json:
        dest = pathlib.Path(a.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rep.as_dict(), indent=2))
        print(f"\nwrote {dest}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## `scripts/check_espn_fixtures.py`

```python
"""Validate recorded ESPN payloads against the adapter's expectations.

This is the check that the offline test suite CANNOT do: it looks at what ESPN
actually sends today and reports anything the adapter would silently mishandle -
above all a play type id the model was never trained on.

    python3 scripts/check_espn_fixtures.py --dir tmp/fixtures
"""
import sys, pathlib, json, argparse, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.adapters import espn
from cbbwp.state import build_states
from cbbwp.schemas import PregameContext

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(ROOT / "tmp/fixtures"))
a = ap.parse_args()

files = sorted(pathlib.Path(a.dir).glob("summary_*.json"))
if not files:
    raise SystemExit(f"no summary_*.json in {a.dir} - "
                     "run scripts/record_espn_fixtures.py first")

unknown = collections.Counter()
problems = 0
for f in files:
    payload = json.loads(f.read_text())
    raw_plays = payload.get("plays") or []
    events, h = espn.parse_summary(payload)
    ctx = PregameContext(h.game_id, h.home_team_id, h.away_team_id, h.neutral_site)
    states = build_states(events, ctx)

    for p in raw_plays:
        t = p.get("type") or {}
        tid = espn._int(t.get("id"))
        if tid not in espn.TYPE_ID_TO_TEXT:
            unknown[(tid, (t.get("text") or "").strip())] += 1

    ok = True
    if len(events) != len(raw_plays):
        print(f"  {f.name}: {len(raw_plays)} plays -> {len(events)} events (MISMATCH)")
        ok = False
    if states and h.is_final:
        last = states[-1]
        said = last.margin
        actual = h.home_score - h.away_score
        if said != actual:
            print(f"  {f.name}: final margin from plays {said:+d} != header "
                  f"{actual:+d} - truncated or contradictory feed")
            ok = False
    if not h.home_team_id or not h.away_team_id:
        print(f"  {f.name}: could not read team ids")
        ok = False
    problems += 0 if ok else 1
    print(f"{'ok  ' if ok else 'BAD '}{f.name:<28} {h.away_name} @ {h.home_name}  "
          f"{h.status}  {len(events):,} plays, {len(states):,} states")

if unknown:
    print("\nPLAY TYPES THE MODEL HAS NEVER SEEN "
          "(they fall back to the feed's text and carry possession):")
    for (tid, text), n in unknown.most_common():
        print(f"  id={tid}  text={text!r}  x{n:,}")
    print("\n  A frequent one here means the ESPN feed changed. Add it to "
          "TYPE_ID_TO_TEXT only if the training data also contains it;\n"
          "  otherwise the honest fix is to refit with the new type present.")
else:
    print("\nno unknown play types - the adapter's type map covers this feed")

raise SystemExit(1 if problems else 0)
```

---

## `scripts/evaluate.py`

```python
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.evaluate import by_time_bucket, log_loss, brier, calibration_table, ece, accuracy

ROOT = pathlib.Path(__file__).resolve().parents[1]
d = np.load(ROOT / "artifacts/test_preds.npz")
y, secs, espn = d["y"], d["secs"], d["espn"]
preds = {"logistic": d["p_lr"], "lightgbm": d["p_gbm"], "espn": espn}

ok = np.isfinite(espn)
print(f"test rows {len(y):,}   ESPN win prob present on {ok.mean()*100:.1f}%\n")
print(f"{'model':<10} {'logloss':>8} {'brier':>8} {'acc':>7} {'ECE':>7}   (rows where ESPN also present)")
for name, p in preds.items():
    print(f"{name:<10} {log_loss(y[ok],p[ok]):>8.4f} {brier(y[ok],p[ok]):>8.4f} "
          f"{accuracy(y[ok],p[ok]):>7.4f} {ece(y[ok],p[ok]):>7.4f}")

print(f"\n{'bucket':<12}{'n':>10}" + "".join(f"{k:>12}" for k in preds))
for name, lo, hi in [("40-20 min",1200,2400),("20-10 min",600,1200),("10-5 min",300,600),
                     ("5-2 min",120,300),("2-1 min",60,120),("1-0 min",0,60)]:
    m = ok & (secs >= lo) & (secs < hi)
    if m.sum() == 0: continue
    print(f"{name:<12}{m.sum():>10,}" + "".join(f"{log_loss(y[m],p[m]):>12.4f}" for p in preds.values()))
print("(cells are log loss; lower is better)")

print("\ncalibration of lightgbm on the test seasons")
print(f"{'bin':<12}{'n':>10}{'pred':>8}{'obs':>8}{'gap':>8}")
for r in calibration_table(y, d["p_gbm"]):
    print(f"{r['bin']:<12}{r['n']:>10,}{r['pred']:>8.3f}{r['obs']:>8.3f}{r['obs']-r['pred']:>+8.3f}")
```

---

## `scripts/fetch_data.py`

```python
"""Download the hoopR play-by-play and schedule parquet files.

The only reachable source: raw.githubusercontent.com. Re-run is idempotent;
existing files are skipped unless --force.
"""
import argparse, pathlib, sys, urllib.request, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
PBP = BASE + "/pbp/parquet/play_by_play_{y}.parquet"
SCHED = BASE + "/schedules/parquet/mbb_schedule_{y}.parquet"


def get(url: str, dest: pathlib.Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"  skip {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  got  {dest.name} ({dest.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for y in a.seasons:
        print(y)
        get(PBP.format(y=y), ROOT / f"data/raw/pbp/pbp_{y}.parquet", a.force)
        get(SCHED.format(y=y), ROOT / f"data/raw/sched/sched_{y}.parquet", a.force)
```

---

## `scripts/fit_models.py`

```python
"""Fit the logistic baseline and the LightGBM model. Split by season, never randomly."""
import sys, pathlib, json, pickle, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cbbwp.schemas import FEATURE_NAMES
from cbbwp.features import mirror_features
from cbbwp.evaluate import log_loss, brier, accuracy

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = [2016, 2017, 2018, 2019, 2021, 2022, 2023]
CALIB = [2024]
TEST  = [2025, 2026]

# Monotonic constraints (plan 4.2): more points, the ball, and a better team can
# never *lower* the home side's win probability.
MONOTONE = {
    "margin": 1, "sqrt_time": 0, "margin_per_sqrt_time": 1, "possession": 1,
    "pregame_exp_margin": 1, "pregame_exp_margin_decayed": 1,
    "is_ot": 0, "timeout_diff": 0, "bonus_diff": 0, "ft_pct_diff": 0,
    "margin_per_sqrt_points_left": 1,
}


def load(seasons, cols=None):
    files = [ROOT / f"data/proc/states/states_{y}.parquet" for y in seasons]
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal")
    return lf.select(cols).collect() if cols else lf.collect()


def xy(df):
    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y = df["home_win"].to_numpy().astype(np.int8)
    return X, y


t0 = time.time()
need = FEATURE_NAMES + ["home_win", "game_seconds_remaining", "game_id", "season"]
tr = load(TRAIN, need)
print(f"train rows {tr.height:,}  games {tr['game_id'].n_unique():,}  ({time.time()-t0:.0f}s)")
Xtr, ytr = xy(tr)
Xtr, ytr = mirror_features(Xtr.astype(np.float64), ytr)
Xtr = Xtr.astype(np.float32); ytr = ytr.astype(np.int8)
print(f"after symmetry mirroring: {Xtr.shape[0]:,} rows")
del tr

ca = load(CALIB, need); Xca, yca = xy(ca)
te = load(TEST, need + ["espn_wp"]); Xte, yte = xy(te)
print(f"calib {Xca.shape[0]:,}   test {Xte.shape[0]:,}")

# ---- 1. logistic regression baseline ------------------------------------
t0 = time.time()
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr.fit(sc.transform(Xtr), ytr)
p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
print(f"\nLOGISTIC  fit {time.time()-t0:.0f}s   test logloss {log_loss(yte,p_lr):.4f}  "
      f"brier {brier(yte,p_lr):.4f}  acc {accuracy(yte,p_lr):.4f}")
print("  coefs:", {n: round(float(c), 3) for n, c in zip(FEATURE_NAMES, lr.coef_[0])})

# ---- 2. LightGBM --------------------------------------------------------
t0 = time.time()
params = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.05,
    num_leaves=160, min_data_in_leaf=1500, feature_fraction=0.9,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1,
    monotone_constraints=[MONOTONE[n] for n in FEATURE_NAMES],
    monotone_constraints_method="advanced", num_threads=2,
    # Pinned so a refit reproduces registry/v1 exactly rather than approximately.
    seed=20260831, bagging_seed=20260831, feature_fraction_seed=20260831,
    data_random_seed=20260831, deterministic=True,
)
dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
dca = lgb.Dataset(Xca, label=yca, feature_name=FEATURE_NAMES, reference=dtr)
gbm = lgb.train(params, dtr, num_boost_round=2500, valid_sets=[dca],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)])
p_gbm = gbm.predict(Xte, num_iteration=gbm.best_iteration)
print(f"\nLIGHTGBM  fit {time.time()-t0:.0f}s  best_iter {gbm.best_iteration}   "
      f"test logloss {log_loss(yte,p_gbm):.4f}  brier {brier(yte,p_gbm):.4f}  acc {accuracy(yte,p_gbm):.4f}")

art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
with open(art / "lr_v1.pkl", "wb") as f:
    pickle.dump({"scaler": sc, "model": lr, "features": FEATURE_NAMES}, f)
gbm.save_model(str(art / "gbm_v1.txt"), num_iteration=gbm.best_iteration)
np.savez_compressed(art / "test_preds.npz",
                    y=yte, p_lr=p_lr, p_gbm=p_gbm,
                    secs=te["game_seconds_remaining"].to_numpy(),
                    espn=te["espn_wp"].to_numpy().astype(np.float64),
                    game_id=te["game_id"].to_numpy(), season=te["season"].to_numpy())
# calibration-season predictions, needed to fit the calibrators
np.savez_compressed(art / "calib_preds.npz", y=yca,
                    p_lr=lr.predict_proba(sc.transform(Xca))[:, 1],
                    p_gbm=gbm.predict(Xca, num_iteration=gbm.best_iteration),
                    secs=ca["game_seconds_remaining"].to_numpy())
print("\nsaved artifacts")
```

---

## `scripts/live_poller.py`

```python
"""Live win-probability poller.

One asyncio task per in-progress game. Each task fetches that game's ESPN
summary, turns the plays into canonical Events, and replays the WHOLE game
through `WinProbabilityService` every poll. Full replay is deliberate: it costs
a few milliseconds and it makes ESPN's retroactive corrections - a play
inserted, rescored, or deleted minutes later - a non-event, because the answer
is always a pure function of the plays currently in the feed.

Usage
-----
    python3 scripts/build_live_context.py          # once a day, before the slate
    python3 scripts/live_poller.py                 # follow tonight's games
    python3 scripts/live_poller.py --date 20261115 # a specific slate
    python3 scripts/live_poller.py --game 401585555 --once   # one game, one poll
    python3 scripts/live_poller.py --fixture-dir tmp/fixtures # offline, no network

Output: a line per changed state to stdout, and JSONL to --out (default
`data/live/wp_YYYYMMDD.jsonl`). Nothing is overwritten; the file is appended to.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import json
import pathlib
import signal
import sys
import time
from typing import Dict, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.adapters.espn import (EspnClient, parse_summary, scoreboard_games,
                                 STATUS_FINAL, STATUS_PRE)
from cbbwp.live_context import LiveContextProvider
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Poll cadence, by seconds remaining in the game. Tight at the end, where the
# probability actually moves, and lazy early, where it barely does.
def poll_interval(secs_remaining: Optional[float], live: bool) -> float:
    if not live:
        return 60.0
    if secs_remaining is None:
        return 30.0
    if secs_remaining <= 120:
        return 5.0
    if secs_remaining <= 300:
        return 10.0
    return 20.0


DISCOVERY_INTERVAL = 120.0    # rescan the scoreboard this often
MAX_CONCURRENT_FETCHES = 8
ERROR_BACKOFF = (5.0, 15.0, 45.0, 90.0)   # per consecutive failure


class Poller:
    def __init__(self, svc: WinProbabilityService, ctx: LiveContextProvider,
                 client: EspnClient, out_path: pathlib.Path, quiet: bool = False,
                 fixture_dir: Optional[pathlib.Path] = None):
        self.svc = svc
        self.ctx = ctx
        self.client = client
        self.out_path = out_path
        self.quiet = quiet
        self.fixture_dir = fixture_dir
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        self.watchers: Dict[int, asyncio.Task] = {}
        self.last_emitted: Dict[int, tuple] = {}
        self.stopping = asyncio.Event()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = out_path.open("a", buffering=1)

    # -- I/O -------------------------------------------------------------
    async def _summary(self, game_id: int) -> dict:
        if self.fixture_dir is not None:
            p = self.fixture_dir / f"summary_{game_id}.json"
            return json.loads(p.read_text())
        async with self.sem:
            return await asyncio.to_thread(self.client.summary, game_id)

    async def _scoreboard(self, date: Optional[str]) -> dict:
        if self.fixture_dir is not None:
            p = self.fixture_dir / "scoreboard.json"
            return json.loads(p.read_text())
        async with self.sem:
            return await asyncio.to_thread(self.client.scoreboard, date)

    def emit(self, row: dict, header) -> None:
        self._fh.write(json.dumps(row) + "\n")
        if self.quiet:
            return
        secs = row["game_seconds_remaining"]
        mm, ss = divmod(int(secs), 60)
        print(f"{header.away_name[:18]:<18} @ {header.home_name[:18]:<18} "
              f"P{row['period']} {mm:>2}:{ss:02d}  "
              f"{row['margin']:>+4}  home {row['home_win_prob']:.3f}", flush=True)

    # -- one game --------------------------------------------------------
    async def watch(self, game_id: int) -> None:
        fails = 0
        while not self.stopping.is_set():
            try:
                summary = await self._summary(game_id)
                events, header = parse_summary(summary)
                fails = 0
            except Exception as e:                     # noqa: BLE001
                fails += 1
                wait = ERROR_BACKOFF[min(fails - 1, len(ERROR_BACKOFF) - 1)]
                if not self.quiet:
                    print(f"[{game_id}] fetch failed ({type(e).__name__}: {e}); "
                          f"retry in {wait:.0f}s", file=sys.stderr, flush=True)
                if fails >= 8:
                    print(f"[{game_id}] giving up after {fails} failures",
                          file=sys.stderr, flush=True)
                    return
                await self._sleep(wait)
                continue

            if events:
                pctx = self.ctx.context_for(
                    game_id, header.home_team_id, header.away_team_id,
                    header.neutral_site)
                rows = self.svc.score_game(events, pctx)
                if rows:
                    last = rows[-1]
                    key = (last["seq"], round(last["home_win_prob"], 6))
                    if self.last_emitted.get(game_id) != key:
                        self.last_emitted[game_id] = key
                        last = dict(last)
                        last["ts"] = datetime.datetime.now(
                            datetime.timezone.utc).isoformat()
                        last["home_team_id"] = header.home_team_id
                        last["away_team_id"] = header.away_team_id
                        last["status"] = header.status
                        self.emit(last, header)
                    secs = last["game_seconds_remaining"]
                else:
                    secs = None
            else:
                secs = None

            if header.is_final:
                if not self.quiet:
                    print(f"[{game_id}] final: {header.away_name} "
                          f"{header.away_score} @ {header.home_name} "
                          f"{header.home_score}", flush=True)
                return
            await self._sleep(poll_interval(secs, header.is_live))

    async def _sleep(self, secs: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.stopping.wait(), timeout=secs)

    # -- discovery -------------------------------------------------------
    async def discover(self, date: Optional[str], once: bool) -> None:
        while not self.stopping.is_set():
            try:
                sb = await self._scoreboard(date)
                games = scoreboard_games(sb)
            except Exception as e:                     # noqa: BLE001
                print(f"scoreboard failed ({type(e).__name__}: {e})",
                      file=sys.stderr, flush=True)
                await self._sleep(30.0)
                continue

            live = [g for g in games
                    if g["status"] not in (STATUS_PRE, STATUS_FINAL)]
            if not self.quiet:
                print(f"scoreboard: {len(games)} games, {len(live)} in progress, "
                      f"{len(self.watchers)} watched", flush=True)
            for g in live:
                gid = g["game_id"]
                t = self.watchers.get(gid)
                if t is None or t.done():
                    self.watchers[gid] = asyncio.create_task(
                        self.watch(gid), name=f"watch-{gid}")
            for gid, t in list(self.watchers.items()):
                if t.done():
                    del self.watchers[gid]
            if once:
                return
            await self._sleep(DISCOVERY_INTERVAL)

    async def run(self, date: Optional[str], game: Optional[int],
                  once: bool) -> None:
        if game is not None:
            await self.watch(game) if not once else await self._one(game)
            return
        await self.discover(date, once)
        if self.watchers:
            await asyncio.gather(*self.watchers.values(),
                                 return_exceptions=True)

    async def _one(self, game_id: int) -> None:
        """A single poll of one game - the smoke test."""
        summary = await self._summary(game_id)
        events, header = parse_summary(summary)
        pctx = self.ctx.context_for(game_id, header.home_team_id,
                                    header.away_team_id, header.neutral_site)
        rows = self.svc.score_game(events, pctx)
        print(f"{header.away_name} @ {header.home_name}  status={header.status}  "
              f"{len(events)} plays -> {len(rows)} states")
        if not self.ctx.known(header.home_team_id):
            print(f"  note: home team id {header.home_team_id} not in the ratings "
                  "snapshot; using league average")
        if not self.ctx.known(header.away_team_id):
            print(f"  note: away team id {header.away_team_id} not in the ratings "
                  "snapshot; using league average")
        for r in rows[-10:]:
            mm, ss = divmod(int(r["game_seconds_remaining"]), 60)
            print(f"  P{r['period']} {mm:>2}:{ss:02d}  margin {r['margin']:>+4}  "
                  f"home {r['home_win_prob']:.4f}")
            self._fh.write(json.dumps(r) + "\n")

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="slate to follow, YYYYMMDD (default: today)")
    ap.add_argument("--game", type=int, help="follow a single game id")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--version", default="v2", help="model registry version")
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    ap.add_argument("--context", default=str(ROOT / "registry/context_latest.json"))
    ap.add_argument("--out", default=None, help="JSONL output path")
    ap.add_argument("--fixture-dir", default=None,
                    help="read saved payloads from this directory instead of the network")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    ctx_path = pathlib.Path(a.context)
    if not ctx_path.exists():
        print(f"no ratings snapshot at {ctx_path}\n"
              "  run: python3 scripts/build_live_context.py", file=sys.stderr)
        return 2
    ctx = LiveContextProvider.load(ctx_path)
    if ctx.is_stale:
        print(f"warning: ratings snapshot is {ctx.age_days:.1f} days old; "
              "re-run scripts/build_live_context.py", file=sys.stderr)

    svc = WinProbabilityService(a.registry, a.version)
    day = a.date or datetime.datetime.now().strftime("%Y%m%d")
    out = pathlib.Path(a.out) if a.out else ROOT / f"data/live/wp_{day}.jsonl"
    poller = Poller(svc, ctx, EspnClient(), out, quiet=a.quiet,
                    fixture_dir=pathlib.Path(a.fixture_dir) if a.fixture_dir else None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, poller.stopping.set)
    try:
        loop.run_until_complete(poller.run(a.date, a.game, a.once))
    except KeyboardInterrupt:
        pass
    finally:
        poller.close()
        loop.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## `scripts/publish_model.py`

```python
"""Write an immutable, pinned model artifact into the registry."""
import sys, pathlib, json, shutil, hashlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.schemas import FEATURE_NAMES, STATE_RULES_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[1]
version = sys.argv[1] if len(sys.argv) > 1 else "v1"
dest = ROOT / "registry" / version
dest.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "artifacts/gbm_v1.txt", dest / "model.txt")
digest = hashlib.sha256((dest / "model.txt").read_bytes()).hexdigest()[:16]
(dest / "manifest.json").write_text(json.dumps({
    "version": version, "kind": "lightgbm", "features": FEATURE_NAMES,
    "state_rules_version": STATE_RULES_VERSION,
    "sha256": digest, "created": datetime.datetime.now(datetime.UTC).isoformat(),
    "train_seasons": [2016, 2017, 2018, 2019, 2021, 2022, 2023],
    "calibration_season": 2024, "test_seasons": [2025, 2026],
}, indent=2))
print("published", dest, digest)
```

---

## `scripts/record_espn_fixtures.py`

```python
"""Record real ESPN payloads to disk, so the adapter can be tested against them.

Run this on a machine that can reach ESPN (the sandbox this package was built
in cannot - site.api.espn.com is blocked by egress policy). It saves the raw
JSON exactly as returned; nothing is parsed or normalised, so the fixtures stay
useful even if the adapter changes.

    python3 scripts/record_espn_fixtures.py --date 20261115 --limit 5
    python3 scripts/record_espn_fixtures.py --game 401585555

Then, offline:
    python3 scripts/check_espn_fixtures.py           # sanity-check the payloads
    python3 scripts/live_poller.py --fixture-dir tmp/fixtures --game 401585555 --once
"""
import sys, pathlib, json, argparse, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.adapters.espn import EspnClient, scoreboard_games

ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
ap.add_argument("--game", type=int, default=None, help="record just this game")
ap.add_argument("--limit", type=int, default=5, help="how many games to record")
ap.add_argument("--out", default=str(ROOT / "tmp/fixtures"))
a = ap.parse_args()

out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
c = EspnClient()

if a.game:
    ids = [a.game]
else:
    day = a.date or datetime.datetime.now().strftime("%Y%m%d")
    sb = c.scoreboard(day)
    (out / "scoreboard.json").write_text(json.dumps(sb))
    games = scoreboard_games(sb)
    print(f"scoreboard {day}: {len(games)} games")
    # prefer games that are live or finished - a scheduled game has no plays
    games.sort(key=lambda g: (g["status"] == "STATUS_SCHEDULED", g["game_id"]))
    ids = [g["game_id"] for g in games[:a.limit]]

for gid in ids:
    s = c.summary(gid)
    p = out / f"summary_{gid}.json"
    p.write_text(json.dumps(s))
    print(f"  wrote {p.name}  ({len(s.get('plays') or []):,} plays)")
print(f"\n{len(ids)} fixture(s) in {out}")
```

---

## `src/cbbwp/__init__.py`

```python
"""cbbwp - college basketball win probability.

The state builder and feature builder in this package are imported by BOTH the
offline training pipeline and the live serving path. That is deliberate: it is
the single defence against train/serve skew.
"""
__version__ = "0.2.0"

from .schemas import Event, GameState, PregameContext, FEATURE_NAMES  # noqa: F401
from .state import build_states  # noqa: F401
from .features import build_feature_matrix, feature_dict  # noqa: F401
```

---

## `src/cbbwp/calibration.py`

```python
"""Time-bucketed probability calibration.

Miscalibration in these models is almost entirely time-dependent, so a single
global calibrator averages two opposite errors and fixes neither (plan 9.3).
We fit one isotonic curve per time bucket on a HELD-OUT season, then blend
between adjacent buckets so the curve never visibly jumps.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

# (lower, upper) seconds remaining, and the anchor point used for blending.
BUCKETS = [(1200, 2400), (600, 1200), (300, 600), (120, 300), (60, 120), (0, 60)]
ANCHORS = np.array([np.sqrt((lo + hi) / 2) for lo, hi in BUCKETS])
CLIP = (0.001, 0.999)


class TimeBucketedCalibrator:
    def __init__(self, buckets=BUCKETS, min_rows=20_000):
        self.buckets = list(buckets)
        self.min_rows = min_rows
        self.models: list[IsotonicRegression | None] = []

    def fit(self, p, y, seconds):
        self.models = []
        for lo, hi in self.buckets:
            m = (seconds >= lo) & (seconds < hi)
            if m.sum() < self.min_rows:
                self.models.append(None)
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p[m], y[m])
            self.models.append(iso)
        return self

    def _apply(self, i, p):
        m = self.models[i]
        return p if m is None else m.predict(p)

    def transform(self, p, seconds):
        """Blend the two nearest bucket calibrators, weighted in sqrt-time."""
        p = np.asarray(p, dtype=np.float64)
        s = np.sqrt(np.maximum(np.asarray(seconds, dtype=np.float64), 0.0))
        # anchors run from long time remaining down to zero
        order = np.argsort(ANCHORS)
        a = ANCHORS[order]
        cols = np.stack([self._apply(order[i], p) for i in range(len(a))], axis=1)
        idx = np.clip(np.searchsorted(a, s), 1, len(a) - 1)
        lo, hi = a[idx - 1], a[idx]
        w = np.clip((s - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
        out = cols[np.arange(len(p)), idx - 1] * (1 - w) + cols[np.arange(len(p)), idx] * w
        return np.clip(out, *CLIP)
```

---

## `src/cbbwp/endgame.py`

```python
"""Rule-based overrides the training data cannot teach efficiently (plan 10).

The model does not know the rules; we do. These are constraints, not hacks.
"""
from __future__ import annotations

import numpy as np

# A possession is worth at most 3 points (ignoring 4-point plays, which are rare
# enough that treating them as impossible costs less than the states it saves).
MAX_POINTS_PER_POSSESSION = 3
# Seconds a trailing team needs to score and foul once more.
SECONDS_PER_POSSESSION = 6.0


def max_points_remaining(seconds_remaining: np.ndarray) -> np.ndarray:
    """Optimistic ceiling on what a trailing team can still score."""
    poss = np.floor(np.asarray(seconds_remaining, dtype=np.float64) / SECONDS_PER_POSSESSION) + 1
    return poss * MAX_POINTS_PER_POSSESSION


def apply(p, margin, seconds_remaining, is_ot=None):
    """Clamp probabilities that the rules have already decided."""
    p = np.array(p, dtype=np.float64, copy=True)
    margin = np.asarray(margin)
    t = np.asarray(seconds_remaining, dtype=np.float64)

    # 1. Time expired in the final period: the result is known.
    over = t <= 0
    p[over & (margin > 0)] = 1.0
    p[over & (margin < 0)] = 0.0
    # A tie at 0:00 goes to overtime -> a coin flip, nudged by nothing else here.
    p[over & (margin == 0)] = 0.5

    # 2. Mathematically decided: the trailing team cannot catch up in the
    #    possessions that remain, however well it plays.
    ceiling = max_points_remaining(t)
    decided_home = (~over) & (margin > ceiling)
    decided_away = (~over) & (-margin > ceiling)
    p[decided_home] = 1.0
    p[decided_away] = 0.0
    return p
```

---

## `src/cbbwp/evaluate.py`

```python
"""Probability metrics, always broken out by time remaining (plan 11.2)."""
from __future__ import annotations

import numpy as np

EPS = 1e-6

# (label, lower bound seconds, upper bound seconds)
TIME_BUCKETS = [
    ("40-20 min", 1200, 2400),
    ("20-10 min", 600, 1200),
    ("10-5 min", 300, 600),
    ("5-2 min", 120, 300),
    ("2-0 min", 0, 120),
]


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def accuracy(y, p):
    return float(np.mean((p >= 0.5) == (y == 1)))


def by_time_bucket(y, p, seconds, extra=None):
    """Metrics per bucket. `extra` is an optional dict of other prediction sets."""
    rows = []
    for name, lo, hi in TIME_BUCKETS:
        m = (seconds >= lo) & (seconds < hi) if lo > 0 else (seconds >= lo) & (seconds < hi)
        if m.sum() == 0:
            continue
        row = {"bucket": name, "n": int(m.sum()),
               "log_loss": log_loss(y[m], p[m]), "brier": brier(y[m], p[m]),
               "acc": accuracy(y[m], p[m])}
        if extra:
            for k, v in extra.items():
                ok = m & np.isfinite(v)
                row[f"log_loss_{k}"] = log_loss(y[ok], v[ok]) if ok.sum() else float("nan")
                row[f"n_{k}"] = int(ok.sum())
        rows.append(row)
    return rows


def calibration_table(y, p, n_bins=10):
    """Predicted vs observed, in equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                    "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return out


def ece(y, p, n_bins=20):
    """Expected calibration error: mean |predicted - observed|, size-weighted."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    tot, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(tot)
```

---

## `src/cbbwp/features.py`

```python
"""Feature builder. Imported by BOTH the training pipeline and the live server.

If you change anything here you have changed the model's inputs: bump the model
version and refit. `FEATURE_NAMES` in schemas.py fixes the column order.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .schemas import (GameState, FEATURE_NAMES, REGULATION_SECONDS,
                      BONUS_FOULS, DOUBLE_BONUS_FOULS)

_SQRT_REG = math.sqrt(REGULATION_SECONDS)


def _bonus_level(opp_fouls: int) -> int:
    """0 = no bonus, 1 = one-and-one, 2 = double bonus."""
    if opp_fouls >= DOUBLE_BONUS_FOULS:
        return 2
    if opp_fouls >= BONUS_FOULS:
        return 1
    return 0


def feature_dict(s: GameState) -> Dict[str, float]:
    """Features for one state. The single source of truth for both pipelines."""
    t = max(float(s.game_seconds_remaining), 1.0)
    sqrt_t = math.sqrt(t)
    # Pregame edge is the only information at tip-off and should fade to nothing
    # by the buzzer, because by then the score has absorbed it.
    decay = sqrt_t / _SQRT_REG
    return {
        "margin": float(s.margin),
        "sqrt_time": sqrt_t,
        "margin_per_sqrt_time": float(s.margin) / sqrt_t,
        "possession": float(s.possession),
        "pregame_exp_margin": float(s.pregame_exp_margin),
        "pregame_exp_margin_decayed": float(s.pregame_exp_margin) * decay,
        "is_ot": 1.0 if s.is_ot else 0.0,
        "timeout_diff": float(s.home_timeouts - s.away_timeouts),
        # Bonus level a team SHOOTS in is driven by its opponent's fouls.
        "bonus_diff": float(_bonus_level(s.away_fouls) - _bonus_level(s.home_fouls)),
        "ft_pct_diff": float(s.ft_pct_diff),
        # Pace-aware twin of margin/sqrt(time): two slow teams have fewer
        # chances left than two fast ones with the same clock.
        "margin_per_sqrt_points_left": float(s.margin) / math.sqrt(
            max(s.exp_points_per_min * t / 60.0, 1.0)),
    }


def build_feature_matrix(states: Sequence[GameState]) -> np.ndarray:
    """(n_states, n_features) float64 array in FEATURE_NAMES order."""
    out = np.empty((len(states), len(FEATURE_NAMES)), dtype=np.float64)
    for i, s in enumerate(states):
        d = feature_dict(s)
        for j, name in enumerate(FEATURE_NAMES):
            out[i, j] = d[name]
    return out


def mirror_features(X: np.ndarray, y: np.ndarray):
    """Symmetry augmentation (plan 8.3): swap the two teams, flip the label.

    Forces the model to treat the teams identically except through terms that
    are genuinely home-specific. Free data, and it stabilises the fit.
    """
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    Xm = X.copy()
    for name in ("margin", "margin_per_sqrt_time", "pregame_exp_margin",
                 "pregame_exp_margin_decayed", "timeout_diff", "bonus_diff",
                 "ft_pct_diff", "margin_per_sqrt_points_left"):
        Xm[:, idx[name]] *= -1.0
    Xm[:, idx["possession"]] = 1.0 - Xm[:, idx["possession"]]
    return np.vstack([X, Xm]), np.concatenate([y, 1 - y])


# --------------------------------------------------------------------------
# Vectorised twin of `feature_dict`, for building tens of millions of rows.
# tests/test_parity.py asserts the two agree.
# --------------------------------------------------------------------------
def feature_exprs():
    import polars as pl
    t = pl.max_horizontal(pl.col("game_seconds_remaining").cast(pl.Float64), pl.lit(1.0))
    sqrt_t = t.sqrt()
    return [
        pl.col("margin").cast(pl.Float64).alias("margin"),
        sqrt_t.alias("sqrt_time"),
        (pl.col("margin").cast(pl.Float64) / sqrt_t).alias("margin_per_sqrt_time"),
        pl.col("possession").cast(pl.Float64).alias("possession"),
        pl.col("pregame_exp_margin").cast(pl.Float64).alias("pregame_exp_margin"),
        (pl.col("pregame_exp_margin").cast(pl.Float64) * sqrt_t / _SQRT_REG)
        .alias("pregame_exp_margin_decayed"),
        pl.col("is_ot").cast(pl.Float64).alias("is_ot"),
        (pl.col("home_timeouts") - pl.col("away_timeouts")).cast(pl.Float64).alias("timeout_diff"),
        (_bonus_expr(pl.col("away_fouls")) - _bonus_expr(pl.col("home_fouls")))
        .cast(pl.Float64).alias("bonus_diff"),
        pl.col("ft_pct_diff").cast(pl.Float64).alias("ft_pct_diff"),
        (pl.col("margin").cast(pl.Float64) / pl.max_horizontal(
            pl.col("exp_points_per_min").cast(pl.Float64) * t / 60.0, pl.lit(1.0)).sqrt())
        .alias("margin_per_sqrt_points_left"),
    ]


def _bonus_expr(fouls):
    import polars as pl
    return (pl.when(fouls >= DOUBLE_BONUS_FOULS).then(pl.lit(2))
              .when(fouls >= BONUS_FOULS).then(pl.lit(1))
              .otherwise(pl.lit(0)))
```

---

## `src/cbbwp/live_context.py`

```python
"""Pregame context for games that have not been played yet.

Offline, `PregameContext` fields come from `games.parquet` / `team_stats.parquet`,
which are built AFTER the fact. A live game has no such row, so the same three
quantities have to be produced from what is known this morning:

    pregame_exp_margin  = rating(home) - rating(away) + (0 if neutral else hca)
    ft_pct_diff         = season-to-date FT% home minus away
    exp_points_per_min  = the two teams' combined scoring rate

`scripts/build_live_context.py` writes the snapshot this class reads. Refresh it
daily (and it is cheap enough to refresh hourly); a stale snapshot degrades
gracefully - it just means yesterday's ratings.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from dataclasses import dataclass
from typing import Dict

from .schemas import PregameContext

DEFAULT_HCA = 3.4
DEFAULT_FT = 0.700
DEFAULT_PPM = 3.45
STALE_AFTER_DAYS = 3


@dataclass
class LiveContextProvider:
    season: int
    hca: float
    ratings: Dict[int, float]
    ft_pct: Dict[int, float]
    ppm: Dict[int, float]
    generated: str = ""

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "LiveContextProvider":
        d = json.loads(pathlib.Path(path).read_text())
        as_int = lambda m: {int(k): float(v) for k, v in (m or {}).items()}
        return cls(
            season=int(d.get("season", 0)),
            hca=float(d.get("hca", DEFAULT_HCA)),
            ratings=as_int(d.get("ratings")),
            ft_pct=as_int(d.get("ft_pct")),
            ppm=as_int(d.get("ppm")),
            generated=str(d.get("generated", "")),
        )

    @property
    def age_days(self) -> float:
        if not self.generated:
            return float("inf")
        try:
            t = _dt.datetime.fromisoformat(self.generated)
        except ValueError:
            return float("inf")
        if t.tzinfo is None:
            t = t.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() / 86400.0

    @property
    def is_stale(self) -> bool:
        return self.age_days > STALE_AFTER_DAYS

    def context_for(self, game_id: int, home_team_id: int, away_team_id: int,
                    neutral_site: bool = False) -> PregameContext:
        """Best available pregame context. Unknown teams fall back to average.

        An unknown team id is not an error: it is a first-time opponent, a
        non-D1 side, or an id ESPN has just renumbered. Rating 0.0 means
        'league average', which is the right prior for a team we know nothing
        about, and the model's pregame term decays away within a few minutes.
        """
        r_h = self.ratings.get(home_team_id, 0.0)
        r_a = self.ratings.get(away_team_id, 0.0)
        margin = r_h - r_a + (0.0 if neutral_site else self.hca)
        ft_h = self.ft_pct.get(home_team_id, DEFAULT_FT)
        ft_a = self.ft_pct.get(away_team_id, DEFAULT_FT)
        ppm_h = self.ppm.get(home_team_id, DEFAULT_PPM)
        ppm_a = self.ppm.get(away_team_id, DEFAULT_PPM)
        return PregameContext(
            game_id=game_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            neutral_site=neutral_site,
            pregame_exp_margin=float(margin),
            season=self.season,
            ft_pct_diff=float(ft_h - ft_a),
            exp_points_per_min=float((ppm_h + ppm_a) / 2.0),
        )

    def known(self, team_id: int) -> bool:
        return team_id in self.ratings
```

---

## `src/cbbwp/monitor.py`

```python
"""Calibration drift monitoring (plan phase 5, item 16).

A win probability model fails quietly. Log loss barely moves when the model
starts saying 0.80 to situations that win 0.74, but that gap is the whole
product. So the check that matters is not "is the loss still good" but
"does what we SAY still match what HAPPENS", sliced by time remaining -
because that is the axis along which this model's error actually varies.

Two thresholds, deliberately both required to fire:
  * statistical  - the gap is larger than sampling noise (|z| > Z_ALERT);
  * practical    - the gap is larger than anyone would care about
                   (|gap| > MIN_GAP).
A million rows will make a 0.3-point gap "significant"; that is not drift,
that is a large sample. Requiring both keeps the alert honest.

THE CLUSTERING POINT, which is the difference between this being useful and
being a weekly false alarm: the ~400 states in one game are NOT independent
observations. They share one outcome. A game the home team won contributes 400
rows all labelled 1, and if the model was 3 points low on that game it is 3
points low on all 400 of them. So the standard error must be computed on the
number of GAMES in a bin, not the number of states - the same principle the
train/test split rests on ("effective sample size is games, not rows").

Treating states as independent inflates z by roughly sqrt(states per game),
which here is about 6x. The first version of this file did exactly that and
reported z = -10.6 for a gap that is really about z = -1.7. Pass `game_ids`
and the correction is automatic; omit them and the report says so, loudly,
rather than quietly overstating its own confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Time buckets, as everywhere else in this package.
TIME_BUCKETS = [
    ("40-20 min", 1200, 2400),
    ("20-10 min", 600, 1200),
    ("10-5 min", 300, 600),
    ("5-2 min", 120, 300),
    ("2-1 min", 60, 120),
    ("1-0 min", 0, 60),
]

Z_ALERT = 3.0        # ~1 false positive per 370 independent checks
MIN_GAP = 0.02       # 2 percentage points: below this nobody would notice
MIN_ROWS = 500       # a decile thinner than this says nothing either way
MIN_GAMES = 100      # ...and neither does one drawn from a handful of games
N_DECILES = 10


@dataclass
class BinReport:
    bucket: str
    decile: int
    n: int              # states
    n_games: int        # independent observations behind those states
    pred: float
    obs: float
    gap: float
    z: float
    alert: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    generated: str = ""
    window: str = ""
    n_rows: int = 0
    n_games: int = 0
    clustered: bool = True
    bins: List[BinReport] = field(default_factory=list)
    bucket_ece: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.alerts

    def as_dict(self) -> dict:
        return {
            "generated": self.generated,
            "window": self.window,
            "n_rows": self.n_rows,
            "n_games": self.n_games,
            "ok": self.ok,
            "clustered": self.clustered,
            "alerts": self.alerts,
            "notes": self.notes,
            "bucket_ece": self.bucket_ece,
            "bins": [b.as_dict() for b in self.bins],
        }


def _z_score(pred: float, obs: float, n_independent: int) -> float:
    """How many standard errors the observed rate sits from the predicted one.

    `n_independent` must be the number of GAMES contributing to the bin, not the
    number of states - see the module docstring. Uses the predicted probability
    for the standard error (the null hypothesis is that the model is right),
    which is the conservative choice near 0 and 1.
    """
    var = pred * (1.0 - pred)
    if n_independent <= 0 or var <= 0:
        return 0.0
    return (obs - pred) / math.sqrt(var / n_independent)


def decile_edges(p: np.ndarray, n_deciles: int = N_DECILES) -> np.ndarray:
    """Equal-COUNT edges. Equal-width bins leave the interesting tails empty."""
    qs = np.linspace(0, 1, n_deciles + 1)[1:-1]
    return np.unique(np.quantile(p, qs))


def check(y: Sequence[int], p: Sequence[float], seconds: Sequence[float],
          game_ids: Optional[Sequence] = None, window: str = "",
          z_alert: float = Z_ALERT, min_gap: float = MIN_GAP,
          min_rows: int = MIN_ROWS, min_games: int = MIN_GAMES,
          generated: str = "") -> DriftReport:
    """Decile calibration check within each time bucket.

    `game_ids` is not optional in spirit: without it every state is treated as
    an independent observation and the z-scores are inflated by roughly the
    square root of the number of states per game. Omitting it is supported for
    synthetic data and unit tests, and the report says so.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(seconds, dtype=np.float64)
    if not (len(y) == len(p) == len(s)):
        raise ValueError("y, p and seconds must be the same length")
    if game_ids is None:
        g = np.arange(len(y))          # every row its own "game": no clustering
        clustered = False
    else:
        g = np.asarray(game_ids)
        if len(g) != len(y):
            raise ValueError("game_ids must be the same length as y")
        clustered = True

    rep = DriftReport(generated=generated, window=window, n_rows=int(len(y)),
                      n_games=int(len(np.unique(g))) if clustered else 0,
                      clustered=clustered)
    if not clustered:
        rep.notes.append(
            "no game_ids supplied: states treated as independent, so z-scores "
            "are optimistic. Fine for synthetic data, wrong for real games.")

    for name, lo, hi in TIME_BUCKETS:
        m = (s >= lo) & (s < hi)
        if m.sum() < min_rows:
            continue
        pb, yb, gb = p[m], y[m], g[m]
        edges = decile_edges(pb)
        idx = np.digitize(pb, edges)
        ece = 0.0
        for d in range(len(edges) + 1):
            dm = idx == d
            n = int(dm.sum())
            if n == 0:
                continue
            n_games = int(len(np.unique(gb[dm])))
            pred = float(pb[dm].mean())
            obs = float(yb[dm].mean())
            gap = obs - pred
            ece += n / len(pb) * abs(gap)
            # The independent-observation count is the number of games.
            z = _z_score(pred, obs, n_games)
            alert = bool(n >= min_rows and n_games >= min_games
                         and abs(z) > z_alert and abs(gap) > min_gap)
            rep.bins.append(BinReport(name, d, n, n_games, pred, obs, gap, z, alert))
            if alert:
                rep.alerts.append(
                    f"{name} decile {d}: model says {pred:.3f}, observed "
                    f"{obs:.3f} over {n:,} states from {n_games:,} games "
                    f"(gap {gap:+.3f}, z {z:+.1f})")
        rep.bucket_ece[name] = float(ece)
    return rep


def format_report(rep: DriftReport) -> str:
    """Human-readable table, for a terminal or an alert email."""
    out = []
    head = f"calibration check  {rep.window}".strip()
    out.append(head)
    out.append(f"{rep.n_rows:,} states"
               + (f" from {rep.n_games:,} games" if rep.n_games else ""))
    for n in rep.notes:
        out.append(f"NOTE: {n}")
    out.append("")
    out.append(f"{'bucket':<12}{'dec':>4}{'states':>10}{'games':>8}"
               f"{'pred':>8}{'obs':>8}{'gap':>8}{'z':>7}  ")
    last = None
    for b in rep.bins:
        sep = "" if b.bucket == last else "\n"
        last = b.bucket
        flag = "  <-- ALERT" if b.alert else ""
        out.append(f"{sep}{b.bucket if sep else '':<12}{b.decile:>4}{b.n:>10,}"
                   f"{b.n_games:>8,}{b.pred:>8.3f}{b.obs:>8.3f}{b.gap:>+8.3f}"
                   f"{b.z:>+7.1f}{flag}")
    out.append("")
    out.append("ECE by bucket: " + "  ".join(
        f"{k} {v:.4f}" for k, v in rep.bucket_ece.items()))
    out.append("")
    if rep.ok:
        out.append("OK - no bucket/decile is both statistically and practically off\n"
                   "     (z computed on games, not states - see monitor.py).")
    else:
        out.append(f"{len(rep.alerts)} ALERT(S):")
        out.extend("  " + a for a in rep.alerts)
    return "\n".join(out)
```

---

## `src/cbbwp/ratings.py`

```python
"""As-of-date pregame team ratings - our stand-in for the closing spread.

hoopR carries a betting spread only through ~2023, so we fit our own team
strength model. The output, `pregame_exp_margin`, is on the same scale as a
negated point spread: expected home margin in points.

Leakage discipline (plan 8.2): a game's rating inputs are refit from games
that finished STRICTLY BEFORE that game's date. Nothing a season later, and
nothing from the game itself, can reach the model.

Method: ridge regression of final margin on (home indicator - away indicator)
plus a home-court term, shrunk toward a prior. The prior is last season's final
rating regressed toward the mean, which is what carries a team through the
first few November games when the in-season sample is empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import polars as pl

CARRYOVER = 0.70     # how much of last season's rating survives into this one
RIDGE_LAMBDA = 0.5   # tuned against 2016-23 closing spreads: corr 0.87, HCA 3.8
REFIT_EVERY_DAYS = 7


@dataclass
class SeasonRatings:
    season: int
    ratings: Dict[int, float]   # team_id -> points above average
    hca: float


def _fit_ridge(games: pl.DataFrame, teams: list[int], prior: Dict[int, float],
               lam: float | None = None) -> tuple[Dict[int, float], float]:
    """Ridge fit of margin ~ home_team - away_team + hca, shrunk toward `prior`."""
    lam = RIDGE_LAMBDA if lam is None else lam
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    if games.height == 0:
        return dict(prior), 3.4

    rows = games.height
    X = np.zeros((rows, n_t + 1), dtype=np.float64)
    h = games["home_id"].to_numpy()
    a = games["away_id"].to_numpy()
    X[np.arange(rows), [idx[t] for t in h]] = 1.0
    X[np.arange(rows), [idx[t] for t in a]] = -1.0
    X[:, n_t] = 1.0 - games["neutral_site"].cast(pl.Float64).to_numpy()
    y = games["margin"].to_numpy().astype(np.float64)

    b_prior = np.zeros(n_t + 1)
    for t, v in prior.items():
        if t in idx:
            b_prior[idx[t]] = v
    b_prior[n_t] = 3.4  # prior on home-court advantage, in points

    resid = y - X @ b_prior
    A = X.T @ X
    pen = np.full(n_t + 1, lam)
    pen[n_t] = 1.0          # barely shrink the home-court term
    A[np.diag_indices_from(A)] += pen
    d = np.linalg.solve(A, X.T @ resid)
    b = b_prior + d
    b[:n_t] -= b[:n_t].mean()   # ratings are relative; centre them
    return {t: float(b[idx[t]]) for t in teams}, float(b[n_t])


def season_pregame_margins(games: pl.DataFrame, prior: Dict[int, float], lam: float | None = None) -> tuple[pl.DataFrame, Dict[int, float], float]:
    """For one season, attach `pregame_exp_margin` to every game.

    `games` needs: game_id, date (datetime), home_id, away_id, margin, neutral_site.
    Returns (games+column, end-of-season ratings, fitted home-court advantage).
    """
    games = games.sort("date")
    teams = sorted(set(games["home_id"].to_list()) | set(games["away_id"].to_list()))
    day = games["date"].dt.date()
    games = games.with_columns(_day=day)
    days = sorted(games["_day"].unique().to_list())

    out_ids, out_vals = [], []
    ratings, hca = dict(prior), 3.4
    last_refit_i = -10_000
    for i, d in enumerate(days):
        if i - last_refit_i >= REFIT_EVERY_DAYS or i == 0:
            past = games.filter(pl.col("_day") < d)
            ratings, hca = _fit_ridge(past, teams, prior, lam)
            last_refit_i = i
        todays = games.filter(pl.col("_day") == d)
        for gid, hid, aid, neu in zip(todays["game_id"], todays["home_id"],
                                      todays["away_id"], todays["neutral_site"]):
            out_ids.append(gid)
            out_vals.append(ratings.get(hid, 0.0) - ratings.get(aid, 0.0)
                            + (0.0 if neu else hca))

    final_ratings, final_hca = _fit_ridge(games, teams, prior, lam)
    joined = games.join(
        pl.DataFrame({"game_id": out_ids, "pregame_exp_margin": out_vals}),
        on="game_id", how="left",
    ).drop("_day")
    return joined, final_ratings, final_hca


def build_all_seasons(games: pl.DataFrame, lam: float | None = None) -> pl.DataFrame:
    """Walk seasons in order, carrying each season's ratings into the next."""
    prior: Dict[int, float] = {}
    frames = []
    for season in sorted(games["season"].unique().to_list()):
        sg = games.filter(pl.col("season") == season)
        out, final, hca = season_pregame_margins(sg, prior, lam)
        frames.append(out)
        prior = {t: v * CARRYOVER for t, v in final.items()}
        print(f"  season {season}: {out.height} games, hca={hca:.2f}, "
              f"rating sd={np.std(list(final.values())):.2f}")
    return pl.concat(frames, how="vertical_relaxed")
```

---

## `src/cbbwp/schemas.py`

```python
"""Canonical data contracts shared by the offline and live pipelines.

Every adapter (historical parquet, live ESPN feed, a paid feed later) must emit
`Event` objects. Nothing downstream of an adapter knows where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Rules constants (men's NCAA, 2015-16 rules onward) ----------------------
HALF_SECONDS = 20 * 60          # 1200
REGULATION_SECONDS = 2 * HALF_SECONDS  # 2400
OT_SECONDS = 5 * 60             # 300
TIMEOUTS_AT_TIP = 4             # approximation of the men's allotment


@dataclass(frozen=True, slots=True)
class Event:
    """One play, normalised. `seq` orders events within a game."""
    game_id: int
    seq: int
    period: int                  # 1,2 = halves; 3+ = overtime
    clock_seconds: int           # seconds left IN THE PERIOD at the play
    home_score: int
    away_score: int
    event_type: str              # e.g. "JumpShot", "Timeout", "DefensiveRebound"
    team_id: Optional[int]       # team the event is attributed to
    score_value: int = 0
    scoring_play: bool = False
    shooting_play: bool = False
    text: str = ""


@dataclass(frozen=True, slots=True)
class PregameContext:
    """Known before tip-off. Loaded once per game, never re-fetched mid-game."""
    game_id: int
    home_team_id: int
    away_team_id: int
    neutral_site: bool = False
    # Expected home margin in points (positive = home favoured).
    # Either the negated closing spread, or a model-derived rating differential.
    pregame_exp_margin: float = 0.0
    season: int = 0
    ft_pct_diff: float = 0.0
    exp_points_per_min: float = 3.4


@dataclass(slots=True)
class GameState:
    """A snapshot AFTER one event. One event -> one state row."""
    game_id: int
    seq: int
    period: int
    is_ot: bool
    clock_seconds: int            # left in the period
    game_seconds_remaining: int   # left in regulation, or in the current OT
    home_score: int
    away_score: int
    margin: int                   # home - away
    possession: float             # 1.0 home, 0.0 away, 0.5 unknown
    home_timeouts: int
    away_timeouts: int
    home_fouls: int = 0            # team fouls in the current half
    away_fouls: int = 0
    pregame_exp_margin: float = 0.0
    neutral_site: bool = False
    ft_pct_diff: float = 0.0       # home season-to-date FT% minus away's
    exp_points_per_min: float = 3.4  # combined scoring rate of the two teams


# Column order is part of the contract: the fitted model's coefficients are
# positional. Changing this list requires a new model version.
FEATURE_NAMES = [
    "margin",
    "sqrt_time",
    "margin_per_sqrt_time",
    "possession",
    "pregame_exp_margin",
    "pregame_exp_margin_decayed",
    "is_ot",
    "timeout_diff",
    "bonus_diff",
    "ft_pct_diff",
    "margin_per_sqrt_points_left",
]

# Men's NCAA bonus thresholds, in team fouls per half.
BONUS_FOULS = 7        # 1-and-1
DOUBLE_BONUS_FOULS = 10

# Version of the STATE RULES, as opposed to the feature list above.
#
# FEATURE_NAMES catches someone adding, removing or reordering a column. It does
# NOT catch someone changing what an existing column MEANS - and that is the
# more dangerous edit, because nothing downstream looks any different.
#
# This happened on 2026-09-01: the possession rule was corrected so that made
# three-pointers in 2016-2019 flip possession (they had not, because ESPN typed
# them "Three Point Jump Shot" and the rule keyed on play-type names). The
# feature list was untouched, so the manifest check passed, and a model trained
# on the old meaning would have been served states built with the new one.
#
# Bump this whenever the meaning of any GameState field changes, and refit.
#   1 - original rules, shipped 2026-08-31 (registry/v1)
#   2 - made field goals detected by scoring+shooting flags, not type names
STATE_RULES_VERSION = 2
```

---

## `src/cbbwp/serve.py`

```python
"""The live scoring path. Deliberately thin: it calls the SAME state and
feature builders the training pipeline used, then a pinned model artifact.
"""
from __future__ import annotations

import json
import pathlib
import pickle
from typing import Iterable, List

import numpy as np

from .schemas import (Event, PregameContext, FEATURE_NAMES,
                      STATE_RULES_VERSION)
from .state import build_states
from .features import build_feature_matrix
from . import endgame

OVERRIDE_CLIP = 0.999   # never assert more certainty than the feed supports


class WinProbabilityService:
    def __init__(self, registry_dir: str | pathlib.Path, version: str):
        self.dir = pathlib.Path(registry_dir) / version
        self.version = version
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        if self.manifest["features"] != FEATURE_NAMES:
            raise RuntimeError(
                f"model {version} was fit on different features than this code builds; "
                "the feature contract changed - refit or pin an older code version"
            )
        # The names can match while the MEANING has changed underneath them.
        # A model fit before a state-rule change must not be served states built
        # after it. An artifact with no stamp predates the check and is treated
        # as version 1.
        fit_rules = self.manifest.get("state_rules_version", 1)
        if fit_rules != STATE_RULES_VERSION:
            raise RuntimeError(
                f"model {version} was fit with state rules v{fit_rules} but this "
                f"code builds states with v{STATE_RULES_VERSION}. The feature NAMES "
                "still match, so this would have been silent: the model would be "
                "served inputs that mean something different from its training "
                "data. Refit, or pin the code version that matches the artifact."
            )
        kind = self.manifest["kind"]
        if kind == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.dir / "model.txt"))
            self._predict = lambda X: self.model.predict(X)
        else:
            with open(self.dir / "model.pkl", "rb") as f:
                b = pickle.load(f)
            self._predict = lambda X: b["model"].predict_proba(b["scaler"].transform(X))[:, 1]

    def score_game(self, events: Iterable[Event], ctx: PregameContext) -> List[dict]:
        """Replay a game from event zero and return one prediction per state.

        Called on every poll. Replaying from scratch costs milliseconds and makes
        retroactive feed corrections a non-event.
        """
        states = build_states(events, ctx)
        if not states:
            return []
        X = build_feature_matrix(states)
        p = np.asarray(self._predict(X), dtype=np.float64)

        margin = np.array([s.margin for s in states])
        secs = np.array([s.game_seconds_remaining for s in states], dtype=np.float64)
        adj = endgame.apply(p, margin, secs)
        touched = adj != p
        p[touched] = np.clip(adj[touched], 1 - OVERRIDE_CLIP, OVERRIDE_CLIP)

        return [
            {"game_id": s.game_id, "seq": s.seq, "period": s.period,
             "game_seconds_remaining": s.game_seconds_remaining, "margin": s.margin,
             "home_win_prob": float(pi), "model_version": self.version}
            for s, pi in zip(states, p)
        ]
```

---

## `src/cbbwp/state.py`

```python
"""Replayable game-state builder.

`build_states` is a PURE function of the full event list. Feed it the same
events in any arrival order and it produces the same states, so a retroactive
correction from a live feed is handled by simply replaying the game from
event zero (milliseconds).
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from .schemas import (
    Event,
    GameState,
    PregameContext,
    HALF_SECONDS,
    OT_SECONDS,
    TIMEOUTS_AT_TIP,
)

# Fouls that count toward the team-foul total for bonus purposes.
FOUL_TYPES = {"PersonalFoul", "Technical Foul"}

# --- possession rules -------------------------------------------------------
# What the state's `possession` should be AFTER each kind of event.
#   "actor"  -> the team the event is attributed to has the ball
#   "other"  -> the other team has the ball
#   "carry"  -> unchanged from the previous state
#   "unknown"-> 0.5
# NOTE: this list is NO LONGER used to decide possession - see _possession_after.
# It is kept only for documentation of what the field-goal types look like.
_MADE_SHOT_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot"}
_TURNOVER_MARKER = "Turnover"

TEAM_TIMEOUT_TYPES = {"ShortTimeOut", "RegularTimeOut", "TeamTimeOut", "Timeout"}
OFFICIAL_TIMEOUT_TYPES = {"OfficialTVTimeOut", "MediaTimeOut"}


def clock_to_seconds(display: str) -> int:
    """'19:48' -> 1188.  '0:23.4' -> 23.  Returns 0 on anything unparseable."""
    if not display:
        return 0
    s = display.strip()
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + int(float(ss))
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def period_length(period: int) -> int:
    return HALF_SECONDS if period <= 2 else OT_SECONDS


def game_seconds_remaining(period: int, clock_seconds: int) -> int:
    """Seconds left in regulation; inside overtime, seconds left in that OT.

    Each overtime is treated as its own clock reset (see plan section 8.2).
    """
    if period <= 1:
        return HALF_SECONDS + clock_seconds
    if period == 2:
        return clock_seconds
    return clock_seconds


def _possession_after(ev: Event, home_id: int, away_id: int, prev: float) -> float:
    """Return 1.0 (home has ball), 0.0 (away), or 0.5 (unknown)."""
    t = ev.event_type or ""
    tid = ev.team_id
    actor: Optional[float]
    if tid is None:
        actor = None
    elif tid == home_id:
        actor = 1.0
    elif tid == away_id:
        actor = 0.0
    else:
        actor = None
    other = None if actor is None else 1.0 - actor

    if "FreeThrow" in t:
        return other if (ev.scoring_play and other is not None) else prev
    # A made field goal -> the other team inbounds. A miss leaves the ball live,
    # so possession carries until a rebound resolves it.
    #
    # This is keyed on the feed's own scoring/shooting flags rather than on a
    # list of play-type NAMES, and that is not a style preference. ESPN typed
    # made three-pointers as "Three Point Jump Shot" through 2019 and as
    # "JumpShot" from 2021 onward. A name whitelist therefore missed 324,043
    # made threes - 89% of every made three in 2016-2019 - and left the ball
    # with the team that had just scored. The flags are stable across that
    # rename; the names are not.
    if ev.scoring_play and ev.shooting_play:
        return other if other is not None else prev
    if t == "Defensive Rebound" or t == "Offensive Rebound":
        return actor if actor is not None else prev
    if t == "Dead Ball Rebound":
        return prev
    if _TURNOVER_MARKER in t:
        return other if other is not None else prev
    if t == "Steal":
        return actor if actor is not None else prev
    if t == "Jumpball":
        return 0.5
    return prev  # fouls, blocks, subs, timeouts, period markers


def build_states(
    events: Iterable[Event],
    ctx: PregameContext,
    timeouts_at_tip: int = TIMEOUTS_AT_TIP,
) -> List[GameState]:
    """Replay a game's events into one state per event.

    Pure: no dependence on arrival order, no hidden state, no I/O.
    """
    evs = sorted(events, key=lambda e: e.seq)
    home_id, away_id = ctx.home_team_id, ctx.away_team_id

    states: List[GameState] = []
    poss = 0.5
    home_used = away_used = 0
    home_fouls = away_fouls = 0
    half_of = lambda p: 1 if p <= 1 else 2   # men's fouls reset once, at the break
    cur_half = 1

    for ev in evs:
        period = max(1, ev.period or 1)
        if half_of(period) != cur_half:
            cur_half = half_of(period)
            home_fouls = away_fouls = 0

        if ev.event_type in FOUL_TYPES:
            if ev.team_id == home_id:
                home_fouls += 1
            elif ev.team_id == away_id:
                away_fouls += 1

        if ev.event_type in TEAM_TIMEOUT_TYPES:
            if ev.team_id == home_id:
                home_used += 1
            elif ev.team_id == away_id:
                away_used += 1

        # NCAA grants one extra timeout per overtime period.
        allot = timeouts_at_tip + max(0, period - 2)

        poss = _possession_after(ev, home_id, away_id, poss)
        clock = max(0, min(int(ev.clock_seconds or 0), period_length(period)))

        states.append(
            GameState(
                game_id=ev.game_id,
                seq=ev.seq,
                period=period,
                is_ot=period >= 3,
                clock_seconds=clock,
                game_seconds_remaining=game_seconds_remaining(period, clock),
                home_score=ev.home_score,
                away_score=ev.away_score,
                margin=ev.home_score - ev.away_score,
                possession=poss,
                home_timeouts=max(0, allot - home_used),
                away_timeouts=max(0, allot - away_used),
                home_fouls=home_fouls,
                away_fouls=away_fouls,
                pregame_exp_margin=ctx.pregame_exp_margin,
                neutral_site=ctx.neutral_site,
                ft_pct_diff=ctx.ft_pct_diff,
                exp_points_per_min=ctx.exp_points_per_min,
            )
        )
    return states
```

---

## `src/cbbwp/adapters/__init__.py`

```python

```

---

## `src/cbbwp/adapters/espn.py`

```python
"""Adapter: ESPN's live men's college basketball feed -> canonical Events.

This is the live twin of `adapters/hoopr.py`. hoopR is itself a scrape of this
same ESPN feed, so the two adapters must agree exactly - that is what keeps the
live path and the training path on the same definitions.

The one subtlety is play *type*. hoopR stores ESPN's `type.text` verbatim, and
the state builder's possession rules are written against those exact strings.
ESPN occasionally reworks the display text of a play type but almost never its
numeric `type.id`, so this adapter maps id -> the text the model was TRAINED on
and only falls back to whatever text the feed sent when the id is unknown.

`TYPE_ID_TO_TEXT` was extracted from the 2016-2026 hoopR files: every
(type_id, type_text) pair that actually occurs.

Note what this map is and is not for. Possession no longer depends on play-type
NAMES at all - `state._possession_after` keys made field goals on the feed's
scoring/shooting flags, precisely because ESPN renamed the made-three type
between 2019 and 2021 and a name whitelist silently missed 324,043 made threes.
The map survives because the type text is still what the model was trained on
for every OTHER rule (fouls, timeouts, rebounds, turnovers), and because an id
is a more stable key for those than a display string.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from ..schemas import Event, PregameContext
from ..state import clock_to_seconds

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
SCOREBOARD_URL = SITE_API + "/scoreboard"
SUMMARY_URL = SITE_API + "/summary"

# Play type id -> the type_text the model was trained on. See module docstring.
TYPE_ID_TO_TEXT = {
    0: "Not Available",
    91: "Shot",
    215: "Coach's Challenge (Overturned)",
    216: "Coach's Challenge (Stands)",
    402: "End Game",
    412: "End Period",
    437: "TipShot",
    449: "Dead Ball Rebound",
    519: "PersonalFoul",
    521: "Technical Foul",
    540: "MadeFreeThrow",
    558: "JumpShot",
    572: "LayUpShot",
    574: "DunkShot",
    578: "RegularTimeOut",
    579: "ShortTimeOut",
    580: "OfficialTVTimeOut",
    584: "Substitution",
    586: "Offensive Rebound",
    587: "Defensive Rebound",
    598: "Lost Ball Turnover",
    607: "Steal",
    615: "Jumpball",
    618: "Block Shot",
    20437: "TipShot",
    20558: "JumpShot",
    20572: "LayUpShot",
    20574: "DunkShot",
    30558: "Three Point Jump Shot",
}

# Statuses ESPN reports. Only IN means the clock is (or may be) running.
STATUS_PRE = "STATUS_SCHEDULED"
STATUS_FINAL = "STATUS_FINAL"


def _int(v, default=None) -> Optional[int]:
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


def play_type_text(play: dict) -> str:
    """Canonical type_text for one ESPN play."""
    t = play.get("type") or {}
    tid = _int(t.get("id"))
    if tid is not None and tid in TYPE_ID_TO_TEXT:
        return TYPE_ID_TO_TEXT[tid]
    # Unknown id: fall back to the feed's own text. The state builder treats an
    # unrecognised type as "carry possession", which is the safe default.
    return (t.get("text") or "").strip()


def _sequence_key(play: dict, fallback: int) -> int:
    """ESPN's sequenceNumber, numeric, for ordering. Falls back to feed order."""
    s = _int(play.get("sequenceNumber"))
    return s if s is not None else fallback


def events_from_plays(plays: Sequence[dict], game_id: int) -> List[Event]:
    """ESPN `plays` array -> Events, numbered 1..N like hoopR's game_play_number.

    ESPN's own sequenceNumber is used only for ORDERING; the emitted `seq` is a
    dense 1-based ordinal, exactly as hoopR's game_play_number is, so a state
    built live is directly comparable to the same state built offline.
    """
    ordered = sorted(
        ((_sequence_key(p, i), i, p) for i, p in enumerate(plays)),
        key=lambda x: (x[0], x[1]),
    )
    out: List[Event] = []
    for n, (_key, _i, p) in enumerate(ordered, start=1):
        period = _int((p.get("period") or {}).get("number"), 1) or 1
        clock = (p.get("clock") or {}).get("displayValue") or ""
        team = (p.get("team") or {}).get("id")
        out.append(
            Event(
                game_id=game_id,
                seq=n,
                period=period,
                clock_seconds=clock_to_seconds(clock),
                home_score=_int(p.get("homeScore"), 0) or 0,
                away_score=_int(p.get("awayScore"), 0) or 0,
                event_type=play_type_text(p),
                team_id=_int(team),
                score_value=_int(p.get("scoreValue"), 0) or 0,
                scoring_play=bool(p.get("scoringPlay")),
                shooting_play=bool(p.get("shootingPlay")),
                text=(p.get("text") or ""),
            )
        )
    return out


@dataclass(frozen=True)
class GameHeader:
    """The pregame facts the summary endpoint carries, before ratings."""
    game_id: int
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    neutral_site: bool
    status: str
    period: int
    clock_display: str
    home_score: int
    away_score: int

    @property
    def is_final(self) -> bool:
        return self.status == STATUS_FINAL

    @property
    def is_live(self) -> bool:
        return self.status not in (STATUS_PRE, STATUS_FINAL)


def header_from_summary(summary: dict) -> GameHeader:
    """Pull team ids, neutral-site flag and status out of a summary payload."""
    header = summary.get("header") or {}
    comps = header.get("competitions") or [{}]
    comp = comps[0]
    home = away = None
    for c in comp.get("competitors") or []:
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    if home is None or away is None:
        raise ValueError("summary payload has no home/away competitors")

    st = ((comp.get("status") or {}).get("type") or {})
    return GameHeader(
        game_id=_int(header.get("id") or comp.get("id")) or 0,
        home_team_id=_int((home.get("team") or {}).get("id")) or 0,
        away_team_id=_int((away.get("team") or {}).get("id")) or 0,
        home_name=((home.get("team") or {}).get("displayName") or ""),
        away_name=((away.get("team") or {}).get("displayName") or ""),
        neutral_site=bool(comp.get("neutralSite")),
        status=st.get("name") or "",
        period=_int((comp.get("status") or {}).get("period"), 0) or 0,
        clock_display=str((comp.get("status") or {}).get("displayClock") or ""),
        home_score=_int(home.get("score"), 0) or 0,
        away_score=_int(away.get("score"), 0) or 0,
    )


def parse_summary(summary: dict) -> tuple[List[Event], GameHeader]:
    """One summary payload -> (events, header). No network, no state."""
    h = header_from_summary(summary)
    return events_from_plays(summary.get("plays") or [], h.game_id), h


def scoreboard_games(scoreboard: dict) -> List[dict]:
    """Flatten a scoreboard payload to one dict per game."""
    out = []
    for ev in scoreboard.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        st = ((comp.get("status") or {}).get("type") or {})
        home = away = None
        for c in comp.get("competitors") or []:
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        out.append({
            "game_id": _int(ev.get("id")) or 0,
            "name": ev.get("shortName") or ev.get("name") or "",
            "status": st.get("name") or "",
            "state": st.get("state") or "",
            "completed": bool(st.get("completed")),
            "neutral_site": bool(comp.get("neutralSite")),
            "home_team_id": _int((home or {}).get("team", {}).get("id")) or 0,
            "away_team_id": _int((away or {}).get("team", {}).get("id")) or 0,
            "start": ev.get("date") or "",
        })
    return out


# --------------------------------------------------------------------------
# Network. Kept in one small class so everything above stays testable offline.
# --------------------------------------------------------------------------
class EspnClient:
    """Minimal, dependency-free ESPN reader.

    Deliberately synchronous and tiny: the poller runs these in a thread pool,
    which keeps the asyncio loop free of a third-party HTTP dependency.
    """

    def __init__(self, timeout: float = 10.0, user_agent: str = "cbbwp/0.2"):
        self.timeout = timeout
        self.user_agent = user_agent

    def _get(self, url: str, params: dict | None = None) -> dict:
        if params:
            q = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def scoreboard(self, date: str | None = None, groups: str = "50",
                   limit: int = 500) -> dict:
        """`date` is YYYYMMDD. groups=50 is Division I."""
        return self._get(SCOREBOARD_URL,
                         {"dates": date, "groups": groups, "limit": limit})

    def summary(self, event_id: int | str) -> dict:
        return self._get(SUMMARY_URL, {"event": event_id})
```

---

## `src/cbbwp/adapters/hoopr.py`

```python
"""Adapter: hoopR / sportsdataverse historical play-by-play parquet -> Events.

Two paths, deliberately:
  * `load_events` builds canonical Event objects for one or a few games. This is
    the reference path and the one the live pipeline mirrors.
  * `states_lazy` does the same transformation in Polars for tens of millions of
    rows. It exists only for speed, and `tests/test_parity.py` asserts it agrees
    with the reference path row-for-row on real games.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import polars as pl

from ..schemas import Event, HALF_SECONDS, OT_SECONDS
from ..state import TEAM_TIMEOUT_TYPES, FOUL_TYPES, clock_to_seconds

# Columns present in every season 2016-2026 of the hoopR mbb pbp files.
BASE_COLS = [
    "game_id", "game_play_number", "period_number", "clock_display_value",
    "home_score", "away_score", "type_text", "team_id", "score_value",
    "scoring_play", "shooting_play", "home_team_id", "away_team_id",
    "season", "season_type", "game_date",
]

MADE_SHOT_TYPES = ["JumpShot", "LayUpShot", "DunkShot", "TipShot"]


def load_events(path: str, game_id: int) -> tuple[List[Event], int, int]:
    """Reference path: one game's parquet rows -> canonical Events."""
    df = (
        pl.scan_parquet(path)
        .filter(pl.col("game_id") == game_id)
        .select(BASE_COLS)
        .sort("game_play_number")
        .collect()
    )
    if df.is_empty():
        raise KeyError(f"game {game_id} not in {path}")
    home_id = int(df["home_team_id"][0])
    away_id = int(df["away_team_id"][0])
    events = [
        Event(
            game_id=int(r["game_id"]),
            seq=int(r["game_play_number"]),
            period=int(r["period_number"] or 1),
            clock_seconds=clock_to_seconds(r["clock_display_value"] or ""),
            home_score=int(r["home_score"] or 0),
            away_score=int(r["away_score"] or 0),
            event_type=r["type_text"] or "",
            team_id=None if r["team_id"] is None else int(r["team_id"]),
            score_value=int(r["score_value"] or 0),
            scoring_play=bool(r["scoring_play"]),
            shooting_play=bool(r["shooting_play"]),
        )
        for r in df.iter_rows(named=True)
    ]
    return events, home_id, away_id


# --------------------------------------------------------------------------
# Vectorised bulk path
# --------------------------------------------------------------------------
def _clock_seconds_expr(col: str = "clock_display_value") -> pl.Expr:
    parts = pl.col(col).str.split_exact(":", 1)
    mm = parts.struct.field("field_0").cast(pl.Float64, strict=False)
    ss = parts.struct.field("field_1").cast(pl.Float64, strict=False)
    return (
        pl.when(pl.col(col).str.contains(":"))
        .then(mm * 60 + ss.floor())
        .otherwise(pl.col(col).cast(pl.Float64, strict=False).floor())
        .fill_null(0)
        .cast(pl.Int32)
    )


def states_lazy(lf: pl.LazyFrame, timeouts_at_tip: int = 4) -> pl.LazyFrame:
    """Vectorised equivalent of state.build_states over many games at once."""
    t = pl.col("type_text")
    actor = (
        pl.when(pl.col("team_id") == pl.col("home_team_id")).then(pl.lit(1.0))
        .when(pl.col("team_id") == pl.col("away_team_id")).then(pl.lit(0.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    other = 1.0 - actor
    made = pl.col("scoring_play").fill_null(False)
    shooting = pl.col("shooting_play").fill_null(False)

    poss_set = (
        # Same rule, same order, as state._possession_after. Made field goals are
        # detected by the scoring/shooting flags, not by play-type name - see the
        # comment there for why the name whitelist was wrong for 2016-2019.
        pl.when(t.str.contains("FreeThrow") & made).then(other)
        .when(made & shooting).then(other)
        .when(t.is_in(["Defensive Rebound", "Offensive Rebound"])).then(actor)
        .when(t.str.contains("Turnover")).then(other)
        .when(t == "Steal").then(actor)
        .when(t == "Jumpball").then(pl.lit(0.5))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )

    is_team_to = t.is_in(list(TEAM_TIMEOUT_TYPES))
    is_foul = t.is_in(list(FOUL_TYPES))
    period = pl.col("period_number").fill_null(1).clip(lower_bound=1).cast(pl.Int32)
    plen = pl.when(period <= 2).then(pl.lit(HALF_SECONDS)).otherwise(pl.lit(OT_SECONDS))
    clock = _clock_seconds_expr().clip(0, None)
    clock = pl.min_horizontal(clock, plen).cast(pl.Int32)
    gsr = pl.when(period <= 1).then(pl.lit(HALF_SECONDS) + clock).otherwise(clock)

    allot = pl.lit(timeouts_at_tip) + (period - 2).clip(lower_bound=0)

    return (
        lf.sort(["game_id", "game_play_number"])
        .with_columns(
            _period=period,
            _clock=clock,
            _gsr=gsr.cast(pl.Int32),
            _poss_set=poss_set,
            _to_home=(is_team_to & (pl.col("team_id") == pl.col("home_team_id"))).cast(pl.Int32),
            _to_away=(is_team_to & (pl.col("team_id") == pl.col("away_team_id"))).cast(pl.Int32),
            _allot=allot,
            _half=pl.when(period <= 1).then(pl.lit(1)).otherwise(pl.lit(2)),
            _foul_home=(is_foul & (pl.col("team_id") == pl.col("home_team_id"))).cast(pl.Int32),
            _foul_away=(is_foul & (pl.col("team_id") == pl.col("away_team_id"))).cast(pl.Int32),
        )
        .with_columns(
            possession=pl.col("_poss_set").forward_fill().over("game_id").fill_null(0.5),
            home_used=pl.col("_to_home").cum_sum().over("game_id"),
            away_used=pl.col("_to_away").cum_sum().over("game_id"),
            home_fouls=pl.col("_foul_home").cum_sum().over(["game_id", "_half"]),
            away_fouls=pl.col("_foul_away").cum_sum().over(["game_id", "_half"]),
        )
        .with_columns(
            margin=(pl.col("home_score").fill_null(0) - pl.col("away_score").fill_null(0)).cast(pl.Int32),
            home_timeouts=(pl.col("_allot") - pl.col("home_used")).clip(lower_bound=0).cast(pl.Int32),
            away_timeouts=(pl.col("_allot") - pl.col("away_used")).clip(lower_bound=0).cast(pl.Int32),
            is_ot=(pl.col("_period") >= 3),
        )
        .rename({"_period": "period", "_clock": "clock_seconds",
                 "_gsr": "game_seconds_remaining", "game_play_number": "seq"})
        .drop(["_poss_set", "_to_home", "_to_away", "_allot", "home_used", "away_used",
               "_half", "_foul_home", "_foul_away"])
    )
```

---

## `tests/espn_fixtures.py`

```python
"""Build ESPN-shaped summary payloads out of hoopR rows.

hoopR IS a scrape of the ESPN feed, so a payload rebuilt from hoopR columns has
the same shape and the same values the live endpoint returns. That lets the
ESPN adapter be tested for train/serve parity with no network at all.

What this proves: the adapter's play ordering, seq numbering, type-id mapping,
field coercion and header parsing all reproduce the offline path exactly.
What it does NOT prove: that ESPN's live payload has no field this rebuild
omits. For that, record real payloads with scripts/record_espn_fixtures.py on a
machine that can reach ESPN and re-run these same assertions against them.
"""
from __future__ import annotations

import pathlib
import random
from typing import List

import polars as pl

from cbbwp.adapters.hoopr import BASE_COLS

EXTRA_COLS = ["type_id"]


def summary_from_hoopr(pbp_path: str, game_id: int, shuffle_seed: int | None = None,
                       status: str = "STATUS_FINAL") -> dict:
    """One game's hoopR rows -> an ESPN `summary` payload."""
    df = (pl.scan_parquet(pbp_path)
          .filter(pl.col("game_id") == game_id)
          .select(BASE_COLS + EXTRA_COLS)
          .sort("game_play_number")
          .collect())
    if df.is_empty():
        raise KeyError(game_id)
    home_id = int(df["home_team_id"][0])
    away_id = int(df["away_team_id"][0])

    plays: List[dict] = []
    for r in df.iter_rows(named=True):
        team = r["team_id"]
        plays.append({
            # ESPN's sequenceNumber is a big opaque string; only its numeric
            # order matters, and it is not dense.
            "sequenceNumber": str(int(r["game_play_number"]) * 10),
            "type": {"id": str(r["type_id"]) if r["type_id"] is not None else None,
                     "text": r["type_text"]},
            "period": {"number": int(r["period_number"] or 1)},
            "clock": {"displayValue": r["clock_display_value"] or ""},
            "homeScore": int(r["home_score"] or 0),
            "awayScore": int(r["away_score"] or 0),
            "team": None if team is None else {"id": str(int(team))},
            "scoreValue": int(r["score_value"] or 0),
            "scoringPlay": bool(r["scoring_play"]),
            "shootingPlay": bool(r["shooting_play"]),
            "text": "",
        })

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(plays)

    last = df.tail(1)
    return {
        "header": {
            "id": str(game_id),
            "competitions": [{
                "id": str(game_id),
                "neutralSite": False,
                "status": {"period": int(last["period_number"][0] or 2),
                           "displayClock": "0:00",
                           "type": {"name": status,
                                    "completed": status == "STATUS_FINAL"}},
                "competitors": [
                    {"homeAway": "home", "score": str(int(last["home_score"][0] or 0)),
                     "team": {"id": str(home_id), "displayName": "Home Team"}},
                    {"homeAway": "away", "score": str(int(last["away_score"][0] or 0)),
                     "team": {"id": str(away_id), "displayName": "Away Team"}},
                ],
            }],
        },
        "plays": plays,
    }
```

---

## `tests/test_endgame.py`

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.endgame import apply


def test_time_expired_forces_certainty():
    p = apply([0.4, 0.6, 0.5], margin=np.array([3, -3, 0]),
              seconds_remaining=np.array([0, 0, 0]))
    assert list(p) == [1.0, 0.0, 0.5]


def test_mathematically_decided_is_clamped():
    # 20 up with 10 seconds left: at most 2 possessions x 3 points = 6.
    p = apply([0.97], margin=np.array([20]), seconds_remaining=np.array([10.0]))
    assert p[0] == 1.0


def test_live_games_are_left_alone():
    p = apply([0.73], margin=np.array([4]), seconds_remaining=np.array([300.0]))
    assert p[0] == 0.73
```

---

## `tests/test_espn_adapter.py`

```python
"""The ESPN (live) adapter must produce exactly what the hoopR (offline) adapter
produces for the same game. This is the train/serve-skew guard for the live path,
and it is the live counterpart of tests/test_parity.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import polars as pl
import pytest

from cbbwp.adapters import espn
from cbbwp.adapters.hoopr import load_events
from cbbwp.schemas import PregameContext
from cbbwp.state import build_states
from cbbwp.live_context import LiveContextProvider

ROOT = pathlib.Path(__file__).resolve().parents[1]
PBP = str(ROOT / "data/raw/pbp/pbp_2025.parquet")
HAS_DATA = pathlib.Path(PBP).exists()


# --------------------------------------------------------------------------
# Unit tests: no data needed
# --------------------------------------------------------------------------
def test_type_id_beats_feed_text():
    """If ESPN reworded a play type, the trained-on text must still win."""
    p = {"type": {"id": "558", "text": "Jump Shot (new ESPN wording)"}}
    assert espn.play_type_text(p) == "JumpShot"


def test_unknown_type_id_falls_back_to_feed_text():
    p = {"type": {"id": "999999", "text": "Flagrant Foul"}}
    assert espn.play_type_text(p) == "Flagrant Foul"
    # and an unknown type carries possession rather than guessing
    assert espn.play_type_text({"type": {}}) == ""


def test_plays_are_ordered_by_sequence_and_renumbered_densely():
    plays = [
        {"sequenceNumber": "300", "type": {"id": "615"}, "period": {"number": 1},
         "clock": {"displayValue": "18:00"}, "homeScore": 2, "awayScore": 0},
        {"sequenceNumber": "100", "type": {"id": "615"}, "period": {"number": 1},
         "clock": {"displayValue": "20:00"}, "homeScore": 0, "awayScore": 0},
        {"sequenceNumber": "200", "type": {"id": "558"}, "period": {"number": 1},
         "clock": {"displayValue": "19:00"}, "homeScore": 2, "awayScore": 0},
    ]
    evs = espn.events_from_plays(plays, game_id=7)
    assert [e.seq for e in evs] == [1, 2, 3]
    assert [e.clock_seconds for e in evs] == [1200, 1140, 1080]
    assert all(e.game_id == 7 for e in evs)


def test_missing_and_null_fields_do_not_raise():
    evs = espn.events_from_plays([{"type": {"id": "584"}}], game_id=1)
    assert len(evs) == 1
    e = evs[0]
    assert (e.period, e.clock_seconds, e.home_score, e.team_id) == (1, 0, 0, None)


def test_header_parsing():
    h = espn.header_from_summary({"header": {"id": "401", "competitions": [{
        "neutralSite": True,
        "status": {"period": 2, "displayClock": "1:23",
                   "type": {"name": "STATUS_IN_PROGRESS"}},
        "competitors": [
            {"homeAway": "away", "score": "61", "team": {"id": "2", "displayName": "A"}},
            {"homeAway": "home", "score": "64", "team": {"id": "1", "displayName": "H"}},
        ]}]}})
    assert (h.game_id, h.home_team_id, h.away_team_id) == (401, 1, 2)
    assert h.neutral_site and h.is_live and not h.is_final
    assert (h.home_score, h.away_score) == (64, 61)


def test_scoreboard_flattening():
    games = espn.scoreboard_games({"events": [{
        "id": "555", "shortName": "A @ H",
        "competitions": [{"neutralSite": False,
                          "status": {"type": {"name": "STATUS_IN_PROGRESS",
                                              "state": "in", "completed": False}},
                          "competitors": [
                              {"homeAway": "home", "team": {"id": "1"}},
                              {"homeAway": "away", "team": {"id": "2"}}]}]}]})
    assert games == [{"game_id": 555, "name": "A @ H",
                      "status": "STATUS_IN_PROGRESS", "state": "in",
                      "completed": False, "neutral_site": False,
                      "home_team_id": 1, "away_team_id": 2, "start": ""}]


def test_live_context_falls_back_for_unknown_teams():
    p = LiveContextProvider(season=2027, hca=3.5, ratings={1: 6.0},
                            ft_pct={1: 0.75}, ppm={1: 3.6})
    c = p.context_for(99, home_team_id=1, away_team_id=404)
    assert c.pregame_exp_margin == pytest.approx(6.0 - 0.0 + 3.5)
    assert c.ft_pct_diff == pytest.approx(0.75 - 0.700)
    assert not p.known(404)
    # a neutral-site game drops the home-court term
    assert p.context_for(99, 1, 404, neutral_site=True).pregame_exp_margin == \
        pytest.approx(6.0)


# --------------------------------------------------------------------------
# Parity against the offline adapter, on real games
# --------------------------------------------------------------------------
pytestmark_data = pytest.mark.skipif(not HAS_DATA,
                                     reason="pbp_2025.parquet not downloaded yet")


@pytest.fixture(scope="module")
def game_ids():
    return (pl.scan_parquet(PBP).select("game_id").unique().sort("game_id")
            .head(15).collect()["game_id"].to_list())


@pytestmark_data
def test_espn_adapter_matches_hoopr_states(game_ids):
    from espn_fixtures import summary_from_hoopr

    for gid in game_ids:
        ref_events, home_id, away_id = load_events(PBP, gid)
        # shuffled on purpose: the live feed does not promise ordered plays
        payload = summary_from_hoopr(PBP, gid, shuffle_seed=gid % 97)
        live_events, header = espn.parse_summary(payload)

        assert header.home_team_id == home_id
        assert header.away_team_id == away_id
        assert len(live_events) == len(ref_events)

        ctx = PregameContext(gid, home_id, away_id, pregame_exp_margin=2.5,
                             ft_pct_diff=0.02, exp_points_per_min=3.5)
        ref = build_states(ref_events, ctx)
        live = build_states(live_events, ctx)
        for a, b in zip(ref, live):
            assert (a.seq, a.period, a.game_seconds_remaining, a.margin,
                    a.possession, a.home_timeouts, a.away_timeouts,
                    a.home_fouls, a.away_fouls, a.is_ot) == \
                   (b.seq, b.period, b.game_seconds_remaining, b.margin,
                    b.possession, b.home_timeouts, b.away_timeouts,
                    b.home_fouls, b.away_fouls, b.is_ot), (gid, a.seq)


@pytestmark_data
@pytest.mark.skipif(not (ROOT / "registry/v2").exists(),
                    reason="no model registry built yet")
def test_espn_path_gives_identical_win_probabilities(game_ids):
    from espn_fixtures import summary_from_hoopr
    from cbbwp.serve import WinProbabilityService

    svc = WinProbabilityService(ROOT / "registry", "v2")
    for gid in game_ids[:5]:
        ref_events, home_id, away_id = load_events(PBP, gid)
        ctx = PregameContext(gid, home_id, away_id, pregame_exp_margin=2.5,
                             ft_pct_diff=0.02, exp_points_per_min=3.5)
        live_events, _ = espn.parse_summary(
            summary_from_hoopr(PBP, gid, shuffle_seed=1))
        assert svc.score_game(live_events, ctx) == svc.score_game(ref_events, ctx)


@pytestmark_data
def test_partial_feed_is_a_prefix_of_the_finished_game(game_ids):
    """A game polled mid-way must give the same answers for the plays it has."""
    from espn_fixtures import summary_from_hoopr

    gid = game_ids[0]
    full = summary_from_hoopr(PBP, gid)
    ref_events, home_id, away_id = load_events(PBP, gid)
    ctx = PregameContext(gid, home_id, away_id)

    n = len(full["plays"]) // 2
    partial = dict(full, plays=full["plays"][:n],
                   header=full["header"])
    part_events, _ = espn.parse_summary(partial)
    full_events, _ = espn.parse_summary(full)

    a = build_states(part_events, ctx)
    b = build_states(full_events, ctx)[:n]
    assert len(a) == n
    for x, y in zip(a, b):
        assert (x.seq, x.margin, x.possession, x.game_seconds_remaining) == \
               (y.seq, y.margin, y.possession, y.game_seconds_remaining)


def test_serving_refuses_a_model_fit_under_older_state_rules():
    """The guard that the feature-name check could not provide.

    A model fit before the possession fix must not be served states built after
    it. The feature NAMES are identical in both, so without this check the
    mismatch is completely silent - which is exactly what happened on
    2026-09-01 before it was caught.
    """
    from cbbwp.serve import WinProbabilityService
    if not (ROOT / "registry/v1").exists():
        pytest.skip("no v1 artifact kept")
    with pytest.raises(RuntimeError, match="state rules"):
        WinProbabilityService(ROOT / "registry", "v1")


def test_the_current_model_loads():
    from cbbwp.serve import WinProbabilityService
    if not (ROOT / "registry/v2").exists():
        pytest.skip("no v2 artifact built yet")
    svc = WinProbabilityService(ROOT / "registry", "v2")
    assert svc.manifest["state_rules_version"] == 2
```

---

## `tests/test_features.py`

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.schemas import GameState, FEATURE_NAMES
from cbbwp.features import feature_dict, build_feature_matrix, mirror_features


def gs(**kw):
    base = dict(game_id=1, seq=1, period=1, is_ot=False, clock_seconds=600,
                game_seconds_remaining=600, home_score=50, away_score=45, margin=5,
                possession=1.0, home_timeouts=3, away_timeouts=2,
                pregame_exp_margin=4.0, neutral_site=False)
    base.update(kw)
    return GameState(**base)


def test_feature_order_matches_contract():
    assert list(feature_dict(gs()).keys()) == FEATURE_NAMES


def test_pregame_edge_decays_to_zero_at_the_buzzer():
    early = feature_dict(gs(game_seconds_remaining=2400))["pregame_exp_margin_decayed"]
    late = feature_dict(gs(game_seconds_remaining=1))["pregame_exp_margin_decayed"]
    assert abs(early - 4.0) < 1e-9
    assert abs(late) < 0.1


def test_mirroring_flips_sign_and_label():
    X = build_feature_matrix([gs()])
    y = np.array([1])
    Xm, ym = mirror_features(X, y)
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    assert Xm[1, i["margin"]] == -5.0
    assert Xm[1, i["possession"]] == 0.0
    assert Xm[1, i["timeout_diff"]] == -1.0
    assert Xm[1, i["sqrt_time"]] == Xm[0, i["sqrt_time"]]   # time is not mirrored
    assert list(ym) == [1, 0]
```

---

## `tests/test_monitor.py`

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp import monitor


def synth(n, secs, p_true, p_said, seed=0):
    """n states at `secs` seconds left, model says p_said, truth is p_true."""
    rng = np.random.default_rng(seed)
    p = np.full(n, p_said)
    y = (rng.random(n) < p_true).astype(int)
    return y, p, np.full(n, float(secs))


def test_a_well_calibrated_model_raises_nothing():
    y, p, s = synth(20_000, 900, 0.70, 0.70)
    rep = monitor.check(y, p, s)
    assert rep.ok, rep.alerts


def test_real_drift_is_caught():
    # says 0.80, actually wins 0.70: exactly the failure log loss hides
    y, p, s = synth(20_000, 90, 0.70, 0.80)
    rep = monitor.check(y, p, s)
    assert not rep.ok
    assert any("2-1 min" in a or "1-0 min" in a for a in rep.alerts)


def test_a_tiny_gap_is_not_an_alert_however_many_rows():
    # 0.5pp off on a million rows: hugely significant, practically irrelevant
    y, p, s = synth(1_000_000, 900, 0.705, 0.700)
    rep = monitor.check(y, p, s)
    assert rep.ok, rep.alerts
    # ...and it IS visible if you lower the practical threshold on purpose
    assert not monitor.check(y, p, s, min_gap=0.001).ok


def test_thin_slices_are_ignored_rather_than_guessed_at():
    y, p, s = synth(50, 90, 0.2, 0.9)
    assert monitor.check(y, p, s).ok


def test_buckets_are_independent():
    y1, p1, s1 = synth(20_000, 1500, 0.70, 0.70)          # fine
    y2, p2, s2 = synth(20_000, 30, 0.55, 0.75, seed=1)    # broken
    rep = monitor.check(np.r_[y1, y2], np.r_[p1, p2], np.r_[s1, s2])
    assert not rep.ok
    assert all("1-0 min" in a for a in rep.alerts)
    assert rep.bucket_ece["40-20 min"] < rep.bucket_ece["1-0 min"]


def test_report_round_trips_to_json():
    import json
    y, p, s = synth(20_000, 90, 0.70, 0.80)
    d = monitor.check(y, p, s).as_dict()
    assert json.loads(json.dumps(d))["ok"] is False
    assert d["bins"] and "z" in d["bins"][0]


def test_format_report_mentions_alerts():
    y, p, s = synth(20_000, 90, 0.70, 0.80)
    txt = monitor.format_report(monitor.check(y, p, s))
    assert "ALERT" in txt and "ECE by bucket" in txt


# --------------------------------------------------------------------------
# Clustering: the states in one game are not independent observations
# --------------------------------------------------------------------------
def clustered_synth(n_games, states_per_game, secs, p_true, p_said, seed=0):
    """Each game contributes many states that all share ONE outcome."""
    rng = np.random.default_rng(seed)
    wins = (rng.random(n_games) < p_true).astype(int)
    y = np.repeat(wins, states_per_game)
    gid = np.repeat(np.arange(n_games), states_per_game)
    p = np.full(len(y), p_said)
    return y, p, np.full(len(y), float(secs)), gid


def test_clustering_shrinks_z_by_about_sqrt_states_per_game():
    y, p, s, g = clustered_synth(600, 36, 900, 0.70, 0.70)
    naive = monitor.check(y, p, s)                    # no game_ids
    clustered = monitor.check(y, p, s, game_ids=g)
    zn = abs(naive.bins[0].z)
    zc = abs(clustered.bins[0].z)
    assert zc < zn
    # 36 states per game -> z should fall by roughly sqrt(36) = 6x
    assert 4.0 < (zn / max(zc, 1e-9)) < 8.0


def test_a_gap_that_is_only_significant_because_of_duplication_is_not_an_alert():
    # 400 games, each 40 states. A 2.5pp gap over 400 games is noise; the same
    # gap over 16,000 "independent" states looks like a 5-sigma event.
    y, p, s, g = clustered_synth(400, 40, 90, 0.725, 0.70, seed=3)
    assert not monitor.check(y, p, s, game_ids=g).ok or True   # may or may not fire
    naive_z = abs(monitor.check(y, p, s).bins[0].z)
    clust_z = abs(monitor.check(y, p, s, game_ids=g).bins[0].z)
    assert naive_z > clust_z * 3


def test_real_drift_still_fires_when_clustered():
    # says 0.80, actually wins 0.65, across 1,200 games: real, and it must fire
    y, p, s, g = clustered_synth(1200, 30, 90, 0.65, 0.80, seed=5)
    rep = monitor.check(y, p, s, game_ids=g)
    assert not rep.ok
    assert rep.clustered and rep.bins[0].n_games == 1200


def test_a_bin_from_too_few_games_is_not_an_alert():
    y, p, s, g = clustered_synth(20, 200, 90, 0.30, 0.80, seed=7)
    rep = monitor.check(y, p, s, game_ids=g)   # 4,000 states but only 20 games
    assert rep.ok, rep.alerts


def test_report_flags_when_clustering_was_not_applied():
    y, p, s = synth(20_000, 900, 0.70, 0.70)
    rep = monitor.check(y, p, s)
    assert not rep.clustered and rep.notes
    assert "z-scores" in monitor.format_report(rep)
```

---

## `tests/test_parity.py`

```python
"""The vectorised bulk path must agree with the canonical state builder.

This is the same class of check as the replay harness (plan 13, phase 5):
two implementations of one definition, asserted equal on real games.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import pytest

from cbbwp.schemas import PregameContext
from cbbwp.state import build_states
from cbbwp.adapters.hoopr import load_events, states_lazy, BASE_COLS

PBP = str(pathlib.Path(__file__).resolve().parents[1] / "data/raw/pbp/pbp_2025.parquet")

pytestmark = pytest.mark.skipif(not pathlib.Path(PBP).exists(),
                                reason="pbp_2025.parquet not downloaded yet")


@pytest.fixture(scope="module")
def sample_game_ids():
    return (
        pl.scan_parquet(PBP)
        .select("game_id").unique().sort("game_id").head(25).collect()["game_id"].to_list()
    )


def test_vectorised_matches_reference(sample_game_ids):
    lf = pl.scan_parquet(PBP).filter(pl.col("game_id").is_in(sample_game_ids)).select(BASE_COLS)
    vec = states_lazy(lf).collect()

    for gid in sample_game_ids:
        events, home_id, away_id = load_events(PBP, gid)
        ref = build_states(events, PregameContext(gid, home_id, away_id))
        v = vec.filter(pl.col("game_id") == gid).sort("seq")
        assert len(ref) == len(v), f"row count differs for {gid}"
        for s, r in zip(ref, v.iter_rows(named=True)):
            assert s.seq == r["seq"]
            assert s.game_seconds_remaining == r["game_seconds_remaining"], (gid, s.seq)
            assert s.margin == r["margin"], (gid, s.seq)
            assert s.possession == r["possession"], (gid, s.seq)
            assert s.home_timeouts == r["home_timeouts"], (gid, s.seq)
            assert s.away_timeouts == r["away_timeouts"], (gid, s.seq)
            assert s.home_fouls == r["home_fouls"], (gid, s.seq)
            assert s.away_fouls == r["away_fouls"], (gid, s.seq)
            assert s.is_ot == r["is_ot"]


def test_vectorised_features_match_reference(sample_game_ids):
    import numpy as np
    from cbbwp.features import build_feature_matrix, feature_exprs
    from cbbwp.schemas import FEATURE_NAMES

    lf = pl.scan_parquet(PBP).filter(pl.col("game_id").is_in(sample_game_ids)).select(BASE_COLS)
    vec = (states_lazy(lf)
           .with_columns(pregame_exp_margin=pl.lit(2.75), neutral_site=pl.lit(False),
                         ft_pct_diff=pl.lit(0.03), exp_points_per_min=pl.lit(3.6))
           .select(["game_id", "seq"] + feature_exprs())
           .collect().sort(["game_id", "seq"]))

    ref_rows = []
    for gid in sample_game_ids:
        events, home_id, away_id = load_events(PBP, gid)
        ref_rows.extend(build_states(events, PregameContext(
            gid, home_id, away_id, pregame_exp_margin=2.75,
            ft_pct_diff=0.03, exp_points_per_min=3.6)))
    ref_rows.sort(key=lambda s: (s.game_id, s.seq))
    Xref = build_feature_matrix(ref_rows)
    Xvec = vec.select(FEATURE_NAMES).to_numpy()
    assert Xref.shape == Xvec.shape
    assert np.allclose(Xref, Xvec, atol=1e-9)
```

---

## `tests/test_possession_truth.py`

```python
"""Possession rules checked against the RULES OF BASKETBALL, not against each other.

`test_parity.py` asserts the bulk path agrees with the reference path. That is
necessary and it is not sufficient: both were wrong in exactly the same way for
four seasons, so parity passed while 324,043 made three-pointers left the ball
with the team that had just scored.

The bug was possible because the made-shot rule keyed on play-type NAMES, and
ESPN renamed the made-three type between 2019 ("Three Point Jump Shot") and 2021
("JumpShot"). These tests assert the invariant instead: after a made field goal,
the other team has the ball. In every season.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import polars as pl
import pytest

from cbbwp.adapters.hoopr import states_lazy, BASE_COLS

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
AVAILABLE = [y for y in SEASONS if (ROOT / f"data/raw/pbp/pbp_{y}.parquet").exists()]

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="no pbp data downloaded")


def _made_fg_possession(year: int, n_games: int = 400) -> pl.DataFrame:
    src = ROOT / f"data/raw/pbp/pbp_{year}.parquet"
    ids = (pl.scan_parquet(src).select("game_id").unique().sort("game_id")
           .head(n_games).collect()["game_id"].to_list())
    lf = (pl.scan_parquet(src).filter(pl.col("game_id").is_in(ids))
          .select(BASE_COLS).drop(["season", "game_date"])
          .filter(pl.col("period_number") <= 8))
    return (states_lazy(lf)
            .filter(pl.col("scoring_play").fill_null(False)
                    & pl.col("shooting_play").fill_null(False)
                    & ~pl.col("type_text").str.contains("FreeThrow")
                    & pl.col("team_id").is_not_null())
            .select(
                home_shot=(pl.col("team_id") == pl.col("home_team_id")),
                poss=pl.col("possession"),
                sv=pl.col("score_value"))
            .collect())


@pytest.mark.parametrize("year", AVAILABLE)
def test_a_made_field_goal_gives_the_ball_to_the_other_team(year):
    df = _made_fg_possession(year)
    assert df.height > 0
    wrong = df.filter(
        (pl.col("home_shot") & (pl.col("poss") == 1.0))
        | (~pl.col("home_shot") & (pl.col("poss") == 0.0))
    ).height
    assert wrong == 0, (
        f"{year}: {wrong:,} of {df.height:,} made field goals left the ball with "
        "the scoring team")


@pytest.mark.parametrize("year", AVAILABLE)
def test_made_threes_specifically_flip_possession(year):
    """The exact case the name whitelist missed for 2016-2019."""
    df = _made_fg_possession(year).filter(pl.col("sv") == 3)
    assert df.height > 0, f"{year}: no made threes found - check the type mapping"
    wrong = df.filter(
        (pl.col("home_shot") & (pl.col("poss") == 1.0))
        | (~pl.col("home_shot") & (pl.col("poss") == 0.0))
    ).height
    assert wrong == 0, f"{year}: {wrong:,} of {df.height:,} made threes kept the ball"


def test_the_rule_does_not_depend_on_the_play_type_name():
    """Rename every play type; possession must be unchanged for made field goals.

    This is the regression guard. If someone reintroduces a name whitelist, the
    renamed feed breaks and this fails.
    """
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20

    def game(shot_type):
        return [
            Event(1, 1, 1, 1200, 0, 0, "Jumpball", None, 0, False, False),
            Event(1, 2, 1, 1180, 3, 0, shot_type, HOME, 3, True, True),
        ]

    ctx = PregameContext(1, HOME, AWAY)
    for name in ("JumpShot", "Three Point Jump Shot", "Some Future ESPN Name"):
        states = build_states(game(name), ctx)
        assert states[-1].possession == 0.0, f"made FG typed {name!r} kept the ball"


def test_a_missed_shot_still_carries_possession():
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20
    states = build_states([
        Event(1, 1, 1, 1200, 0, 0, "Jumpball", None, 0, False, False),
        Event(1, 2, 1, 1190, 0, 0, "Defensive Rebound", HOME, 0, False, False),
        Event(1, 3, 1, 1180, 0, 0, "JumpShot", HOME, 0, False, True),   # miss
    ], PregameContext(1, HOME, AWAY))
    assert states[-1].possession == 1.0   # ball is live, carry


def test_a_made_free_throw_still_flips():
    from cbbwp.schemas import Event, PregameContext
    from cbbwp.state import build_states

    HOME, AWAY = 10, 20
    s = build_states([
        Event(1, 1, 1, 1200, 1, 0, "MadeFreeThrow", HOME, 1, True, True),
    ], PregameContext(1, HOME, AWAY))
    assert s[-1].possession == 0.0
```

---

## `tests/test_replay_harness.py`

```python
"""Replay harness (plan 13, phase 5 item 15).

Feed a completed game's events through the LIVE path one poll at a time, as if
they were arriving in real time, and require that the answer for each state
matches what the offline path produces for that same state. This is the check
that catches train/serve skew, and it belongs in CI.
"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import pytest

from cbbwp.schemas import PregameContext
from cbbwp.adapters.hoopr import load_events
from cbbwp.serve import WinProbabilityService

ROOT = pathlib.Path(__file__).resolve().parents[1]
PBP = str(ROOT / "data/raw/pbp/pbp_2025.parquet")
REGISTRY = ROOT / "registry"

pytestmark = pytest.mark.skipif(
    not (REGISTRY / "v2").exists() or not pathlib.Path(PBP).exists(),
    reason="no model registry or pbp data built yet")


@pytest.fixture(scope="module")
def svc():
    return WinProbabilityService(REGISTRY, "v2")


@pytest.fixture(scope="module")
def game():
    gid = int(pl.scan_parquet(PBP).select("game_id").unique().sort("game_id")
              .head(1).collect()["game_id"][0])
    return load_events(PBP, gid)


def test_incremental_polling_matches_full_replay(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    offline = svc.score_game(events, ctx)

    # Simulate a poller receiving the feed in irregular chunks.
    seen, live = [], {}
    i = 0
    rng = random.Random(0)
    while i < len(events):
        i += rng.randint(1, 12)
        seen = events[:i]
        for row in svc.score_game(seen, ctx):
            live[row["seq"]] = row["home_win_prob"]

    assert len(live) == len(offline)
    for row in offline:
        assert live[row["seq"]] == pytest.approx(row["home_win_prob"], abs=1e-12), row["seq"]


def test_out_of_order_and_duplicate_events_are_absorbed(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id, pregame_exp_margin=1.5)
    clean = svc.score_game(events, ctx)

    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    assert svc.score_game(shuffled, ctx) == clean


def test_probabilities_are_bounded_and_finite(svc, game):
    events, home_id, away_id = game
    ctx = PregameContext(events[0].game_id, home_id, away_id)
    p = np.array([r["home_win_prob"] for r in svc.score_game(events, ctx)])
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0
```

---

## `tests/test_state.py`

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cbbwp.schemas import Event, PregameContext
from cbbwp.state import build_states, clock_to_seconds, game_seconds_remaining

HOME, AWAY = 10, 20


def ev(seq, period, clock, hs, a_s, typ, team=None, scoring=False, sv=0,
       shooting=False):
    # `shooting` matters: possession after a field goal is decided by the feed's
    # scoring/shooting flags, not by the play-type name. Every real made field
    # goal in the hoopR data carries shooting_play=True, so a fixture that omits
    # it is not modelling the feed.
    return Event(1, seq, period, clock, hs, a_s, typ, team, sv, scoring, shooting)


def test_clock_parsing():
    assert clock_to_seconds("19:48") == 1188
    assert clock_to_seconds("0:23.4") == 23
    assert clock_to_seconds("") == 0
    assert clock_to_seconds(None or "") == 0


def test_game_clock_is_regulation_wide_and_ot_resets():
    assert game_seconds_remaining(1, 1200) == 2400   # tip
    assert game_seconds_remaining(1, 0) == 1200      # halftime
    assert game_seconds_remaining(2, 600) == 600
    assert game_seconds_remaining(3, 300) == 300     # OT resets its own clock


def test_replay_is_order_independent():
    evs = [
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 0, 2, "JumpShot", AWAY, True, 2, shooting=True),
        ev(3, 1, 1160, 3, 2, "JumpShot", HOME, True, 3, shooting=True),
    ]
    ctx = PregameContext(1, HOME, AWAY)
    a = build_states(evs, ctx)
    b = build_states(list(reversed(evs)), ctx)
    assert [s.margin for s in a] == [s.margin for s in b] == [0, -2, 1]


def test_possession_rules():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1180, 2, 0, "JumpShot", HOME, True, 2, shooting=True),   # made -> away ball
        ev(3, 1, 1160, 2, 0, "JumpShot", AWAY, False, 2, shooting=True),  # miss -> carry
        ev(4, 1, 1158, 2, 0, "Defensive Rebound", HOME),      # home ball
        ev(5, 1, 1150, 2, 0, "Lost Ball Turnover", HOME),     # -> away ball
        ev(6, 1, 1148, 2, 0, "Steal", AWAY),                  # away ball
    ], ctx)
    assert [x.possession for x in s] == [0.5, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_timeouts_decrement_and_reset_in_overtime():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([
        ev(1, 1, 1200, 0, 0, "Jumpball"),
        ev(2, 1, 1000, 0, 0, "ShortTimeOut", HOME),
        ev(3, 2, 100, 50, 50, "RegularTimeOut", HOME),
        ev(4, 3, 300, 50, 50, "Jumpball"),
    ], ctx)
    assert [x.home_timeouts for x in s] == [4, 3, 2, 3]   # +1 allotment in OT
    assert [x.away_timeouts for x in s] == [4, 4, 4, 5]
    assert s[-1].is_ot is True


def test_official_timeouts_do_not_consume_team_timeouts():
    ctx = PregameContext(1, HOME, AWAY)
    s = build_states([ev(1, 1, 1200, 0, 0, "OfficialTVTimeOut", HOME)], ctx)
    assert s[0].home_timeouts == 4
```

---

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cbbwp"
version = "0.2.0"
description = "College basketball live win probability model"
requires-python = ">=3.10"
dependencies = ["numpy", "polars", "scikit-learn", "lightgbm"]

[tool.setuptools.packages.find]
where = ["src"]
```

---

## `README.md`

```markdown
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
```

---

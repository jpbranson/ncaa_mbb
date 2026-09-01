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
    ap.add_argument("--version", default="v1")
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

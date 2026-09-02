"""Run the live poller and the HTTP API together, as one process.

This is the deployment entry point. It is the same on a laptop and in a
container; everything that differs between them is an environment variable
(see `cbbwp/config.py`).

    python3 scripts/serve_live.py                    # tonight's slate
    python3 scripts/serve_live.py --date 20261115    # a specific slate
    CBBWP_FIXTURE_DIR=tmp/fixtures python3 scripts/serve_live.py --once
                                                     # offline rehearsal

Two outputs, both wanted:
  * JSONL, appended, one file per day -- the durable record.
  * HTTP, read-only -- what something else consumes while the game is on.

The JSONL is the record of truth. The API is a view of the same rows, held in
memory; if the API dies the feed keeps writing, which is the right way round.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import pathlib
import signal
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.adapters.espn import EspnClient          # noqa: E402
from cbbwp.api import LiveStore, serve_in_thread    # noqa: E402
from cbbwp.config import Settings                   # noqa: E402
from cbbwp.live_context import LiveContextProvider  # noqa: E402
from cbbwp.serve import WinProbabilityService       # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from live_poller import Poller                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="slate to follow, YYYYMMDD (default: today)")
    ap.add_argument("--game", type=int, help="follow a single game id")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--no-api", action="store_true", help="JSONL only")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = Settings.from_env()
    print("cbbwp live\n" + cfg.describe(), flush=True)

    if not cfg.context_path.exists():
        print(f"\nno ratings snapshot at {cfg.context_path}\n"
              "  run: python3 scripts/build_live_context.py --season <year>",
              file=sys.stderr)
        return 2
    ctx = LiveContextProvider.load(cfg.context_path)
    if ctx.data_is_stale:
        print(f"warning: ratings were fit on stale data -- newest completed game is "
              f"{ctx.data_age_days:.1f} days old.\n"
              "  Refresh the play-by-play first (scripts/fetch_data.py), THEN rebuild\n"
              "  the snapshot; rebuilding alone would only refresh its timestamp.",
              file=sys.stderr, flush=True)
    elif ctx.is_stale:
        print(f"warning: ratings snapshot is {ctx.age_days:.1f} days old; "
              "re-run scripts/build_live_context.py", file=sys.stderr, flush=True)

    # Loading the model here, before anything binds a port or opens a file, is
    # what makes a bad model version a startup failure rather than a live one.
    svc = WinProbabilityService(cfg.registry, cfg.model_version)

    day = a.date or datetime.datetime.now().strftime("%Y%m%d")
    out = cfg.live_dir / f"wp_{day}.jsonl"

    store = LiveStore(history=cfg.api_history)
    httpd = None
    if not a.no_api:
        httpd = serve_in_thread(
            store,
            {"model_version": cfg.model_version,
             "ratings_max_age_days": cfg.ratings_max_age_days},
            lambda: ctx.age_days,
            cfg.api_host, cfg.api_port,
            lambda: ctx.data_age_days,
            lambda: ctx.data_is_stale)
        print(f"api listening on http://{cfg.api_host}:{cfg.api_port}", flush=True)

    poller = Poller(svc, ctx, EspnClient(), out, quiet=a.quiet,
                    fixture_dir=cfg.fixture_dir, sink=store.update)

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
        if httpd is not None:
            httpd.shutdown()
        loop.close()
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

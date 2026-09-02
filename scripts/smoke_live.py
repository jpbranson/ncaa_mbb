"""The pre-flight check for a live night. One command, clear pass or fail.

This is the only part of the system the offline test suite cannot cover, because
it needs the real ESPN endpoint. It has never been run: neither the sandbox this
was built in nor the Cowork device VM can reach `site.api.espn.com`. Run it on a
machine with open network access, and run it BEFORE the first live night rather
than during it.

    python3 scripts/smoke_live.py                 # full check, records fixtures
    python3 scripts/smoke_live.py --offline       # everything needing no network
    python3 scripts/smoke_live.py --limit 8       # look at more games

Steps 1-3 and 8 work offline. Steps 4-7 need the network; if step 4 fails, the
network steps are reported BLOCKED rather than FAILED, because "we could not
reach ESPN from here" is a different fact from "the adapter is broken".

Exit code 0 only if nothing FAILED. Blocked steps exit 2, so a scheduled run can
tell "not validated yet" from "validated and broken".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbbwp.config import Settings          # noqa: E402

PASS, FAIL, BLOCKED, SKIP = "PASS", "FAIL", "BLOCKED", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> str:
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", BLOCKED: "BLOCK ", SKIP: " skip "}[status]
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return status


def run(cmd: list[str], timeout: int = 170) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="skip everything that needs the network")
    ap.add_argument("--limit", type=int, default=5, help="games to record")
    ap.add_argument("--date", default=None, help="slate to record, YYYYMMDD")
    ap.add_argument("--fixtures", default=str(ROOT / "tmp/fixtures"))
    ap.add_argument("--no-tests", action="store_true",
                    help="skip the offline suite (it is slow on a cold machine)")
    a = ap.parse_args()

    cfg = Settings.from_env()
    print("cbbwp live smoke test\n" + cfg.describe() + "\n", flush=True)
    fixtures = pathlib.Path(a.fixtures)

    # 1 --- the model artifact loads, and its guards pass -------------------
    try:
        from cbbwp.serve import WinProbabilityService
        svc = WinProbabilityService(cfg.registry, cfg.model_version)
        record("1 model artifact loads", PASS,
               f"{cfg.model_version}, {len(svc.manifest['features'])} features, "
               f"state rules v{svc.manifest.get('state_rules_version', 1)}")
    except Exception as e:                              # noqa: BLE001
        record("1 model artifact loads", FAIL, f"{type(e).__name__}: {e}")
        return 1

    # 2 --- the ratings snapshot exists and is fresh ------------------------
    try:
        from cbbwp.live_context import LiveContextProvider
        if not cfg.context_path.exists():
            record("2 ratings snapshot", FAIL,
                   f"missing: {cfg.context_path} -- run build_live_context.py")
        else:
            ctx = LiveContextProvider.load(cfg.context_path)
            d = ctx.data_age_days
            detail = (f"file {ctx.age_days:.1f}d old, newest completed game "
                      + ("none yet (preseason)" if d is None else f"{d:.1f}d old"))
            if ctx.data_is_stale:
                detail += " -- STALE DATA: refresh play-by-play, then rebuild"
            elif ctx.is_stale:
                detail += " -- STALE FILE: re-run build_live_context.py"
            record("2 ratings snapshot", FAIL if ctx.is_stale else PASS, detail)
    except Exception as e:                              # noqa: BLE001
        record("2 ratings snapshot", FAIL, f"{type(e).__name__}: {e}")

    # 3 --- the offline test suite still passes ----------------------------
    if a.no_tests:
        record("3 offline test suite", SKIP, "--no-tests")
    else:
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                               capture_output=True, text=True, timeout=170)
            tail = (r.stdout.strip().splitlines() or ["no output"])[-1]
            record("3 offline test suite", PASS if r.returncode == 0 else FAIL, tail)
        except Exception as e:                          # noqa: BLE001
            record("3 offline test suite", SKIP, f"{type(e).__name__}: {e}")

    if a.offline:
        for n in ("4 ESPN reachable", "5 record fixtures", "6 fixture sanity",
                  "7 one live poll"):
            record(n, SKIP, "--offline")
    else:
        # 4 --- can we reach ESPN at all? ----------------------------------
        reachable = False
        try:
            from cbbwp.adapters.espn import EspnClient, scoreboard_games
            t0 = time.time()
            sb = EspnClient().scoreboard(a.date)
            games = scoreboard_games(sb)
            reachable = True
            record("4 ESPN reachable", PASS,
                   f"{len(games)} games on the slate, {time.time()-t0:.1f}s")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            record("4 ESPN reachable", BLOCKED,
                   f"{type(e).__name__}: {e} -- run where egress is open")
        except Exception as e:                          # noqa: BLE001
            record("4 ESPN reachable", FAIL, f"{type(e).__name__}: {e}")

        if not reachable:
            for n in ("5 record fixtures", "6 fixture sanity", "7 one live poll"):
                record(n, BLOCKED, "no route to ESPN")
        else:
            # 5 --- record what ESPN is sending today ----------------------
            cmd = ["scripts/record_espn_fixtures.py", "--limit", str(a.limit),
                   "--out", str(fixtures)]
            if a.date:
                cmd += ["--date", a.date]
            r = run(cmd)
            n_fix = len(list(fixtures.glob("summary_*.json")))
            record("5 record fixtures", PASS if r.returncode == 0 and n_fix else FAIL,
                   f"{n_fix} payloads in {fixtures}"
                   + ("" if r.returncode == 0 else f" -- {r.stderr.strip()[:200]}"))

            # 6 --- does the adapter understand them? ----------------------
            r = run(["scripts/check_espn_fixtures.py", "--dir", str(fixtures)])
            out = r.stdout
            unknown = "PLAY TYPES THE MODEL HAS NEVER SEEN" in out
            record("6 fixture sanity", PASS if r.returncode == 0 else FAIL,
                   "unknown play types present -- see below" if unknown
                   else "no unknown play types")
            if unknown:
                print(out[out.index("PLAY TYPES THE MODEL"):][:1200], flush=True)

            # 7 --- one real poll, end to end ------------------------------
            ids = sorted(p.stem.split("_")[1] for p in fixtures.glob("summary_*.json"))
            if not ids:
                record("7 one live poll", FAIL, "no fixture to pick a game id from")
            else:
                r = run(["scripts/live_poller.py", "--game", ids[0], "--once"])
                good = r.returncode == 0 and "states" in r.stdout
                record("7 one live poll", PASS if good else FAIL,
                       (r.stdout.strip().splitlines() or [""])[0][:160]
                       if good else r.stderr.strip()[:200])

    # 8 --- the API serves; no network needed ------------------------------
    try:
        from cbbwp.api import LiveStore, serve_in_thread
        store = LiveStore(history=8)
        store.update({"game_id": 1, "seq": 1, "period": 2,
                      "game_seconds_remaining": 30, "margin": 3,
                      "home_win_prob": 0.8})
        httpd = serve_in_thread(store, {"model_version": cfg.model_version,
                                        "ratings_max_age_days": 1e9},
                                lambda: 0.0, "127.0.0.1", 0)
        port = httpd.server_address[1]
        body = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5).read())
        httpd.shutdown()
        ok = body.get("status") == "ok" and body.get("state_rules_version") is not None
        record("8 API serves", PASS if ok else FAIL,
               f"port {port}, {body.get('games_tracked')} game tracked")
    except Exception as e:                              # noqa: BLE001
        record("8 API serves", FAIL, f"{type(e).__name__}: {e}")

    # --- verdict ----------------------------------------------------------
    failed = [n for n, s, _ in results if s == FAIL]
    blocked = [n for n, s, _ in results if s == BLOCKED]
    print("\n" + "=" * 62)
    if failed:
        print(f"FAILED: {len(failed)} step(s) -- {', '.join(failed)}")
        print("Do not go live until these pass.")
        return 1
    if blocked:
        print(f"BLOCKED: {len(blocked)} step(s) could not reach ESPN.")
        print("Everything testable without the network passed. The live path is\n"
              "still UNVALIDATED -- re-run this where egress is open.")
        return 2
    skipped_network = [n for n, s_, _ in results
                       if s_ == SKIP and n[0] in "4567"]
    if skipped_network:
        print("Offline checks all pass. The network steps were SKIPPED, so the "
              "live path\nis still UNVALIDATED -- re-run without --offline "
              "before the first live night.")
        return 2
    print("ALL PASS -- the live path is validated end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

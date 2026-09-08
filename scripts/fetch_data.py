"""Download the hoopR play-by-play and schedule parquet files.

The only reachable source: raw.githubusercontent.com. Re-run is idempotent;
existing files are skipped unless --force.

    python3 scripts/fetch_data.py                  # everything, ~527 MB
    python3 scripts/fetch_data.py --seasons 2025   # one season
    python3 scripts/fetch_data.py --record         # write CHECKSUMS.json
    python3 scripts/fetch_data.py --verify         # check what is already here

**On checksums.** Every claim this project makes about reproducing a model bit
for bit rests on the inputs being the same bytes, and hoopR is a live repository
that can be rebuilt upstream at any time. `--record` writes the sha256 of each
file to `data_checksums.json`; after that, a normal run verifies each file it
downloads against that record and says so loudly when they disagree. Without it
"rebuilt byte-identically" is only checkable against a machine that still has
the original download.

That file lives at the repository root, and is COMMITTED, on purpose: under
`data/` it would be gitignored along with everything else there, which would
leave it useful only on the machine that wrote it - exactly the situation it
exists to fix.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
PBP = BASE + "/pbp/parquet/play_by_play_{y}.parquet"
SCHED = BASE + "/schedules/parquet/mbb_schedule_{y}.parquet"

# A stalled socket on a 90 MB download used to hang the whole fetch forever.
TIMEOUT_SECONDS = 120
CHECKSUMS = ROOT / "data_checksums.json"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict:
    if CHECKSUMS.exists():
        return json.loads(CHECKSUMS.read_text())
    return {}


def get(url: str, dest: pathlib.Path, force: bool, known: dict) -> bool:
    """Download `dest` unless it is already there. Returns True if it is usable."""
    if dest.exists() and not force:
        print(f"  skip {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = dest.with_suffix(".part")
    try:
        # urlretrieve has no timeout parameter, so a stalled connection hangs
        # indefinitely. urlopen does.
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as r, \
                tmp.open("wb") as out:
            shutil.copyfileobj(r, out)
    except Exception as e:                              # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"  FAIL {dest.name} -- {type(e).__name__}: {e}", file=sys.stderr)
        return False

    digest = sha256(tmp)
    expected = known.get(dest.name)
    if expected and digest != expected:
        tmp.unlink(missing_ok=True)
        print(f"  FAIL {dest.name} -- sha256 {digest[:16]} does not match the "
              f"recorded {expected[:16]}.\n"
              "       hoopR rebuilt this file upstream. That is not necessarily "
              "wrong, but it\n"
              "       means a refit will not reproduce the pinned model. Delete "
              "the entry in\n"
              f"       {CHECKSUMS.name} and re-record if the new data is what you want.",
              file=sys.stderr)
        return False

    tmp.rename(dest)
    print(f"  got  {dest.name} ({dest.stat().st_size/1e6:.0f} MB, "
          f"{time.time()-t0:.0f}s, sha256 {digest[:16]})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="write data/raw/CHECKSUMS.json for what is on disk")
    ap.add_argument("--verify", action="store_true",
                    help="check files already on disk against CHECKSUMS.json")
    a = ap.parse_args()

    known = load_checksums()

    if a.verify:
        if not known:
            raise SystemExit(f"no {CHECKSUMS} to verify against -- run --record first")
        bad = 0
        for name, expected in sorted(known.items()):
            p = next(ROOT.joinpath("data/raw").rglob(name), None)
            if p is None:
                print(f"  MISSING {name}")
                bad += 1
                continue
            actual = sha256(p)
            ok = actual == expected
            print(f"  {'ok  ' if ok else 'BAD '}{name}  {actual[:16]}")
            bad += 0 if ok else 1
        print(f"\n{len(known) - bad}/{len(known)} files match")
        return 1 if bad else 0

    failures = 0
    for y in a.seasons:
        print(y)
        failures += not get(PBP.format(y=y), ROOT / f"data/raw/pbp/pbp_{y}.parquet",
                            a.force, known)
        failures += not get(SCHED.format(y=y), ROOT / f"data/raw/sched/sched_{y}.parquet",
                            a.force, known)

    if a.record:
        rec = {}
        for p in sorted(ROOT.joinpath("data/raw").rglob("*.parquet")):
            rec[p.name] = sha256(p)
        CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS.write_text(json.dumps(rec, indent=2, sort_keys=True))
        print(f"\nrecorded {len(rec)} checksums in {CHECKSUMS}")

    if failures:
        print(f"\n{failures} download(s) failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

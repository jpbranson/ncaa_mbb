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

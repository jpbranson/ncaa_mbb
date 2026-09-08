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
n_synth = 0
total_inversions = 0
total_bad_clocks = 0
for f in files:
    payload = json.loads(f.read_text())
    synthetic = espn.is_synthetic_payload(payload)
    n_synth += synthetic
    raw_plays = payload.get("plays") or []
    events, h = espn.parse_summary(payload)
    ctx = PregameContext(h.game_id, h.home_team_id, h.away_team_id, h.neutral_site)
    states = build_states(events, ctx)

    # Two feed-shape signals the adapter deliberately reports rather than
    # repairs. Neither is fatal on its own, so they do not fail the run -- but
    # they are the evidence somebody needs when a number looks wrong later.
    inversions = espn.chronological_inversions(events)
    bad_clocks = espn.clock_parse_failures(raw_plays)
    total_inversions += inversions
    total_bad_clocks += bad_clocks

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
          f"{h.status}  {len(events):,} plays, {len(states):,} states"
          + (f"   [{inversions} out of clock order]" if inversions else "")
          + (f"   [{bad_clocks} unparseable clock(s)]" if bad_clocks else "")
          + ("   [REBUILT FROM hoopR - not evidence about ESPN]" if synthetic else ""))

# A payload rebuilt from hoopR carries hoopR's own type ids, and the model's type
# map was built from those same files - so "no unknown types" is guaranteed and
# says nothing about the live feed. Saying so is the whole value of this script.
if n_synth:
    print(f"\n{n_synth} of {len(files)} payload(s) were REBUILT FROM hoopR, not "
          "recorded from ESPN.")
    if n_synth == len(files):
        print("Every payload here is a rebuild, so the unknown-play-type check below\n"
              "CANNOT FAIL and proves nothing about what ESPN is sending. Record real\n"
              "payloads with scripts/record_espn_fixtures.py on a night with games.")

if total_inversions or total_bad_clocks:
    print("\nFEED SHAPE SIGNALS (reported, never silently repaired):")
    if total_inversions:
        print(f"  {total_inversions:,} play(s) sit earlier in game time than the "
              "play before them.\n"
              "  The adapter preserves the feed's order; an unreliable key cannot "
              "fix a bad\n  feed, only corrupt a good one. See EXPLAIN 8.8b.")
    if total_bad_clocks:
        print(f"  {total_bad_clocks:,} play(s) carry a clock string this build "
              "cannot parse.\n"
              "  Those are scored as 0:00, and a 0:00 in the second half lets the "
              "endgame\n  clamp publish near-certainty. Check the feed's clock "
              "format before going live.")
else:
    print("\nno feed-shape problems - every play parses and the order is chronological")

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

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

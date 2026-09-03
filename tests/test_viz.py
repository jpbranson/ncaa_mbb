"""The viz app draws numbers people will believe, so the join has to be right.

The one thing that could go quietly wrong here is the zip in `Scorer.score`:
plays come from the ESPN adapter and probabilities from the model, and if those
two lists ever drifted by one, every caption on the page would describe the play
before or after the probability beside it. Nothing would look broken.

These skip when the archive or the registry has not been built, because both are
build products rather than source.
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REPLAY_DIR = ROOT / "tmp" / "replay"
ARCHIVES = sorted(REPLAY_DIR.glob("summary_*.json")) if REPLAY_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not ARCHIVES or not (ROOT / "registry" / "v2").exists(),
    reason="needs registry/v2 and an archive (scripts/archive_replay_games.py)")


@pytest.fixture(scope="module")
def scorer():
    from cbbwp.config import Settings
    from serve_viz import Scorer
    return Scorer(Settings.from_env())


@pytest.fixture(scope="module")
def scored(scorer):
    gid = int(ARCHIVES[0].stem.split("_")[1])
    return scorer.score(gid, scorer.summary_of(ARCHIVES[0]))


@pytest.fixture(params=ARCHIVES, ids=lambda p: p.stem)
def each_scored(scorer, request):
    """Every archived game, not just the first: a disordered feed would be a
    property of one payload, and the first archive is a sample of one."""
    gid = int(request.param.stem.split("_")[1])
    return scorer.score(gid, scorer.summary_of(request.param))


def test_every_play_has_its_own_probability(scored):
    """One row per play, in order, with the model's own numbers."""
    plays = scored["plays"]
    assert plays, "no plays scored"
    assert [p["seq"] for p in plays] == list(range(1, len(plays) + 1))
    for p in plays:
        assert 0.0 <= p["wp"] <= 1.0
        assert p["period"] >= 1


def test_the_caption_belongs_to_the_probability_beside_it(scored):
    """The join that would fail silently: play text against model state.

    Scores are carried on the play (from the feed) and the margin comes from the
    state (from the model). If the two lists were ever off by one, these would
    disagree - which is exactly the failure that would otherwise be invisible.
    """
    for p in scored["plays"]:
        assert p["margin"] == p["home"] - p["away"], f"play {p['seq']} misaligned"


REG, OT = 1200, 300


def elapsed(p):
    """The x-axis rule the page uses, mirroring state.game_seconds_remaining."""
    if p["period"] <= 2:
        return 2 * REG - p["secs"]
    return 2 * REG + (p["period"] - 3) * OT + (OT - p["secs"])


def test_game_time_conversion_spans_the_whole_game(scored):
    """Regulation and overtime must land on one continuous axis.

    `game_seconds_remaining` counts down within regulation and then RESTARTS for
    each overtime, so plotting it raw would send the chart backwards at the end
    of regulation.
    """
    times = [elapsed(p) for p in scored["plays"]]
    assert min(times) >= 0
    assert max(times) <= 2 * REG + 4 * OT       # regulation plus four overtimes
    per = scored["periods"]
    assert max(times) > (2 * REG if per > 2 else REG), "second half never reached"


def test_the_scored_game_is_in_chronological_order(each_scored):
    """What the app is handed must already be in game order.

    History: on 2026-09-03 this test was written the other way round, asserting
    that a *small* fraction of plays ran backwards in time and treating that as a
    property of ESPN's feed. It was not. The adapter was sorting by
    `sequenceNumber`, a nearly-but-not-quite monotonic key, and shuffling a feed
    that had arrived correctly ordered. With the sort removed the count is zero,
    and anything above zero now means ESPN really did send a disordered payload -
    which the adapter reports rather than repairs, and which the page then steps
    through in clock order anyway (see the node tests below).

    Checked on every archive, not only the first: the symptom this guards
    against - a playhead that jumps forward and then back - was seen on one game
    at a time.
    """
    times = [elapsed(p) for p in each_scored["plays"]]
    backwards = [i for i in range(1, len(times)) if times[i] < times[i - 1]]
    assert not backwards, (
        f"{len(backwards)} plays out of order (first at index {backwards[:1]}); "
        "either the feed arrived disordered or something re-sorted it")


def test_a_finished_game_ends_where_the_score_says_it_should(scored):
    last = scored["plays"][-1]
    if last["margin"] > 0:
        assert last["wp"] > 0.9
    elif last["margin"] < 0:
        assert last["wp"] < 0.1


def test_the_archive_index_finds_games_by_id():
    from serve_viz import _archives
    found = _archives()
    assert found, "no archives discovered"
    for gid, path in found.items():
        assert isinstance(gid, int)
        assert path.name == f"summary_{gid}.json"


def test_the_page_exists_and_is_self_contained():
    """No build step and no CDN: the app must work on a laptop with no network.

    A live-scoring tool that needs to reach a CDN to render is a tool that stops
    working exactly when someone's network is having a bad night.
    """
    from serve_viz import PAGE
    assert PAGE.exists(), f"missing {PAGE}"
    html = PAGE.read_text()
    for bad in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr", "<script src"):
        assert bad not in html, f"page reaches outside for {bad!r}"


# ---------------------------------------------------------------------------
# The page's own stepping logic, run under node.
#
# The page is one self-contained file with no build step, so its pure functions
# (game-time conversion and moment building) sit between two marker comments
# and are evaluated here without a DOM. What the browser then steps through is
# the array these functions return, so this is the ordering the playhead follows.
# ---------------------------------------------------------------------------

NODE = shutil.which("node")


def _page_pure_block() -> str:
    from serve_viz import PAGE
    html = PAGE.read_text()
    start = html.index("// -- pure: begin")
    end = html.index("// -- pure: end")
    return html[start:end]


def _moments_under_node(plays: list) -> list:
    """Run the page's buildMoments() on `plays`; return what it would step through."""
    script = _page_pure_block() + """
const plays = %s;
const out = buildMoments(plays).map(m => ({
  period: m.period, secs: m.secs, elapsed: elapsed(m), wp: m.wp,
  seqs: m.plays.map(p => p.seq), displaced: m.displaced,
  deltas: m.plays.map(p => +(p.wp - p.prevWp).toFixed(5)),
}));
process.stdout.write(JSON.stringify(out));
""" % json.dumps(plays)
    r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _play(seq, period, clock, wp):
    """A play the way /api/game emits it (secs mirrors state.game_seconds_remaining)."""
    return {"seq": seq, "period": period, "clock": clock,
            "secs": clock + REG if period <= 1 else clock,
            "wp": wp, "margin": 0, "home": 0, "away": 0, "type": "t", "text": ""}


@pytest.mark.skipif(NODE is None, reason="needs node to run the page's script")
def test_the_page_steps_an_ordered_feed_as_delivered():
    """The normal case: an ordered feed is stepped exactly as it arrived, with
    plays sharing a clock folded into one moment and per-play changes intact."""
    plays = [_play(1, 1, 1200, .50), _play(2, 1, 1180, .52),
             _play(3, 1, 1180, .55), _play(4, 1, 1180, .55),   # one clock, three plays
             _play(5, 2, 1200, .60), _play(6, 2, 300, .70)]
    ms = _moments_under_node(plays)
    assert [m["seqs"] for m in ms] == [[1], [2, 3, 4], [5], [6]]
    assert [m["elapsed"] for m in ms] == sorted(m["elapsed"] for m in ms)
    assert not any(m["displaced"] for m in ms)
    assert ms[1]["deltas"] == [.02, .03, 0.0]
    assert ms[1]["wp"] == .55, "a moment's value is the state after its last play"


@pytest.mark.skipif(NODE is None, reason="needs node to run the page's script")
def test_the_page_steps_a_disordered_feed_in_clock_order():
    """The guard: a feed that arrives out of order must not make the playhead
    jump forward and then back. The page steps in clock order, keeps every
    probability exactly as the model produced it in feed order, and marks the
    moment it had to move rather than hiding that it did.
    """
    plays = [_play(1, 1, 1200, .50), _play(2, 1, 1100, .55),
             _play(3, 1, 900, .60),                 # feed skips ahead ...
             _play(4, 1, 1000, .58),                # ... then comes back 100s
             _play(5, 1, 800, .65), _play(6, 2, 1200, .66)]
    ms = _moments_under_node(plays)
    times = [m["elapsed"] for m in ms]
    assert times == sorted(times), f"page would step backwards in time: {times}"
    assert [m["seqs"] for m in ms] == [[1], [2], [4], [3], [5], [6]]

    # Only the two plays the feed swapped are flagged, each with where the feed
    # actually put it; the plays around them were delivered in order.
    moved = ms[2]
    assert moved["displaced"] == {"after": 3}       # play 4 arrived after play 3
    assert ms[3]["displaced"] == {"before": 4}      # play 3 arrived before play 4
    assert all(m["displaced"] is None for m in (ms[0], ms[1], ms[4], ms[5]))

    # The model walked the feed's order, so the change each play made is
    # measured against the play the model saw just before it - not against
    # whatever now sits to its left on screen.
    assert moved["deltas"] == [-.02]                # .58 after .60, in feed order
    assert ms[3]["deltas"] == [.05]                 # .60 after .55
    assert [m["wp"] for m in ms] == [.50, .55, .58, .60, .65, .66]


@pytest.mark.skipif(NODE is None, reason="needs node to run the page's script")
def test_overtime_lands_after_regulation_on_the_page():
    """Each overtime restarts its clock; the page must still step through it
    after the end of regulation, not interleaved with the second half."""
    plays = [_play(1, 2, 60, .5), _play(2, 2, 0, .5), _play(3, 3, 300, .5),
             _play(4, 3, 10, .6), _play(5, 4, 300, .6), _play(6, 4, 0, .9)]
    ms = _moments_under_node(plays)
    assert [m["seqs"][0] for m in ms] == [1, 2, 3, 4, 5, 6]
    assert [m["elapsed"] for m in ms] == [2340, 2400, 2400, 2690, 2700, 3000]

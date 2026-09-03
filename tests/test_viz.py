"""The viz app draws numbers people will believe, so the join has to be right.

The one thing that could go quietly wrong here is the zip in `Scorer.score`:
plays come from the ESPN adapter and probabilities from the model, and if those
two lists ever drifted by one, every caption on the page would describe the play
before or after the probability beside it. Nothing would look broken.

These skip when the archive or the registry has not been built, because both are
build products rather than source.
"""
import pathlib
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
def scored():
    from cbbwp.config import Settings
    from serve_viz import Scorer
    s = Scorer(Settings.from_env())
    gid = int(ARCHIVES[0].stem.split("_")[1])
    return s.score(gid, s.summary_of(ARCHIVES[0]))


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


def test_the_feed_is_not_always_in_chronological_order(scored):
    """A documented property of ESPN's feed, pinned so it is not mistaken for a bug.

    ESPN orders plays by `sequenceNumber`, and on a minority of plays that
    disagrees with the clock those same plays carry - measured 2026-09-03 at 31
    of 4,782 plays (0.65%) across ten archived games, but with single jumps as
    large as 986 seconds.

    The adapter deliberately keeps sequenceNumber order (see adapters/espn.py),
    so the app must not assume monotonic game time. It draws the curve in
    game-time order and captions the out-of-order plays rather than hiding them.
    This test exists so that if the adapter's ordering ever changes, somebody has
    to come and read this comment.
    """
    times = [elapsed(p) for p in scored["plays"]]
    backwards = sum(1 for i in range(1, len(times)) if times[i] < times[i - 1])
    assert backwards < len(times) * 0.05, (
        f"{backwards} of {len(times)} plays out of order -- far more than the "
        "0.65% measured; the feed or the adapter's ordering has changed")


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

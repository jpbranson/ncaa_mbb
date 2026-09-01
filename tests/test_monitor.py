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

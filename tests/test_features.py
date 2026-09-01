import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.schemas import GameState, FEATURE_NAMES
from cbbwp.features import feature_dict, build_feature_matrix, mirror_features


def gs(**kw):
    base = dict(game_id=1, seq=1, period=1, is_ot=False, clock_seconds=600,
                game_seconds_remaining=600, home_score=50, away_score=45, margin=5,
                possession=1.0, home_timeouts=3, away_timeouts=2,
                pregame_exp_margin=4.0, neutral_site=False)
    base.update(kw)
    return GameState(**base)


def test_feature_order_matches_contract():
    assert list(feature_dict(gs()).keys()) == FEATURE_NAMES


def test_pregame_edge_decays_to_zero_at_the_buzzer():
    early = feature_dict(gs(game_seconds_remaining=2400))["pregame_exp_margin_decayed"]
    late = feature_dict(gs(game_seconds_remaining=1))["pregame_exp_margin_decayed"]
    assert abs(early - 4.0) < 1e-9
    assert abs(late) < 0.1


def test_mirroring_flips_sign_and_label():
    X = build_feature_matrix([gs()])
    y = np.array([1])
    Xm, ym = mirror_features(X, y)
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    assert Xm[1, i["margin"]] == -5.0
    assert Xm[1, i["possession"]] == 0.0
    assert Xm[1, i["timeout_diff"]] == -1.0
    assert Xm[1, i["sqrt_time"]] == Xm[0, i["sqrt_time"]]   # time is not mirrored
    assert list(ym) == [1, 0]

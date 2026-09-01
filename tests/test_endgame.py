import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.endgame import apply


def test_time_expired_forces_certainty():
    p = apply([0.4, 0.6, 0.5], margin=np.array([3, -3, 0]),
              seconds_remaining=np.array([0, 0, 0]))
    assert list(p) == [1.0, 0.0, 0.5]


def test_mathematically_decided_is_clamped():
    # 20 up with 10 seconds left: at most 2 possessions x 3 points = 6.
    p = apply([0.97], margin=np.array([20]), seconds_remaining=np.array([10.0]))
    assert p[0] == 1.0


def test_live_games_are_left_alone():
    p = apply([0.73], margin=np.array([4]), seconds_remaining=np.array([300.0]))
    assert p[0] == 0.73

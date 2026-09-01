"""cbbwp - college basketball win probability.

The state builder and feature builder in this package are imported by BOTH the
offline training pipeline and the live serving path. That is deliberate: it is
the single defence against train/serve skew.
"""
__version__ = "0.2.0"

from .schemas import Event, GameState, PregameContext, FEATURE_NAMES  # noqa: F401
from .state import build_states  # noqa: F401
from .features import build_feature_matrix, feature_dict  # noqa: F401

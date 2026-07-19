"""QQN (Quadratic Quasi-Newton) optimizer for JAX.

QQN combines steepest descent and L-BFGS through a quadratic interpolation
path:

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

and uses a line search over this path to select the optimal blend of
gradient and quasi-Newton directions.
"""

from qqn_jax.line_search.strong_wolfe import strong_wolfe_search
from qqn_jax.oracles.adam import AdamOracle
from qqn_jax.oracles.fallback import Fallback
from qqn_jax.oracles.lbfgs import LBFGSOracle
from qqn_jax.oracles.momentum import MomentumOracle
from qqn_jax.oracles.path_history import PathHistoryMomentumOracle
from qqn_jax.oracles.secant import SecantOracle
from qqn_jax.oracles.shampoo import ShampooOracle
from qqn_jax.oracles import Oracle, OracleInfo
from qqn_jax.regions.box import BoxRegion
from qqn_jax.regions.sequence import Sequential
from qqn_jax.regions.trustregion import TrustRegion
from qqn_jax.experimental.rolling_window_activation import (
    make_rolling_window,
    rolling_sin_diff,
    rolling_atan2_ramp,
)
from qqn_jax.solver import QQN, QQNState
from qqn_jax.line_search.backtracking import backtracking_search
from qqn_jax.spline_search import spline_wrap, spline_search
from qqn_jax.oracles.anderson import AndersonOracle
from qqn_jax.regions.strategy import (
    Region,
    RegionInfo,
    IdentityRegion,
)
from qqn_jax.regions.orthant import OrthantRegion

__version__ = "0.1.0"

__all__ = [
    "QQN",
    "QQNState",
    "strong_wolfe_search",
    "backtracking_search",
    "spline_wrap",
    "spline_search",
    "Oracle",
    "OracleInfo",
    "LBFGSOracle",
    "MomentumOracle",
    "AdamOracle",
    "PathHistoryMomentumOracle",
    "ShampooOracle",
    "SecantOracle",
    "AndersonOracle",
    "Fallback",
    "Region",
    "RegionInfo",
    "make_rolling_window",
    "rolling_sin_diff",
    "rolling_atan2_ramp",
    "IdentityRegion",
    "BoxRegion",
    "OrthantRegion",
    "TrustRegion",
    "Sequential",
    "__version__",
]

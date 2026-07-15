"""QQN (Quadratic Quasi-Newton) optimizer for JAX.

QQN combines steepest descent and L-BFGS through a quadratic interpolation
path:

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

and uses a line search over this path to select the optimal blend of
gradient and quasi-Newton directions.
"""

from qqn_jax.rolling_window_activation import (
    make_rolling_window,
    rolling_sin_diff,
    rolling_atan2_ramp,
)
from qqn_jax.solver import QQN, QQNState
from qqn_jax.line_search import (
    strong_wolfe_search,
    backtracking_search,
    backtracking_temperature_search,
)
from qqn_jax.spline_search import spline_wrap, spline_search
from qqn_jax.oracles import (
    Oracle,
    OracleInfo,
    LBFGSOracle,
    MomentumOracle,
    AdamOracle,
    PathHistoryMomentumOracle,
    ShampooOracle,
    SecantOracle,
    AndersonOracle,
    Fallback,
)
from qqn_jax.regions import (
    Region,
    RegionInfo,
    IdentityRegion,
    BoxRegion,
    OrthantRegion,
    TrustRegion,
    Sequential,
)

__version__ = "0.1.0"

__all__ = [
    "QQN",
    "QQNState",
    "strong_wolfe_search",
    "backtracking_search",
    "backtracking_temperature_search",
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

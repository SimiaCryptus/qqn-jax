from qqn_jax.oracles.strategy import resolve_oracle
from qqn_jax.oracles.oracle import Oracle, OracleInfo
from qqn_jax.oracles.lbfgs import LBFGSOracle
from qqn_jax.oracles.adam import AdamOracle, AdamState
from qqn_jax.oracles.momentum import MomentumOracle, MomentumState
from qqn_jax.oracles.path_history import (
    PathHistoryMomentumOracle,
    PathHistoryMomentumState,
)
from qqn_jax.oracles.secant import SecantOracle, SecantState
from qqn_jax.oracles.anderson import AndersonOracle, AndersonState
from qqn_jax.oracles.shampoo import ShampooOracle, ShampooState
from qqn_jax.oracles.fallback import Fallback

__all__ = [
    "Oracle",
    "OracleInfo",
    "resolve_oracle",
    "LBFGSOracle",
    "AdamOracle",
    "AdamState",
    "MomentumOracle",
    "MomentumState",
    "PathHistoryMomentumOracle",
    "PathHistoryMomentumState",
    "SecantOracle",
    "SecantState",
    "AndersonOracle",
    "AndersonState",
    "ShampooOracle",
    "ShampooState",
    "Fallback",
]

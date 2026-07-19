from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.anderson import AndersonOracle
from qqn_jax.oracles.ams_qn import AnchoredMultiSecantOracle
from qqn_jax.oracles.adam import AdamOracle
from qqn_jax.oracles.fallback import Fallback
from qqn_jax.oracles.lbfgs import LBFGSOracle
from qqn_jax.oracles.momentum import MomentumOracle
from qqn_jax.oracles.path_history import PathHistoryMomentumOracle
from qqn_jax.oracles.secant import SecantOracle
from qqn_jax.oracles.shampoo import ShampooOracle


def resolve_oracle(oracle, history_size: int = 10, max_probe_replay: int = 2) -> Oracle:
    """Map a string shortcut or ``Oracle`` instance to a concrete oracle."""
    if oracle is None or oracle == "lbfgs":
        return LBFGSOracle(history_size=history_size, max_probe_replay=max_probe_replay)
    if isinstance(oracle, str):
        if oracle == "momentum":
            return MomentumOracle()
        if oracle == "adam":
            return AdamOracle()
        if oracle == "path_momentum":
            return PathHistoryMomentumOracle(history_size=history_size)
        if oracle == "shampoo":
            return ShampooOracle()
        if oracle == "secant":
            return SecantOracle()
        if oracle == "anderson":
            return AndersonOracle()
        if oracle == "ams_qn":
            return AnchoredMultiSecantOracle(window=history_size)
        if oracle == "anderson+secant":
            # The variational ideal, safeguarded by a featherweight secant —
            # a strictly-dominant pairing when the residual solve degenerates.
            return Fallback([AndersonOracle(window=5), SecantOracle()])
        if oracle == "lbfgs+secant":
            # Your data's "best zero-storage safety net": deep curvature while
            # healthy, finite curvature the instant the history collapses.
            return Fallback(
                [
                    LBFGSOracle(
                        history_size=history_size,
                        max_probe_replay=max_probe_replay,
                    ),
                    SecantOracle(),
                ]
            )
        raise ValueError(
            f"Unknown oracle: {oracle!r}. "
            "Available: 'lbfgs', 'momentum', 'adam', 'path_momentum', "
            "'shampoo', 'secant', 'anderson', 'anderson+secant', "
            "'ams_qn', 'lbfgs+secant' or an Oracle instance."
        )
    if isinstance(oracle, Oracle):
        return oracle
    raise TypeError(f"oracle must be a string, Oracle, or None; got {type(oracle)!r}.")

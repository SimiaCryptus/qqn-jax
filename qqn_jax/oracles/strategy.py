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
    if oracle is None:
        return Fallback([LBFGSOracle(history_size=50), AdamOracle(learning_rate=1e-3)])
    elif isinstance(oracle, Oracle):
        return oracle
    elif isinstance(oracle, str):
        if oracle == "momentum":
            return MomentumOracle()
        elif oracle == "lbfgs":
            return LBFGSOracle(history_size=history_size, max_probe_replay=max_probe_replay)
        elif oracle == "adam":
            return AdamOracle()
        elif oracle == "path_momentum":
            return PathHistoryMomentumOracle(history_size=history_size)
        elif oracle == "shampoo":
            return ShampooOracle()
        elif oracle == "secant":
            return SecantOracle()
        elif oracle == "anderson":
            return AndersonOracle()
        elif oracle == "ams_qn":
            return AnchoredMultiSecantOracle(window=history_size)
        elif oracle == "anderson+secant":
            return Fallback([AndersonOracle(window=5), SecantOracle()])
        elif oracle == "lbfgs+secant":
            return Fallback(
                [
                    LBFGSOracle(
                        history_size=history_size,
                        max_probe_replay=max_probe_replay,
                    ),
                    SecantOracle(),
                ]
            )
        else:
            raise ValueError(
            f"Unknown oracle: {oracle!r}. "
            "Available: 'lbfgs', 'momentum', 'adam', 'path_momentum', "
            "'shampoo', 'secant', 'anderson', 'anderson+secant', "
            "'ams_qn', 'lbfgs+secant' or an Oracle instance."
        )
    else:
        raise TypeError(f"oracle must be a string, Oracle, or None; got {type(oracle)!r}.")

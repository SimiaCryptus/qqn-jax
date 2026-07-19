from typing import NamedTuple, Callable, Any, Tuple

from qqn_jax import AndersonOracle
from qqn_jax.oracles.adam import AdamOracle
from qqn_jax.oracles.fallback import Fallback
from qqn_jax.oracles.lbfgs import LBFGSOracle
from qqn_jax.oracles.momentum import MomentumOracle
from qqn_jax.oracles.path_history import PathHistoryMomentumOracle
from qqn_jax.oracles.secant import SecantOracle
from qqn_jax.oracles.shampoo import ShampooOracle


class OracleInfo(NamedTuple):
    """Information passed to ``Oracle.update`` after a step is accepted.

    Attributes:
        params: iterate ``x`` before the step.
        new_params: accepted iterate ``x_new``.
        grad: gradient ``∇f(x)`` before the step.
        new_grad: gradient ``∇f(x_new)`` after the step.
        t: chosen interpolation parameter.
        step_size: accepted step size ``α``.
         probe_params: optional ``(k, n)`` buffer of line-search probe points.
         probe_grads: optional ``(k, n)`` buffer of probe gradients.
         probe_valid: optional ``(k,)`` boolean mask of filled probe slots.
    """

    params: Any = None
    new_params: Any = None
    grad: Any = None
    new_grad: Any = None
    t: Any = None
    step_size: Any = None
    probe_params: Any = None
    probe_grads: Any = None
    probe_valid: Any = None
    probe_alphas: Any = None


class Oracle(NamedTuple):
    """Pure, swappable oracle interface.

    Attributes:
        init: ``params -> oracle_state`` (use ``()`` when stateless).
        direction: ``(params, grad, state) -> (direction, new_state)``.
            ``direction`` is the ``t = 1`` endpoint ``-H∇f``.
        update: ``(state, info) -> state`` (no-op for stateless oracles).
    """

    init: Callable[[Any], Any]
    direction: Callable[[Any, Any, Any], Tuple[Any, Any]]
    update: Callable[[Any, Any], Any]


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
            "'lbfgs+secant' or an Oracle instance."
        )
    if isinstance(oracle, Oracle):
        return oracle
    raise TypeError(f"oracle must be a string, Oracle, or None; got {type(oracle)!r}.")

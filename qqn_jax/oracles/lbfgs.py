import jax
from jax import numpy as jnp

from qqn_jax.lbfgs import (
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
    update_lbfgs_history_batch,
)
from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.secant import _ordered_probe_secants


def LBFGSOracle(history_size: int = 10, max_probe_replay: int = 2) -> Oracle:
    """Limited-memory BFGS quasi-Newton oracle.

    Wraps the existing ``qqn_jax.lbfgs`` two-loop recursion so the default
    behavior is byte-for-byte equivalent to the original optimizer.
    ``max_probe_replay`` caps how many line-search probes are folded into the
    curvature history per step. Probes are *collinear* (they all lie on the
    single ray ``x + α·d``), so replaying many of them only re-estimates the
    same 1-D curvature while *evicting* genuine cross-iteration curvature from
    the fixed-size buffer. Capping to a small number (default 2) keeps the
    bulk of the real history intact and limits the probe contribution to a
    mild secant refinement of the t=1 endpoint.
    """

    def init(params):

        grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        return init_lbfgs_state(params, grad, history_size)

    def direction(params, grad, state):
        d = lbfgs_direction(state, grad)
        return d, state

    def update(state, info):

        ordered = _ordered_probe_secants(info, max_replay=max_probe_replay)
        if ordered is None:
            return update_lbfgs_history(
                state, info.new_params, info.new_grad, history_size
            )

        params_seq, grad_seq, valid_seq = ordered
        return update_lbfgs_history_batch(
            state, params_seq, grad_seq, valid_seq, history_size
        )

    return Oracle(init=init, direction=direction, update=update)

import jax
from jax import numpy as jnp

from qqn_jax.lbfgs import (
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
    update_lbfgs_history_batch,
)
from qqn_jax.oracles.strategy import Oracle
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
        # ``grad`` is unknown at init; use zeros for ``prev_grad`` so the
        # very first curvature pair is computed once a real gradient lands.
        grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        return init_lbfgs_state(params, grad, history_size)

    def direction(params, grad, state):
        d = lbfgs_direction(state, grad)
        return d, state

    def update(state, info):
        # When line-search probes are supplied, replay them oldest-first so
        # every gradient evaluated along the path contributes curvature, then
        # finish with the accepted point as the newest pair. Otherwise fall
        # back to the single-pair update (byte-for-byte legacy behavior).
        # The replay path additionally needs probe_alphas (to sort the probes
        # into monotone-α order) and probe_valid (to mask empty slots). Some
        # line searches (e.g. the spline-wrapped variant) record probe params
        # and grads but neither alphas nor a valid mask — in that case we
        # cannot meaningfully replay, so fall back to the single-pair update.
        #
        # All probes are COLLINEAR (they lie on the single ray x + α·d), so
        # they only ever re-estimate curvature *along d*. Replaying many of
        # them flushes the fixed-size buffer of genuine cross-iteration
        # curvature. ``_ordered_probe_secants`` caps the replay count so the
        # bulk of the real history survives and probes only add a mild
        # endpoint refinement.
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

import jax
from jax import numpy as jnp

from typing import NamedTuple
from qqn_jax.oracles.strategy import Oracle
from qqn_jax.oracles.secant import _ordered_probe_secants


class AdamState(NamedTuple):
    """State for the ADAM oracle.
    Attributes:
        m: first-moment (momentum) estimate of the gradient.
        v: second-moment (energy) estimate of the gradient.
        step: iteration counter for bias correction.
    """

    m: jnp.ndarray
    v: jnp.ndarray
    step: jnp.ndarray


def AdamOracle(
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> Oracle:
    """ADAM (adaptive moment estimation) oracle.
    The ``t = 1`` endpoint is the classical ADAM update direction, formed by
    integrating a decaying-weight *momentum* (first moment ``m``) and a
    decaying-weight *energy* (second moment ``v``) of the gradients, with the
    standard bias correction::
        m      = β1·m + (1 − β1)·∇f
        v      = β2·v + (1 − β2)·∇f²
        m̂      = m / (1 − β1^t)
        v̂      = v / (1 − β2^t)
        direction = − m̂ / (√v̂ + ε)
    The moments are integrated in ``update`` (committed once a step is
    accepted) so the persisted state accumulates across iterations. The very
    first step (before any accepted gradient) reduces to plain (scaled)
    steepest descent, preserving the ``d'(0)`` anchor.
    """

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(
            m=zeros,
            v=zeros,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        # Fold the *current* gradient into the moment estimates to form the
        # endpoint, but do NOT persist here (the solver discards the returned
        # oracle state; ``update`` commits the accepted moments).
        t = state.step + 1
        m = jax.tree_util.tree_map(
            lambda mi, g: beta1 * mi + (1.0 - beta1) * g, state.m, grad
        )
        v = jax.tree_util.tree_map(
            lambda vi, g: beta2 * vi + (1.0 - beta2) * (g * g), state.v, grad
        )
        bc1 = 1.0 - beta1 ** t.astype(grad.dtype)
        bc2 = 1.0 - beta2 ** t.astype(grad.dtype)
        d = jax.tree_util.tree_map(
            lambda mi, vi: -(mi / bc1) / (jnp.sqrt(vi / bc2) + epsilon),
            m,
            v,
        )
        return d, state

    def update(state, info):
        # Commit the accepted gradient into the running moment estimates. This
        # is the only state the solver persists across iterations.
        # When line-search probes are populated, fold *every* gradient
        # evaluated along the path into the moment estimates (oldest-first,
        # ordered by increasing α), finishing with the accepted gradient as
        # the newest sample. Each valid probe supplies an additional gradient
        # observation for the first/second-moment averages, so the adaptive
        # scaling reflects the whole probed ray rather than a single point.
        # Absent probes we fall back to the single-gradient update.
        ordered = _ordered_probe_secants(info)

        def fold(m, v, g):
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            return m, v

        if ordered is None:
            m, v = fold(state.m, state.v, info.grad)
            return AdamState(m=m, v=v, step=state.step + 1)

        _, grad_seq, valid_seq = ordered

        def body(carry, elem):
            m, v = carry
            g, valid = elem
            m_new, v_new = fold(m, v, g)
            # Skip empty probe slots: retain the moments unchanged.
            m = jnp.where(valid, m_new, m)
            v = jnp.where(valid, v_new, v)
            return (m, v), None

        (m, v), _ = jax.lax.scan(body, (state.m, state.v), (grad_seq, valid_seq))
        # Advance the step counter once per accepted step (bias correction is
        # keyed to accepted iterations, not probe count).
        return AdamState(m=m, v=v, step=state.step + 1)

    return Oracle(init=init, direction=direction, update=update)

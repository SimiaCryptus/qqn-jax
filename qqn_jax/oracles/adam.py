import jax
from jax import numpy as jnp

from typing import NamedTuple
from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.point_history import publish


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
    learning_rate: float = 1e-3,
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
         direction = − learning_rate · m̂ / (√v̂ + ε)
    The moments are integrated in ``update`` (committed once a step is
    accepted) so the persisted state accumulates across iterations. The very
    first step (before any accepted gradient) reduces to plain (scaled)
    steepest descent, preserving the ``d'(0)`` anchor.

     The ``learning_rate`` scaling makes the ``t = 1`` oracle endpoint a
     *genuine* ADAM step, so that a fixed unit step along the quadratic path
     (``line_search="fixed"``) reproduces plain ADAM's per-iteration behavior
     rather than an unscaled — and typically catastrophically large — update.
    """

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(
            m=zeros,
            v=zeros,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):

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
            lambda mi, vi: -learning_rate * (mi / bc1) / (jnp.sqrt(vi / bc2) + epsilon),
            m,
            v,
        )
        return d, state

    def update(state, info):

        def fold(m, v, g):
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            return m, v

        points = publish(info)
        if points is None:
            m, v = fold(state.m, state.v, info.grad)
            return AdamState(m=m, v=v, step=state.step + 1)

        grad_seq = points.grad_seq
        valid_seq = points.valid_seq

        def body(carry, elem):
            m, v = carry
            g, valid = elem
            m_new, v_new = fold(m, v, g)

            m = jnp.where(valid, m_new, m)
            v = jnp.where(valid, v_new, v)
            return (m, v), None

        (m, v), _ = jax.lax.scan(body, (state.m, state.v), (grad_seq, valid_seq))

        return AdamState(m=m, v=v, step=state.step + 1)

    return Oracle(init=init, direction=direction, update=update)

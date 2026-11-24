from typing import Any, NamedTuple

import jax
from jax import numpy as jnp

from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.point_history import publish
from qqn_jax.utils import tree_l2_norm, tree_scalar_dtype, tree_where


class AdamState(NamedTuple):
    """State for the ADAM oracle.
    Attributes:
        m: first-moment (momentum) estimate of the gradient.
        v: second-moment (energy) estimate of the gradient.
        step: iteration counter for bias correction.
    """

    m: Any
    v: Any
    step: jnp.ndarray


def AdamOracle(
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    learning_rate: float = 1e-3,
    grad_clip_norm: float = 0.0,
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

    def _clip_grad(g):
        if grad_clip_norm <= 0.0:
            return g
        scalar_dtype = tree_scalar_dtype(g)
        gnorm = tree_l2_norm(g)
        clip = jnp.asarray(grad_clip_norm, dtype=scalar_dtype)
        scale = jnp.minimum(
            jnp.asarray(1.0, dtype=scalar_dtype),
            clip / (gnorm.astype(scalar_dtype) + jnp.asarray(1e-12, dtype=scalar_dtype)),
        )
        return jax.tree_util.tree_map(
            lambda gi: (scale.astype(gi.dtype) * gi).astype(gi.dtype), g
        )

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(
            m=zeros,
            v=zeros,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        del params
        grad_eff = _clip_grad(grad)
        t = state.step + 1
        scalar_dtype = tree_scalar_dtype(grad_eff)
        m = jax.tree_util.tree_map(
            lambda mi, g: beta1 * mi + (1.0 - beta1) * g.astype(mi.dtype),
            state.m,
            grad_eff,
        )
        v = jax.tree_util.tree_map(
            lambda vi, g: beta2 * vi
            + (1.0 - beta2) * (g.astype(vi.dtype) * g.astype(vi.dtype)),
            state.v,
            grad_eff,
        )
        bc1 = 1.0 - beta1 ** t.astype(scalar_dtype)
        bc2 = 1.0 - beta2 ** t.astype(scalar_dtype)
        d = jax.tree_util.tree_map(
            lambda mi, vi, g: (
                -learning_rate
                * (mi.astype(scalar_dtype) / bc1)
                / (jnp.sqrt(vi.astype(scalar_dtype) / bc2) + epsilon)
            ).astype(g.dtype),
            m,
            v,
            grad,
        )
        return d, state

    def update(state, info):

        def fold(m, v, g):
            g_eff = _clip_grad(g)
            m = jax.tree_util.tree_map(
                lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi.astype(mi.dtype),
                m,
                g_eff,
            )
            v = jax.tree_util.tree_map(
                lambda vi, gi: beta2 * vi
                + (1.0 - beta2) * (gi.astype(vi.dtype) * gi.astype(vi.dtype)),
                v,
                g_eff,
            )
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

            m = tree_where(valid, m_new, m)
            v = tree_where(valid, v_new, v)
            return (m, v), None

        (m, v), _ = jax.lax.scan(body, (state.m, state.v), (grad_seq, valid_seq))

        return AdamState(m=m, v=v, step=state.step + 1)

    return Oracle(init=init, direction=direction, update=update)

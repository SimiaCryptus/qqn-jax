import jax
from jax import numpy as jnp

from typing import NamedTuple
from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.secant import _ordered_probe_secants


class MomentumState(NamedTuple):
    velocity: jnp.ndarray


def MomentumOracle(beta: float = 0.9) -> Oracle:
    """First-order accelerated (heavy-ball) oracle.

    The ``t = 1`` endpoint blends the current steepest-descent move with a
    decaying-weight average of the *actual per-iteration deltas* Δx = x_new − x
    that the solver has already realized::

        (committed in update, after each accepted step)
        v_new      = β · v + (1 − β) · Δx        Δx = x_new − x

        (returned by direction at the current iterate)
        direction  = -∇f + β · v

    Here ``v`` is the running momentum of realized steps, so the oracle nudges
    the t=1 endpoint along the direction the optimizer has actually been
    travelling — true heavy-ball momentum — rather than along an average of raw
    gradients. On the very first step ``v = 0`` and the endpoint reduces to
    plain steepest descent, preserving the ``d'(0)`` anchor.
    """

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return MomentumState(velocity=zeros)

    def direction(params, grad, state):

        d = jax.tree_util.tree_map(lambda g, v: -g + beta * v, grad, state.velocity)
        return d, state

    def update(state, info):

        ordered = _ordered_probe_secants(info)
        if ordered is None:
            delta = jax.tree_util.tree_map(
                lambda xn, x: xn - x, info.new_params, info.params
            )
            v_new = jax.tree_util.tree_map(
                lambda v, dx: beta * v + (1.0 - beta) * dx, state.velocity, delta
            )
            return MomentumState(velocity=v_new)

        params_seq, _, valid_seq = ordered

        anchored = jnp.concatenate([info.params[None, :], params_seq], axis=0)
        deltas = anchored[1:] - anchored[:-1]

        def body(v, elem):
            dx, valid = elem
            v_new = beta * v + (1.0 - beta) * dx
            return jnp.where(valid, v_new, v), None

        v, _ = jax.lax.scan(body, state.velocity, (deltas, valid_seq))
        return MomentumState(velocity=v)

    return Oracle(init=init, direction=direction, update=update)

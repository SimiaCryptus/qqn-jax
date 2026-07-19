from typing import Sequence

from jax import numpy as jnp
from qqn_jax.oracles.oracle import Oracle
from qqn_jax.utils import tree_negative


def Fallback(oracles: Sequence[Oracle]) -> Oracle:
    """Use the first oracle's direction when valid, else fall back.

    Validity is detected as a finite, non-zero direction (e.g. an L-BFGS
    oracle with an empty history returns ``-H∇f = -∇f`` which is valid;
    a degenerate ``NaN``/``inf`` direction triggers the fallback). All
    selection uses ``jnp.where`` / ``lax.select`` — no Python conditionals.
    """
    oracles = tuple(oracles)

    def init(params):
        return tuple(o.init(params) for o in oracles)

    def direction(params, grad, state):
        new_states = []
        chosen = None
        chosen_valid = None
        for o, s in zip(oracles, state):
            d, ns = o.direction(params, grad, s)

            gd = jnp.vdot(grad, d)
            finite = jnp.all(jnp.isfinite(d))
            nonzero = jnp.vdot(d, d) > jnp.asarray(0.0, dtype=d.dtype)
            descent = gd < jnp.asarray(0.0, dtype=d.dtype)
            valid = finite & nonzero & descent
            if chosen is None:
                chosen = d
                chosen_valid = valid
            else:
                assert chosen is not None
                assert chosen_valid is not None
                take_prev = chosen_valid
                chosen = jnp.where(take_prev, chosen, d)
                chosen_valid = chosen_valid | valid

        neg_grad = tree_negative(grad)
        assert chosen is not None
        assert chosen_valid is not None
        chosen = jnp.where(chosen_valid, chosen, neg_grad)
        return chosen, tuple(new_states)

    def update(state, info):
        return tuple(o.update(s, info) for o, s in zip(oracles, state))

    return Oracle(init=init, direction=direction, update=update)

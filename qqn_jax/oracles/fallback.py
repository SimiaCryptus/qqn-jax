from collections.abc import Sequence

from jax import numpy as jnp

from qqn_jax.oracles.oracle import Oracle
from qqn_jax.utils import (
    tree_all_finite,
    tree_negative,
    tree_scalar_dtype,
    tree_vdot,
    tree_where,
)


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
        scalar_dtype = tree_scalar_dtype(grad)
        zero = jnp.asarray(0.0, dtype=scalar_dtype)
        for o, s in zip(oracles, state):
            d, ns = o.direction(params, grad, s)
            new_states.append(ns)

            gd = tree_vdot(grad, d, accum_dtype=scalar_dtype)
            finite = tree_all_finite(d)
            nonzero = tree_vdot(d, d, accum_dtype=scalar_dtype) > zero
            descent = gd < zero
            valid = finite & nonzero & descent
            if chosen is None:
                chosen = d
                chosen_valid = valid
            else:
                assert chosen is not None
                assert chosen_valid is not None
                take_prev = chosen_valid
                chosen = tree_where(take_prev, chosen, d)
                chosen_valid = chosen_valid | valid

        neg_grad = tree_negative(grad)
        assert chosen is not None
        assert chosen_valid is not None
        chosen = tree_where(chosen_valid, chosen, neg_grad)
        return chosen, tuple(new_states)

    def update(state, info):
        return tuple(o.update(s, info) for o, s in zip(oracles, state))

    return Oracle(init=init, direction=direction, update=update)

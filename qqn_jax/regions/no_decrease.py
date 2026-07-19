from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.regions.strategy import (
    Region,
    _tree_sub,
    _tree_add,
)
from qqn_jax.regions.identity import _identity_init, _identity_update


def NoDecreaseRegion(secondary_grad_fn: Callable) -> Region:
    """Project each step onto the half-space ``{s : ⟨∇g, s⟩ ≤ 0}``.
    Given a secondary objective ``g`` whose gradient is supplied by
    ``secondary_grad_fn(params) -> ∇g``, this region removes only the
    component of the proposed step that would *increase* ``g`` — preserving
    fitness on a protected objective while optimizing the primary one. This
    is the geometry of continual learning and constrained fine-tuning: the
    step is free to move in any direction that does not climb ``g``.
    The projection is the orthogonal removal of the offending component::
        step = candidate - x
        c    = ⟨∇g, step⟩
        s_proj = step - relu(c) / (‖∇g‖² + eps) · ∇g
    Only the *positive* (g-increasing) component is removed; descent on ``g``
    is permitted to pass through untouched.
    """
    eps = 1e-12

    def project(params, candidate, state):
        g = secondary_grad_fn(params)
        step = _tree_sub(candidate, params)
        c = sum(
            jnp.vdot(gi, si)
            for gi, si in zip(
                jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(step)
            )
        )
        gg = sum(jnp.vdot(gi, gi) for gi in jax.tree_util.tree_leaves(g))
        # Remove only the g-increasing component (relu(c) gates the sign).
        coeff = jnp.maximum(c, 0.0) / (gg + eps)
        s_proj = jax.tree_util.tree_map(lambda si, gi: si - coeff * gi, step, g)
        return _tree_add(params, s_proj)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )

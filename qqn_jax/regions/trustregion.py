from typing import NamedTuple

import jax
from jax import numpy as jnp

from qqn_jax.regions.types import Region, _tree_sub
from qqn_jax.utils import tree_l2_norm


class TrustRegionState(NamedTuple):
    radius: jnp.ndarray


def TrustRegion(
    radius: float = 1.0,
    radius_max: float = 1e3,
    adaptive: bool = True,
    shrink: float = 0.5,
    expand: float = 2.0,
    eta_lo: float = 0.1,
    eta_hi: float = 0.75,
) -> Region:
    """Enforce ``‖x_new − x‖₂ ≤ Δ`` by radially clipping the step.

    With ``adaptive=True`` the radius grows/shrinks according to the ratio
     ``ρ = ared / pred`` of actual to predicted reduction.
     Esoteric note (from the Andromeda gradient-clusters): on a *curved*
     path the chord-length the radius constrains and the arc-length the
     predicted-reduction model integrates are different coordinates. We
     therefore (a) only shrink on a genuinely poor ``ρ`` (``< eta_lo``),
     (b) shrink *gently* (``shrink``, default 0.5 not 0.25), and (c) hold
     the radius in the wide acceptable band ``[eta_lo, eta_hi]`` so the
     adaptive feedback does not over-react to the chord/arc mismatch that
     stalls the naive ``ρ < 0.25`` rule.
    """
    eps = 1e-12

    def init(params):
        dtype = jax.tree_util.tree_leaves(params)[0].dtype
        return TrustRegionState(radius=jnp.asarray(radius, dtype=dtype))

    def project(params, candidate, state):
        step = _tree_sub(candidate, params)
        n = tree_l2_norm(step)
        scale = jnp.minimum(1.0, state.radius / (n + eps))
        return jax.tree_util.tree_map(lambda x, s: x + scale * s, params, step)

    def update(state, info):
        if not adaptive:
            return state
        pred = info.pred_reduction
        ared = info.actual_reduction
        rho = ared / (pred + eps)
        step = _tree_sub(info.new_params, info.params)
        n = tree_l2_norm(step)
        at_boundary = n >= state.radius - 1e-6

        new_radius = jnp.where(
            rho < eta_lo,
            shrink * state.radius,
            jnp.where(
                jnp.logical_and(rho > eta_hi, at_boundary),
                jnp.minimum(expand * state.radius, radius_max),
                state.radius,
            ),
        )

        made_progress = ared > 0.0
        floor = jnp.where(made_progress, n, eps)
        new_radius = jnp.maximum(new_radius, floor)
        return TrustRegionState(radius=new_radius)

    return Region(init=init, project=project, update=update)

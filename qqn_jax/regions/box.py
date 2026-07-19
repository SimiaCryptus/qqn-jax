import jax
from jax import numpy as jnp

from qqn_jax.regions.strategy import Region
from qqn_jax.regions.identity import _identity_init, _identity_update


def BoxRegion(lo=None, hi=None) -> Region:
    """Enforce elementwise bounds ``lo ≤ x_new ≤ hi``.

    ``lo``/``hi`` may be scalars, pytrees broadcastable to the parameter
    structure, or ``None`` (mapped to ∓inf).
    """
    lo_val = -jnp.inf if lo is None else lo
    hi_val = jnp.inf if hi is None else hi

    def project(params, candidate, state):
        return jax.tree_util.tree_map(lambda c: jnp.clip(c, lo_val, hi_val), candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )

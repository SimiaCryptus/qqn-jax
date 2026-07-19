import jax
from jax import numpy as jnp


from qqn_jax.regions.strategy import Region
from qqn_jax.regions.identity import _identity_init, _identity_update


def OrthantRegion(l1: float = 0.0) -> Region:
    """Constrain each step to remain within the orthant of the current
    point's signs.

    The region merely guarantees that the post-step position does not change
    the sign of any coordinate: a coordinate that would cross zero is clamped
    *at* zero. There is no ``l1`` term and no modification of the fitness — the
    region is a pure geometric projection onto the current orthant.

    For a coordinate with current value ``x``, the projected candidate ``c`` is
    clamped so that ``sign(c)`` equals ``sign(x)`` (zero is the wall). A
    coordinate that starts at exactly zero stays at zero.
    """

    def project(params, candidate, state):
        def proj_leaf(x, c):

            zero = jnp.zeros((), dtype=c.dtype)
            c = jnp.where(x > 0.0, jnp.maximum(c, zero), c)
            c = jnp.where(x < 0.0, jnp.minimum(c, zero), c)
            c = jnp.where(x == 0.0, zero, c)
            return c

        return jax.tree_util.tree_map(proj_leaf, params, candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )

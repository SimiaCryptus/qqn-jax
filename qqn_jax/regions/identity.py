from qqn_jax.regions.strategy import Region


def _identity_init(params):
    return ()


def _identity_project(params, candidate, state):
    return candidate


def _identity_update(state, info):
    return state


def IdentityRegion() -> Region:
    """The trivial region: projection is the identity (no constraints)."""
    return Region(
        init=_identity_init,
        project=_identity_project,
        update=_identity_update,
    )

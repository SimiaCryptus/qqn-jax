from typing import Sequence


from qqn_jax.regions.strategy import (
    Region,
)


def Sequential(regions: Sequence[Region]) -> Region:
    """Compose regions by applying their projections in order.

    ``project = R_k ∘ ... ∘ R_1``. State is a tuple of child states and
    ``update`` fans out to each child.
    """
    regions = tuple(regions)

    def init(params):
        return tuple(r.init(params) for r in regions)

    def project(params, candidate, state):
        c = candidate
        for r, s in zip(regions, state):
            c = r.project(params, c, s)
        return c

    def update(state, info):
        return tuple(r.update(s, info) for r, s in zip(regions, state))

    return Region(init=init, project=project, update=update)

"""Projective regions for QQN.

A *projective region* remaps a proposed parameter update onto a feasible
(or otherwise preferred) set before it is applied. Because QQN searches a
single continuous quadratic path ``d(t)``, regions integrate cleanly: the
line search navigates the *projected* path

    d_R(t) = project_R(x, x + d(t)) - x

All regions are pure, functional JAX so they compose with ``jit``,
``vmap``, ``pmap`` and ``grad``. When the region is the identity
(``IdentityRegion`` / ``region=None``), behavior is byte-for-byte
equivalent to the un-regioned optimizer.
"""

from typing import Any, Optional

from qqn_jax.regions.types import Region, RegionInfo, _tree_add, _tree_sub
from qqn_jax.regions.identity import IdentityRegion
from qqn_jax.regions.trustregion import TrustRegionState


def resolve_region(region: Optional[Region]) -> Region:
    """Return ``region`` or the identity region when ``None``."""
    return IdentityRegion() if region is None else region


# Backwards-compat alias used in docstrings/specs.
RegionState = Any


__all__ = [
    "Region",
    "RegionInfo",
    "RegionState",
    "TrustRegionState",
    "resolve_region",
    "_tree_add",
    "_tree_sub",
]

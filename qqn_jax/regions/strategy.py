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

from typing import Any, Callable, NamedTuple, Optional

import jax

from qqn_jax.regions.identity import IdentityRegion
from qqn_jax.regions.trustregion import TrustRegionState


class Region(NamedTuple):
    """Pure, composable projection interface.

    Attributes:
        init: ``params -> region_state`` (use ``()`` when stateless).
        project: ``(params, candidate, state) -> projected_candidate``.
        update: ``(state, info) -> state`` (no-op for stateless regions).
    """

    init: Callable[[Any], Any]
    project: Callable[[Any, Any, Any], Any]
    update: Callable[[Any, Any], Any]


class RegionInfo(NamedTuple):
    """Information passed to ``Region.update`` after a step.

    Attributes:
        params: iterate ``x`` before the step.
        new_params: accepted iterate ``x + α·d_R(t)``.
        pred_reduction: predicted reduction from the along-path model.
        actual_reduction: actual reduction ``f(x) - f(x_new)``.
        t: chosen interpolation parameter.
        step_size: accepted step size ``α``.
    """

    params: Any = None
    new_params: Any = None
    pred_reduction: Any = None
    actual_reduction: Any = None
    t: Any = None
    step_size: Any = None


# --- Tree helpers -----------------------------------------------------


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


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
]

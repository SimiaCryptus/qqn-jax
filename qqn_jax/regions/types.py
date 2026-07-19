"""Dependency-free primitives shared across region modules.

This module exists purely to break the circular-import cycle that would
otherwise arise between ``strategy.py`` (which wants to re-export
``TrustRegionState`` from ``trustregion.py``) and ``identity.py`` /
``trustregion.py`` (which both need ``Region``/``_tree_sub`` defined in
``strategy.py``). ``types.py`` has no imports from sibling region modules,
so it can be imported first by all of them.
"""

from typing import Any, Callable, NamedTuple

import jax


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


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


__all__ = [
    "Region",
    "RegionInfo",
    "_tree_add",
    "_tree_sub",
]

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

from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_l2_norm, tree_vdot


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


# --- Identity (default, zero-overhead) --------------------------------


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


# --- Box / Min-Max Region ---------------------------------------------


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


# --- Orthant Region (OWL-QN style sparsity) ---------------------------


def OrthantRegion(l1: float = 0.0) -> Region:
    """Constrain each step to remain within the orthant of the current
    point's signs, zeroing coordinates that would cross zero.

    When ``l1 > 0`` the pseudo-gradient ``∇f + l1·sign(x)`` chooses the
    orthant for zero coordinates (OWL-QN convention). The pseudo-gradient
    is approximated using ``candidate - params`` as a step proxy, which is
    the direction the line search proposes.
    """

    def project(params, candidate, state):
        def proj_leaf(x, c):
            # Chosen orthant sign ξ.
            step = c - x
            xi = jnp.where(x != 0.0, jnp.sign(x), jnp.sign(step))
            keep = jnp.sign(c) == xi
            return jnp.where(keep, c, 0.0)

        return jax.tree_util.tree_map(proj_leaf, params, candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )


# --- Trust-Region Sphere ----------------------------------------------


class TrustRegionState(NamedTuple):
    radius: jnp.ndarray


def TrustRegion(
    radius: float = 1.0,
    radius_max: float = 1e3,
    adaptive: bool = True,
) -> Region:
    """Enforce ``‖x_new − x‖₂ ≤ Δ`` by radially clipping the step.

    With ``adaptive=True`` the radius grows/shrinks according to the ratio
    ``ρ = ared / pred`` of actual to predicted reduction.
    """
    eps = 1e-12

    def init(params):
        return TrustRegionState(radius=jnp.asarray(radius, dtype=jnp.float32))

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
            rho < 0.25,
            0.25 * state.radius,
            jnp.where(
                jnp.logical_and(rho > 0.75, at_boundary),
                jnp.minimum(2.0 * state.radius, radius_max),
                state.radius,
            ),
        )
        return TrustRegionState(radius=new_radius)

    return Region(init=init, project=project, update=update)


# --- Combinator: Sequential -------------------------------------------


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


def resolve_region(region: Optional[Region]) -> Region:
    """Return ``region`` or the identity region when ``None``."""
    return IdentityRegion() if region is None else region


__all__ = [
    "Region",
    "RegionInfo",
    "RegionState",
    "IdentityRegion",
    "BoxRegion",
    "OrthantRegion",
    "TrustRegion",
    "TrustRegionState",
    "Sequential",
    "resolve_region",
]

# Backwards-compat alias used in docstrings/specs.
RegionState = Any

"""Shared path-strategy abstraction.

A *path strategy* is the shared component that remaps the scalar
line-search parameter ``t`` into the multidimensional probe point
``x + d(t)`` (and its velocity ``d'(t)``, used to project a measured
gradient onto the directional derivative ``⟨∇f, d'(t)⟩``). Every path
module (``qqn_jax.paths.linear``, ``qqn_jax.paths.quadratic``,
``qqn_jax.paths.spline``) builds its probes through this shared interface
so that, regardless of which curve is being traversed, every consumer
(line searches and their augmentations) constructs probe points
identically.
"""

from typing import Callable, NamedTuple

from qqn_jax.utils import tree_add_scaled, tree_negative, tree_vdot


class PathStrategy(NamedTuple):
    """Maps the scalar path parameter ``t`` to a probe offset and velocity.

    Attributes:
        offset: ``(t, grad_dir, direction) -> d(t)`` pytree, the
            displacement from ``params`` at parameter ``t``.
        velocity: ``(t, grad_dir, direction) -> d'(t)`` pytree, the path's
            tangent at ``t``. Used to project a measured gradient into the
            directional derivative ``⟨∇f, d'(t)⟩`` without needing to
            re-derive the curve's analytic derivative at each call site.
    """

    offset: Callable
    velocity: Callable


def make_evaluator(
    value_and_grad_fn: Callable,
    params,
    grad,
    direction,
    region,
    region_state,
    path: PathStrategy,
    *args,
):
    """Build an ``eval_at(t) -> (projected_params, value, grad, slope)``
    closure for a given path strategy, projecting every probe through
     ``region``. This is exactly the scalar 1-D problem handed to the
     line searches (which are otherwise entirely path-unaware).

    ``grad`` is the gradient measured at ``t = 0`` (i.e. at ``params``); it
    fixes ``grad_dir = -grad``, the path's steepest-descent tangent, which
    every path strategy receives alongside the oracle ``direction``.

    This is the shared component referenced throughout ``qqn_jax.paths``:
    it is the single place where the 1-D parameter ``t`` is remapped into
    the multidimensional probe point, so ``linear``, ``quadratic`` and
    ``spline`` all build their candidates identically modulo the
    ``PathStrategy`` they are given.
    """
    grad_dir = tree_negative(grad)

    def project(candidate):
        return region.project(params, candidate, region_state)

    def eval_at(t):
        d = path.offset(t, grad_dir, direction)
        raw = tree_add_scaled(params, 1.0, d)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        v = path.velocity(t, grad_dir, direction)
        slope = tree_vdot(g, v)
        return projected, val, g, slope

    return eval_at


__all__ = ["PathStrategy", "make_evaluator"]

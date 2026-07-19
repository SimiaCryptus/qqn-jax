"""Linear (chord) path augmentation for QQN line searches.

This is the deliberate opposite of the spline/quadratic paths: where the
quadratic path blends the steepest-descent tangent with the oracle
endpoint, and the spline reuses every probe's *gradient*, the linear path
throws all of that information away and samples the objective along the
straight chord from the origin (``t = 0``) to the oracle endpoint
(``t = 1``, i.e. ``params + direction``). It keeps the lowest-value
feasible sample found.

The chord is expressed as a ``PathStrategy`` (``LINEAR_PATH``) so its
probes are built through the same shared ``t -> point`` remapping
component used by ``qqn_jax.paths.quadratic`` and ``qqn_jax.paths.spline``.

``linear_refine`` takes an already-computed inner ``LineSearchResult`` plus
the shared scalar ``eval_at`` and returns a possibly-improved result by
sampling interior points of the chord (value-only) and keeping the best.
"""

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_scale
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.paths.base import PathStrategy


def _linear_offset(t, grad_dir, direction):
    """The straight chord ``d(t) = t · direction``.

    Deliberately ignores ``grad_dir``: the linear path is the control that
    discards curvature/gradient information entirely.
    """
    del grad_dir
    return tree_scale(t, direction)


def _linear_velocity(t, grad_dir, direction):
    """Constant tangent ``d'(t) = direction``."""
    del t, grad_dir
    return direction


LINEAR_PATH = PathStrategy(offset=_linear_offset, velocity=_linear_velocity)


def linear_refine(inner, eval_at, dtype, num_samples: int = 8) -> LineSearchResult:
    """Value-only *linear* refinement of an inner line-search result.

    Samples ``params + t·direction`` for ``t ∈ (0, 1]`` (via the shared
    scalar ``eval_at``) and keeps the better of ``inner`` and the best
    sample found.

    Args:
        inner: the baseline ``LineSearchResult`` to refine.
        eval_at: shared scalar evaluator ``t -> (params, value, grad, slope)``
            built from ``LINEAR_PATH``.
        dtype: dtype for the sampling grid.
        num_samples: number of interior samples of ``t ∈ (0, 1]`` to probe.
    """
    inner_evals = inner.num_evals
    if inner_evals is None:
        inner_evals = jnp.asarray(1, jnp.int32)

    n = num_samples
    alphas = (jnp.arange(1, n + 1, dtype=dtype)) / jnp.asarray(n, dtype=dtype)

    def sample(alpha):
        p, v, g, _slope = eval_at(alpha)
        return alpha, v, p, g

    s_alpha, s_val, s_params, s_grad = jax.vmap(sample)(alphas)

    best_sample_idx = jnp.argmin(s_val)
    best_sample_val = s_val[best_sample_idx]
    best_sample_alpha = s_alpha[best_sample_idx]
    best_sample_params = s_params[best_sample_idx]
    best_sample_grad = s_grad[best_sample_idx]

    use_sample = best_sample_val < inner.new_value
    fa = jnp.where(use_sample, best_sample_alpha, inner.step_size)
    fv = jnp.where(use_sample, best_sample_val, inner.new_value)
    fp = jax.tree_util.tree_map(
        lambda s, i: jnp.where(use_sample, s, i),
        best_sample_params,
        inner.new_params,
    )
    fg = jax.tree_util.tree_map(
        lambda s, i: jnp.where(use_sample, s, i),
        best_sample_grad,
        inner.new_grad,
    )

    done = jnp.logical_or(inner.done, use_sample)
    return LineSearchResult(
        step_size=fa,
        new_value=fv,
        new_grad=fg,
        new_params=fp,
        done=done,
        probe_params=inner.probe_params,
        probe_grads=inner.probe_grads,
        probe_valid=inner.probe_valid,
        probe_values=inner.probe_values,
        probe_alphas=inner.probe_alphas,
        num_evals=inner_evals + jnp.asarray(n, jnp.int32),
    )


__all__ = ["LINEAR_PATH", "linear_refine"]

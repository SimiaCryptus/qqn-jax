"""Linear (chord) path augmentation for QQN line searches.

This is the deliberate opposite of the spline/quadratic paths: where the
quadratic path blends the steepest-descent tangent with the oracle
endpoint, and the spline reuses every probe's *gradient*, the linear path
throws all of that information away and samples the objective along the
straight chord from the origin (``t = 0``) to the oracle endpoint
(``t = 1``, i.e. ``params + direction``). It keeps the lowest-value
feasible sample found.

``linear_wrap(inner_search)`` returns a line-search-compatible callable
that first runs ``inner_search``, then samples ``num_samples`` interior
points of the chord and keeps the better of the inner result and the best
sample.

The chord is expressed as a ``PathStrategy`` (``LINEAR_PATH``) so its
probes are built through the same shared ``t -> point`` remapping
component used by ``qqn_jax.paths.quadratic`` and ``qqn_jax.paths.spline``.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_scale
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.line_search.backtracking import backtracking_search
from qqn_jax.paths.base import PathStrategy, make_evaluator


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


# The shared strategy object for the chord path.
LINEAR_PATH = PathStrategy(offset=_linear_offset, velocity=_linear_velocity)


def linear_wrap(
    inner_search: Callable,
    num_samples: int = 8,
    path: PathStrategy = LINEAR_PATH,
) -> Callable:
    """Augment ``inner_search`` with a *linear* (value-only) refinement.

    Because the path direction is consistent, sampling ``params +
    t·direction`` for ``t ∈ [0, 1]`` traces exactly the chord from the
    current iterate to the oracle endpoint. When the direction degenerates
    to the (negated) gradient (no genuine oracle point), the samples still
    interpolate along the gradient ray, so a sensible step is recovered.

    Args:
        inner_search: any registered line-search strategy to seed the baseline.
        num_samples: number of interior samples of ``t ∈ (0, 1]`` to probe.
        path: the ``PathStrategy`` used to remap ``t`` into a probe point.
            Defaults to ``LINEAR_PATH`` (the straight chord); overriding
            this is mostly useful for testing other curves against the
            same value-only sampling scheme.
     ``path`` is forwarded explicitly to ``inner_search`` (as a first-class
     ``path=path`` keyword), not merely relied upon as a convention that the
     inner search happens to default to the same curve. This keeps the
     chord this wrapper samples and the curve the inner search itself
     traverses structurally in sync.
    """

    def wrapped(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        region=None,
        region_state=None,
        **inner_kwargs,
    ) -> LineSearchResult:
        region = resolve_region(region)
        eval_at = make_evaluator(
            value_and_grad_fn,
            params,
            grad,
            direction,
            region,
            region_state,
            path,
            *args,
        )

        dtype = value.dtype

        # 1. Baseline from the inner search.
        inner = inner_search(
            value_and_grad_fn,
            params,
            direction,
            value,
            grad,
            *args,
            region=region,
            region_state=region_state,
            path=path,
            **inner_kwargs,
        )

        inner_evals = inner.num_evals
        if inner_evals is None:
            inner_evals = jnp.asarray(1, jnp.int32)

        # 2. Sample the chord from origin (t=0) to oracle point (t=1),
        #    discarding all gradient/curvature information. Interior samples
        #    at t = k / num_samples for k = 1..num_samples.
        n = num_samples
        alphas = (jnp.arange(1, n + 1, dtype=dtype)) / jnp.asarray(n, dtype=dtype)

        def sample(alpha):
            p, v, g, _slope = eval_at(alpha)
            return alpha, v, p, g

        s_alpha, s_val, s_params, s_grad = jax.vmap(sample)(alphas)

        # 3. Pick the best of {inner accepted point, linear samples}.
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

    return wrapped


# A ready-to-use linear line search: value-only chord interpolation wrapped
# around the default backtracking (Armijo) inner search. This is a
# convenience for testing ``linear_wrap`` in isolation; production use should
# prefer the orthogonal ``linear=True`` solver flag, which wraps whichever
# line search the caller actually selected (linear interpolation is a *path*
# choice, not a line search in its own right).
linear_search = linear_wrap(backtracking_search)


__all__ = ["linear_wrap", "linear_search", "LINEAR_PATH"]

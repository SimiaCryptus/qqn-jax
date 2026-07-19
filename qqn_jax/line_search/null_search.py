from typing import Callable

from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _empty_probes,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_vdot, tree_add_scaled


def null_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    step_size: float = 1.0,
    grad_scale: float = 1.0,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """ "Null" line search: unconditionally accept the ``t = 1`` oracle point.
    The ``direction`` handed to the line search is the oracle endpoint
    ``-H∇f`` (the ``t = 1`` point of the quadratic path). This search performs
    *no* acceptance test and simply steps to ``params + step_size·direction``.
    When the oracle degenerates and hands back the raw (negated) gradient — the
    Fallback oracle's terminal safety net returns ``-∇f`` — this reduces to a
    plain scaled-gradient step. The ``grad_scale`` parameter lets callers
    rescale that case: it is applied as an *additional* multiplier when the
    supplied direction is (anti-)parallel to the gradient (i.e. no genuine
    curvature was available).
     When ``temperature == 0`` this always reports ``done=True``. When
     ``temperature > 0`` the Metropolis meta-rule gates ``done`` on descent
     OR an accepted uphill move (probability ``exp(−ΔE / T)``).
    """
    region = resolve_region(region)
    base_alpha = jnp.asarray(step_size, dtype=value.dtype)

    dd = tree_vdot(direction, direction)
    gg = tree_vdot(grad, grad)
    dg = tree_vdot(direction, grad)
    denom = jnp.sqrt(dd * gg)
    cos_sim = jnp.where(denom > 0.0, dg / denom, jnp.asarray(0.0, dtype=value.dtype))
    is_grad = jnp.abs(cos_sim) >= (1.0 - 1e-6)
    scale = jnp.where(is_grad, jnp.asarray(grad_scale, dtype=value.dtype), 1.0)
    alpha = jnp.minimum(base_alpha * scale, jnp.asarray(max_step, dtype=value.dtype))
    raw_params = tree_add_scaled(params, alpha, direction)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_g, new_val, alpha, max_probes
    )
    return LineSearchResult(
        step_size=alpha,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=jnp.asarray(True),
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=jnp.asarray(1, dtype=jnp.int32),
    )

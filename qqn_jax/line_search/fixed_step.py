from typing import Callable, Optional

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _metropolis_accept,
    _empty_probes,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.paths import PathStrategy, QUADRATIC_PATH
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_add_scaled, tree_negative


def fixed_step_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    step_size: float = 1.0,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
    path: Optional[PathStrategy] = None,
) -> LineSearchResult:
    """Trivial line search using a constant step size.
    Useful for debugging, benchmarking against a baseline, or when the
    quadratic path scaling already provides a sensible step. Always reports
     ``done=True`` when ``temperature == 0``. When ``temperature > 0`` the
     Metropolis meta-rule gates ``done`` on descent OR an accepted uphill
     move (probability ``exp(−ΔE / T)``).
    The ``max_step`` parameter is accepted for interface uniformity; the
    fixed step is clipped to it so callers cannot overshoot the cap.
     Likewise ``path`` is accepted for interface uniformity with every other
     line search — the solver (``QQN.__init__``) unconditionally threads its
     configured ``PathStrategy`` through the *selected* search, fixed-step
     included. Honoring it here (rather than dropping the kwarg, which used
     to raise a ``TypeError``) also keeps the point actually visited in sync
     with the along-path predicted-reduction model built from the same
     ``PathStrategy`` in ``QQN.update``. At the default ``step_size == 1.0``
     this is a no-op: every supported path's ``t = 1`` endpoint is exactly
     the raw oracle ``direction``.
    """
    region = resolve_region(region)
    path = path if path is not None else QUADRATIC_PATH
    alpha = jnp.minimum(
        jnp.asarray(step_size, dtype=value.dtype),
        jnp.asarray(max_step, dtype=value.dtype),
    )
    grad_dir = tree_negative(grad)
    offset = path.offset(alpha, grad_dir, direction)
    raw_params = tree_add_scaled(params, 1.0, offset)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)

    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    stochastic, _key = _metropolis_accept(
        new_val - value, temp0, jax.random.PRNGKey(seed), value.dtype
    )
    done = jnp.where(
        temp0 > 0.0,
        jnp.logical_or(new_val < value, stochastic),
        jnp.asarray(True),
    )
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_g, new_val, alpha, max_probes
    )
    return LineSearchResult(
        step_size=alpha,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=done,
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=jnp.asarray(1, dtype=jnp.int32),
    )

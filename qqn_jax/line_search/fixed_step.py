from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _metropolis_accept,
    _empty_probes,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult


def fixed_step_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
    step_size: float = 1.0,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Trivial line search using a constant step size on the 1-D problem.

    Evaluates ``φ(t)`` at a single fixed ``t = step_size`` (clipped to
    ``max_step``). The path/region are already folded into ``eval_at`` by
    the solver, so this is fully path-agnostic. Always reports ``done=True``
    when ``temperature == 0``; otherwise the Metropolis meta-rule gates
    ``done`` on descent OR an accepted uphill move.
    """
    del slope0
    alpha = jnp.minimum(
        jnp.asarray(step_size, dtype=value.dtype),
        jnp.asarray(max_step, dtype=value.dtype),
    )
    new_params, new_val, new_g, _slope = eval_at(alpha)

    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    stochastic, _key = _metropolis_accept(
        new_val - value, temp0, jax.random.PRNGKey(seed), value.dtype
    )
    done = jnp.where(
        temp0 > 0.0,
        jnp.logical_or(new_val < value, stochastic),
        jnp.asarray(True),
    )
    eff_probes = max_probes if record_probes else 1
    pp, pg, pv, pval, pa = _empty_probes(params, eff_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_g, new_val, alpha, eff_probes
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

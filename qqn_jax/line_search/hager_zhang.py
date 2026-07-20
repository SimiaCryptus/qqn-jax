from typing import Callable

import jax
import optax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _metropolis_accept,
    _empty_probes,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult


def hager_zhang_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
     init_step: float = 1.0,
    c1: float = 0.1,
    max_iter: int = 30,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Hager-Zhang line search via Optax ``scale_by_backtracking_linesearch``.

    Path-agnostic: operates on the scalar 1-D problem ``φ(t)`` exposed by
    ``eval_at`` (the path/region were pre-baked by the solver). Optax
    discovers the step ``t`` on this scalar problem; value/grad are then
    recomputed along the real path at ``t``.
    """
    dtype = value.dtype
    del init_step  # Optax backtracking uses its own initial-guess strategy.
    t0 = jnp.zeros((1,), dtype=dtype)
    unit = jnp.ones((1,), dtype=dtype)

    def phi(tvec):
        _, v, _, _ = eval_at(tvec[0])
        return v

    scalar_grad = jnp.asarray([slope0], dtype=dtype)

    ls = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_iter,
        slope_rtol=c1,
        decrease_factor=0.8,
        increase_factor=jnp.minimum(1.0, float(max_step)),
        store_grad=False,
    )
    ls_state = ls.init(t0)
    scaled_updates, _new_state = ls.update(
        updates=unit,
        state=ls_state,
        params=t0,
        value=value,
        grad=scalar_grad,
        value_fn=phi,
    )
    step_size = jnp.asarray(scaled_updates)[0]
    new_params, new_value, new_grad, _slope = eval_at(step_size)

    stochastic, _key = _metropolis_accept(
        new_value - value, temperature, jax.random.PRNGKey(seed), new_value.dtype
    )
    done = jnp.logical_or(new_value < value, stochastic)
    eff_probes = max_probes if record_probes else 1
    pp, pg, pv, pval, pa = _empty_probes(params, eff_probes)
    pp, pg, pv, pval, pa = _record_probe(
         pp, pg, pv, pval, pa, 0, new_params, new_grad, new_value, step_size, eff_probes
    )
    return LineSearchResult(
        step_size=step_size,
        new_value=new_value,
        new_grad=new_grad,
        new_params=new_params,
        done=done,
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=jnp.asarray(max_iter + 1, dtype=jnp.int32),
    )
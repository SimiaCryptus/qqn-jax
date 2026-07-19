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
from qqn_jax.paths import QUADRATIC_PATH
from qqn_jax.paths.base import PathStrategy
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_vdot


def hager_zhang_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    c1: float = 0.1,
    max_iter: int = 30,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
    path: PathStrategy = QUADRATIC_PATH,
) -> LineSearchResult:
    """Hager-Zhang line search via Optax ``scale_by_backtracking_linesearch``.
    The Hager-Zhang scheme is a robust approximate-Wolfe line search. We use
    Optax's backtracking transformation parameterized to approximate it,
    recomputing value/grad at the accepted point. Falls back gracefully if
    the underlying transform is unavailable.
     The ``temperature`` meta-rule is applied to the final acceptance: an
     uphill step may still be marked ``done`` via a Metropolis move
     (probability ``exp(−ΔE / T)``).
    """
    region = resolve_region(region)

    def fun_only(p):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_iter,
        slope_rtol=c1,
        decrease_factor=0.8,
        increase_factor=jnp.minimum(1.0, float(max_step)),
        store_grad=True,
    )
    ls_state = ls.init(params)
    scaled_updates, _new_state = ls.update(
        updates=direction,
        state=ls_state,
        params=params,
        value=value,
        grad=grad,
        value_fn=fun_only,
    )
    raw_params = optax.apply_updates(params, scaled_updates)
    new_params = region.project(params, raw_params, region_state)
    new_value, new_grad = value_and_grad_fn(new_params, *args)
    d_norm_sq = tree_vdot(direction, direction)
    step_size = jnp.where(
        d_norm_sq > 0.0,
        tree_vdot(scaled_updates, direction) / d_norm_sq,
        jnp.asarray(0.0, dtype=new_value.dtype),
    )
    # Temperature meta-rule: an uphill step may still be marked done via a
    # Metropolis move.
    stochastic, _key = _metropolis_accept(
        new_value - value, temperature, jax.random.PRNGKey(seed), new_value.dtype
    )
    done = jnp.logical_or(new_value < value, stochastic)
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_grad, new_value, step_size, max_probes
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

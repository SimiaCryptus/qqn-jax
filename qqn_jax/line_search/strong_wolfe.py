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


def strong_wolfe_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
    c1: float = 1e-3,
    c2: float = 0.7,
    max_iter: int = 10,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Strong Wolfe line search via Optax ``scale_by_zoom_linesearch``.

    Path-agnostic: the whole multidimensional path/region is pre-baked into
    ``eval_at``. We hand Optax a *scalar* 1-D problem — a single-element
    parameter ``t`` with unit update and value function ``φ(t)`` — and let
    it discover the step ``t`` satisfying Armijo + strong-curvature on
    ``φ``, then recompute value/grad along the real path at that ``t``.
    """
    dtype = value.dtype
    t0 = jnp.zeros((1,), dtype=dtype)
    unit = jnp.ones((1,), dtype=dtype)

    def phi(tvec):
        _, v, _, _ = eval_at(tvec[0])
        return v

                                                             
    scalar_grad = jnp.asarray([slope0], dtype=dtype)

    ls = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_iter,
        curv_rtol=c2,
        slope_rtol=c1,
        tol=c1,
        initial_guess_strategy="one",
        max_learning_rate=float(max_step),
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

    delta_e = new_value - value
    stochastic, _key = _metropolis_accept(
        delta_e, temperature, jax.random.PRNGKey(seed), new_value.dtype
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
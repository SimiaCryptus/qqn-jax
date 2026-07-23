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
    init_step: float = 1.0,
    c1: float = 1e-6,
    c2: float = 0.9,
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

    Because Optax's zoom search enforces the Armijo condition internally,
    it will (when it converges) essentially always return a strictly
    improving point. That means a naive "accept if improved, else maybe
    accept stochastically" check *after* the fact never actually lets
    ``temperature`` influence the outcome — the improving branch always
    wins. To give ``temperature`` real effect (mirroring the reference
    Armijo–Wolfe search), we first probe the initial trial step and give
    the Metropolis criterion a chance to accept it *before* running the
    (temperature-agnostic) Optax search.
    """
    dtype = value.dtype
    t0 = jnp.zeros((1,), dtype=dtype)
    unit = jnp.ones((1,), dtype=dtype)

    def phi(tvec):
        _, v, _, _ = eval_at(tvec[0])
        return v

    scalar_grad = jnp.asarray([slope0], dtype=dtype)

    key0 = jax.random.PRNGKey(seed)

    # Probe the initial trial step first so that ``temperature`` has a
    # real chance to short-circuit the (always-improving) Optax search.
    init_alpha = jnp.asarray(min(float(init_step), float(max_step)), dtype=dtype)
    p_init, v_init, g_init, _s_init = eval_at(init_alpha)
    stochastic_init, key0 = _metropolis_accept(
        v_init - value, temperature, key0, value.dtype
    )

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

    ls_step_size = jnp.asarray(scaled_updates)[0]
    ls_new_params, ls_new_value, ls_new_grad, _ls_slope = eval_at(ls_step_size)

    step_size = jnp.where(stochastic_init, init_alpha, ls_step_size)
    new_value = jnp.where(stochastic_init, v_init, ls_new_value)
    new_params = jnp.where(stochastic_init, p_init, ls_new_params)
    new_grad = jnp.where(stochastic_init, g_init, ls_new_grad)

    delta_e = new_value - value
    stochastic, _key = _metropolis_accept(
        delta_e, temperature, key0, new_value.dtype
    )
    done = jnp.logical_or(
        jnp.logical_or(new_value < value, stochastic), stochastic_init
    )

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
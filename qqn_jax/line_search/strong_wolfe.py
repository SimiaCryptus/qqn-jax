from typing import Callable

import jax
import optax
from jax import numpy as jnp

from qqn_jax.line_search.strategy import (
    LineSearchResult,
    _metropolis_accept,
    _empty_probes,
    _record_probe,
)
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_vdot


def strong_wolfe_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    c1: float = 1e-3,
    c2: float = 0.7,
    max_iter: int = 10,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Strong Wolfe line search via Optax ``scale_by_zoom_linesearch``.

    Enforces Armijo sufficient decrease and the strong curvature
    condition, which keeps the L-BFGS curvature updates well-conditioned.
     The ``temperature`` meta-rule is applied to the *final* acceptance: even
     if Optax's Wolfe step fails to descend, a Metropolis uphill move
     (probability ``exp(−ΔE / T)``) may still mark the step ``done``.

    Optax's zoom line search is a ``GradientTransformationExtraArgs`` whose
    ``update`` step rescales the provided *updates* (here, ``direction``)
    by the discovered step size. We wrap a value-only objective for it and
    recompute value/grad at the accepted point.
     When a ``region`` is supplied, the recovered step is projected onto the
     region before value/grad are recomputed.
    """
    region = resolve_region(region)

    def fun_only(p):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_iter,
        curv_rtol=c2,  # strong Wolfe curvature constant
        slope_rtol=c1,  # sufficient decrease (Armijo) constant
        tol=c1,  # sufficient decrease tolerance
        initial_guess_strategy="one",
        max_learning_rate=float(max_step),
    )
    ls_state = ls.init(params)

    # The zoom line search expects ``updates`` to be the search direction
    # and uses value_fn / grad to find the step. It returns rescaled
    # updates equal to ``alpha * direction``.
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
    # Temperature meta-rule: even if the Wolfe step failed to descend, a
    # Metropolis uphill move may still accept it.
    delta_e = new_value - value
    stochastic, _key = _metropolis_accept(
        delta_e, temperature, jax.random.PRNGKey(seed), new_value.dtype
    )
    done = jnp.logical_or(new_value < value, stochastic)

    # Recover the step size from the scaling of the direction.
    d_norm_sq = tree_vdot(direction, direction)
    step_size = jnp.where(
        d_norm_sq > 0.0,
        tree_vdot(scaled_updates, direction) / d_norm_sq,
        jnp.asarray(0.0, dtype=new_value.dtype),
    )
    # Optax's zoom search hides its intermediate probes; expose the single
    # accepted point as a probe so the oracle still benefits.
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
        # Optax's zoom search does not expose its internal eval count; report
        # the recompute (1) plus the budget as an upper bound so downstream
        # totals are conservative rather than silently undercounting.
        num_evals=jnp.asarray(max_iter + 1, dtype=jnp.int32),
    )

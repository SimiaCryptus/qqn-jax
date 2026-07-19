from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _make_projected_point,
    _empty_probes,
    _metropolis_accept,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_vdot, tree_add_scaled


def backtracking_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-2,
    shrink: float = 0.5,
    max_iter: int = 5,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Backtracking line search (Armijo), self-contained for Optax.

    Repeatedly shrinks the step size by ``shrink`` until the Armijo
    sufficient-decrease condition ``f(x + α d) ≤ f(x) + c1 α gᵀd`` holds
    or ``max_iter`` is reached. Implemented with ``lax.while_loop`` to stay
    JIT/vmap compatible.
     When ``max_step > init_step`` an *extrapolation* phase runs first: the
     step is grown by ``1/shrink`` (capped at ``max_step``) while Armijo keeps
     holding and the objective keeps improving, letting the search explore
     ``t > 1`` (past the oracle endpoint). Once growth stops improving (or the
     cap is hit) the usual backtracking shrink phase takes over.
     If a ``region`` is supplied, the candidate point ``x + α·d`` is projected
     onto the region before evaluation, so the search navigates the feasible
     (projected) path.
    When ``temperature > 0`` a Metropolis-style stochastic acceptance is
    layered on top of the Armijo test: a step that fails Armijo may still be
    accepted (an *uphill climb*) with probability ``exp(−ΔE / T)`` where
    ``ΔE = f(x + α·d) − f(x)`` and ``T`` is the (geometrically cooled)
    temperature. With the default ``temperature = 0.0`` this stochastic path
    is disabled entirely and the search reduces to plain Armijo backtracking.
    A ``seed`` seeds a deterministic PRNG so the search stays JIT/vmap
    compatible and reproducible.
    """
    region = resolve_region(region)
    project = _make_projected_point(region, region_state, params)
    dg = tree_vdot(grad, direction)

    def eval_at(alpha):
        raw = tree_add_scaled(params, alpha, direction)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        return projected, val, g

    eff_probes = max_probes if record_probes else 1
    init_pp, init_pg, init_pv, init_pval, init_pa = _empty_probes(params, eff_probes)
    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    key0 = jax.random.PRNGKey(seed)

    def accept(alpha, val, temp, key):
        """Return (accepted, new_key). Armijo OR (optional) Metropolis test."""
        armijo = val <= value + c1 * alpha * dg
        delta_e = val - value
        stochastic, key = _metropolis_accept(delta_e, temp, key, value.dtype)
        return jnp.logical_or(armijo, stochastic), key

    def cond(carry):
        (
            alpha,
            i,
            evals,
            val,
            _g,
            _p,
            accepted,
            _temp,
            _key,
            _pp,
            _pg,
            _pv,
            _pval,
            _pa,
        ) = carry
        return jnp.logical_and(jnp.logical_not(accepted), i < max_iter)

    def body(carry):
        (
            alpha,
            i,
            evals,
            _val,
            _g,
            _p,
            _accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry
        alpha = alpha * shrink
        new_params, new_val, new_g = eval_at(alpha)
        accepted, key = accept(alpha, new_val, temp, key)
        temp = temp * cooling

        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i, new_params, new_g, new_val, alpha, eff_probes
        )

        return (
            alpha,
            i + 1,
            evals + 1,
            new_val,
            new_g,
            new_params,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        )

    init_alpha = jnp.asarray(init_step, dtype=value.dtype)
    max_alpha = jnp.asarray(max_step, dtype=value.dtype)
    grow = jnp.asarray(1.0 / shrink, dtype=value.dtype)

    init_params, init_val, init_g = eval_at(init_alpha)
    init_accepted, key1 = accept(init_alpha, init_val, temp0, key0)
    temp1 = temp0 * cooling

    init_pp, init_pg, init_pv, init_pval, init_pa = _record_probe(
        init_pp,
        init_pg,
        init_pv,
        init_pval,
        init_pa,
        0,
        init_params,
        init_g,
        init_val,
        init_alpha,
        eff_probes,
    )

    def extrap_cond(carry):
        alpha, i, evals, val, _g, _p, _acc, _pp, _pg, _pv, _pval, _pa = carry
        next_alpha = alpha * grow
        can_grow = jnp.logical_and(next_alpha <= max_alpha, i < max_iter)
        return jnp.logical_and(can_grow, _acc)

    def extrap_body(carry):
        alpha, i, evals, prev_val, _g, _p, _acc, pp, pg, pv, pval, pa = carry
        new_alpha = alpha * grow
        new_params, new_val, new_g = eval_at(new_alpha)
        armijo = new_val <= value + c1 * new_alpha * dg
        improved = new_val < prev_val
        keep = jnp.logical_and(armijo, improved)
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i, new_params, new_g, new_val, new_alpha, eff_probes
        )

        out_alpha = jnp.where(keep, new_alpha, alpha)
        out_val = jnp.where(keep, new_val, prev_val)
        out_g = jnp.where(keep, new_g, _g)
        out_p = jnp.where(keep, new_params, _p)
        return (
            out_alpha,
            i + 1,
            evals + 1,
            out_val,
            out_g,
            out_p,
            keep,
            pp,
            pg,
            pv,
            pval,
            pa,
        )

    use_extrap = max_alpha > init_alpha

    (
        ex_alpha,
        ex_i,
        ex_evals,
        ex_val,
        ex_g,
        ex_p,
        ex_acc,
        ex_pp,
        ex_pg,
        ex_pv,
        ex_pval,
        ex_pa,
    ) = jax.lax.cond(
        jnp.logical_and(use_extrap, init_accepted),
        lambda c: jax.lax.while_loop(extrap_cond, extrap_body, c),
        lambda c: c,
        (
            init_alpha,
            jnp.asarray(1),
            jnp.asarray(1, jnp.int32),
            init_val,
            init_g,
            init_params,
            init_accepted,
            init_pp,
            init_pg,
            init_pv,
            init_pval,
            init_pa,
        ),
    )

    init_alpha = ex_alpha
    init_val = ex_val
    init_g = ex_g
    init_params = ex_p
    init_accepted = jnp.logical_or(init_accepted, ex_acc)
    init_pp, init_pg, init_pv, init_pval, init_pa = ex_pp, ex_pg, ex_pv, ex_pval, ex_pa

    (
        alpha,
        n_iters,
        eval_count,
        final_val,
        final_g,
        new_params,
        accepted,
        _temp,
        _key,
        probe_params,
        probe_grads,
        probe_valid,
        probe_values,
        probe_alphas,
    ) = jax.lax.while_loop(
        cond,
        body,
        (
            init_alpha,
            ex_i,
            ex_evals,
            init_val,
            init_g,
            init_params,
            init_accepted,
            temp1,
            key1,
            init_pp,
            init_pg,
            init_pv,
            init_pval,
            init_pa,
        ),
    )

    num_evals = eval_count
    return LineSearchResult(
        step_size=alpha,
        new_value=final_val,
        new_grad=final_g,
        new_params=new_params,
        done=accepted,
        probe_params=probe_params,
        probe_grads=probe_grads,
        probe_valid=probe_valid,
        probe_values=probe_values,
        probe_alphas=probe_alphas,
        num_evals=num_evals,
    )

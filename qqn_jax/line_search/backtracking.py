from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _empty_probes,
    _metropolis_accept,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult


def backtracking_search(
      eval_at: Callable,
      params,
      value,
      grad,
      slope0,
      *,
      init_step: float = 1.0,
      c1: float = 1e-2,
      shrink: float = 0.5,
      max_iter: int = 5,
      temperature: float = 0.0,
      cooling: float = 0.95,
      seed: int = 0,
      max_probes: int = 32,
      record_probes: bool = True,
      max_step: float = 1.0,
  ) -> LineSearchResult:
    """Backtracking line search (Armijo) over a 1-D scalar problem.

    Operates purely on the scalar problem ``φ(t)`` exposed via
    ``eval_at(t) -> (params, value, grad, slope)`` and the directional
    derivative ``slope0 = φ'(0)``. It has *no* knowledge of the path,
    direction or region — those were baked into ``eval_at`` by the solver.

    Repeatedly shrinks the step by ``shrink`` until the Armijo condition
    ``φ(t) ≤ φ(0) + c1·t·φ'(0)`` holds or ``max_iter`` is reached.
    When ``max_step > init_step`` an extrapolation phase runs first,
    growing ``t`` while Armijo holds and ``φ`` keeps improving. When
    ``temperature > 0`` a Metropolis stochastic acceptance is layered on
    top of the Armijo test.
    """
    dg = slope0

    def eval_pvg(alpha):
        p, val, g, _slope = eval_at(alpha)
        return p, val, g

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
        new_params, new_val, new_g = eval_pvg(alpha)
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

    init_params, init_val, init_g = eval_pvg(init_alpha)
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
        new_params, new_val, new_g = eval_pvg(new_alpha)
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

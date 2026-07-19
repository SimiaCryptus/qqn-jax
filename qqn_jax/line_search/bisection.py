from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _empty_probes,
    _record_probe,
    _metropolis_accept,
)
from qqn_jax.line_search.result import LineSearchResult


def bisection_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
    init_step: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 25,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Bisection line search that seeks a *true* one-dimensional minimum.

    Operates purely on the scalar problem ``φ(t)`` (with slope ``φ'(t)``)
    exposed by ``eval_at``; it has no knowledge of the underlying path.
    Bisects on ``φ'(t)`` to drive it toward zero, locating a genuine
    stationary point of the objective along the (pre-baked) path.
    """
    dg = slope0
    max_alpha = jnp.asarray(max_step, dtype=value.dtype)

    eff_probes = max_probes if record_probes else 1
    init_pp, init_pg, init_pv, init_pval, init_pa = _empty_probes(params, eff_probes)
    zero = jnp.asarray(0.0, dtype=value.dtype)
    hi0 = jnp.asarray(init_step, dtype=value.dtype)

    p_hi, v_hi, g_hi, s_hi = eval_at(hi0)
    init_pp, init_pg, init_pv, init_pval, init_pa = _record_probe(
        init_pp,
        init_pg,
        init_pv,
        init_pval,
        init_pa,
        0,
        p_hi,
        g_hi,
        v_hi,
        hi0,
        eff_probes,
    )

    def bracket_cond(carry):
        hi, s_hi, v_hi, i, evals, _pp, _pg, _pv, _pval, _pa = carry

        return jnp.logical_and(
            jnp.logical_and(s_hi < 0.0, i < max_iter), hi < max_alpha
        )

    def bracket_body(carry):
        hi, _s_hi, _v_hi, i, evals, pp, pg, pv, pval, pa = carry
        new_hi = jnp.minimum(hi * 2.0, max_alpha)
        p, v, g, s = eval_at(new_hi)
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i, p, g, v, new_hi, eff_probes
        )
        return new_hi, s, v, i + 1, evals + 1, pp, pg, pv, pval, pa

    (
        hi,
        s_hi_final,
        _v_hi_final,
        bracket_iters,
        bracket_evals,
        pp,
        pg,
        pv,
        pval,
        pa,
    ) = jax.lax.while_loop(
        bracket_cond,
        bracket_body,
        (
            hi0,
            s_hi,
            v_hi,
            jnp.asarray(1),
            jnp.asarray(1, jnp.int32),
            init_pp,
            init_pg,
            init_pv,
            init_pval,
            init_pa,
        ),
    )
    bracketed = s_hi_final >= 0.0

    def bisect_cond(carry):
        (lo, hi, i, evals, best_a, best_v, best_p, best_g, pp, pg, pv, pval, pa) = carry
        return i < max_iter

    def bisect_body(carry):
        (lo, hi, i, evals, best_a, best_v, best_p, best_g, pp, pg, pv, pval, pa) = carry
        mid = 0.5 * (lo + hi)
        p, v, g, s = eval_at(mid)
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, bracket_iters + i, p, g, v, mid, eff_probes
        )

        improved = v < best_v
        best_a = jnp.where(improved, mid, best_a)
        best_v = jnp.where(improved, v, best_v)
        best_p = jnp.where(improved, p, best_p)
        best_g = jnp.where(improved, g, best_g)

        go_right = s < 0.0
        new_lo = jnp.where(go_right, mid, lo)
        new_hi = jnp.where(go_right, hi, mid)
        return (
            new_lo,
            new_hi,
            i + 1,
            evals + 1,
            best_a,
            best_v,
            best_p,
            best_g,
            pp,
            pg,
            pv,
            pval,
            pa,
        )

    (
        _lo,
        _hi,
        bisect_iters,
        total_evals,
        best_alpha,
        best_value,
        best_params,
        best_grad,
        pp,
        pg,
        pv,
        pval,
        pa,
    ) = jax.lax.while_loop(
        bisect_cond,
        bisect_body,
        (
            zero,
            hi,
            jnp.asarray(0),
            bracket_evals,
            hi,
            v_hi,
            p_hi,
            g_hi,
            pp,
            pg,
            pv,
            pval,
            pa,
        ),
    )

    armijo = best_value <= value + c1 * best_alpha * dg

    stochastic, _key = _metropolis_accept(
        best_value - value, temperature, jax.random.PRNGKey(seed), value.dtype
    )
    done = jnp.logical_or(jnp.logical_or(armijo, bracketed), stochastic)
    return LineSearchResult(
        step_size=best_alpha,
        new_value=best_value,
        new_grad=best_grad,
        new_params=best_params,
        done=done,
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=total_evals,
    )

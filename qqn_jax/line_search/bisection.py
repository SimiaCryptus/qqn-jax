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

    A Metropolis-style acceptance test with a cooling ``temperature`` is
    evaluated *during* both the bracketing and bisection phases (not just
    once, after the fact, against the final best value). This lets
    ``temperature`` actually influence which candidate step is returned
    (early-accepting a currently-probed point), rather than merely
    flipping the ``done`` flag on an already-fixed result.
    """
    dg = slope0
    max_alpha = jnp.asarray(max_step, dtype=value.dtype)

    eff_probes = max_probes if record_probes else 1
    init_pp, init_pg, init_pv, init_pval, init_pa = _empty_probes(params, eff_probes)
    zero = jnp.asarray(0.0, dtype=value.dtype)
    hi0 = jnp.asarray(init_step, dtype=value.dtype)

    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    key0 = jax.random.PRNGKey(seed)

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

    stoch0, key0 = _metropolis_accept(v_hi - value, temp0, key0, value.dtype)
    temp0 = temp0 * cooling

    def bracket_cond(carry):
        (
            hi,
            s_hi,
            v_hi,
            p_hi,
            g_hi,
            i,
            evals,
            accepted,
            temp,
            key,
            _pp,
            _pg,
            _pv,
            _pval,
            _pa,
        ) = carry
        keep_growing = jnp.logical_and(
            jnp.logical_and(s_hi < 0.0, i < max_iter), hi < max_alpha
        )
        return jnp.logical_and(jnp.logical_not(accepted), keep_growing)

    def bracket_body(carry):
        (
            hi,
            _s_hi,
            _v_hi,
            _p_hi,
            _g_hi,
            i,
            evals,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry
        new_hi = jnp.minimum(hi * 2.0, max_alpha)
        p, v, g, s = eval_at(new_hi)
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i, p, g, v, new_hi, eff_probes
        )
        new_accepted, key = _metropolis_accept(v - value, temp, key, value.dtype)
        temp = temp * cooling
        return (
            new_hi,
            s,
            v,
            p,
            g,
            i + 1,
            evals + 1,
            new_accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        )

    (
        hi,
        s_hi_final,
        v_hi_final,
        p_hi_final,
        g_hi_final,
        bracket_iters,
        bracket_evals,
        bracket_accepted,
        bracket_temp,
        bracket_key,
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
            p_hi,
            g_hi,
            jnp.asarray(1),
            jnp.asarray(1, jnp.int32),
            stoch0,
            temp0,
            key0,
            init_pp,
            init_pg,
            init_pv,
            init_pval,
            init_pa,
        ),
    )
    bracketed = jnp.logical_or(s_hi_final >= 0.0, bracket_accepted)

    def bisect_cond(carry):
        (
            lo,
            hi,
            i,
            evals,
            best_a,
            best_v,
            best_p,
            best_g,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry
        return jnp.logical_and(jnp.logical_not(accepted), i < max_iter)

    def bisect_body(carry):
        (
            lo,
            hi,
            i,
            evals,
            best_a,
            best_v,
            best_p,
            best_g,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry
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

        stoch, key = _metropolis_accept(v - value, temp, key, value.dtype)
        temp = temp * cooling
        # A temperature-driven acceptance directly adopts the currently
        # probed point (even if it isn't the best value seen so far) and
        # stops the search early, mirroring the Armijo-Wolfe reference.
        best_a = jnp.where(stoch, mid, best_a)
        best_v = jnp.where(stoch, v, best_v)
        best_p = jnp.where(stoch, p, best_p)
        best_g = jnp.where(stoch, g, best_g)

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
            stoch,
            temp,
            key,
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
        bisect_accepted,
        _bisect_temp,
        _bisect_key,
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
            v_hi_final,
            p_hi_final,
            g_hi_final,
            bracket_accepted,
            bracket_temp,
            bracket_key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ),
    )

    armijo = best_value <= value + c1 * best_alpha * dg

    done = jnp.logical_or(jnp.logical_or(armijo, bracketed), bisect_accepted)
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
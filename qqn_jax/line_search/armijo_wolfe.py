from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _empty_probes,
    _record_probe,
    _metropolis_accept,
)
from qqn_jax.line_search.result import LineSearchResult


def armijo_wolfe_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
    init_step: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Combined Armijo–Wolfe line search over a 1-D scalar problem.

    Textbook two-phase bracketing + zoom (Nocedal & Wright, Alg. 3.5/3.6)
    enforcing Armijo sufficient decrease and the strong Wolfe curvature
    condition on ``φ(t) = f(x + d(t))``, ``φ'(t)`` where the path ``d(t)``
    and region projection are pre-baked into ``eval_at`` by the solver.
    This search is therefore fully path-agnostic.
    """
    dg = slope0
    max_alpha = jnp.asarray(max_step, dtype=value.dtype)
    zero = jnp.asarray(0.0, dtype=value.dtype)

    eff_probes = max_probes if record_probes else 1
    pp, pg, pv, pval, pa = _empty_probes(params, eff_probes)
    abs_dg = jnp.abs(dg)

    a0 = jnp.asarray(init_step, dtype=value.dtype)
    a0 = jnp.minimum(a0, max_alpha)

    p0, v0, g0, s0 = eval_at(a0)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, p0, g0, v0, a0, eff_probes
    )

    init_armijo = v0 <= value + c1 * a0 * dg
    init_found = jnp.logical_and(init_armijo, jnp.abs(s0) <= c2 * abs_dg)
    init_bracket_a = jnp.logical_not(init_armijo)
    init_bracket_c = jnp.logical_and(
        init_armijo, jnp.logical_and(jnp.logical_not(init_found), s0 >= 0.0)
    )
    init_bracketed = jnp.logical_or(init_bracket_a, init_bracket_c)

    init_lo = jnp.where(init_bracket_c, a0, zero)
    init_hi = jnp.where(init_bracket_c, zero, a0)
    init_phi_lo = jnp.where(init_bracket_c, v0, value)
    init_s_lo = jnp.where(init_bracket_c, s0, dg)
    init_phi_hi = jnp.where(init_bracket_c, value, v0)
    init_s_hi = jnp.where(init_bracket_c, dg, s0)

    def bracket_cond(carry):
        (
            a_prev,
            phi_prev,
            s_prev,
            a_cur,
            phi_cur,
            s_cur,
            g_cur,
            p_cur,
            found,
            bracketed,
            lo,
            hi,
            phi_lo,
            s_lo,
            phi_hi,
            s_hi,
            p_lo,
            g_lo,
            p_hi,
            g_hi,
            best_a,
            best_v,
            best_p,
            best_g,
            i,
            evals,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry
        stop = jnp.logical_or(found, bracketed)
        can_grow = jnp.logical_and(i < max_iter, a_cur < max_alpha)
        return jnp.logical_and(jnp.logical_not(stop), can_grow)

    def bracket_body(carry):
        (
            a_prev,
            phi_prev,
            s_prev,
            a_cur,
            phi_cur,
            s_cur,
            g_cur,
            p_cur,
            found,
            bracketed,
            lo,
            hi,
            phi_lo,
            s_lo,
            phi_hi,
            s_hi,
            p_lo,
            g_lo,
            p_hi,
            g_hi,
            best_a,
            best_v,
            best_p,
            best_g,
            i,
            evals,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry

        improved = phi_cur < best_v
        best_a = jnp.where(improved, a_cur, best_a)
        best_v = jnp.where(improved, phi_cur, best_v)
        best_p = jnp.where(improved, p_cur, best_p)
        best_g = jnp.where(improved, g_cur, best_g)

        armijo_cur = phi_cur <= value + c1 * a_cur * dg
        cond_a = jnp.logical_or(
            jnp.logical_not(armijo_cur),
            jnp.logical_and(i > 0, phi_cur >= phi_prev),
        )

        cond_found = jnp.abs(s_cur) <= c2 * abs_dg

        cond_c = s_cur >= 0.0

        use_a = cond_a
        use_c = jnp.logical_and(
            jnp.logical_not(cond_a),
            jnp.logical_and(jnp.logical_not(cond_found), cond_c),
        )
        new_bracketed = jnp.logical_or(use_a, use_c)
        new_found = jnp.logical_and(jnp.logical_not(cond_a), cond_found)

        lo_a, hi_a = a_prev, a_cur
        phi_lo_a, phi_hi_a = phi_prev, phi_cur
        s_lo_a, s_hi_a = s_prev, s_cur

        lo_c, hi_c = a_cur, a_prev
        phi_lo_c, phi_hi_c = phi_cur, phi_prev
        s_lo_c, s_hi_c = s_cur, s_prev
        new_lo = jnp.where(use_a, lo_a, jnp.where(use_c, lo_c, lo))
        new_hi = jnp.where(use_a, hi_a, jnp.where(use_c, hi_c, hi))
        new_phi_lo = jnp.where(use_a, phi_lo_a, jnp.where(use_c, phi_lo_c, phi_lo))
        new_phi_hi = jnp.where(use_a, phi_hi_a, jnp.where(use_c, phi_hi_c, phi_hi))
        new_s_lo = jnp.where(use_a, s_lo_a, jnp.where(use_c, s_lo_c, s_lo))
        new_s_hi = jnp.where(use_a, s_hi_a, jnp.where(use_c, s_hi_c, s_hi))

        next_alpha = jnp.minimum(a_cur * 2.0, max_alpha)
        p_n, v_n, g_n, s_n = eval_at(next_alpha)
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i + 1, p_n, g_n, v_n, next_alpha, eff_probes
        )
        stop_now = jnp.logical_or(new_found, new_bracketed)

        best_a = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), a_cur, best_a)
        best_v = jnp.where(
            jnp.logical_and(new_found, phi_cur < best_v), phi_cur, best_v
        )
        best_p = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), p_cur, best_p)
        best_g = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), g_cur, best_g)

        out_a_prev = jnp.where(stop_now, a_prev, a_cur)
        out_phi_prev = jnp.where(stop_now, phi_prev, phi_cur)
        out_s_prev = jnp.where(stop_now, s_prev, s_cur)
        out_a_cur = jnp.where(stop_now, a_cur, next_alpha)
        out_phi_cur = jnp.where(stop_now, phi_cur, v_n)
        out_s_cur = jnp.where(stop_now, s_cur, s_n)
        out_g_cur = jnp.where(stop_now, g_cur, g_n)
        out_p_cur = jnp.where(stop_now, p_cur, p_n)
        return (
            out_a_prev,
            out_phi_prev,
            out_s_prev,
            out_a_cur,
            out_phi_cur,
            out_s_cur,
            out_g_cur,
            out_p_cur,
            new_found,
            new_bracketed,
            new_lo,
            new_hi,
            new_phi_lo,
            new_s_lo,
            new_phi_hi,
            new_s_hi,
            p_cur,
            g_cur,
            p_cur,
            g_cur,
            best_a,
            best_v,
            best_p,
            best_g,
            i + 1,
            evals + 1,
            pp,
            pg,
            pv,
            pval,
            pa,
        )

    (
        _a_prev,
        _phi_prev,
        _s_prev,
        a_cur,
        phi_cur,
        s_cur,
        g_cur,
        p_cur,
        found,
        bracketed,
        lo,
        hi,
        phi_lo,
        s_lo,
        phi_hi,
        s_hi,
        p_lo,
        g_lo,
        p_hi,
        g_hi,
        best_a,
        best_v,
        best_p,
        best_g,
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
            zero,
            value,
            dg,
            a0,
            v0,
            s0,
            g0,
            p0,
            init_found,
            init_bracketed,
            init_lo,
            init_hi,
            init_phi_lo,
            init_s_lo,
            init_phi_hi,
            init_s_hi,
            p0,
            g0,
            p0,
            g0,
            a0,
            v0,
            p0,
            g0,
            jnp.asarray(1),
            jnp.asarray(1, jnp.int32),
            pp,
            pg,
            pv,
            pval,
            pa,
        ),
    )

    found_a = a_cur
    found_v = phi_cur
    found_p = p_cur
    found_g = g_cur

    def zoom_cond(carry):
        (
            lo,
            hi,
            phi_lo,
            s_lo,
            i,
            evals,
            z_found,
            z_a,
            z_v,
            z_p,
            z_g,
            best_a,
            best_v,
            best_p,
            best_g,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry
        keep = jnp.logical_and(jnp.logical_not(z_found), i < max_iter)

        return jnp.logical_and(keep, bracketed)

    def zoom_body(carry):
        (
            lo,
            hi,
            phi_lo,
            s_lo,
            i,
            evals,
            z_found,
            z_a,
            z_v,
            z_p,
            z_g,
            best_a,
            best_v,
            best_p,
            best_g,
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
        armijo = v <= value + c1 * mid * dg
        higher = v >= phi_lo

        shrink_hi = jnp.logical_or(jnp.logical_not(armijo), higher)
        curv_ok = jnp.abs(s) <= c2 * abs_dg
        this_found = jnp.logical_and(armijo, curv_ok)

        flip = jnp.logical_and(jnp.logical_not(shrink_hi), s * (hi - lo) >= 0.0)
        new_hi = jnp.where(shrink_hi, mid, jnp.where(flip, lo, hi))
        new_lo = jnp.where(shrink_hi, lo, mid)
        new_phi_lo = jnp.where(shrink_hi, phi_lo, v)
        new_s_lo = jnp.where(shrink_hi, s_lo, s)
        z_a = jnp.where(this_found, mid, z_a)
        z_v = jnp.where(this_found, v, z_v)
        z_p = jnp.where(this_found, p, z_p)
        z_g = jnp.where(this_found, g, z_g)
        return (
            new_lo,
            new_hi,
            new_phi_lo,
            new_s_lo,
            i + 1,
            evals + 1,
            jnp.logical_or(z_found, this_found),
            z_a,
            z_v,
            z_p,
            z_g,
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
        _phi_lo,
        _s_lo,
        zoom_iters,
        total_evals,
        zoom_found,
        z_a,
        z_v,
        z_p,
        z_g,
        best_a,
        best_v,
        best_p,
        best_g,
        pp,
        pg,
        pv,
        pval,
        pa,
    ) = jax.lax.while_loop(
        zoom_cond,
        zoom_body,
        (
            lo,
            hi,
            phi_lo,
            s_lo,
            jnp.asarray(0),
            bracket_evals,
            jnp.asarray(False),
            best_a,
            best_v,
            best_p,
            best_g,
            best_a,
            best_v,
            best_p,
            best_g,
            pp,
            pg,
            pv,
            pval,
            pa,
        ),
    )

    use_found = found
    use_zoom = jnp.logical_and(jnp.logical_not(found), zoom_found)
    fb_improved = best_v <= value
    fb_a = jnp.where(fb_improved, best_a, jnp.asarray(0.0, dtype=value.dtype))
    fb_v = jnp.where(fb_improved, best_v, value)
    out_a = jnp.where(use_found, found_a, jnp.where(use_zoom, z_a, fb_a))
    out_v = jnp.where(use_found, found_v, jnp.where(use_zoom, z_v, fb_v))
    out_p = jnp.where(use_found, found_p, jnp.where(use_zoom, z_p, best_p))
    out_g = jnp.where(use_found, found_g, jnp.where(use_zoom, z_g, best_g))

    stochastic, _key = _metropolis_accept(
        out_v - value, temperature, jax.random.PRNGKey(seed), value.dtype
    )
    done = jnp.logical_or(jnp.logical_or(use_found, use_zoom), stochastic)
    return LineSearchResult(
        step_size=out_a,
        new_value=out_v,
        new_grad=out_g,
        new_params=out_p,
        done=done,
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=total_evals,
    )

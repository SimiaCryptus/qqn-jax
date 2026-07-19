from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _make_projected_point,
    _empty_probes,
    _record_probe,
    _metropolis_accept,
)
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_vdot, tree_add_scaled


def armijo_wolfe_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Combined Armijo–Wolfe line search (the classic L-BFGS scheme).
    This is the textbook two-phase *bracketing + zoom* line search (Nocedal &
    Wright, Alg. 3.5/3.6) that enforces both the Armijo sufficient-decrease
    condition
        φ(α) ≤ φ(0) + c1·α·φ'(0)
    and the (strong) Wolfe curvature condition
        |φ'(α)| ≤ c2·|φ'(0)|,
    where ``φ(α) = f(x + α·d)`` and ``φ'(α) = ⟨∇f(x + α·d), d⟩``. Unlike the
    permissive Armijo backtracking search, this scheme keeps the L-BFGS
    curvature pairs well conditioned by guaranteeing the Wolfe condition.
    Phase 1 (*bracket*) grows the trial step (capped at ``max_step``) until it
    finds an interval known to contain a point satisfying the Wolfe
    conditions. Phase 2 (*zoom*) shrinks that interval by bisection until the
    conditions hold or the budget is exhausted. Implemented with
    ``lax.while_loop`` to stay JIT/vmap compatible.
     The ``temperature`` meta-rule is layered on the final acceptance: when no
     Wolfe point is found, the best-value fallback may still be marked
     ``done`` via a Metropolis uphill move (probability ``exp(−ΔE / T)``).
    """
    region = resolve_region(region)
    project = _make_projected_point(region, region_state, params)
    dg = tree_vdot(grad, direction)  # φ'(0)
    max_alpha = jnp.asarray(max_step, dtype=value.dtype)
    zero = jnp.asarray(0.0, dtype=value.dtype)

    def eval_at(alpha):
        raw = tree_add_scaled(params, alpha, direction)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        slope = tree_vdot(g, direction)
        return projected, val, g, slope

    eff_probes = max_probes if record_probes else 1
    pp, pg, pv, pval, pa = _empty_probes(params, eff_probes)
    abs_dg = jnp.abs(dg)

    def wolfe_ok(alpha, val, slope):
        armijo = val <= value + c1 * alpha * dg
        curv = jnp.abs(slope) <= c2 * abs_dg
        return jnp.logical_and(armijo, curv)

    # --- Phase 1: bracket an interval [lo, hi] containing a Wolfe point. --
    # We track a "previous" trial (alpha_prev, phi_prev, slope_prev) and a
    # current trial; the classic conditions decide when a bracket is found.
    a0 = jnp.asarray(init_step, dtype=value.dtype)
    p0, v0, g0, s0 = eval_at(a0)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, p0, g0, v0, a0, eff_probes
    )

    # Bracket carry:
    #  alpha_prev, phi_prev, slope_prev  : previous trial
    #  alpha_cur,  phi_cur,  slope_cur   : current trial
    #  found      : a Wolfe point already satisfied at the current trial
    #  lo/hi + associated phi/slope/params/grad : the bracket, once set
    #  bracketed  : whether a bracket was produced
    #  best_*     : lowest-value probe seen (fallback return)
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
        # Track best-value point for graceful fallback.
        improved = phi_cur < best_v
        best_a = jnp.where(improved, a_cur, best_a)
        best_v = jnp.where(improved, phi_cur, best_v)
        best_p = jnp.where(improved, p_cur, best_p)
        best_g = jnp.where(improved, g_cur, best_g)
        # Condition A: Armijo violated, or (i>0 and phi_cur >= phi_prev)
        #  => bracket = [prev, cur]
        armijo_cur = phi_cur <= value + c1 * a_cur * dg
        cond_a = jnp.logical_or(
            jnp.logical_not(armijo_cur),
            jnp.logical_and(i > 0, phi_cur >= phi_prev),
        )
        # Condition B: Wolfe curvature already satisfied => done (found).
        cond_found = jnp.abs(s_cur) <= c2 * abs_dg
        # Condition C: slope non-negative => bracket = [cur, prev]
        cond_c = s_cur >= 0.0
        # Decide the bracket (only relevant when we stop this iteration).
        # cond_a -> [prev, cur]; cond_c -> [cur, prev].
        use_a = cond_a
        use_c = jnp.logical_and(
            jnp.logical_not(cond_a),
            jnp.logical_and(jnp.logical_not(cond_found), cond_c),
        )
        new_bracketed = jnp.logical_or(use_a, use_c)
        new_found = jnp.logical_and(jnp.logical_not(cond_a), cond_found)
        # [prev, cur] bracket
        lo_a, hi_a = a_prev, a_cur
        phi_lo_a, phi_hi_a = phi_prev, phi_cur
        s_lo_a, s_hi_a = s_prev, s_cur
        # For prev endpoint we do not retain params/grad; zoom re-evaluates
        # midpoints, so endpoint params/grad are only used as fallback. Reuse
        # current point as a safe placeholder.
        # [cur, prev] bracket
        lo_c, hi_c = a_cur, a_prev
        phi_lo_c, phi_hi_c = phi_cur, phi_prev
        s_lo_c, s_hi_c = s_cur, s_prev
        new_lo = jnp.where(use_a, lo_a, jnp.where(use_c, lo_c, lo))
        new_hi = jnp.where(use_a, hi_a, jnp.where(use_c, hi_c, hi))
        new_phi_lo = jnp.where(use_a, phi_lo_a, jnp.where(use_c, phi_lo_c, phi_lo))
        new_phi_hi = jnp.where(use_a, phi_hi_a, jnp.where(use_c, phi_hi_c, phi_hi))
        new_s_lo = jnp.where(use_a, s_lo_a, jnp.where(use_c, s_lo_c, s_lo))
        new_s_hi = jnp.where(use_a, s_hi_a, jnp.where(use_c, s_hi_c, s_hi))
        # Grow the current step for the next iteration (only used if not stopped).
        next_alpha = jnp.minimum(a_cur * 2.0, max_alpha)
        p_n, v_n, g_n, s_n = eval_at(next_alpha)
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i + 1, p_n, g_n, v_n, next_alpha, eff_probes
        )
        stop_now = jnp.logical_or(new_found, new_bracketed)
        # If we stop, freeze the current trial as the "best found" candidate.
        best_a = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), a_cur, best_a)
        best_v = jnp.where(
            jnp.logical_and(new_found, phi_cur < best_v), phi_cur, best_v
        )
        best_p = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), p_cur, best_p)
        best_g = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), g_cur, best_g)
        # Advance the trial window when not stopping.
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
            jnp.asarray(False),
            jnp.asarray(False),
            zero,
            a0,
            value,
            dg,
            v0,
            s0,
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
    # If the current trial already satisfied Wolfe during bracketing, adopt it.
    found_a = a_cur
    found_v = phi_cur
    found_p = p_cur
    found_g = g_cur

    # --- Phase 2: zoom within [lo, hi] via bisection. --------------------
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
        # Only zoom if we actually bracketed and haven't already found a point.
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
        # If Armijo fails or value not below lo endpoint, shrink from the right.
        shrink_hi = jnp.logical_or(jnp.logical_not(armijo), higher)
        curv_ok = jnp.abs(s) <= c2 * abs_dg
        this_found = jnp.logical_and(armijo, curv_ok)
        # Standard zoom update of the bracket.
        # If not shrinking hi, mid becomes new lo; if slope*(hi-lo) >= 0 flip hi->lo.
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
    # Resolve the returned point:
    #  1. a Wolfe point found during bracketing, else
    #  2. a Wolfe point found during zoom, else
    #  3. the best-value probe seen (graceful fallback).
    use_found = found
    use_zoom = jnp.logical_and(jnp.logical_not(found), zoom_found)
    out_a = jnp.where(use_found, found_a, jnp.where(use_zoom, z_a, best_a))
    out_v = jnp.where(use_found, found_v, jnp.where(use_zoom, z_v, best_v))
    out_p = jnp.where(use_found, found_p, jnp.where(use_zoom, z_p, best_p))
    out_g = jnp.where(use_found, found_g, jnp.where(use_zoom, z_g, best_g))
    # Temperature meta-rule: if no Wolfe point was found, the best-value
    # fallback may still be accepted via a Metropolis uphill move.
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

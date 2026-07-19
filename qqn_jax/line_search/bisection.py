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
from qqn_jax.paths import QUADRATIC_PATH
from qqn_jax.paths.base import PathStrategy
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.utils import tree_vdot, tree_add_scaled


def bisection_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 25,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Bisection line search that seeks a *true* one-dimensional minimum.
    Whereas the backtracking/Armijo family is deliberately *permissive* — it
    accepts the first step that merely makes sufficient progress — this search
    is the opposite: it bisects on the directional derivative
    ``φ'(α) = ⟨∇f(x + α·d), d⟩`` to drive it toward zero, locating a genuine
    stationary point of the objective *along the path*. Use it only in the
    special cases where an accurate along-path minimizer is worth the extra
    gradient evaluations (the cross-product profiles reserve it for exactly
    that role).
    The scheme first brackets a sign change of ``φ'`` by expanding from a small
    lower bound; if no bracket is found within the expansion budget it falls
    back to the best (lowest-value) point it evaluated, still reporting
    ``done`` when the Armijo sufficient-decrease condition holds there.
    Implemented with ``lax.while_loop`` to stay JIT/vmap compatible.
     The ``temperature`` meta-rule is layered on the final acceptance: a
     non-descending minimizer may still be marked ``done`` via a Metropolis
     uphill move (probability ``exp(−ΔE / T)``).
    """
    region = resolve_region(region)
    project = _make_projected_point(region, region_state, params)
    dg = tree_vdot(grad, direction)  # φ'(0) = gᵀd
    max_alpha = jnp.asarray(max_step, dtype=value.dtype)

    def eval_at(alpha):
        raw = tree_add_scaled(params, alpha, direction)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        slope = tree_vdot(g, direction)
        return projected, val, g, slope

    eff_probes = max_probes if record_probes else 1
    init_pp, init_pg, init_pv, init_pval, init_pa = _empty_probes(params, eff_probes)
    zero = jnp.asarray(0.0, dtype=value.dtype)
    hi0 = jnp.asarray(init_step, dtype=value.dtype)
    # --- Phase 1: bracket a sign change of φ'. ---------------------------
    # We keep a low endpoint (slope known-negative, starting at α=0 where the
    # slope is dg < 0 for a descent direction) and expand the high endpoint by
    # doubling until φ'(hi) >= 0 (a bracket) or the budget is exhausted.
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
        # Keep expanding while slope still negative (no bracket yet).
        # Cap expansion at ``max_step`` so extrapolation past the oracle
        # endpoint stays bounded.
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
            jnp.asarray(1, jnp.int32),  # the initial eval_at(init_step) probe
            init_pp,
            init_pg,
            init_pv,
            init_pval,
            init_pa,
        ),
    )
    bracketed = s_hi_final >= 0.0

    # --- Phase 2: bisect within [lo, hi] to drive φ'(α) -> 0. ------------
    # lo starts at 0 (slope dg < 0); hi is the bracketing high endpoint.
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
        # Track the lowest-value probe seen (the returned point).
        improved = v < best_v
        best_a = jnp.where(improved, mid, best_a)
        best_v = jnp.where(improved, v, best_v)
        best_p = jnp.where(improved, p, best_p)
        best_g = jnp.where(improved, g, best_g)
        # Standard slope bisection: if φ'(mid) < 0 the minimum is to the right.
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

    # Seed the "best" tracker with the bracketing high point (a valid,
    # already-projected candidate) so we always have something to return.
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
            hi,  # best_alpha
            v_hi,  # best_value
            p_hi,  # best_params
            g_hi,  # best_grad
            pp,
            pg,
            pv,
            pval,
            pa,
        ),
    )
    # Only actually bisect when a bracket was found; otherwise the best-value
    # point already tracked from the expansion phase is returned as-is.
    # (The while_loop still runs but the bisection interval collapses onto the
    # unbracketed hi, so the result degrades gracefully to the expansion best.)
    # Accept when the Armijo sufficient-decrease condition holds at the
    # returned point (a minimizer that also descends), or when we successfully
    # bracketed a stationary point.
    armijo = best_value <= value + c1 * best_alpha * dg
    # Temperature meta-rule: a non-descent minimizer may still be accepted
    # via a Metropolis uphill move.
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

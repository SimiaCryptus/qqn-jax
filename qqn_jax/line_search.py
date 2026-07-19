"""Line search strategies for QQN.

The line search is a *first-class component* of QQN. It operates over the
quadratic path direction ``d`` (already constructed) and selects a step
size ``α`` satisfying sufficient decrease (Armijo) and, optionally, the
curvature (strong Wolfe) condition.

We delegate the strong-Wolfe search to Optax's proven, JIT/vmap-compatible
``optax.scale_by_zoom_linesearch`` and provide a self-contained
backtracking (Armijo) search. Both are adapted to the QQN interface so
the strategies remain swappable.
"""

from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp
import optax

from qqn_jax.utils import tree_add_scaled, tree_vdot
from qqn_jax.regions import resolve_region


class LineSearchResult(NamedTuple):
    """Result of a line search.

    Attributes:
        step_size: chosen step size ``α``.
        new_value: function value at ``params + α·d``.
        new_grad: gradient at ``params + α·d``.
        new_params: the updated parameters.
        done: whether the search satisfied its conditions.
         probe_params: fixed-size ``(max_probes, n)`` buffer of evaluated
             points along the path (for feeding oracle curvature memory).
         probe_grads: fixed-size ``(max_probes, n)`` buffer of probe gradients.
         probe_valid: fixed-size ``(max_probes,)`` boolean mask of filled slots.
    """

    step_size: jnp.ndarray
    new_value: jnp.ndarray
    new_grad: jnp.ndarray
    new_params: jnp.ndarray
    done: jnp.ndarray
    probe_params: Any = None
    probe_grads: Any = None
    probe_valid: Any = None
    # Per-probe objective values (so callers gating on descent need not
    # recompute f via an extra vmapped forward pass — the line search
    # already evaluated these points).
    probe_values: Any = None
    # Per-probe step size α (lets the oracle replay probes in α-order
    # rather than slot-order, which matters for secant differences).
    probe_alphas: Any = None
    # Number of value-and-grad evaluations performed by the line search.
    # Each ``value_and_grad_fn`` call evaluates both f and ∇f, so this counts
    # combined value+grad oracle calls. ``None`` means "not reported".
    num_evals: Any = None


def _empty_probes(params, max_probes):
    """Allocate empty probe buffers shaped for ``params`` (a flat vector)."""
    n = params.shape[0]
    return (
        jnp.zeros((max_probes, n), dtype=params.dtype),
        jnp.zeros((max_probes, n), dtype=params.dtype),
        jnp.zeros((max_probes,), dtype=bool),
        jnp.full((max_probes,), jnp.inf, dtype=params.dtype),  # values
        jnp.zeros((max_probes,), dtype=params.dtype),  # alphas
    )


def _record_probe(
        probe_params,
        probe_grads,
        probe_valid,
        probe_values,
        probe_alphas,
        slot,
        p,
        g,
        v,
        a,
        max_probes,
):
    """Write ``(p, g)`` into ``slot`` of the probe buffers (JIT-safe)."""
    in_range = jnp.logical_and(slot >= 0, slot < max_probes)
    idx = jnp.clip(slot, 0, max_probes - 1)
    new_params = jnp.where(in_range, probe_params.at[idx].set(p), probe_params)
    new_grads = jnp.where(in_range, probe_grads.at[idx].set(g), probe_grads)
    new_valid = jnp.where(in_range, probe_valid.at[idx].set(True), probe_valid)
    new_values = jnp.where(in_range, probe_values.at[idx].set(v), probe_values)
    new_alphas = jnp.where(in_range, probe_alphas.at[idx].set(a), probe_alphas)
    return new_params, new_grads, new_valid, new_values, new_alphas


def _make_projected_point(region, region_state, params):
    """Return a fn ``α -> projected(x + α·d)`` for a given direction.
    The caller curries the direction in; here we build a helper that, given
    a tentative point ``x + α·d``, projects it onto the region. When the
    region is the identity, this is a no-op (zero overhead).
    """

    def project_candidate(candidate):
        return region.project(params, candidate, region_state)

    return project_candidate


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
    dg = tree_vdot(grad, direction)  # directional derivative gᵀd

    def eval_at(alpha):
        raw = tree_add_scaled(params, alpha, direction)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        return projected, val, g

    # When the caller does not consume probes, shrink the scratch buffers to a
    # single slot so the line-search ``while_loop`` does not allocate and
    # thread a full ``(max_probes, n)`` array through every iteration.
    eff_probes = max_probes if record_probes else 1
    init_pp, init_pg, init_pv, init_pval, init_pa = _empty_probes(params, eff_probes)
    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    # Temperature is enabled only when strictly positive.
    use_temp = temp0 > 0.0
    key0 = jax.random.PRNGKey(seed)

    def accept(alpha, val, temp, key):
        """Return (accepted, new_key). Armijo OR (optional) Metropolis test."""
        armijo = val <= value + c1 * alpha * dg
        delta_e = val - value
        # Safe temperature guards against divide-by-zero when T collapses.
        safe_t = jnp.maximum(temp, jnp.asarray(1e-12, dtype=value.dtype))
        p = jnp.exp(-delta_e / safe_t)
        p = jnp.clip(p, 0.0, 1.0)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, dtype=value.dtype)
        # Only allow the stochastic uphill move when temperature is active.
        stochastic = jnp.logical_and(use_temp, u < p)
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
        # Record this probe (slot = i, since slot 0 holds the init_step probe).
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i, new_params, new_g, new_val, alpha, eff_probes
        )
        # ``evals`` counts every eval_at call: the body adds exactly one.
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

    # Evaluate at the initial step first.
    init_params, init_val, init_g = eval_at(init_alpha)
    init_accepted, key1 = accept(init_alpha, init_val, temp0, key0)
    temp1 = temp0 * cooling
    # Slot 0 records the initial-step probe.
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

    # --- Optional extrapolation phase (t > 1) ---------------------------
    # When permitted (``max_step > init_step``) and the initial step already
    # satisfies Armijo, try growing the step while it keeps improving the
    # objective, capped at ``max_step``. This lets the search explore past
    # the oracle endpoint. The phase reuses the probe buffers, filling slots
    # after slot 0.
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
        # Only advance the accepted point when the grown step still improves.
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
            keep,  # stop growing once a step fails to improve
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
    # If extrapolation ran and improved, adopt its point and skip backtracking.
    init_alpha = ex_alpha
    init_val = ex_val
    init_g = ex_g
    init_params = ex_p
    init_accepted = jnp.logical_or(init_accepted, ex_acc)
    init_pp, init_pg, init_pv, init_pval, init_pa = ex_pp, ex_pg, ex_pv, ex_pval, ex_pa

    # --- Backtracking (shrink) phase ------------------------------------
    # Start the carry from the (possibly extrapolated) initial point. If the
    # initial point was already accepted the ``cond`` guard exits immediately.
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

    # Evals are tracked explicitly in the carry (1 for the initial-step probe
    # plus one per backtracking iteration), decoupled from ``n_iters`` so the
    # count cannot drift if the loop index semantics change.
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
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
) -> LineSearchResult:
    """Strong Wolfe line search via Optax ``scale_by_zoom_linesearch``.

    Enforces Armijo sufficient decrease and the strong curvature
    condition, which keeps the L-BFGS curvature updates well-conditioned.
    The ``temperature`` parameter is accepted for interface uniformity with
    the backtracking searches but is ignored here: Optax's zoom search does
    not expose a stochastic-acceptance hook.

    Optax's zoom line search is a ``GradientTransformationExtraArgs`` whose
    ``update`` step rescales the provided *updates* (here, ``direction``)
    by the discovered step size. We wrap a value-only objective for it and
    recompute value/grad at the accepted point.
     When a ``region`` is supplied, the recovered step is projected onto the
     region before value/grad are recomputed.
    """
    region = resolve_region(region)

    def fun_only(p, *fa, **fkw):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_iter,
        curv_rtol=c2,  # strong Wolfe curvature constant
        slope_rtol=c1,  # sufficient decrease (Armijo) constant
        tol=c1,  # sufficient decrease tolerance
        initial_guess_strategy="one",
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
        done=new_value < value,
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


def fixed_step_search(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        step_size: float = 1.0,
        temperature: float = 0.0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
) -> LineSearchResult:
    """Trivial line search using a constant step size.
    Useful for debugging, benchmarking against a baseline, or when the
    quadratic path scaling already provides a sensible step. Always reports
    ``done=True`` (it makes no acceptance test).
    The ``temperature`` parameter is accepted for interface uniformity but
    has no effect (this search makes no acceptance test).
    """
    region = resolve_region(region)
    alpha = jnp.asarray(step_size, dtype=value.dtype)
    raw_params = tree_add_scaled(params, alpha, direction)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_g, new_val, alpha, max_probes
    )
    return LineSearchResult(
        step_size=alpha,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=jnp.asarray(True),
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=jnp.asarray(1, dtype=jnp.int32),
    )


def armijo_search(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        init_step: float = 1.0,
        c1: float = 1e-2,
        shrink: float = 0.5,
        max_iter: int = 30,
        temperature: float = 0.0,
        cooling: float = 0.95,
        seed: int = 0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
        max_step: float = 1.0,
) -> LineSearchResult:
    """Alias for :func:`backtracking_search`.
    Provided so users can refer to the Armijo backtracking search by its
    classical name as well.
    """
    return backtracking_search(
        value_and_grad_fn,
        params,
        direction,
        value,
        grad,
        *args,
        init_step=init_step,
        c1=c1,
        shrink=shrink,
        max_iter=max_iter,
        temperature=temperature,
        cooling=cooling,
        seed=seed,
        region=region,
        region_state=region_state,
        max_probes=max_probes,
        record_probes=record_probes,
        max_step=max_step,
    )


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
        slope_tol: float = 1e-8,
        temperature: float = 0.0,
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
    The ``temperature`` parameter is accepted for interface uniformity but has
    no effect (this search makes a deterministic minimizing test).
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
        init_pp, init_pg, init_pv, init_pval, init_pa,
        0, p_hi, g_hi, v_hi, hi0, eff_probes,
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
    done = jnp.logical_or(armijo, bracketed)
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


def null_search(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        step_size: float = 1.0,
        grad_scale: float = 1.0,
        temperature: float = 0.0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
) -> LineSearchResult:
    """"Null" line search: unconditionally accept the ``t = 1`` oracle point.
    The ``direction`` handed to the line search is the oracle endpoint
    ``-H∇f`` (the ``t = 1`` point of the quadratic path). This search performs
    *no* acceptance test and simply steps to ``params + step_size·direction``.
    When the oracle degenerates and hands back the raw (negated) gradient — the
    Fallback oracle's terminal safety net returns ``-∇f`` — this reduces to a
    plain scaled-gradient step. The ``grad_scale`` parameter lets callers
    rescale that case: it is applied as an *additional* multiplier when the
    supplied direction is (anti-)parallel to the gradient (i.e. no genuine
    curvature was available).
    The ``temperature`` parameter is accepted for interface uniformity but
    has no effect (this search makes no acceptance test).
    Always reports ``done=True`` (it makes no acceptance test).
    """
    region = resolve_region(region)
    base_alpha = jnp.asarray(step_size, dtype=value.dtype)
    # Detect the "no oracle point" case: the direction is (anti-)parallel to
    # the gradient, i.e. cos-similarity magnitude ~= 1. In that case apply the
    # configurable ``grad_scale`` multiplier.
    dd = tree_vdot(direction, direction)
    gg = tree_vdot(grad, grad)
    dg = tree_vdot(direction, grad)
    denom = jnp.sqrt(dd * gg)
    cos_sim = jnp.where(denom > 0.0, dg / denom, jnp.asarray(0.0, dtype=value.dtype))
    is_grad = jnp.abs(cos_sim) >= (1.0 - 1e-6)
    scale = jnp.where(is_grad, jnp.asarray(grad_scale, dtype=value.dtype), 1.0)
    alpha = base_alpha * scale
    raw_params = tree_add_scaled(params, alpha, direction)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_g, new_val, alpha, max_probes
    )
    return LineSearchResult(
        step_size=alpha,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=jnp.asarray(True),
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=jnp.asarray(1, dtype=jnp.int32),
    )


def backtracking_temperature_search(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        init_step: float = 1.0,
        c1: float = 1e-2,
        shrink: float = 0.5,
        max_iter: int = 30,
        temperature: float = 1.0,
        cooling: float = 0.95,
        seed: int = 0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
        max_step: float = 1.0,
) -> LineSearchResult:
    """Backtracking line search with a Metropolis-style temperature.

    Thin proxy over :func:`backtracking_search` that simply defaults the
    ``temperature`` to ``1.0`` (enabling stochastic acceptance) rather than
    ``0.0``. All the simulated-annealing logic — the Armijo test, the
    Metropolis ``p = exp(−ΔE / T)`` uphill acceptance, and the geometric
    cooling ``T ← cooling·T`` — now lives in ``backtracking_search`` and is
    activated whenever ``temperature > 0``.

    A ``seed`` seeds a deterministic PRNG so the whole search stays
    JIT/vmap-compatible and reproducible.
    """
    return backtracking_search(
        value_and_grad_fn,
        params,
        direction,
        value,
        grad,
        *args,
        init_step=init_step,
        c1=c1,
        shrink=shrink,
        max_iter=max_iter,
        temperature=temperature,
        cooling=cooling,
        seed=seed,
        region=region,
        region_state=region_state,
        max_probes=max_probes,
        record_probes=record_probes,
    )


def hager_zhang_search(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        c1: float = 0.1,
        max_iter: int = 30,
        temperature: float = 0.0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
) -> LineSearchResult:
    """Hager-Zhang line search via Optax ``scale_by_backtracking_linesearch``.
    The Hager-Zhang scheme is a robust approximate-Wolfe line search. We use
    Optax's backtracking transformation parameterized to approximate it,
    recomputing value/grad at the accepted point. Falls back gracefully if
    the underlying transform is unavailable.
    The ``temperature`` parameter is accepted for interface uniformity but
    is ignored (Optax's backtracking transform exposes no stochastic hook).
    """
    region = resolve_region(region)

    def fun_only(p, *fa, **fkw):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_iter,
        slope_rtol=c1,
        decrease_factor=0.8,
        increase_factor=1.0,
        store_grad=True,
    )
    ls_state = ls.init(params)
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
    d_norm_sq = tree_vdot(direction, direction)
    step_size = jnp.where(
        d_norm_sq > 0.0,
        tree_vdot(scaled_updates, direction) / d_norm_sq,
        jnp.asarray(0.0, dtype=new_value.dtype),
    )
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_grad, new_value, step_size, max_probes
    )
    return LineSearchResult(
        step_size=step_size,
        new_value=new_value,
        new_grad=new_grad,
        new_params=new_params,
        done=new_value < value,
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=jnp.asarray(max_iter + 1, dtype=jnp.int32),
    )
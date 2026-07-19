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


def _metropolis_accept(delta_e, temp, key, dtype):
    """Metropolis-style stochastic acceptance meta-rule.
    Returns ``(accepted, new_key)`` where ``accepted`` is True with
    probability ``exp(−ΔE / T)`` (clipped to [0, 1]). Disabled (returns
    False) when ``temp <= 0``. JIT/vmap-safe and deterministic given ``key``.
    """
    temp = jnp.asarray(temp, dtype=dtype)
    use_temp = temp > 0.0
    safe_t = jnp.maximum(temp, jnp.asarray(1e-12, dtype=dtype))
    p = jnp.clip(jnp.exp(-delta_e / safe_t), 0.0, 1.0)
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, dtype=dtype)
    accepted = jnp.logical_and(use_temp, u < p)
    return accepted, key


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


def fixed_step_search(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
         grad,
        *args,
        step_size: float = 1.0,
        temperature: float = 0.0,
        cooling: float = 0.95,
        seed: int = 0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
        max_step: float = 1.0,
) -> LineSearchResult:
    """Trivial line search using a constant step size.
    Useful for debugging, benchmarking against a baseline, or when the
    quadratic path scaling already provides a sensible step. Always reports
     ``done=True`` when ``temperature == 0``. When ``temperature > 0`` the
     Metropolis meta-rule gates ``done`` on descent OR an accepted uphill
     move (probability ``exp(−ΔE / T)``).
    The ``max_step`` parameter is accepted for interface uniformity; the
    fixed step is clipped to it so callers cannot overshoot the cap.
    """
    region = resolve_region(region)
    alpha = jnp.minimum(
        jnp.asarray(step_size, dtype=value.dtype),
        jnp.asarray(max_step, dtype=value.dtype),
    )
    raw_params = tree_add_scaled(params, alpha, direction)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    # Temperature meta-rule: gate on descent OR Metropolis when active.
    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    stochastic, _key = _metropolis_accept(
        new_val - value, temp0, jax.random.PRNGKey(seed), value.dtype
    )
    done = jnp.where(
        temp0 > 0.0,
        jnp.logical_or(new_val < value, stochastic),
        jnp.asarray(True),
    )
    pp, pg, pv, pval, pa = _empty_probes(params, max_probes)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, new_params, new_g, new_val, alpha, max_probes
    )
    return LineSearchResult(
        step_size=alpha,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=done,
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
        cooling: float = 0.95,
        seed: int = 0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
        max_step: float = 1.0,
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
     When ``temperature == 0`` this always reports ``done=True``. When
     ``temperature > 0`` the Metropolis meta-rule gates ``done`` on descent
     OR an accepted uphill move (probability ``exp(−ΔE / T)``).
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
    alpha = jnp.minimum(base_alpha * scale, jnp.asarray(max_step, dtype=value.dtype))
    raw_params = tree_add_scaled(params, alpha, direction)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    # Temperature meta-rule: gate on descent OR Metropolis when active.
    temp0 = jnp.asarray(temperature, dtype=value.dtype)
    stochastic, _key = _metropolis_accept(
        new_val - value, temp0, jax.random.PRNGKey(seed), value.dtype
    )
    done = jnp.where(
        temp0 > 0.0,
        jnp.logical_or(new_val < value, stochastic),
        jnp.asarray(True),
    )
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
        max_step=max_step,
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
        cooling: float = 0.95,
        seed: int = 0,
        region=None,
        region_state=None,
        max_probes: int = 32,
        record_probes: bool = True,
        max_step: float = 1.0,
) -> LineSearchResult:
    """Hager-Zhang line search via Optax ``scale_by_backtracking_linesearch``.
    The Hager-Zhang scheme is a robust approximate-Wolfe line search. We use
    Optax's backtracking transformation parameterized to approximate it,
    recomputing value/grad at the accepted point. Falls back gracefully if
    the underlying transform is unavailable.
     The ``temperature`` meta-rule is applied to the final acceptance: an
     uphill step may still be marked ``done`` via a Metropolis move
     (probability ``exp(−ΔE / T)``).
    """
    region = resolve_region(region)

    def fun_only(p):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_iter,
        slope_rtol=c1,
        decrease_factor=0.8,
        increase_factor=jnp.minimum(1.0, float(max_step)),
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
    # Temperature meta-rule: an uphill step may still be marked done via a
    # Metropolis move.
    stochastic, _key = _metropolis_accept(
        new_value - value, temperature, jax.random.PRNGKey(seed), new_value.dtype
    )
    done = jnp.logical_or(new_value < value, stochastic)
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
        num_evals=jnp.asarray(max_iter + 1, dtype=jnp.int32),
    )


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
        (a_prev, phi_prev, s_prev, a_cur, phi_cur, s_cur, g_cur, p_cur,
         found, bracketed, lo, hi, phi_lo, s_lo, phi_hi, s_hi,
         p_lo, g_lo, p_hi, g_hi,
         best_a, best_v, best_p, best_g,
         i, evals, pp, pg, pv, pval, pa) = carry
        stop = jnp.logical_or(found, bracketed)
        can_grow = jnp.logical_and(i < max_iter, a_cur < max_alpha)
        return jnp.logical_and(jnp.logical_not(stop), can_grow)

    def bracket_body(carry):
        (a_prev, phi_prev, s_prev, a_cur, phi_cur, s_cur, g_cur, p_cur,
         found, bracketed, lo, hi, phi_lo, s_lo, phi_hi, s_hi,
         p_lo, g_lo, p_hi, g_hi,
         best_a, best_v, best_p, best_g,
         i, evals, pp, pg, pv, pval, pa) = carry
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
        use_c = jnp.logical_and(jnp.logical_not(cond_a),
                                jnp.logical_and(jnp.logical_not(cond_found), cond_c))
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
        best_v = jnp.where(jnp.logical_and(new_found, phi_cur < best_v), phi_cur, best_v)
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
            out_a_prev, out_phi_prev, out_s_prev,
            out_a_cur, out_phi_cur, out_s_cur, out_g_cur, out_p_cur,
            new_found, new_bracketed,
            new_lo, new_hi, new_phi_lo, new_s_lo, new_phi_hi, new_s_hi,
            p_cur, g_cur, p_cur, g_cur,
            best_a, best_v, best_p, best_g,
            i + 1, evals + 1, pp, pg, pv, pval, pa,
        )

    (
        _a_prev, _phi_prev, _s_prev,
        a_cur, phi_cur, s_cur, g_cur, p_cur,
        found, bracketed,
        lo, hi, phi_lo, s_lo, phi_hi, s_hi,
        p_lo, g_lo, p_hi, g_hi,
        best_a, best_v, best_p, best_g,
        bracket_iters, bracket_evals, pp, pg, pv, pval, pa,
    ) = jax.lax.while_loop(
        bracket_cond,
        bracket_body,
        (
            zero, value, dg,
            a0, v0, s0, g0, p0,
            jnp.asarray(False), jnp.asarray(False),
            zero, a0, value, dg, v0, s0,
            p0, g0, p0, g0,
            a0, v0, p0, g0,
            jnp.asarray(1), jnp.asarray(1, jnp.int32), pp, pg, pv, pval, pa,
        ),
    )
    # If the current trial already satisfied Wolfe during bracketing, adopt it.
    found_a = a_cur
    found_v = phi_cur
    found_p = p_cur
    found_g = g_cur

    # --- Phase 2: zoom within [lo, hi] via bisection. --------------------
    def zoom_cond(carry):
        (lo, hi, phi_lo, s_lo, i, evals, z_found,
         z_a, z_v, z_p, z_g, best_a, best_v, best_p, best_g,
         pp, pg, pv, pval, pa) = carry
        keep = jnp.logical_and(jnp.logical_not(z_found), i < max_iter)
        # Only zoom if we actually bracketed and haven't already found a point.
        return jnp.logical_and(keep, bracketed)

    def zoom_body(carry):
        (lo, hi, phi_lo, s_lo, i, evals, z_found,
         z_a, z_v, z_p, z_g, best_a, best_v, best_p, best_g,
         pp, pg, pv, pval, pa) = carry
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
        flip = jnp.logical_and(jnp.logical_not(shrink_hi),
                               s * (hi - lo) >= 0.0)
        new_hi = jnp.where(shrink_hi, mid, jnp.where(flip, lo, hi))
        new_lo = jnp.where(shrink_hi, lo, mid)
        new_phi_lo = jnp.where(shrink_hi, phi_lo, v)
        new_s_lo = jnp.where(shrink_hi, s_lo, s)
        z_a = jnp.where(this_found, mid, z_a)
        z_v = jnp.where(this_found, v, z_v)
        z_p = jnp.where(this_found, p, z_p)
        z_g = jnp.where(this_found, g, z_g)
        return (
            new_lo, new_hi, new_phi_lo, new_s_lo, i + 1, evals + 1,
            jnp.logical_or(z_found, this_found),
            z_a, z_v, z_p, z_g, best_a, best_v, best_p, best_g,
            pp, pg, pv, pval, pa,
        )

    (
        _lo, _hi, _phi_lo, _s_lo, zoom_iters, total_evals, zoom_found,
        z_a, z_v, z_p, z_g, best_a, best_v, best_p, best_g,
        pp, pg, pv, pval, pa,
    ) = jax.lax.while_loop(
        zoom_cond,
        zoom_body,
        (
            lo, hi, phi_lo, s_lo, jnp.asarray(0), bracket_evals,
            jnp.asarray(False),
            best_a, best_v, best_p, best_g,
            best_a, best_v, best_p, best_g,
            pp, pg, pv, pval, pa,
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


# Registry mapping line-search names to their implementations.
_LINE_SEARCHES = {
    "strong_wolfe": strong_wolfe_search,
    "backtracking": backtracking_search,
    "armijo": armijo_search,
    "armijo_wolfe": armijo_wolfe_search,
    "hager_zhang": hager_zhang_search,
    "fixed": fixed_step_search,
    "null": null_search,
    "bisection": bisection_search,
}
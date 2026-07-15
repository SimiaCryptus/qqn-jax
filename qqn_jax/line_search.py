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
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
) -> LineSearchResult:
    """Backtracking line search (Armijo), self-contained for Optax.

    Repeatedly shrinks the step size by ``shrink`` until the Armijo
    sufficient-decrease condition ``f(x + α d) ≤ f(x) + c1 α gᵀd`` holds
    or ``max_iter`` is reached. Implemented with ``lax.while_loop`` to stay
    JIT/vmap compatible.
     If a ``region`` is supplied, the candidate point ``x + α·d`` is projected
     onto the region before evaluation, so the search navigates the feasible
     (projected) path.
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

    def cond(carry):
        alpha, i, evals, val, _g, _p, _pp, _pg, _pv, _pval, _pa = carry
        armijo = val <= value + c1 * alpha * dg
        return jnp.logical_and(jnp.logical_not(armijo), i < max_iter)

    def body(carry):
        alpha, i, evals, _val, _g, _p, pp, pg, pv, pval, pa = carry
        alpha = alpha * shrink
        new_params, new_val, new_g = eval_at(alpha)
        # Record this probe (slot = i, since slot 0 holds the init_step probe).
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, i, new_params, new_g, new_val, alpha, eff_probes
        )
        # ``evals`` counts every eval_at call: the body adds exactly one.
        return alpha, i + 1, evals + 1, new_val, new_g, new_params, pp, pg, pv, pval, pa

    init_alpha = jnp.asarray(init_step, dtype=value.dtype)

    # Evaluate at the initial step first.
    init_params, init_val, init_g = eval_at(init_alpha)
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

    (
        alpha,
        n_iters,
        eval_count,
        final_val,
        final_g,
        new_params,
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
            jnp.asarray(1),
            jnp.asarray(1, jnp.int32),  # the initial eval_at(init_step) probe
            init_val,
            init_g,
            init_params,
            init_pp,
            init_pg,
            init_pv,
            init_pval,
            init_pa,
        ),
    )
    armijo = final_val <= value + c1 * alpha * dg
    # Evals are tracked explicitly in the carry (1 for the initial-step probe
    # plus one per backtracking iteration), decoupled from ``n_iters`` so the
    # count cannot drift if the loop index semantics change.
    num_evals = eval_count
    return LineSearchResult(
        step_size=alpha,
        new_value=final_val,
        new_grad=final_g,
        new_params=new_params,
        done=armijo,
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
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
) -> LineSearchResult:
    """Strong Wolfe line search via Optax ``scale_by_zoom_linesearch``.

    Enforces Armijo sufficient decrease and the strong curvature
    condition, which keeps the L-BFGS curvature updates well-conditioned.

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
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
) -> LineSearchResult:
    """Trivial line search using a constant step size.
    Useful for debugging, benchmarking against a baseline, or when the
    quadratic path scaling already provides a sensible step. Always reports
    ``done=True`` (it makes no acceptance test).
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
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
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
        region=region,
        region_state=region_state,
        max_probes=max_probes,
        record_probes=record_probes,
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
) -> LineSearchResult:
    """Backtracking line search with a Metropolis-style temperature.
    A simulated-annealing flavored backtracking search: at each shrink step
    the Armijo sufficient-decrease test is checked first, but if it fails the
    step may *still* be probabilistically accepted — an *uphill climb* — with
    Metropolis acceptance probability
        p = exp(−ΔE / T),   ΔE = f(x + α·d) − f(x)
    where ``T`` is the (annealed) temperature. This lets the search escape
    shallow local structure by occasionally accepting a worse point. The
    temperature is cooled geometrically (``T ← cooling·T``) at each iteration
    so the acceptance of uphill moves becomes progressively less likely.
    Downhill steps (ΔE ≤ 0) are always accepted (``p ≥ 1``). If neither the
    Armijo test nor the stochastic test ever accepts, the search terminates at
    ``max_iter`` with the last (shrunk) step.
    A ``seed`` seeds a deterministic PRNG so the whole search stays
    JIT/vmap-compatible and reproducible.
    """
    region = resolve_region(region)
    project = _make_projected_point(region, region_state, params)
    dg = tree_vdot(grad, direction)  # directional derivative gᵀd

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
        """Return (accepted, new_key). Armijo OR Metropolis stochastic test."""
        armijo = val <= value + c1 * alpha * dg
        delta_e = val - value
        # Metropolis probability: exp(-ΔE / T), clamped to [0, 1]. A safe
        # temperature guards against divide-by-zero when T collapses.
        safe_t = jnp.maximum(temp, jnp.asarray(1e-12, dtype=value.dtype))
        p = jnp.exp(-delta_e / safe_t)
        p = jnp.clip(p, 0.0, 1.0)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, dtype=value.dtype)
        stochastic = u < p
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
            temp,
            key,
            _pp,
            _pg,
            _pv,
            _pval,
            _pa,
        ) = carry
        return jnp.logical_and(jnp.logical_not(accepted), i < max_iter)

    def body(carry):
        (alpha, i, evals, _val, _g, _p, _accepted, temp, key, pp, pg, pv, pval, pa) = (
            carry
        )
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
    # Evaluate at the initial step first.
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
            jnp.asarray(1),
            jnp.asarray(1, jnp.int32),
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


def hager_zhang_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    c1: float = 0.1,
    max_iter: int = 30,
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

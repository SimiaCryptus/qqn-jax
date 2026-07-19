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

from typing import NamedTuple, Any

import jax
import jax.numpy as jnp

from qqn_jax import backtracking_search
from qqn_jax.line_search.armijo import armijo_search
from qqn_jax.line_search.armijo_wolfe import armijo_wolfe_search
from qqn_jax.line_search.bisection import bisection_search
from qqn_jax.line_search.fixed_step import fixed_step_search
from qqn_jax.line_search.hager_zhang import hager_zhang_search
from qqn_jax.line_search.null_search import null_search
from qqn_jax.line_search.strong_wolfe import strong_wolfe_search


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

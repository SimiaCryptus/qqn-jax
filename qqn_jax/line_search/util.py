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

import jax
import jax.numpy as jnp


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


from qqn_jax.utils import tree_scalar_dtype


def _empty_probes(params, max_probes):
    """Allocate empty probe buffers shaped for ``params`` (flat or pytree)."""
    scalar_dtype = tree_scalar_dtype(params)
    probe_params = jax.tree_util.tree_map(
        lambda x: jnp.zeros((max_probes, *x.shape), dtype=x.dtype), params
    )
    probe_grads = jax.tree_util.tree_map(
        lambda x: jnp.zeros((max_probes, *x.shape), dtype=x.dtype), params
    )
    return (
        probe_params,
        probe_grads,
        jnp.zeros((max_probes,), dtype=bool),
        jnp.full((max_probes,), jnp.inf, dtype=scalar_dtype),
        jnp.zeros((max_probes,), dtype=scalar_dtype),
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
    """Write ``(p, g)`` into ``slot`` of the probe buffers (JIT-safe, pytree-safe).

    When ``probe_params`` is ``None`` (probe recording disabled), this is a
    zero-overhead no-op.
    """
    if probe_params is None:
        return probe_params, probe_grads, probe_valid, probe_values, probe_alphas
    in_range = jnp.logical_and(slot >= 0, slot < max_probes)
    idx = jnp.clip(slot, 0, max_probes - 1)
    new_params = jax.tree_util.tree_map(
        lambda buf, leaf: jnp.where(in_range, buf.at[idx].set(leaf), buf),
        probe_params,
        p,
    )
    new_grads = jax.tree_util.tree_map(
        lambda buf, leaf: jnp.where(in_range, buf.at[idx].set(leaf), buf),
        probe_grads,
        g,
    )
    new_valid = jnp.where(in_range, probe_valid.at[idx].set(True), probe_valid)
    new_values = jnp.where(
        in_range,
        probe_values.at[idx].set(jnp.asarray(v, dtype=probe_values.dtype)),
        probe_values,
    )
    new_alphas = jnp.where(
        in_range,
        probe_alphas.at[idx].set(jnp.asarray(a, dtype=probe_alphas.dtype)),
        probe_alphas,
    )
    return new_params, new_grads, new_valid, new_values, new_alphas

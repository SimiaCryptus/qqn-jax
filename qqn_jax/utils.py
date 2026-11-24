"""Shared utilities for QQN-JAX."""

from typing import Callable

import jax
import jax.numpy as jnp


def tree_vdot(a, b, accum_dtype=None):
    """Inner product over (possibly) pytree-structured arrays.

    For flat arrays this is just ``jnp.vdot``. When leaves are low-precision
    (e.g. ``bfloat16`` or ``float16``), accumulation defaults to ``float32``
    to prevent underflow and summation drift on large parameter trees.
    """
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if not leaves_a:
        return jnp.asarray(0.0, dtype=accum_dtype or jnp.float32)

    def _leaf_dot(x, y):
        if accum_dtype is not None:
            return jnp.vdot(x.astype(accum_dtype), y.astype(accum_dtype))
        if x.dtype in (jnp.bfloat16, jnp.float16) or y.dtype in (
            jnp.bfloat16,
            jnp.float16,
        ):
            return jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
        return jnp.vdot(x, y)

    total = _leaf_dot(leaves_a[0], leaves_b[0])
    for x, y in zip(leaves_a[1:], leaves_b[1:]):
        total = total + _leaf_dot(x, y)
    return total


def tree_all_finite(tree):
    """Return a scalar boolean indicating whether all leaves are finite."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    acc = jnp.all(jnp.isfinite(leaves[0]))
    for x in leaves[1:]:
        acc = jnp.logical_and(acc, jnp.all(jnp.isfinite(x)))
    return acc


def tree_where(cond, a, b):
    """Elementwise ``jnp.where(cond, a, b)`` mapped across pytrees."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def tree_dtype(tree, default=jnp.float32):
    """Return the dtype of the first leaf in ``tree`` (or ``default``)."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return default
    return leaves[0].dtype


def tree_scalar_dtype(tree, default=jnp.float32):
    """Return a numerically safe scalar dtype (promoting bf16/fp16 to fp32)."""
    dt = tree_dtype(tree, default=default)
    if dt in (jnp.bfloat16, jnp.float16):
        return jnp.float32
    return dt


def tree_add_scaled(tree, scale, other):
    """Compute ``tree + scale * other`` over pytrees, preserving ``tree`` leaf dtypes."""
    return jax.tree_util.tree_map(
        lambda t, o: (t + jnp.asarray(scale, dtype=t.dtype) * o.astype(t.dtype)).astype(
            t.dtype
        ),
        tree,
        other,
    )


def tree_scale(scale, tree):
    """Compute ``scale * tree`` over pytrees, preserving leaf dtypes."""
    return jax.tree_util.tree_map(
        lambda t: (jnp.asarray(scale, dtype=t.dtype) * t).astype(t.dtype), tree
    )


def tree_negative(tree):
    """Compute ``-tree`` over pytrees."""
    return jax.tree_util.tree_map(lambda t: -t, tree)


def tree_l2_norm(tree):
    """L2 norm over a pytree."""
    return jnp.sqrt(tree_vdot(tree, tree))


def make_value_and_grad(fun: Callable, has_aux: bool = False) -> Callable:
    """Build a value-and-grad function, transparently handling ``has_aux``."""
    return jax.value_and_grad(fun, has_aux=has_aux)

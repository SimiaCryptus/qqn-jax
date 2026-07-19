"""Shared utilities for QQN-JAX."""

from typing import Callable

import jax
import jax.numpy as jnp


def tree_vdot(a, b):
    """Inner product over (possibly) pytree-structured arrays.

    For flat arrays this is just ``jnp.vdot``.
    """
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return sum(jnp.vdot(x, y) for x, y in zip(leaves_a, leaves_b))


def tree_add_scaled(tree, scale, other):
    """Compute ``tree + scale * other`` over pytrees."""
    return jax.tree_util.tree_map(lambda t, o: t + scale * o, tree, other)


def tree_scale(scale, tree):
    """Compute ``scale * tree`` over pytrees."""
    return jax.tree_util.tree_map(lambda t: scale * t, tree)


def tree_negative(tree):
    """Compute ``-tree`` over pytrees."""
    return jax.tree_util.tree_map(lambda t: -t, tree)


def tree_l2_norm(tree):
    """L2 norm over a pytree."""
    return jnp.sqrt(tree_vdot(tree, tree))


def make_value_and_grad(fun: Callable, has_aux: bool = False) -> Callable:
    """Build a value-and-grad function, transparently handling ``has_aux``."""
    return jax.value_and_grad(fun, has_aux=has_aux)

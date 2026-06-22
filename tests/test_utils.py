"""Tests for qqn_jax.utils."""

import jax.numpy as jnp
import numpy as np

from qqn_jax.utils import (
    quadratic_path,
    quadratic_path_derivative,
    tree_add_scaled,
    tree_l2_norm,
    tree_negative,
    tree_scale,
    tree_vdot,
)


def test_quadratic_path_endpoints():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])

    # t = 0 -> zero (both terms vanish).
    d0 = quadratic_path(0.0, grad_dir, qn_dir)
    np.testing.assert_allclose(d0, jnp.zeros(3), atol=1e-7)

    # t = 1 -> pure L-BFGS direction.
    d1 = quadratic_path(1.0, grad_dir, qn_dir)
    np.testing.assert_allclose(d1, qn_dir, atol=1e-7)


def test_quadratic_path_derivative_at_zero():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])
    # d'(0) = (-∇f), i.e. grad_dir.
    dprime = quadratic_path_derivative(0.0, grad_dir, qn_dir)
    np.testing.assert_allclose(dprime, grad_dir, atol=1e-7)


def test_tree_vdot_and_norm():
    a = jnp.array([3.0, 4.0])
    assert float(tree_vdot(a, a)) == 25.0
    assert float(tree_l2_norm(a)) == 5.0


def test_quadratic_path_midpoint():
    grad_dir = jnp.array([1.0, 0.0])
    qn_dir = jnp.array([0.0, 1.0])
    # d(0.5) = 0.25*grad_dir + 0.25*qn_dir.
    d = quadratic_path(0.5, grad_dir, qn_dir)
    np.testing.assert_allclose(d, jnp.array([0.25, 0.25]), atol=1e-7)


def test_quadratic_path_derivative_at_one():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])
    # d'(1) = (1-2)*grad_dir + 2*qn_dir = -grad_dir + 2*qn_dir.
    dprime = quadratic_path_derivative(1.0, grad_dir, qn_dir)
    np.testing.assert_allclose(dprime, -grad_dir + 2.0 * qn_dir, atol=1e-7)


def test_tree_add_scaled():
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    out = tree_add_scaled(a, 2.0, b)
    np.testing.assert_allclose(out, jnp.array([7.0, 10.0]))


def test_tree_scale_and_negative():
    a = jnp.array([1.0, -2.0])
    np.testing.assert_allclose(tree_scale(3.0, a), jnp.array([3.0, -6.0]))
    np.testing.assert_allclose(tree_negative(a), jnp.array([-1.0, 2.0]))


def test_tree_ops_on_pytrees():
    a = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
    b = {"w": jnp.array([1.0, 1.0]), "b": jnp.array([1.0])}
    # vdot over a pytree sums the leaf vdots.
    assert float(tree_vdot(a, b)) == float(1 + 2 + 3)
    out = tree_add_scaled(a, 1.0, b)
    np.testing.assert_allclose(out["w"], jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(out["b"], jnp.array([4.0]))

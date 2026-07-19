"""Tests for qqn_jax.utils."""

import jax

import jax.numpy as jnp
import numpy as np

from qqn_jax.paths.quadratic import _quadratic_path, quadratic_path_derivative
from qqn_jax.utils import (
    make_value_and_grad,
    tree_add_scaled,
    tree_l2_norm,
    tree_negative,
    tree_scale,
    tree_vdot,
)


def test_quadratic_path_endpoints():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])

    d0 = _quadratic_path(0.0, grad_dir, qn_dir)
    np.testing.assert_allclose(d0, jnp.zeros(3), atol=1e-7)

    d1 = _quadratic_path(1.0, grad_dir, qn_dir)
    np.testing.assert_allclose(d1, qn_dir, atol=1e-7)


def test_quadratic_path_derivative_at_zero():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])

    dprime = quadratic_path_derivative(0.0, grad_dir, qn_dir)
    np.testing.assert_allclose(dprime, grad_dir, atol=1e-7)


def test_tree_vdot_and_norm():
    a = jnp.array([3.0, 4.0])
    assert float(tree_vdot(a, a)) == 25.0
    assert float(tree_l2_norm(a)) == 5.0


def test_quadratic_path_midpoint():
    grad_dir = jnp.array([1.0, 0.0])
    qn_dir = jnp.array([0.0, 1.0])

    d = _quadratic_path(0.5, grad_dir, qn_dir)
    np.testing.assert_allclose(d, jnp.array([0.25, 0.25]), atol=1e-7)


def test_quadratic_path_derivative_at_one():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])

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

    assert float(tree_vdot(a, b)) == float(1 + 2 + 3)
    out = tree_add_scaled(a, 1.0, b)
    np.testing.assert_allclose(out["w"], jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(out["b"], jnp.array([4.0]))


def test_quadratic_path_equals_derivative_integral_endpoints():

    grad_dir = jnp.array([1.0, -2.0, 0.5])
    qn_dir = jnp.array([-0.5, 1.0, 2.0])
    eps = 1e-4
    for t in (0.1, 0.4, 0.9):
        d_plus = _quadratic_path(t + eps, grad_dir, qn_dir)
        d_minus = _quadratic_path(t - eps, grad_dir, qn_dir)
        numeric = (d_plus - d_minus) / (2 * eps)
        analytic = quadratic_path_derivative(t, grad_dir, qn_dir)
        np.testing.assert_allclose(numeric, analytic, atol=1e-3)


def test_tree_l2_norm_matches_flat_norm():
    tree = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([12.0])}

    np.testing.assert_allclose(float(tree_l2_norm(tree)), 13.0, atol=1e-6)


def test_tree_scale_on_pytrees():
    tree = {"a": jnp.array([1.0, -2.0]), "b": jnp.array([3.0])}
    out = tree_scale(-2.0, tree)
    np.testing.assert_allclose(out["a"], jnp.array([-2.0, 4.0]))
    np.testing.assert_allclose(out["b"], jnp.array([-6.0]))


def test_tree_negative_is_involution():
    tree = {"a": jnp.array([1.0, -2.0]), "b": jnp.array([3.0])}
    out = tree_negative(tree_negative(tree))
    np.testing.assert_allclose(out["a"], tree["a"])
    np.testing.assert_allclose(out["b"], tree["b"])


def test_make_value_and_grad_basic():
    def f(x):
        return jnp.sum(x**2)

    vg = make_value_and_grad(f)
    x = jnp.array([1.0, 2.0, 3.0])
    value, grad = vg(x)
    np.testing.assert_allclose(float(value), 14.0, atol=1e-6)
    np.testing.assert_allclose(grad, 2.0 * x, atol=1e-6)


def test_make_value_and_grad_has_aux():
    def f(x):
        return jnp.sum(x**2), {"norm": jnp.linalg.norm(x)}

    vg = make_value_and_grad(f, has_aux=True)
    x = jnp.array([3.0, 4.0])
    (value, aux), grad = vg(x)
    np.testing.assert_allclose(float(value), 25.0, atol=1e-6)
    np.testing.assert_allclose(float(aux["norm"]), 5.0, atol=1e-6)
    np.testing.assert_allclose(grad, 2.0 * x, atol=1e-6)


def test_quadratic_path_is_jittable():
    grad_dir = jnp.array([1.0, 2.0])
    qn_dir = jnp.array([3.0, 4.0])
    fn = jax.jit(_quadratic_path)
    out = fn(0.5, grad_dir, qn_dir)
    np.testing.assert_allclose(out, jnp.array([1.0, 1.5]), atol=1e-6)


def test_tree_vdot_symmetry():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(
        float(tree_vdot(a, b)), float(tree_vdot(b, a)), atol=1e-6
    )

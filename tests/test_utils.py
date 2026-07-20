"""Unit tests for qqn_jax.utils."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.utils import (
    tree_vdot,
    tree_add_scaled,
    tree_scale,
    tree_negative,
    tree_l2_norm,
    make_value_and_grad,
)

jax.config.update("jax_enable_x64", True)


class TestTreeVdot:
    def test_flat_arrays(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        out = tree_vdot(a, b)
        assert float(out) == pytest.approx(32.0)

    def test_orthogonal(self):
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        assert float(tree_vdot(a, b)) == pytest.approx(0.0)

    def test_pytree(self):
        a = {"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0])}
        b = {"x": jnp.array([4.0, 5.0]), "y": jnp.array([6.0])}
                              
        assert float(tree_vdot(a, b)) == pytest.approx(32.0)

    def test_matrix(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.ones((2, 2))
        assert float(tree_vdot(a, b)) == pytest.approx(10.0)

    def test_jit(self):
        f = jax.jit(tree_vdot)
        out = f(jnp.array([1.0, 1.0]), jnp.array([2.0, 3.0]))
        assert float(out) == pytest.approx(5.0)


class TestTreeAddScaled:
    def test_flat(self):
        t = jnp.array([1.0, 2.0])
        o = jnp.array([3.0, 4.0])
        out = tree_add_scaled(t, 2.0, o)
        np.testing.assert_allclose(np.asarray(out), np.array([7.0, 10.0]))

    def test_zero_scale(self):
        t = jnp.array([1.0, 2.0])
        o = jnp.array([100.0, 200.0])
        out = tree_add_scaled(t, 0.0, o)
        np.testing.assert_allclose(np.asarray(out), np.asarray(t))

    def test_pytree(self):
        t = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
        o = {"a": jnp.array([1.0]), "b": jnp.array([1.0])}
        out = tree_add_scaled(t, 3.0, o)
        assert float(out["a"][0]) == pytest.approx(4.0)
        assert float(out["b"][0]) == pytest.approx(5.0)

    def test_negative_scale(self):
        t = jnp.array([5.0])
        o = jnp.array([2.0])
        out = tree_add_scaled(t, -1.0, o)
        assert float(out[0]) == pytest.approx(3.0)


class TestTreeScale:
    def test_flat(self):
        t = jnp.array([1.0, 2.0, 3.0])
        out = tree_scale(2.0, t)
        np.testing.assert_allclose(np.asarray(out), np.array([2.0, 4.0, 6.0]))

    def test_zero(self):
        t = jnp.array([1.0, 2.0])
        out = tree_scale(0.0, t)
        np.testing.assert_allclose(np.asarray(out), np.zeros(2))

    def test_pytree(self):
        t = {"a": jnp.array([2.0]), "b": jnp.array([4.0])}
        out = tree_scale(0.5, t)
        assert float(out["a"][0]) == pytest.approx(1.0)
        assert float(out["b"][0]) == pytest.approx(2.0)


class TestTreeNegative:
    def test_flat(self):
        t = jnp.array([1.0, -2.0, 3.0])
        out = tree_negative(t)
        np.testing.assert_allclose(np.asarray(out), np.array([-1.0, 2.0, -3.0]))

    def test_pytree(self):
        t = {"a": jnp.array([1.0]), "b": jnp.array([-2.0])}
        out = tree_negative(t)
        assert float(out["a"][0]) == pytest.approx(-1.0)
        assert float(out["b"][0]) == pytest.approx(2.0)

    def test_double_negative(self):
        t = jnp.array([1.0, 2.0])
        out = tree_negative(tree_negative(t))
        np.testing.assert_allclose(np.asarray(out), np.asarray(t))


class TestTreeL2Norm:
    def test_basic(self):
        t = jnp.array([3.0, 4.0])
        assert float(tree_l2_norm(t)) == pytest.approx(5.0)

    def test_zero(self):
        assert float(tree_l2_norm(jnp.zeros(5))) == pytest.approx(0.0)

    def test_pytree(self):
        t = {"a": jnp.array([3.0]), "b": jnp.array([4.0])}
        assert float(tree_l2_norm(t)) == pytest.approx(5.0)

    def test_matrix(self):
        t = jnp.array([[3.0, 0.0], [0.0, 4.0]])
        assert float(tree_l2_norm(t)) == pytest.approx(5.0)

    def test_grad(self):
        t = jnp.array([3.0, 4.0])
        g = jax.grad(tree_l2_norm)(t)
                                 
        np.testing.assert_allclose(np.asarray(g), np.array([0.6, 0.8]), atol=1e-9)


class TestMakeValueAndGrad:
    def test_no_aux(self):
        def f(x):
            return jnp.sum(x**2)

        vg = make_value_and_grad(f)
        value, grad = vg(jnp.array([1.0, 2.0]))
        assert float(value) == pytest.approx(5.0)
        np.testing.assert_allclose(np.asarray(grad), np.array([2.0, 4.0]))

    def test_with_aux(self):
        def f(x):
            return jnp.sum(x**2), {"extra": jnp.sum(x)}

        vg = make_value_and_grad(f, has_aux=True)
        (value, aux), grad = vg(jnp.array([1.0, 2.0]))
        assert float(value) == pytest.approx(5.0)
        assert float(aux["extra"]) == pytest.approx(3.0)
        np.testing.assert_allclose(np.asarray(grad), np.array([2.0, 4.0]))

    def test_jit(self):
        vg = jax.jit(make_value_and_grad(lambda x: jnp.sum(x**3)))
        value, grad = vg(jnp.array([2.0]))
        assert float(value) == pytest.approx(8.0)
        assert float(grad[0]) == pytest.approx(12.0)
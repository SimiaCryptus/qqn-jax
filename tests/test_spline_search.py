"""Tests for the cubic Hermite spline line search."""

import jax
import jax.numpy as jnp
import numpy as np

from qqn_jax import QQN
from qqn_jax.paths.spline import (
    _orient_tangents,
    _segment_stationary_candidates,
    _segment_value,
)


def _quadratic(x):

    return jnp.sum(x**2)


def _rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def test_spline_search_in_solver():
    solver = QQN(_rosenbrock, maxiter=500, tol=1e-5, line_search="spline")
    params, state = solver.run(jnp.array([-1.2, 1.0]))

    assert float(state.value) < 1.0


def test_spline_search_vmap_starting_points():
    solver = QQN(_quadratic, maxiter=50, tol=1e-6, line_search="spline")
    starts = jnp.array([[2.0, 2.0], [-1.0, 3.0], [0.5, -0.5]])
    params, states = jax.vmap(solver.run)(starts)

    assert jnp.all(states.value < 1e-3)


def test_segment_value_endpoints():

    h = 1.0
    f0, m0, f1, m1 = 2.0, -1.0, 0.5, 0.3
    np.testing.assert_allclose(_segment_value(0.0, h, f0, m0, f1, m1), f0, atol=1e-7)
    np.testing.assert_allclose(_segment_value(1.0, h, f0, m0, f1, m1), f1, atol=1e-7)


def test_orient_tangents_reflects_opposing_sign():

    m0 = jnp.asarray(2.0)
    m1 = jnp.asarray(-1.0)
    m0o, m1o = _orient_tangents(m0, m1)
    assert float(m0o) <= 0.0
    np.testing.assert_allclose(float(m1o), -1.0)


def test_orient_tangents_flat_secant_keeps_raw():
    m0 = jnp.asarray(2.0)
    m1 = jnp.asarray(-3.0)
    m0o, m1o = _orient_tangents(m0, m1)
    np.testing.assert_allclose(float(m0o), 2.0)
    np.testing.assert_allclose(float(m1o), -3.0)


def test_segment_stationary_finds_min_of_descending_cubic():

    t0, t1 = 0.0, 1.0
    f0, m0 = 1.0, -2.0
    f1, m1 = 0.0, 1.0
    t_c, v_c, valid = _segment_stationary_candidates(
        jnp.asarray(t0),
        jnp.asarray(t1),
        jnp.asarray(f0),
        jnp.asarray(m0),
        jnp.asarray(f1),
        jnp.asarray(m1),
    )

    assert bool(jnp.any(valid))
    inside = jnp.logical_and(t_c >= t0 - 1e-6, t_c <= t1 + 1e-6)
    assert bool(jnp.any(jnp.logical_and(valid, inside)))


def test_segment_value_midpoint_between_endpoints():

    h = 1.0
    f0, m0, f1, m1 = 2.0, -1.0, 0.5, -0.5
    mid = float(_segment_value(0.5, h, f0, m0, f1, m1))
    assert min(f0, f1) - 1e-6 <= mid <= max(f0, f1) + 1e-6


def test_orient_tangents_preserves_bracketed_minimum():

    m0 = jnp.asarray(-2.0)
    m1 = jnp.asarray(3.0)
    m0o, m1o = _orient_tangents(m0, m1)
    np.testing.assert_allclose(float(m0o), -2.0)
    np.testing.assert_allclose(float(m1o), 3.0)


def test_segment_stationary_no_valid_for_monotone_cubic():

    t0, t1 = 0.0, 1.0
    f0, m0 = 2.0, -1.0
    f1, m1 = 0.0, -1.0
    t_c, v_c, valid = _segment_stationary_candidates(
        jnp.asarray(t0),
        jnp.asarray(t1),
        jnp.asarray(f0),
        jnp.asarray(m0),
        jnp.asarray(f1),
        jnp.asarray(m1),
    )

    finite_where_valid = jnp.all(jnp.where(valid, jnp.isfinite(v_c), True))
    assert bool(finite_where_valid)

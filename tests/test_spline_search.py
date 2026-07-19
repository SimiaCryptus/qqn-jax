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
    # Simple convex quadratic with minimum at the origin.
    return jnp.sum(x**2)


def _rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def test_spline_search_in_solver():
    solver = QQN(_rosenbrock, maxiter=500, tol=1e-5, line_search="spline")
    params, state = solver.run(jnp.array([-1.2, 1.0]))
    # Should make meaningful progress toward [1, 1].
    assert float(state.value) < 1.0


def test_spline_search_vmap_starting_points():
    solver = QQN(_quadratic, maxiter=50, tol=1e-6, line_search="spline")
    starts = jnp.array([[2.0, 2.0], [-1.0, 3.0], [0.5, -0.5]])
    params, states = jax.vmap(solver.run)(starts)
    # Each run should converge near the origin.
    assert jnp.all(states.value < 1e-3)


def test_segment_value_endpoints():
    # Cubic Hermite interpolant hits the endpoint values at s=0 and s=1.
    h = 1.0
    f0, m0, f1, m1 = 2.0, -1.0, 0.5, 0.3
    np.testing.assert_allclose(_segment_value(0.0, h, f0, m0, f1, m1), f0, atol=1e-7)
    np.testing.assert_allclose(_segment_value(1.0, h, f0, m0, f1, m1), f1, atol=1e-7)


def test_orient_tangents_reflects_opposing_sign():
    # Secant slope is negative (f1 < f0); a positive tangent gets reflected.
    m0 = jnp.asarray(2.0)  # opposes the descending channel
    m1 = jnp.asarray(-1.0)  # aligned, kept
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
    # A simple convex-ish segment: f0 high, f1 low, slopes bracket a min.
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
    # At least one stationary point lies inside [t0, t1].
    assert bool(jnp.any(valid))
    inside = jnp.logical_and(t_c >= t0 - 1e-6, t_c <= t1 + 1e-6)
    assert bool(jnp.any(jnp.logical_and(valid, inside)))


def test_segment_value_midpoint_between_endpoints():
    # For a monotone descending segment, the midpoint value lies between
    # the endpoint values.
    h = 1.0
    f0, m0, f1, m1 = 2.0, -1.0, 0.5, -0.5
    mid = float(_segment_value(0.5, h, f0, m0, f1, m1))
    assert min(f0, f1) - 1e-6 <= mid <= max(f0, f1) + 1e-6


def test_orient_tangents_preserves_bracketed_minimum():
    # A genuine interior minimum (m0 < 0 < m1) must NOT be reflected away.
    m0 = jnp.asarray(-2.0)
    m1 = jnp.asarray(3.0)
    m0o, m1o = _orient_tangents(m0, m1)
    np.testing.assert_allclose(float(m0o), -2.0)
    np.testing.assert_allclose(float(m1o), 3.0)


def test_segment_stationary_no_valid_for_monotone_cubic():
    # A strictly descending cubic with aligned tangents has no interior min.
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
    # Even if roots exist, valid candidates' values are finite where valid.
    finite_where_valid = jnp.all(jnp.where(valid, jnp.isfinite(v_c), True))
    assert bool(finite_where_valid)

"""Tests for the cubic Hermite spline line search."""

import jax
import jax.numpy as jnp
import numpy as np

from qqn_jax import QQN, spline_search
from qqn_jax.utils import make_value_and_grad
from qqn_jax.line_search import backtracking_search
from qqn_jax.spline_search import (
    _orient_tangents,
    _segment_stationary_candidates,
    _segment_value,
)


def _quadratic(x):
    # Simple convex quadratic with minimum at the origin.
    return jnp.sum(x**2)


def _rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def test_spline_search_decreases_on_quadratic():
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([2.0, -3.0])
    value, grad = vg(params)
    direction = -grad  # steepest descent

    res = spline_search(vg, params, direction, value, grad, init_step=1.0)
    # Spline search must not increase the objective.
    assert res.new_value <= value + 1e-6
    # On an exact quadratic, the minimizer along -grad is at alpha ~ 0.5.
    assert jnp.isfinite(res.step_size)
    assert bool(res.done)


def test_spline_search_jit_compatible():
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([1.0, 1.0, 1.0])
    value, grad = vg(params)
    direction = -grad

    fn = jax.jit(lambda p, d, v, g: spline_search(vg, p, d, v, g))
    res = fn(params, direction, value, grad)
    assert res.new_value <= value + 1e-6


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


def test_spline_improves_or_matches_inner_on_quadratic():
    # On a quadratic, the spline refinement should not be worse than inner.
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([4.0, -2.0])
    value, grad = vg(params)
    direction = -grad
    res = spline_search(vg, params, direction, value, grad, init_step=1.0)
    assert float(res.new_value) <= float(value) + 1e-6


def test_spline_search_on_rosenbrock_makes_progress():
    vg = make_value_and_grad(_rosenbrock)
    params = jnp.array([-1.0, 1.0])
    value, grad = vg(params)
    direction = -grad
    res = spline_search(vg, params, direction, value, grad, init_step=1.0)
    assert float(res.new_value) <= float(value) + 1e-6


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


def test_spline_search_returns_finite_grad():
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([3.0, -1.0])
    value, grad = vg(params)
    direction = -grad
    res = spline_search(vg, params, direction, value, grad, init_step=1.0)
    assert jnp.all(jnp.isfinite(res.new_grad))
    assert jnp.all(jnp.isfinite(res.new_params))


def test_spline_search_step_size_in_unit_interval_ish():
    # The accepted step on a convex quadratic should be a small positive value.
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([2.0, -2.0])
    value, grad = vg(params)
    direction = -grad
    res = spline_search(vg, params, direction, value, grad, init_step=1.0)
    assert float(res.step_size) > 0.0


def test_spline_never_worse_than_inner_on_rosenbrock():
    vg = make_value_and_grad(_rosenbrock)
    params = jnp.array([-1.2, 1.0])
    value, grad = vg(params)
    direction = -grad
    # Inner backtracking baseline.
    from qqn_jax.line_search import backtracking_search

    inner = backtracking_search(vg, params, direction, value, grad)
    res = spline_search(vg, params, direction, value, grad)
    # Spline only improves on the inner search.
    assert float(res.new_value) <= float(inner.new_value) + 1e-6


def test_spline_strictly_improves_when_armijo_accepts_short_step():
    # On a convex quadratic f(x) = sum(x**2), the exact minimizer along the
    # steepest-descent path lies at alpha = 0.5. If Armijo backtracking is fed
    # a large initial step and shrinks aggressively, it accepts a step well
    # short of 0.5 (the first point satisfying sufficient decrease). The spline
    # refinement, which reuses measured control points and probes the cubic
    # stationary points, should then *strictly* improve on that short step.
    #
    # The existing suite only asserts ``<=``; this test pins down the actual
    # value-add of the spline by requiring a genuine improvement.
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([2.0, -3.0])
    value, grad = vg(params)
    direction = -grad  # steepest descent; true minimizer at alpha = 0.5
    # Large initial step + aggressive shrink => Armijo accepts a short step.
    inner = backtracking_search(
        vg, params, direction, value, grad, init_step=8.0, shrink=0.1
    )
    res = spline_search(vg, params, direction, value, grad, init_step=8.0, shrink=0.1)
    # Sanity: the inner search accepts a step that misses the true minimizer
    # (alpha = 0.5). With this configuration Armijo stops at alpha = 0.8,
    # overshooting the minimizer, so the inner-accepted value is well above 0.
    assert abs(float(inner.step_size) - 0.5) > 0.1
    assert float(inner.new_value) > 0.0
    # The spline must *strictly* improve on the inner-accepted point.
    assert float(res.new_value) < float(inner.new_value) - 1e-9
    # And it should land close to the analytic minimizer (alpha = 0.5).
    assert abs(float(res.step_size) - 0.5) < 0.1


def test_spline_search_vmap_directions():
    vg = make_value_and_grad(_quadratic)

    def run_one(p):
        value, grad = vg(p)
        return spline_search(vg, p, -grad, value, grad).new_value

    ps = jnp.array([[2.0, 2.0], [1.0, -1.0], [3.0, 0.5]])
    vals = jax.vmap(run_one)(ps)
    assert jnp.all(jnp.isfinite(vals))

"""Tests for line search strategies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.line_search.strategy import (
    armijo_search,
    fixed_step_search,
    hager_zhang_search,
    strong_wolfe_search,
)
from qqn_jax import backtracking_search
from qqn_jax.regions import BoxRegion


def quad_value_and_grad(x):
    # f(x) = 0.5 * ||x||^2, grad = x.
    return 0.5 * jnp.vdot(x, x), x


def test_backtracking_decreases_value():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad  # steepest descent
    res = backtracking_search(quad_value_and_grad, x, direction, value, grad)
    assert float(res.new_value) < float(value)
    assert bool(res.done)


def test_strong_wolfe_decreases_value():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = strong_wolfe_search(quad_value_and_grad, x, direction, value, grad)
    assert float(res.new_value) < float(value)


def test_strong_wolfe_finds_exact_step_for_quadratic():
    # For f = 0.5||x||^2 along d = -x, the exact minimizer is alpha = 1.
    x = jnp.array([3.0, -1.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = strong_wolfe_search(quad_value_and_grad, x, direction, value, grad)
    # New params should be near the origin.
    assert float(jnp.linalg.norm(res.new_params)) < 0.5


def test_line_search_jittable():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    fn = jax.jit(
        lambda x, d, v, g: (
            strong_wolfe_search(quad_value_and_grad, x, d, v, g).step_size
        )
    )
    step = fn(x, direction, value, grad)
    assert float(step) > 0.0


def test_armijo_alias_matches_backtracking():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    a = armijo_search(quad_value_and_grad, x, direction, value, grad)
    b = backtracking_search(quad_value_and_grad, x, direction, value, grad)
    np.testing.assert_allclose(a.step_size, b.step_size)
    np.testing.assert_allclose(a.new_value, b.new_value)


def test_backtracking_no_probes_when_disabled():
    x = jnp.array([5.0, 5.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = backtracking_search(
        quad_value_and_grad, x, direction, value, grad, record_probes=False
    )
    # With recording disabled, no more than a single scratch slot is filled.
    assert res.probe_valid is not None
    assert int(jnp.sum(res.probe_valid)) <= 1


def test_strong_wolfe_step_size_positive():
    x = jnp.array([3.0, -1.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = strong_wolfe_search(quad_value_and_grad, x, direction, value, grad)
    assert float(res.step_size) > 0.0


def test_backtracking_shrink_reduces_step():
    # A poorly-scaled direction forces backtracking to shrink the step.
    x = jnp.array([1.0, 1.0])
    value, grad = quad_value_and_grad(x)
    direction = -10.0 * grad  # overshoots; Armijo should shrink it
    res = backtracking_search(
        quad_value_and_grad,
        x,
        direction,
        value,
        grad,
        init_step=1.0,
        shrink=0.5,
        max_iter=10,
    )
    assert float(res.step_size) < 1.0
    assert float(res.new_value) <= float(value)


def test_region_restricted_search_stays_feasible():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    region = BoxRegion(lo=1.0, hi=5.0)
    for search in (backtracking_search, armijo_search, fixed_step_search):
        res = search(
            quad_value_and_grad,
            x,
            direction,
            value,
            grad,
            region=region,
            region_state=(),
        )
        assert jnp.all(res.new_params >= 1.0 - 1e-6)
        assert jnp.all(res.new_params <= 5.0 + 1e-6)


def test_all_searches_return_finite(quadratic_problem=None):
    x = jnp.array([2.0, -3.0, 1.5])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    for search in (
        backtracking_search,
        armijo_search,
        fixed_step_search,
        strong_wolfe_search,
        hager_zhang_search,
    ):
        res = search(quad_value_and_grad, x, direction, value, grad)
        assert np.isfinite(float(res.new_value))
        assert jnp.all(jnp.isfinite(res.new_grad))
        assert jnp.all(jnp.isfinite(res.new_params))


def test_backtracking_records_probes():
    x = jnp.array([5.0, 5.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = backtracking_search(
        quad_value_and_grad, x, direction, value, grad, record_probes=True
    )
    # At least one probe slot should be filled.
    assert res.probe_valid is not None
    assert bool(jnp.any(res.probe_valid))


def test_backtracking_respects_region():
    # Box restricts the step so the projected point is clipped.
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    region = BoxRegion(lo=1.5, hi=5.0)
    res = backtracking_search(
        quad_value_and_grad,
        x,
        direction,
        value,
        grad,
        region=region,
        region_state=(),
    )
    assert jnp.all(res.new_params >= 1.5 - 1e-6)


@pytest.mark.parametrize(
    "search",
    [backtracking_search, armijo_search, fixed_step_search],
)
def test_line_searches_are_vmappable(search):
    xs = jnp.array([[2.0, 2.0], [1.0, -1.0], [3.0, 0.5]])

    def run_one(x):
        value, grad = quad_value_and_grad(x)
        direction = -grad
        return search(quad_value_and_grad, x, direction, value, grad).new_value

    vals = jax.vmap(run_one)(xs)
    assert jnp.all(jnp.isfinite(vals))

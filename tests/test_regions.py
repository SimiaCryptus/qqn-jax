"""Tests for projective regions."""

import jax
import jax.numpy as jnp
import numpy as np

from qqn_jax import QQN
from qqn_jax.regions import (
    BoxRegion,
    IdentityRegion,
    NoDecreaseRegion,
    OrthantRegion,
    RegionInfo,
    Sequential,
    TrustRegion,
    resolve_region,
)


def quadratic(x):
    return 0.5 * jnp.sum(x**2)


# --- Identity ---------------------------------------------------------


def test_identity_region_is_noop():
    region = IdentityRegion()
    params = jnp.array([1.0, 2.0])
    candidate = jnp.array([3.0, 4.0])
    state = region.init(params)
    out = region.project(params, candidate, state)
    np.testing.assert_allclose(out, candidate)


def test_resolve_region_none_is_identity():
    region = resolve_region(None)
    params = jnp.array([1.0])
    candidate = jnp.array([99.0])
    np.testing.assert_allclose(
        region.project(params, candidate, region.init(params)), candidate
    )


def test_resolve_region_passthrough():
    r = BoxRegion(lo=0.0, hi=1.0)
    assert resolve_region(r) is r


# --- Box --------------------------------------------------------------


def test_box_region_clips():
    region = BoxRegion(lo=-1.0, hi=1.0)
    params = jnp.array([0.0, 0.0, 0.0])
    candidate = jnp.array([-5.0, 0.5, 5.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([-1.0, 0.5, 1.0]))


def test_box_region_unbounded_sides():
    region = BoxRegion(lo=0.0, hi=None)
    params = jnp.array([0.0, 0.0])
    candidate = jnp.array([-3.0, 100.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([0.0, 100.0]))


# --- Orthant ----------------------------------------------------------


def test_orthant_region_zeros_sign_crossing():
    region = OrthantRegion(l1=0.0)
    params = jnp.array([1.0, -1.0])
    # candidate flips both signs -> both zeroed.
    candidate = jnp.array([-2.0, 2.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([0.0, 0.0]))


def test_orthant_region_keeps_same_orthant():
    region = OrthantRegion(l1=0.0)
    params = jnp.array([1.0, -1.0])
    candidate = jnp.array([0.5, -0.5])  # same signs
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, candidate)


# --- Trust Region -----------------------------------------------------


def test_trust_region_clips_step_norm():
    region = TrustRegion(radius=1.0, adaptive=False)
    params = jnp.array([0.0, 0.0])
    candidate = jnp.array([3.0, 4.0])  # step norm = 5
    state = region.init(params)
    out = region.project(params, candidate, state)
    # Step should be radially scaled to norm 1.
    np.testing.assert_allclose(jnp.linalg.norm(out - params), 1.0, atol=1e-6)


def test_trust_region_leaves_small_step():
    region = TrustRegion(radius=10.0, adaptive=False)
    params = jnp.array([0.0, 0.0])
    candidate = jnp.array([0.3, 0.4])  # norm 0.5 < radius
    state = region.init(params)
    out = region.project(params, candidate, state)
    np.testing.assert_allclose(out, candidate, atol=1e-6)


def test_trust_region_adaptive_expands_on_good_agreement():
    region = TrustRegion(radius=1.0, adaptive=True, expand=2.0)
    params = jnp.array([0.0, 0.0])
    state = region.init(params)
    # A boundary step with excellent agreement (rho ~ 1) should expand.
    new_params = jnp.array([1.0, 0.0])  # step norm = radius
    info = RegionInfo(
        params=params,
        new_params=new_params,
        pred_reduction=jnp.asarray(1.0),
        actual_reduction=jnp.asarray(1.0),
    )
    new_state = region.update(state, info)
    assert float(new_state.radius) >= 1.0


def test_trust_region_floor_respects_successful_step():
    region = TrustRegion(radius=1.0, adaptive=True, shrink=0.5)
    params = jnp.array([0.0, 0.0])
    state = region.init(params)
    new_params = jnp.array([0.8, 0.0])
    # Poor predicted agreement but actual progress was made.
    info = RegionInfo(
        params=params,
        new_params=new_params,
        pred_reduction=jnp.asarray(10.0),
        actual_reduction=jnp.asarray(0.1),  # rho small -> would shrink
    )
    new_state = region.update(state, info)
    # Radius must not fall below the realized step length.
    assert float(new_state.radius) >= 0.8 - 1e-6


# --- NoDecrease -------------------------------------------------------


def test_no_decrease_removes_increasing_component():
    # Secondary g(x) = 0.5||x||^2, grad g = x.
    region = NoDecreaseRegion(lambda p: p)
    params = jnp.array([1.0, 0.0])
    # Step purely along +grad g would increase g.
    candidate = jnp.array([2.0, 0.0])
    out = region.project(params, candidate, region.init(params))
    step = out - params
    # The g-increasing component should have been removed.
    assert float(jnp.vdot(params, step)) <= 1e-6


def test_no_decrease_passes_descent_through():
    region = NoDecreaseRegion(lambda p: p)
    params = jnp.array([1.0, 0.0])
    # Step that decreases g (toward origin).
    candidate = jnp.array([0.5, 0.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, candidate, atol=1e-6)


# --- Sequential -------------------------------------------------------


def test_sequential_applies_in_order():
    region = Sequential([BoxRegion(lo=-2.0, hi=2.0), BoxRegion(lo=-1.0, hi=1.0)])
    params = jnp.array([0.0])
    candidate = jnp.array([5.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([1.0]))


def test_sequential_update_fans_out():
    region = Sequential(
        [TrustRegion(radius=1.0, adaptive=True), BoxRegion(lo=-5.0, hi=5.0)]
    )
    params = jnp.array([0.0, 0.0])
    state = region.init(params)
    info = RegionInfo(
        params=params,
        new_params=jnp.array([0.5, 0.0]),
        pred_reduction=jnp.asarray(1.0),
        actual_reduction=jnp.asarray(1.0),
    )
    new_state = region.update(state, info)
    assert isinstance(new_state, tuple)
    assert len(new_state) == 2


def test_box_region_pytree_bounds():
    region = BoxRegion(lo=jnp.array([-1.0, 0.0]), hi=jnp.array([1.0, 2.0]))
    params = jnp.array([0.0, 0.0])
    candidate = jnp.array([-5.0, 5.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([-1.0, 2.0]))


def test_trust_region_non_adaptive_update_is_noop():
    region = TrustRegion(radius=2.0, adaptive=False)
    params = jnp.array([0.0, 0.0])
    state = region.init(params)
    info = RegionInfo(
        params=params,
        new_params=jnp.array([1.0, 0.0]),
        pred_reduction=jnp.asarray(0.01),
        actual_reduction=jnp.asarray(-5.0),  # terrible agreement
    )
    new_state = region.update(state, info)
    # Non-adaptive: radius is unchanged.
    np.testing.assert_allclose(float(new_state.radius), 2.0, atol=1e-6)


def test_trust_region_shrinks_on_poor_agreement():
    region = TrustRegion(radius=4.0, adaptive=True, shrink=0.5, eta_lo=0.1)
    params = jnp.array([0.0, 0.0])
    state = region.init(params)
    # A small step (no floor pressure) with poor agreement should shrink.
    info = RegionInfo(
        params=params,
        new_params=jnp.array([0.01, 0.0]),
        pred_reduction=jnp.asarray(10.0),
        actual_reduction=jnp.asarray(-1.0),  # rho < 0 -> shrink
    )
    new_state = region.update(state, info)
    assert float(new_state.radius) < 4.0


def test_orthant_region_zeros_only_crossing_coordinate():
    region = OrthantRegion(l1=0.0)
    params = jnp.array([1.0, 1.0])
    # First coordinate flips sign, second stays positive.
    candidate = jnp.array([-0.5, 0.5])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([0.0, 0.5]))


def test_no_decrease_region_orthogonality():
    # After projection, the step's component along ∇g must be non-positive.
    region = NoDecreaseRegion(lambda p: p)
    params = jnp.array([1.0, 2.0])
    candidate = jnp.array([3.0, 5.0])  # large increasing step
    out = region.project(params, candidate, region.init(params))
    step = out - params
    assert float(jnp.vdot(params, step)) <= 1e-6


def test_identity_region_update_is_noop():
    region = IdentityRegion()
    params = jnp.array([1.0])
    state = region.init(params)
    info = RegionInfo(params=params, new_params=params)
    assert region.update(state, info) == state


def test_resolve_region_passthrough_trust():
    r = TrustRegion(radius=1.0)
    assert resolve_region(r) is r


def test_solver_with_sequential_region_is_jittable():
    region = Sequential([BoxRegion(lo=-10.0, hi=10.0), TrustRegion(radius=2.0)])
    solver = QQN(quadratic, maxiter=100, tol=1e-6, region=region)
    x0 = jnp.array([5.0, -3.0, 2.0])
    run_jit = jax.jit(solver.run)
    _, state = run_jit(x0)
    assert np.isfinite(float(state.value))


# --- End-to-end with solver -------------------------------------------


def test_solver_with_box_region_respects_bounds():
    # Minimum of quadratic is at origin; box keeps us within [0.5, 5].
    region = BoxRegion(lo=0.5, hi=5.0)
    solver = QQN(quadratic, maxiter=100, tol=1e-6, region=region)
    x0 = jnp.array([3.0, 4.0])
    params, _ = solver.run(x0)
    assert jnp.all(params >= 0.5 - 1e-5)
    assert jnp.all(params <= 5.0 + 1e-5)


def test_solver_with_trust_region_converges():
    region = TrustRegion(radius=1.0, adaptive=True)
    solver = QQN(quadratic, maxiter=200, tol=1e-5, region=region)
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.value) < 1e-2


def test_solver_with_region_is_jittable():
    region = BoxRegion(lo=-10.0, hi=10.0)
    solver = QQN(quadratic, maxiter=100, tol=1e-6, region=region)
    x0 = jnp.array([5.0, -3.0, 2.0])
    run_jit = jax.jit(solver.run)
    _, state = run_jit(x0)
    assert np.isfinite(float(state.value))

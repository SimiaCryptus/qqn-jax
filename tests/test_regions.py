"""Tests for projective regions."""

import jax
import jax.numpy as jnp
import numpy as np

from qqn_jax import QQN, OrthantRegion, BoxRegion, TrustRegion, Sequential
from qqn_jax.regions.no_decrease import NoDecreaseRegion
from qqn_jax.regions.strategy import (
    IdentityRegion,
    RegionInfo,
    resolve_region,
)


def quadratic(x):
    return 0.5 * jnp.sum(x**2)


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


def test_orthant_region_zeros_sign_crossing():
    region = OrthantRegion()
    params = jnp.array([1.0, -1.0])

    candidate = jnp.array([-2.0, 2.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([0.0, 0.0]))


def test_orthant_region_keeps_same_orthant():
    region = OrthantRegion()
    params = jnp.array([1.0, -1.0])
    candidate = jnp.array([0.5, -0.5])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, candidate)


def test_trust_region_clips_step_norm():
    region = TrustRegion(radius=1.0, adaptive=False)
    params = jnp.array([0.0, 0.0])
    candidate = jnp.array([3.0, 4.0])
    state = region.init(params)
    out = region.project(params, candidate, state)

    np.testing.assert_allclose(jnp.linalg.norm(out - params), 1.0, atol=1e-6)


def test_trust_region_leaves_small_step():
    region = TrustRegion(radius=10.0, adaptive=False)
    params = jnp.array([0.0, 0.0])
    candidate = jnp.array([0.3, 0.4])
    state = region.init(params)
    out = region.project(params, candidate, state)
    np.testing.assert_allclose(out, candidate, atol=1e-6)


def test_trust_region_adaptive_expands_on_good_agreement():
    region = TrustRegion(radius=1.0, adaptive=True, expand=2.0)
    params = jnp.array([0.0, 0.0])
    state = region.init(params)

    new_params = jnp.array([1.0, 0.0])
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

    info = RegionInfo(
        params=params,
        new_params=new_params,
        pred_reduction=jnp.asarray(10.0),
        actual_reduction=jnp.asarray(0.1),
    )
    new_state = region.update(state, info)

    assert float(new_state.radius) >= 0.8 - 1e-6


def test_no_decrease_removes_increasing_component():

    region = NoDecreaseRegion(lambda p: p)
    params = jnp.array([1.0, 0.0])

    candidate = jnp.array([2.0, 0.0])
    out = region.project(params, candidate, region.init(params))
    step = out - params

    assert float(jnp.vdot(params, step)) <= 1e-6


def test_no_decrease_passes_descent_through():
    region = NoDecreaseRegion(lambda p: p)
    params = jnp.array([1.0, 0.0])

    candidate = jnp.array([0.5, 0.0])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, candidate, atol=1e-6)


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
        actual_reduction=jnp.asarray(-5.0),
    )
    new_state = region.update(state, info)

    np.testing.assert_allclose(float(new_state.radius), 2.0, atol=1e-6)


def test_trust_region_shrinks_on_poor_agreement():
    region = TrustRegion(radius=4.0, adaptive=True, shrink=0.5, eta_lo=0.1)
    params = jnp.array([0.0, 0.0])
    state = region.init(params)

    info = RegionInfo(
        params=params,
        new_params=jnp.array([0.01, 0.0]),
        pred_reduction=jnp.asarray(10.0),
        actual_reduction=jnp.asarray(-1.0),
    )
    new_state = region.update(state, info)
    assert float(new_state.radius) < 4.0


def test_orthant_region_zeros_only_crossing_coordinate():
    region = OrthantRegion()
    params = jnp.array([1.0, 1.0])

    candidate = jnp.array([-0.5, 0.5])
    out = region.project(params, candidate, region.init(params))
    np.testing.assert_allclose(out, jnp.array([0.0, 0.5]))


def test_no_decrease_region_orthogonality():

    region = NoDecreaseRegion(lambda p: p)
    params = jnp.array([1.0, 2.0])
    candidate = jnp.array([3.0, 5.0])
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


def test_solver_with_box_region_respects_bounds():

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

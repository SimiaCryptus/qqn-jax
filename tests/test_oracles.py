"""Tests for the oracle abstraction and its concrete implementations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax import QQN, AndersonOracle, Oracle, OracleInfo
from qqn_jax.oracles.strategy import (
    Fallback,
    LBFGSOracle,
    MomentumOracle,
    SecantOracle,
    ShampooOracle,
    resolve_oracle,
)


def quadratic(x):
    A = jnp.diag(jnp.array([1.0, 5.0, 10.0]))
    return 0.5 * x @ A @ x


def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


# --- Oracle direction sanity ------------------------------------------


def test_lbfgs_oracle_initial_direction_is_negative_gradient():
    oracle = LBFGSOracle(history_size=5)
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    np.testing.assert_allclose(d, -grad, atol=1e-7)


def test_momentum_oracle_accumulates_velocity():
    oracle = MomentumOracle(beta=0.9)
    params = jnp.array([0.0, 0.0])
    grad = jnp.array([1.0, 2.0])
    state = oracle.init(params)
    d1, _ = oracle.direction(params, grad, state)
    # First step: velocity v = 0, so the endpoint reduces to plain steepest
    # descent d = -grad (preserving the d'(0) anchor; see MomentumOracle).
    np.testing.assert_allclose(d1, -grad, atol=1e-6)
    # Velocity is committed in ``update`` (mirroring the solver), not in
    # ``direction`` (whose returned state the solver discards).
    # The momentum oracle accumulates the *realized* per-iteration delta
    # Δx = x_new − x, so the update must reflect an actual step taken
    # (here a steepest-descent move) for the velocity to grow.
    new_params = params - grad
    info = OracleInfo(params=params, new_params=new_params, grad=grad, new_grad=grad)
    state = oracle.update(state, info)
    d2, _ = oracle.direction(params, grad, state)
    # Velocity grows on a repeated gradient -> direction magnitude grows.
    assert float(jnp.linalg.norm(d2)) > float(jnp.linalg.norm(d1))


def test_secant_oracle_first_step_is_scaled_descent():
    oracle = SecantOracle(alpha0=1.0)
    params = jnp.array([1.0, 1.0])
    grad = jnp.array([2.0, -1.0])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    np.testing.assert_allclose(d, -grad, atol=1e-6)


def test_secant_oracle_updates_curvature():
    oracle = SecantOracle(alpha0=1.0)
    params = jnp.array([1.0, 1.0])
    grad = jnp.array([1.0, 10.0])  # grad of 0.5 x^T diag(1,10) x at (1,1)
    state = oracle.init(params)
    new_params = jnp.array([0.5, 0.5])
    new_grad = jnp.array([0.5, 5.0])
    info = OracleInfo(
        params=params, new_params=new_params, grad=grad, new_grad=new_grad
    )
    new_state = oracle.update(state, info)
    assert int(new_state.step_count) == 1
    # BB step should be finite and positive.
    assert float(new_state.alpha) > 0.0
    assert np.isfinite(float(new_state.alpha))


def test_anderson_oracle_safeguards_to_descent_when_empty():
    oracle = AndersonOracle(window=5)
    params = jnp.array([1.0, 2.0, 3.0])
    grad = jnp.array([0.1, 0.2, 0.3])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    # With no history, the safeguard falls back to -grad.
    np.testing.assert_allclose(d, -grad, atol=1e-6)


def test_shampoo_oracle_direction_is_finite():
    oracle = ShampooOracle(update_freq=1, epsilon=1e-6)
    params = jnp.array([1.0, 2.0, 3.0, 4.0])
    grad = jnp.array([0.5, -0.5, 1.0, -1.0])
    state = oracle.init(params)
    d, new_state = oracle.direction(params, grad, state)
    assert jnp.all(jnp.isfinite(d))
    assert int(new_state.step) == 1


def test_lbfgs_oracle_is_descent_after_update():
    oracle = LBFGSOracle(history_size=5)
    params = jnp.array([1.0, 1.0])
    grad = jnp.array([1.0, 10.0])
    state = oracle.init(params)
    new_params = jnp.array([0.5, 0.5])
    new_grad = jnp.array([0.5, 5.0])
    info = OracleInfo(
        params=params, new_params=new_params, grad=grad, new_grad=new_grad
    )
    state = oracle.update(state, info)
    d, _ = oracle.direction(new_params, new_grad, state)
    assert float(jnp.vdot(d, new_grad)) < 0.0


def test_momentum_oracle_direction_is_descent_on_first_step():
    oracle = MomentumOracle(beta=0.9)
    params = jnp.array([0.0, 0.0])
    grad = jnp.array([1.0, 2.0])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    assert float(jnp.vdot(d, grad)) < 0.0


def test_secant_oracle_rejects_negative_curvature():
    oracle = SecantOracle(alpha0=1.0)
    params = jnp.array([1.0, 1.0])
    grad = jnp.array([1.0, 1.0])
    state = oracle.init(params)
    # Construct a pair with sy <= 0 (non-positive curvature).
    new_params = jnp.array([2.0, 1.0])  # s = (1, 0)
    new_grad = jnp.array([-2.0, 1.0])  # y = (-3, 0), sy = -3 < 0
    info = OracleInfo(
        params=params, new_params=new_params, grad=grad, new_grad=new_grad
    )
    new_state = oracle.update(state, info)
    # Alpha should be retained (unchanged) on non-positive curvature.
    np.testing.assert_allclose(float(new_state.alpha), 1.0, atol=1e-6)


def test_anderson_oracle_descent_after_history():
    oracle = AndersonOracle(window=5)
    params = jnp.array([1.0, 2.0, 3.0])
    grad = jnp.array([0.1, 0.2, 0.3])
    state = oracle.init(params)
    # Seed some history.
    for _ in range(3):
        new_params = params * 0.9
        new_grad = grad * 0.9
        info = OracleInfo(
            params=params, new_params=new_params, grad=grad, new_grad=new_grad
        )
        state = oracle.update(state, info)
        params, grad = new_params, new_grad
    d, _ = oracle.direction(params, grad, state)
    assert jnp.all(jnp.isfinite(d))


def test_shampoo_oracle_keep_branch_uses_gradient():
    # On a non-refresh step the direction falls back to -grad.
    oracle = ShampooOracle(update_freq=5, epsilon=1e-6)
    params = jnp.array([1.0, 2.0, 3.0])
    grad = jnp.array([0.5, -0.5, 1.0])
    state = oracle.init(params)
    # Step 0 refreshes; advance to a non-refresh step.
    _, state = oracle.direction(params, grad, state)
    d, _ = oracle.direction(params, grad, state)
    np.testing.assert_allclose(d, -grad, atol=1e-6)


def test_fallback_update_fans_out():
    oracle = Fallback([LBFGSOracle(history_size=5), SecantOracle()])
    params = jnp.array([1.0, 1.0])
    grad = jnp.array([1.0, 10.0])
    state = oracle.init(params)
    new_params = jnp.array([0.5, 0.5])
    new_grad = jnp.array([0.5, 5.0])
    info = OracleInfo(
        params=params, new_params=new_params, grad=grad, new_grad=new_grad
    )
    new_state = oracle.update(state, info)
    # State remains a tuple of the two child states.
    assert isinstance(new_state, tuple)
    assert len(new_state) == 2


def test_oracle_direction_is_jittable():
    oracle = AndersonOracle(window=5)
    params = jnp.array([1.0, 2.0, 3.0])
    grad = jnp.array([0.1, 0.2, 0.3])
    state = oracle.init(params)
    fn = jax.jit(lambda p, g, s: oracle.direction(p, g, s)[0])
    d = fn(params, grad, state)
    assert jnp.all(jnp.isfinite(d))


# --- Fallback combinator ----------------------------------------------


def test_fallback_uses_first_valid_direction():
    # A "broken" oracle that always emits NaNs.
    def _broken():
        def init(params):
            return ()

        def direction(params, grad, state):
            return jnp.full_like(grad, jnp.nan), ()

        def update(state, info):
            return state

        return Oracle(init=init, direction=direction, update=update)

    oracle = Fallback([_broken(), LBFGSOracle(history_size=5)])
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    # Should fall through to the L-BFGS oracle -> -grad.
    np.testing.assert_allclose(d, -grad, atol=1e-6)


def test_fallback_rejects_uphill_direction():
    # An oracle whose direction points uphill (ascent).
    def _uphill():
        def init(params):
            return ()

        def direction(params, grad, state):
            return grad, ()  # +grad is ascent

        def update(state, info):
            return state

        return Oracle(init=init, direction=direction, update=update)

    oracle = Fallback([_uphill(), LBFGSOracle(history_size=5)])
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    # Uphill is rejected; fall back to -grad.
    assert float(jnp.vdot(grad, d)) < 0.0


def test_fallback_all_invalid_yields_steepest_descent():
    def _broken():
        def init(params):
            return ()

        def direction(params, grad, state):
            return jnp.full_like(grad, jnp.nan), ()

        def update(state, info):
            return state

        return Oracle(init=init, direction=direction, update=update)

    oracle = Fallback([_broken(), _broken()])
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = oracle.init(params)
    d, _ = oracle.direction(params, grad, state)
    np.testing.assert_allclose(d, -grad, atol=1e-6)


# --- resolve_oracle ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "lbfgs",
        "momentum",
        "shampoo",
        "secant",
        "anderson",
        "anderson+secant",
        "lbfgs+secant",
    ],
)
def test_resolve_oracle_known_strings(name):
    oracle = resolve_oracle(name)
    assert isinstance(oracle, Oracle)


def test_resolve_oracle_none_is_lbfgs():
    oracle = resolve_oracle(None)
    assert isinstance(oracle, Oracle)


def test_resolve_oracle_passthrough_instance():
    inst = MomentumOracle()
    assert resolve_oracle(inst) is inst


def test_resolve_oracle_unknown_string_raises():
    with pytest.raises(ValueError):
        resolve_oracle("not_an_oracle")


def test_resolve_oracle_bad_type_raises():
    with pytest.raises(TypeError):
        resolve_oracle(42)


# --- End-to-end convergence with non-default oracles ------------------


@pytest.mark.parametrize("oracle", ["lbfgs", "secant", "anderson", "lbfgs+secant"])
def test_solver_converges_with_oracle_on_quadratic(oracle):
    solver = QQN(quadratic, maxiter=200, tol=1e-6, oracle=oracle)
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert float(state.value) < 1e-2


def test_solver_with_momentum_oracle_decreases():
    solver = QQN(quadratic, maxiter=100, tol=1e-6, oracle="momentum")
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.value) < float(quadratic(x0))


def test_solver_with_anderson_secant_converges():
    solver = QQN(quadratic, maxiter=200, tol=1e-6, oracle="anderson+secant")
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.value) < 1e-2


def test_solver_with_shampoo_decreases():
    solver = QQN(quadratic, maxiter=100, tol=1e-6, oracle="shampoo")
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.value) < float(quadratic(x0))


def test_oracle_run_is_vmappable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6, oracle="lbfgs+secant")
    x0_batch = jnp.array([[5.0, -3.0, 2.0], [1.0, 1.0, 1.0], [-2.0, 4.0, -1.0]])
    params, states = jax.vmap(solver.run)(x0_batch)
    assert params.shape == (3, 3)
    assert jnp.all(jnp.isfinite(states.value))


def test_oracle_is_jittable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6, oracle="anderson")
    x0 = jnp.array([5.0, -3.0, 2.0])
    run_jit = jax.jit(solver.run)
    _, state = run_jit(x0)
    assert np.isfinite(float(state.value))

"""Tests for the oracle abstraction and its concrete implementations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax import QQN
from qqn_jax.oracles import (
    AndersonOracle,
    Fallback,
    LBFGSOracle,
    MomentumOracle,
    Oracle,
    OracleInfo,
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
    d1, state = oracle.direction(params, grad, state)
    # First step: v = 0.1 * grad, so d = -0.1 * grad.
    np.testing.assert_allclose(d1, -0.1 * grad, atol=1e-6)
    d2, state = oracle.direction(params, grad, state)
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
    assert int(new_state.count) == 1
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


def test_oracle_is_jittable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6, oracle="anderson")
    x0 = jnp.array([5.0, -3.0, 2.0])
    run_jit = jax.jit(solver.run)
    _, state = run_jit(x0)
    assert np.isfinite(float(state.value))

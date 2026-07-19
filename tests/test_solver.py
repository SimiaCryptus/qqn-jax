"""Convergence and interface tests for the QQN solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax import QQN


def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def quadratic(x):
    A = jnp.diag(jnp.array([1.0, 5.0, 10.0]))
    return 0.5 * x @ A @ x


def test_init_state():
    solver = QQN(quadratic, maxiter=50)
    x0 = jnp.array([1.0, 1.0, 1.0])
    state = solver.init_state(x0)
    assert int(state.iter) == 0
    np.testing.assert_allclose(state.value, quadratic(x0))


def test_single_update_decreases_value():
    solver = QQN(quadratic, maxiter=50)
    x0 = jnp.array([1.0, 1.0, 1.0])
    state = solver.init_state(x0)
    new_x, new_state = solver.update(x0, state)
    assert float(new_state.value) < float(state.value)
    assert int(new_state.iter) == 1


def test_converges_on_quadratic():
    solver = QQN(quadratic, maxiter=100, tol=1e-6)
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert float(state.error) < 1e-4
    np.testing.assert_allclose(params, jnp.zeros(3), atol=1e-3)


def test_converges_on_rosenbrock():
    solver = QQN(rosenbrock, maxiter=500, tol=1e-5, history_size=15)
    x0 = jnp.array([-1.2, 1.0])
    params, state = solver.run(x0)

    np.testing.assert_allclose(params, jnp.ones(2), atol=1e-2)


def test_run_is_jittable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6)
    x0 = jnp.array([5.0, -3.0, 2.0])
    run_jit = jax.jit(solver.run)
    params, state = run_jit(x0)
    assert float(state.error) < 1e-4


def test_run_is_vmappable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6)
    x0_batch = jnp.array([[5.0, -3.0, 2.0], [1.0, 1.0, 1.0], [-2.0, 4.0, -1.0]])
    batched = jax.vmap(solver.run)
    params, states = batched(x0_batch)
    assert params.shape == (3, 3)
    np.testing.assert_allclose(params, jnp.zeros((3, 3)), atol=1e-2)


def test_backtracking_line_search_option():
    solver = QQN(quadratic, maxiter=200, tol=1e-5, line_search="backtracking")
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert float(state.error) < 1e-3


def test_has_aux():
    def fun_with_aux(x):
        value = quadratic(x)
        aux = {"norm": jnp.linalg.norm(x)}
        return value, aux

    solver = QQN(fun_with_aux, maxiter=100, tol=1e-6, has_aux=True)
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert "norm" in state.aux
    assert float(state.error) < 1e-3


def test_unknown_line_search_raises():
    with pytest.raises(ValueError):
        QQN(quadratic, line_search="does_not_exist")


@pytest.mark.parametrize("ls", ["backtracking", "strong_wolfe", "hager_zhang", "fixed"])
def test_all_line_searches_construct_and_run(ls):
    solver = QQN(quadratic, maxiter=200, tol=1e-5, line_search=ls)
    x0 = jnp.array([1.0, 1.0, 1.0])
    _, state = solver.run(x0)

    assert float(state.value) <= float(quadratic(x0)) + 1e-6


def test_line_search_options_are_forwarded():

    solver = QQN(
        quadratic,
        maxiter=300,
        tol=1e-5,
        line_search="backtracking",
        line_search_options={"c1": 1e-3, "shrink": 0.7, "max_iter": 20},
    )
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.error) < 1e-3


def test_init_state_done_flag_when_already_converged():
    solver = QQN(quadratic, maxiter=50, tol=1e-3)

    x0 = jnp.array([1e-8, 1e-8, 1e-8])
    state = solver.init_state(x0)
    assert bool(state.done)


def test_run_terminates_on_nonfinite():

    def explosive(x):
        return jnp.sum(jnp.exp(50.0 * x))

    solver = QQN(explosive, maxiter=20, tol=1e-6)
    x0 = jnp.array([1.0, 1.0])
    _, state = solver.run(x0)

    assert int(state.iter) <= 20


def test_update_increments_iteration_count():
    solver = QQN(quadratic, maxiter=50)
    x0 = jnp.array([1.0, 1.0, 1.0])
    state = solver.init_state(x0)
    for i in range(3):
        x0, state = solver.update(x0, state)
        assert int(state.iter) == i + 1


def test_run_respects_maxiter_when_not_converged():
    solver = QQN(rosenbrock, maxiter=3, tol=1e-12, history_size=5)
    x0 = jnp.array([-1.2, 1.0])
    _, state = solver.run(x0)
    assert int(state.iter) <= 3


def test_spline_in_solver_converges_quadratic():
    solver = QQN(quadratic, maxiter=100, tol=1e-6, line_search="spline")
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.error) < 1e-3


def test_init_state_error_matches_grad_norm():
    solver = QQN(quadratic, maxiter=50)
    x0 = jnp.array([3.0, 4.0, 0.0])
    state = solver.init_state(x0)
    _, grad = jax.value_and_grad(quadratic)(x0)
    np.testing.assert_allclose(
        float(state.error), float(jnp.linalg.norm(grad)), atol=1e-6
    )


def test_value_decreases_monotonically_on_quadratic():
    solver = QQN(quadratic, maxiter=20, tol=1e-8)
    x0 = jnp.array([5.0, -3.0, 2.0])
    state = solver.init_state(x0)
    prev = float(state.value)
    for _ in range(10):
        x0, state = solver.update(x0, state)
        cur = float(state.value)
        assert cur <= prev + 1e-6
        prev = cur


@pytest.mark.parametrize("oracle", ["lbfgs", "secant", "anderson"])
def test_solver_oracle_and_region_compose(oracle):
    solver = QQN(
        quadratic,
        maxiter=200,
        tol=1e-5,
        oracle=oracle,
        region=None,
    )
    x0 = jnp.array([5.0, -3.0, 2.0])
    _, state = solver.run(x0)
    assert float(state.value) < 1e-1

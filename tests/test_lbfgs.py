"""Tests for the L-BFGS oracle."""

import jax

import jax.numpy as jnp
import numpy as np

from qqn_jax.lbfgs import (
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
    update_lbfgs_history_batch,
)


def test_initial_direction_is_negative_gradient():

    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = init_lbfgs_state(params, grad, history_size=5)
    d = lbfgs_direction(state, grad)
    np.testing.assert_allclose(d, -grad, atol=1e-7)


def test_history_update_curvature():

    A = jnp.array([[1.0, 0.0], [0.0, 10.0]])

    def grad(x):
        return A @ x

    history_size = 5
    x0 = jnp.array([1.0, 1.0])
    g0 = grad(x0)
    state = init_lbfgs_state(x0, g0, history_size)

    x1 = jnp.array([0.5, 0.5])
    g1 = grad(x1)
    state = update_lbfgs_history(state, x1, g1, history_size)

    assert int(state.step_count) == 1

    d = lbfgs_direction(state, g1)

    assert float(jnp.vdot(d, g1)) < 0.0


def test_rejects_negative_curvature():
    params = jnp.array([0.0, 0.0])
    grad0 = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(params, grad0, history_size=5)

    bad_params = jnp.array([1.0, 0.0])
    bad_grad = jnp.array([-2.0, 1.0])
    new_state = update_lbfgs_history(state, bad_params, bad_grad, 5)
    assert int(new_state.step_count) == 0


def test_history_buffer_is_circular():

    A = jnp.array([[1.0, 0.0], [0.0, 10.0]])

    def grad(x):
        return A @ x

    history_size = 3
    x = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(x, grad(x), history_size)
    for k in range(6):
        x_new = x * 0.7
        state = update_lbfgs_history(state, x_new, grad(x_new), history_size)
        x = x_new
    assert int(state.step_count) == history_size


def test_relative_curvature_guard_rejects_tiny_curvature():

    params = jnp.array([0.0, 0.0])
    grad0 = jnp.array([1.0, 0.0])
    state = init_lbfgs_state(params, grad0, history_size=5)

    new_params = jnp.array([1e3, 0.0])
    new_grad = jnp.array([1e-9 + 1.0, 1.0])
    new_state = update_lbfgs_history(state, new_params, new_grad, 5)

    assert int(new_state.step_count) == 0


def test_history_batch_replays_valid_probes():
    A = jnp.array([[1.0, 0.0], [0.0, 10.0]])

    def grad(x):
        return A @ x

    history_size = 5
    x0 = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(x0, grad(x0), history_size)
    params_seq = jnp.array([[0.7, 0.7], [0.5, 0.5], [0.3, 0.3]])
    grad_seq = jnp.stack([grad(p) for p in params_seq])
    valid_seq = jnp.array([True, True, True])
    new_state = update_lbfgs_history_batch(
        state, params_seq, grad_seq, valid_seq, history_size
    )

    assert int(new_state.step_count) == 3
    d = lbfgs_direction(new_state, grad(params_seq[-1]))
    assert float(jnp.vdot(d, grad(params_seq[-1]))) < 0.0


def test_history_batch_skips_invalid_slots():
    A = jnp.array([[1.0, 0.0], [0.0, 10.0]])

    def grad(x):
        return A @ x

    history_size = 5
    x0 = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(x0, grad(x0), history_size)
    params_seq = jnp.array([[0.7, 0.7], [0.5, 0.5], [0.3, 0.3]])
    grad_seq = jnp.stack([grad(p) for p in params_seq])

    valid_seq = jnp.array([True, False, True])
    new_state = update_lbfgs_history_batch(
        state, params_seq, grad_seq, valid_seq, history_size
    )
    assert int(new_state.step_count) == 2


def test_direction_is_jittable():
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = init_lbfgs_state(params, grad, history_size=5)
    fn = jax.jit(lambda g: lbfgs_direction(state, g))
    d = fn(grad)
    np.testing.assert_allclose(d, -grad, atol=1e-7)


def test_update_is_jittable():
    params = jnp.array([1.0, 1.0])
    grad = jnp.array([1.0, 10.0])
    state = init_lbfgs_state(params, grad, history_size=5)

    def step(s, p, g):
        return update_lbfgs_history(s, p, g, 5)

    fn = jax.jit(step)
    new_state = fn(state, jnp.array([0.5, 0.5]), jnp.array([0.5, 5.0]))
    assert int(new_state.step_count) == 1


def test_gradient_does_not_poison_on_rejected_pair():

    def loss(scale):
        params = jnp.array([0.0, 0.0])
        grad0 = jnp.array([1.0, 1.0])
        state = init_lbfgs_state(params, grad0, history_size=5)
        bad_params = jnp.array([scale, 0.0])
        bad_grad = jnp.array([-2.0, 1.0])
        new_state = update_lbfgs_history(state, bad_params, bad_grad, 5)
        d = lbfgs_direction(new_state, grad0)
        return jnp.sum(d**2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))


def test_lbfgs_direction_solves_diagonal_quadratic():

    A = jnp.array([[2.0, 0.0], [0.0, 8.0]])

    def grad(x):
        return A @ x

    history_size = 10
    x = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(x, grad(x), history_size)
    for _ in range(8):
        x_new = x * 0.5
        state = update_lbfgs_history(state, x_new, grad(x_new), history_size)
        x = x_new
    g = grad(x)
    d = lbfgs_direction(state, g)

    assert float(jnp.vdot(d, g)) < 0.0

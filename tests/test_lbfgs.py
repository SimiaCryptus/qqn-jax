"""Tests for the L-BFGS oracle."""

import jax.numpy as jnp
import numpy as np

from qqn_jax.lbfgs import (
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
    update_lbfgs_history_batch,
)


def test_initial_direction_is_negative_gradient():
    # With no history, H0 = gamma*I = I, so direction = -grad.
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = init_lbfgs_state(params, grad, history_size=5)
    d = lbfgs_direction(state, grad)
    np.testing.assert_allclose(d, -grad, atol=1e-7)


def test_history_update_curvature():
    # Quadratic f(x) = 0.5 xᵀA x with A = diag(1, 10).
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

    # Count should have incremented (positive curvature).
    assert int(state.count) == 1

    d = lbfgs_direction(state, g1)
    # The L-BFGS direction should be a descent direction.
    assert float(jnp.vdot(d, g1)) < 0.0


def test_rejects_negative_curvature():
    params = jnp.array([0.0, 0.0])
    grad0 = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(params, grad0, history_size=5)

    # Construct an update with yᵀs < 0 (negative curvature).
    bad_params = jnp.array([1.0, 0.0])  # s = (1, 0)
    bad_grad = jnp.array([-2.0, 1.0])  # y = (-3, 0), yᵀs = -3 < 0
    new_state = update_lbfgs_history(state, bad_params, bad_grad, 5)
    assert int(new_state.count) == 0

    def test_history_buffer_is_circular():
        # Push more pairs than history_size; count saturates at history_size.
        A = jnp.array([[1.0, 0.0], [0.0, 10.0]])

        def grad(x):
            return A @ x

        history_size = 3
        x = jnp.array([1.0, 1.0])
        state = init_lbfgs_state(x, grad(x), history_size)
        for k in range(6):
            x_new = x * 0.7  # always positive curvature toward origin
            state = update_lbfgs_history(state, x_new, grad(x_new), history_size)
            x = x_new
        assert int(state.count) == history_size

    def test_relative_curvature_guard_rejects_tiny_curvature():
        # A near-flat update (yᵀs almost zero relative to ‖y‖‖s‖) is rejected.
        params = jnp.array([0.0, 0.0])
        grad0 = jnp.array([1.0, 0.0])
        state = init_lbfgs_state(params, grad0, history_size=5)
        # s = (1e3, 0), y = (1e-9, 1.0) -> ys = 1e-6 but ‖y‖‖s‖ ~ 1e3.
        new_params = jnp.array([1e3, 0.0])
        new_grad = jnp.array([1e-9 + 1.0, 1.0])
        new_state = update_lbfgs_history(state, new_params, new_grad, 5)
        # Curvature is tiny relative to scale; should be rejected.
        assert int(new_state.count) == 0

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
        # All three positive-curvature probes should be admitted.
        assert int(new_state.count) == 3
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
        # Middle probe marked invalid -> skipped (history unchanged for it).
        valid_seq = jnp.array([True, False, True])
        new_state = update_lbfgs_history_batch(
            state, params_seq, grad_seq, valid_seq, history_size
        )
        assert int(new_state.count) == 2

"""Comprehensive unit tests for all oracle implementations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.oracles.oracle import Oracle, OracleInfo
from qqn_jax.oracles.adam import AdamOracle, AdamState
from qqn_jax.oracles.momentum import MomentumOracle, MomentumState
from qqn_jax.oracles.path_history import (
    PathHistoryMomentumOracle,
    PathHistoryMomentumState,
)
from qqn_jax.oracles.secant import SecantOracle, SecantState
from qqn_jax.oracles.anderson import AndersonOracle
from qqn_jax.oracles.ams_qn import AnchoredMultiSecantOracle
from qqn_jax.oracles.shampoo import ShampooOracle
from qqn_jax.oracles.fallback import Fallback


N = 4


def quad_grad(x, A=None):
    """Gradient of a simple quadratic 0.5 x^T A x."""
    if A is None:
        A = jnp.diag(jnp.arange(1, N + 1, dtype=jnp.float32))
    return A @ x


def make_info(params, new_params, grad, new_grad, step_size=1.0):
    return OracleInfo(
        params=params,
        new_params=new_params,
        grad=grad,
        new_grad=new_grad,
        t=jnp.asarray(1.0),
        step_size=jnp.asarray(step_size, dtype=jnp.float32),
    )


ORACLE_FACTORIES = {
    "adam": lambda: AdamOracle(),
    "momentum": lambda: MomentumOracle(),
    "path_momentum": lambda: PathHistoryMomentumOracle(history_size=5),
    "secant": lambda: SecantOracle(),
    "anderson": lambda: AndersonOracle(window=3),
    "ams_qn": lambda: AnchoredMultiSecantOracle(window=3),
    "shampoo": lambda: ShampooOracle(update_freq=1),
}


@pytest.fixture
def x0():
    return jnp.ones(N, dtype=jnp.float32)


class TestOracleContract:
    """Every oracle must satisfy the basic init/direction/update contract."""

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_init_returns_state(self, name, x0):
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        assert state is not None

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_direction_shape(self, name, x0):
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, new_state = oracle.direction(x0, grad, state)
        assert d.shape == x0.shape

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_direction_finite(self, name, x0):
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        assert np.all(np.isfinite(np.asarray(d)))

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_first_step_is_descent(self, name, x0):
        """On the first step (empty history) direction should be a
        descent direction: <grad, d> < 0."""
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        assert float(jnp.vdot(grad, d)) < 0.0

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_update_returns_state(self, name, x0):
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        new_grad = quad_grad(new_x)
        info = make_info(x0, new_x, grad, new_grad)
        new_state = oracle.update(state, info)
        assert new_state is not None

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_jittable(self, name, x0):
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        grad = quad_grad(x0)

        @jax.jit
        def step(state, x, g):
            d, ns = oracle.direction(x, g, state)
            new_x = x + d
            new_g = quad_grad(new_x)
            info = make_info(x, new_x, g, new_g)
            ns2 = oracle.update(ns, info)
            return d, ns2

        d, ns = step(state, x0, grad)
        assert np.all(np.isfinite(np.asarray(d)))

    @pytest.mark.parametrize("name", list(ORACLE_FACTORIES.keys()))
    def test_convergence_on_quadratic(self, name, x0):
        """Running the oracle should reduce the gradient norm on a
        well-conditioned quadratic."""
        oracle = ORACLE_FACTORIES[name]()
        state = oracle.init(x0)
        A = jnp.diag(jnp.linspace(1.0, 3.0, N))
        x = x0
        g0 = A @ x
        g = g0
        lr = 0.05
        for _ in range(200):
            d, state = oracle.direction(x, g, state)
            new_x = x + lr * d
            new_g = A @ new_x
            info = make_info(x, new_x, g, new_g, step_size=lr)
            state = oracle.update(state, info)
            x, g = new_x, new_g
        assert float(jnp.linalg.norm(g)) < float(jnp.linalg.norm(g0))


class TestAdam:
    def test_init(self, x0):
        oracle = AdamOracle()
        state = oracle.init(x0)
        assert isinstance(state, AdamState)
        np.testing.assert_allclose(state.m, 0.0)
        np.testing.assert_allclose(state.v, 0.0)
        assert int(state.step) == 0

    def test_learning_rate_scaling(self, x0):
        lr = 1e-2
        oracle = AdamOracle(learning_rate=lr)
        state = oracle.init(x0)
        grad = jnp.ones(N, dtype=jnp.float32)
        d, _ = oracle.direction(x0, grad, state)

        assert float(jnp.max(jnp.abs(d))) <= lr + 1e-4

    def test_update_increments_step(self, x0):
        oracle = AdamOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        info = make_info(x0, x0 - 0.1 * grad, grad, grad)._replace(grad=grad)
        new_state = oracle.update(state, info)
        assert int(new_state.step) == 1

    def test_moments_accumulate(self, x0):
        oracle = AdamOracle(beta1=0.5, beta2=0.5)
        state = oracle.init(x0)
        grad = jnp.ones(N, dtype=jnp.float32)
        info = OracleInfo(
            params=x0,
            new_params=x0,
            grad=grad,
            new_grad=grad,
            step_size=jnp.asarray(1.0),
        )
        new_state = oracle.update(state, info)

        np.testing.assert_allclose(new_state.m, 0.5, rtol=1e-6)
        np.testing.assert_allclose(new_state.v, 0.5, rtol=1e-6)


class TestMomentum:
    def test_init_zero_velocity(self, x0):
        oracle = MomentumOracle()
        state = oracle.init(x0)
        assert isinstance(state, MomentumState)
        np.testing.assert_allclose(state.velocity, 0.0)

    def test_first_step_steepest_descent(self, x0):
        oracle = MomentumOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        np.testing.assert_allclose(d, -grad, rtol=1e-6)

    def test_velocity_update(self, x0):
        beta = 0.9
        oracle = MomentumOracle(beta=beta)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        info = make_info(x0, new_x, grad, quad_grad(new_x))
        new_state = oracle.update(state, info)
        delta = new_x - x0
        expected_v = beta * 0.0 + (1 - beta) * delta
        np.testing.assert_allclose(new_state.velocity, expected_v, rtol=1e-6)

    def test_direction_includes_momentum(self, x0):
        beta = 0.9
        oracle = MomentumOracle(beta=beta)
        state = MomentumState(velocity=jnp.ones(N, dtype=jnp.float32))
        grad = jnp.ones(N, dtype=jnp.float32)
        d, _ = oracle.direction(x0, grad, state)
        expected = -grad + beta * state.velocity
        np.testing.assert_allclose(d, expected, rtol=1e-6)


class TestPathHistoryMomentum:
    def test_init(self, x0):
        oracle = PathHistoryMomentumOracle(history_size=5)
        state = oracle.init(x0)
        assert isinstance(state, PathHistoryMomentumState)
        assert state.delta_history.shape == (5, N)
        assert int(state.step_count) == 0

    def test_first_step_steepest_descent(self, x0):
        oracle = PathHistoryMomentumOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        np.testing.assert_allclose(d, -grad, rtol=1e-6)

    def test_delta_pushed_newest_first(self, x0):
        oracle = PathHistoryMomentumOracle(history_size=5)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        info = make_info(x0, new_x, grad, quad_grad(new_x))
        new_state = oracle.update(state, info)
        delta = new_x - x0
        np.testing.assert_allclose(new_state.delta_history[0], delta, rtol=1e-6)
        assert int(new_state.step_count) == 1

    def test_step_count_capped(self, x0):
        hs = 3
        oracle = PathHistoryMomentumOracle(history_size=hs)
        state = oracle.init(x0)
        x = x0
        grad = quad_grad(x)
        for _ in range(5):
            new_x = x - 0.05 * grad
            info = make_info(x, new_x, grad, quad_grad(new_x))
            state = oracle.update(state, info)
            x, grad = new_x, quad_grad(new_x)
        assert int(state.step_count) == hs

    def test_weighted_sum(self, x0):
        beta = 0.5
        hs = 3
        oracle = PathHistoryMomentumOracle(history_size=hs, beta=beta)
        deltas = jnp.stack(
            [jnp.ones(N) * 1.0, jnp.ones(N) * 2.0, jnp.ones(N) * 3.0]
        ).astype(jnp.float32)
        state = PathHistoryMomentumState(
            delta_history=deltas, step_count=jnp.asarray(3, dtype=jnp.int32)
        )
        grad = jnp.zeros(N, dtype=jnp.float32)
        d, _ = oracle.direction(x0, grad, state)
        weights = beta ** jnp.arange(hs)
        expected_v = jnp.tensordot(weights, deltas, axes=(0, 0))
        np.testing.assert_allclose(d, expected_v, rtol=1e-6)


class TestSecant:
    def test_init(self, x0):
        oracle = SecantOracle(alpha0=2.0)
        state = oracle.init(x0)
        assert isinstance(state, SecantState)
        assert float(state.alpha) == pytest.approx(2.0)
        assert int(state.step_count) == 0

    def test_first_direction_scaled_descent(self, x0):
        oracle = SecantOracle(alpha0=2.0)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        np.testing.assert_allclose(d, -2.0 * grad, rtol=1e-6)

    def test_bb_step_computation(self, x0):
        oracle = SecantOracle()
        state = oracle.init(x0)
        new_x = x0 + jnp.ones(N)
        grad = jnp.zeros(N, dtype=jnp.float32)
        new_grad = jnp.ones(N, dtype=jnp.float32) * 2.0
        info = make_info(x0, new_x, grad, new_grad)
        new_state = oracle.update(state, info)
        s = new_x - x0
        y = new_grad - grad
        expected_alpha = float(jnp.vdot(s, s) / jnp.vdot(s, y))
        assert float(new_state.alpha) == pytest.approx(expected_alpha, rel=1e-5)

    def test_negative_curvature_keeps_alpha(self, x0):
        oracle = SecantOracle(alpha0=1.5)
        state = oracle.init(x0)
        new_x = x0 + jnp.ones(N)
        grad = jnp.ones(N, dtype=jnp.float32)
        new_grad = jnp.zeros(N, dtype=jnp.float32)
        info = make_info(x0, new_x, grad, new_grad)
        new_state = oracle.update(state, info)

        assert float(new_state.alpha) == pytest.approx(1.5)

    def test_alpha_clipped(self, x0):
        oracle = SecantOracle(alpha_max=1.0)
        state = oracle.init(x0)
        new_x = x0 + jnp.ones(N) * 10.0
        grad = jnp.zeros(N, dtype=jnp.float32)
        new_grad = jnp.ones(N, dtype=jnp.float32) * 1e-3
        info = make_info(x0, new_x, grad, new_grad)
        new_state = oracle.update(state, info)
        assert float(new_state.alpha) <= 1.0 + 1e-6


class TestAnderson:
    def test_first_step_steepest_descent(self, x0):
        oracle = AndersonOracle(window=3)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)

        np.testing.assert_allclose(d, -grad, rtol=1e-6)

    def test_window_one_is_secant_like(self, x0):
        oracle = AndersonOracle(window=1)
        state = oracle.init(x0)

        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        info = make_info(x0, new_x, grad, quad_grad(new_x))
        state = oracle.update(state, info)
        d, _ = oracle.direction(new_x, quad_grad(new_x), state)
        assert np.all(np.isfinite(np.asarray(d)))

    def test_state_history_shape(self, x0):
        oracle = AndersonOracle(window=4)
        state = oracle.init(x0)
        assert state.g_history.shape == (4, N)
        assert state.x_history.shape == (4, N)


class TestAMSQN:
    def test_first_step_steepest_descent(self, x0):
        oracle = AnchoredMultiSecantOracle(window=3)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        np.testing.assert_allclose(d, -grad, rtol=1e-6)

    def test_invalid_kernel_raises(self, x0):
        oracle = AnchoredMultiSecantOracle(window=3, kernel="bogus")
        state = oracle.init(x0)
        grad = quad_grad(x0)
        with pytest.raises(ValueError):
            oracle.direction(x0, grad, state)

    def test_gaussian_kernel(self, x0):
        oracle = AnchoredMultiSecantOracle(window=3, kernel="gaussian")
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        info = make_info(x0, new_x, grad, quad_grad(new_x))
        state = oracle.update(state, info)
        d, _ = oracle.direction(new_x, quad_grad(new_x), state)
        assert np.all(np.isfinite(np.asarray(d)))

    def test_history_roll(self, x0):
        oracle = AnchoredMultiSecantOracle(window=3)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        new_grad = quad_grad(new_x)
        info = make_info(x0, new_x, grad, new_grad)
        new_state = oracle.update(state, info)
        np.testing.assert_allclose(new_state.x_history[0], new_x, rtol=1e-6)
        np.testing.assert_allclose(new_state.g_history[0], new_grad, rtol=1e-6)
        assert int(new_state.step_count) == 1


class TestShampoo:
    def test_init(self, x0):
        oracle = ShampooOracle()
        state = oracle.init(x0)
        assert state.L.shape == (N, N)
        assert int(state.step) == 0

    def test_direction_updates_state(self, x0):
        oracle = ShampooOracle(update_freq=1)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, new_state = oracle.direction(x0, grad, state)
        assert d.shape == x0.shape
        assert int(new_state.step) == 1

    def test_update_noop(self, x0):
        oracle = ShampooOracle()
        state = oracle.init(x0)
        info = make_info(x0, x0, quad_grad(x0), quad_grad(x0))
        assert oracle.update(state, info) is state

    def test_keep_branch_returns_grad(self, x0):

        oracle = ShampooOracle(update_freq=100)
        state = oracle.init(x0)
        grad = quad_grad(x0)

        _, state = oracle.direction(x0, grad, state)
        d, _ = oracle.direction(x0, grad, state)
        np.testing.assert_allclose(d, -grad, rtol=1e-6)


class TestFallback:
    def test_uses_first_valid(self, x0):

        o1 = MomentumOracle()
        o2 = SecantOracle()
        fb = Fallback([o1, o2])
        state = fb.init(x0)
        grad = quad_grad(x0)
        d, _ = fb.direction(x0, grad, state)

        np.testing.assert_allclose(d, -grad, rtol=1e-6)

    def test_init_tuple(self, x0):
        fb = Fallback([MomentumOracle(), SecantOracle()])
        state = fb.init(x0)
        assert isinstance(state, tuple)
        assert len(state) == 2

    def test_update_all(self, x0):
        fb = Fallback([MomentumOracle(), SecantOracle()])
        state = fb.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        info = make_info(x0, new_x, grad, quad_grad(new_x))
        new_state = fb.update(state, info)
        assert isinstance(new_state, tuple)
        assert len(new_state) == 2

    def test_fallback_to_neg_grad_on_nan(self, x0):

        def nan_direction(params, grad, state):
            return jnp.full_like(grad, jnp.nan), state

        bad = Oracle(
            init=lambda p: (),
            direction=nan_direction,
            update=lambda s, i: s,
        )
        fb = Fallback([bad])
        state = fb.init(x0)
        grad = quad_grad(x0)
        d, _ = fb.direction(x0, grad, state)
        np.testing.assert_allclose(d, -grad, rtol=1e-6)

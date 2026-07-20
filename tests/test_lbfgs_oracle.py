"""Unit tests for the LBFGSOracle wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.oracles.lbfgs import LBFGSOracle
from qqn_jax.oracles.oracle import OracleInfo


N = 5


def quad_grad(x):
    A = jnp.diag(jnp.linspace(1.0, 4.0, N))
    return A @ x


@pytest.fixture
def x0():
    return jnp.ones(N, dtype=jnp.float32)


def make_info(params, new_params, grad, new_grad):
    return OracleInfo(
        params=params,
        new_params=new_params,
        grad=grad,
        new_grad=new_grad,
        t=jnp.asarray(1.0),
        step_size=jnp.asarray(1.0, dtype=jnp.float32),
    )


class TestLBFGSOracle:
    def test_init(self, x0):
        oracle = LBFGSOracle(history_size=7)
        state = oracle.init(x0)
        assert state is not None

    def test_first_direction_is_neg_grad(self, x0):
        oracle = LBFGSOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        # empty history: -H∇f = -∇f
        np.testing.assert_allclose(d, -grad, rtol=1e-5)

    def test_direction_is_descent(self, x0):
        oracle = LBFGSOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        d, _ = oracle.direction(x0, grad, state)
        assert float(jnp.vdot(grad, d)) < 0.0

    def test_update_no_probes(self, x0):
        oracle = LBFGSOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        new_grad = quad_grad(new_x)
        info = make_info(x0, new_x, grad, new_grad)
        new_state = oracle.update(state, info)
        assert new_state is not None

    def test_convergence(self, x0):
        oracle = LBFGSOracle()
        state = oracle.init(x0)
        x = x0
        g0 = quad_grad(x)
        g = g0
        for _ in range(50):
            d, state = oracle.direction(x, g, state)
            new_x = x + 0.1 * d
            new_g = quad_grad(new_x)
            info = make_info(x, new_x, g, new_g)
            state = oracle.update(state, info)
            x, g = new_x, new_g
        assert float(jnp.linalg.norm(g)) < float(jnp.linalg.norm(g0))

    def test_jittable(self, x0):
        oracle = LBFGSOracle()
        state = oracle.init(x0)
        grad = quad_grad(x0)

        @jax.jit
        def step(state, x, g):
            d, ns = oracle.direction(x, g, state)
            new_x = x + 0.1 * d
            new_g = quad_grad(new_x)
            info = make_info(x, new_x, g, new_g)
            return oracle.update(ns, info)

        new_state = step(state, x0, grad)
        assert new_state is not None

    def test_probe_replay_cap(self, x0):
        """With probes present, max_probe_replay controls folding."""
        oracle = LBFGSOracle(history_size=10, max_probe_replay=2)
        state = oracle.init(x0)
        grad = quad_grad(x0)
        new_x = x0 - 0.1 * grad
        new_grad = quad_grad(new_x)
        probe_alphas = jnp.asarray([0.02, 0.05, 0.08], dtype=jnp.float32)
        probe_params = jnp.stack([x0 - a * grad for a in [0.02, 0.05, 0.08]])
        probe_grads = jnp.stack([quad_grad(p) for p in probe_params])
        probe_valid = jnp.asarray([True, True, True])
        info = OracleInfo(
            params=x0,
            new_params=new_x,
            grad=grad,
            new_grad=new_grad,
            t=jnp.asarray(1.0),
            step_size=jnp.asarray(0.1, dtype=jnp.float32),
            probe_params=probe_params,
            probe_grads=probe_grads,
            probe_valid=probe_valid,
            probe_alphas=probe_alphas,
        )
        new_state = oracle.update(state, info)
        assert new_state is not None

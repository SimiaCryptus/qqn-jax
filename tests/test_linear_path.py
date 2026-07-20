"""Unit tests for qqn_jax.paths.linear (chord path & linear_refine)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.paths.linear import (
    LINEAR_PATH,
    _linear_offset,
    _linear_velocity,
    linear_refine,
)


def _close(a, b, tol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=tol)


GRAD_DIR = {"w": jnp.array([1.0, -2.0])}
DIRECTION = {"w": jnp.array([4.0, 0.5])}


class TestLinearGeometry:
    def test_offset_is_scaled_direction(self):
        d = _linear_offset(0.5, GRAD_DIR, DIRECTION)
        assert _close(d["w"], 0.5 * DIRECTION["w"])

    def test_offset_ignores_grad_dir(self):
        d1 = _linear_offset(0.3, GRAD_DIR, DIRECTION)
        other_grad = {"w": jnp.array([99.0, -99.0])}
        d2 = _linear_offset(0.3, other_grad, DIRECTION)
        assert _close(d1["w"], d2["w"])

    def test_offset_endpoints(self):
        d0 = _linear_offset(0.0, GRAD_DIR, DIRECTION)
        assert _close(d0["w"], jnp.zeros(2))
        d1 = _linear_offset(1.0, GRAD_DIR, DIRECTION)
        assert _close(d1["w"], DIRECTION["w"])

    def test_velocity_constant(self):
        v0 = _linear_velocity(0.0, GRAD_DIR, DIRECTION)
        v1 = _linear_velocity(1.0, GRAD_DIR, DIRECTION)
        assert _close(v0["w"], DIRECTION["w"])
        assert _close(v1["w"], DIRECTION["w"])

    def test_strategy_is_stateless(self):
        assert LINEAR_PATH.stateful is False
        assert LINEAR_PATH.init_state is None


class TestLinearRefine:
    def _make_inner(self, value, step_size, dtype=jnp.float32):
        return LineSearchResult(
            step_size=jnp.asarray(step_size, dtype),
            new_value=jnp.asarray(value, dtype),
            new_grad=jnp.array([1.0]),
            new_params=jnp.array([1.0]),
            done=jnp.asarray(False),
            probe_params=jnp.zeros((1, 1), dtype),
            probe_grads=jnp.zeros((1, 1), dtype),
            probe_valid=jnp.array([True]),
            probe_values=jnp.array([0.9], dtype),
            probe_alphas=jnp.array([0.5], dtype),
            num_evals=jnp.asarray(2, jnp.int32),
        )

    def test_refine_finds_better_sample(self):
        dtype = jnp.float32

        def eval_at(t):
            p = jnp.array([t])
            v = (t - 0.5) ** 2  # min at t=0.5
            g = jnp.array([2.0 * (t - 0.5)])
            return p, v, g, 0.0

        inner = self._make_inner(value=0.25, step_size=1.0)  # v at t=1
        result = linear_refine(inner, eval_at, dtype, num_samples=8)
        assert float(result.new_value) < float(inner.new_value)
        assert bool(result.done)
        assert int(result.num_evals) == int(inner.num_evals) + 8

    def test_refine_keeps_inner_when_no_improvement(self):
        dtype = jnp.float32

        def eval_at(t):
            p = jnp.array([t])
            v = 10.0 + t  # always worse than inner
            return p, v, jnp.array([1.0]), 0.0

        inner = self._make_inner(value=0.1, step_size=1.0)
        result = linear_refine(inner, eval_at, dtype, num_samples=4)
        assert _close(result.new_value, inner.new_value)
        assert _close(result.step_size, inner.step_size)

    def test_refine_jittable(self):
        dtype = jnp.float32

        def eval_at(t):
            p = jnp.array([t])
            v = (t - 0.5) ** 2
            return p, v, jnp.array([1.0]), 0.0

        inner = self._make_inner(value=0.25, step_size=1.0)
        jitted = jax.jit(lambda: linear_refine(inner, eval_at, dtype, 8))
        result = jitted()
        assert float(result.new_value) < 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

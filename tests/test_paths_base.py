"""Unit tests for qqn_jax.paths.base (PathStrategy & make_evaluator)."""

import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.paths.base import PathStrategy, make_evaluator
from qqn_jax.paths.quadratic import QUADRATIC_PATH
from qqn_jax.paths.linear import LINEAR_PATH


def _close(a, b, tol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=tol)


class _IdentityRegion:
    """A trivial region that leaves candidates untouched."""

    def project(self, params, candidate, region_state):
        del params, region_state
        return candidate


class TestPathStrategyDefaults:
    def test_stateless_defaults(self):
        ps = PathStrategy(offset=lambda t, g, d: d, velocity=lambda t, g, d: d)
        assert ps.init_state is None
        assert ps.observe is None
        assert ps.propose is None
        assert ps.stateful is False


class TestMakeEvaluator:
    def _setup(self, path):
        params = {"w": jnp.array([0.0, 0.0])}
        grad = {"w": jnp.array([2.0, -4.0])}  # grad_dir = -grad
        direction = {"w": jnp.array([1.0, 1.0])}

        def value_and_grad_fn(p):
            # f(w) = 0.5 * ||w||^2 ; grad = w.
            v = 0.5 * jnp.sum(p["w"] ** 2)
            g = {"w": p["w"]}
            return v, g

        region = _IdentityRegion()
        eval_at = make_evaluator(
            value_and_grad_fn,
            params,
            grad,
            direction,
            region,
            None,
            path,
        )
        return eval_at, params, grad, direction

    def test_quadratic_probe_at_endpoint(self):
        eval_at, params, grad, direction = self._setup(QUADRATIC_PATH)
        # At t=1, offset d(1) = direction, so probe = params + direction.
        projected, val, g, slope = eval_at(1.0)
        expected = params["w"] + direction["w"]
        assert _close(projected["w"], expected)
        # value = 0.5 * ||expected||^2
        assert _close(val, 0.5 * jnp.sum(expected**2))

    def test_quadratic_probe_at_origin(self):
        eval_at, params, grad, direction = self._setup(QUADRATIC_PATH)
        projected, val, g, slope = eval_at(0.0)
        # d(0) = 0 -> probe == params
        assert _close(projected["w"], params["w"])
        assert _close(val, 0.0)

    def test_slope_uses_path_velocity(self):
        # slope = <grad(probe), d'(t)>. At t=0, d'(0) = grad_dir = -grad.
        eval_at, params, grad, direction = self._setup(QUADRATIC_PATH)
        _, _, g, slope = eval_at(0.0)
        grad_dir = -grad["w"]
        expected_slope = float(jnp.sum(g["w"] * grad_dir))
        assert _close(slope, expected_slope)

    def test_linear_probe(self):
        eval_at, params, grad, direction = self._setup(LINEAR_PATH)
        projected, val, g, slope = eval_at(0.5)
        expected = params["w"] + 0.5 * direction["w"]
        assert _close(projected["w"], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

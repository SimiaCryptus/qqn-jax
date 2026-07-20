"""Unit tests for qqn_jax.regularizers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.regularizers import (
    l1_penalty,
    l2_penalty,
    elastic_net_penalty,
    quantization_delta_penalty,
    select_weights,
    round_to_grid,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# round_to_grid
# ---------------------------------------------------------------------------
class TestRoundToGrid:
    def test_requires_bits_or_step(self):
        x = jnp.array([0.0, 0.5])
        with pytest.raises(ValueError):
            round_to_grid(x)

    def test_step_rounding_basic(self):
        x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        out = round_to_grid(x, step=0.5, lo=-1.0, hi=1.0)
        np.testing.assert_allclose(np.asarray(out), np.asarray(x), atol=1e-12)

    def test_step_rounds_to_nearest(self):
        x = jnp.array([0.24, 0.26, -0.24, -0.26])
        out = round_to_grid(x, step=0.5, lo=-1.0, hi=1.0)
        # grid points: -1, -0.5, 0, 0.5, 1
        expected = np.array([0.0, 0.5, 0.0, -0.5])
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)

    def test_bits_grid_spacing(self):
        # bits=1 => levels = 2**1 - 1 = 1, delta = (hi-lo)/1 = 2.0
        x = jnp.array([-1.0, -0.4, 0.6, 1.0])
        out = round_to_grid(x, bits=1, lo=-1.0, hi=1.0)
        # grid points only at -1 and 1
        expected = np.array([-1.0, -1.0, 1.0, 1.0])
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)

    def test_clipping(self):
        x = jnp.array([-5.0, 5.0])
        out = round_to_grid(x, step=0.5, lo=-1.0, hi=1.0)
        expected = np.array([-1.0, 1.0])
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)

    def test_output_on_grid(self):
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.uniform(-1.0, 1.0, size=100))
        out = round_to_grid(x, bits=4, lo=-1.0, hi=1.0)
        delta = 2.0 / (2**4 - 1)
        k = (np.asarray(out) + 1.0) / delta
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)

    def test_preserves_dtype(self):
        x = jnp.array([0.3], dtype=jnp.float32)
        out = round_to_grid(x, step=0.5)
        assert out.dtype == jnp.float32

    def test_jit_compatible(self):
        f = jax.jit(lambda x: round_to_grid(x, step=0.5))
        out = f(jnp.array([0.3, 0.7]))
        assert out.shape == (2,)


# ---------------------------------------------------------------------------
# select_weights
# ---------------------------------------------------------------------------
class TestSelectWeights:
    def test_mlp_list_of_dicts(self):
        params = [
            {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)},
            {"w": jnp.ones((3, 1)), "b": jnp.zeros(1)},
        ]
        ws = select_weights(params)
        assert len(ws) == 2
        assert ws[0].shape == (2, 3)
        assert ws[1].shape == (3, 1)

    def test_custom_key(self):
        params = [{"kernel": jnp.ones((2, 2))}]
        ws = select_weights(params, key="kernel")
        assert len(ws) == 1
        assert ws[0].shape == (2, 2)

    def test_missing_key_skipped(self):
        params = [{"w": jnp.ones((2, 2))}, {"b": jnp.zeros(2)}]
        ws = select_weights(params)
        assert len(ws) == 1

    def test_flat_array_falls_back_to_leaves(self):
        params = jnp.ones((5,))
        ws = select_weights(params)
        assert len(ws) == 1
        assert ws[0].shape == (5,)

    def test_generic_pytree(self):
        params = {"a": jnp.ones(2), "b": jnp.ones(3)}
        ws = select_weights(params)
        assert len(ws) == 2

    def test_empty_list(self):
        ws = select_weights([])
        assert ws == []


# ---------------------------------------------------------------------------
# l1_penalty
# ---------------------------------------------------------------------------
class TestL1Penalty:
    def test_basic(self):
        params = jnp.array([1.0, -2.0, 3.0])
        out = l1_penalty(params, scale=1.0)
        assert float(out) == pytest.approx(6.0)

    def test_scale(self):
        params = jnp.array([1.0, -1.0])
        out = l1_penalty(params, scale=0.5)
        assert float(out) == pytest.approx(1.0)

    def test_zero_params(self):
        params = jnp.zeros((4,))
        out = l1_penalty(params, scale=1.0)
        assert float(out) == pytest.approx(0.0)

    def test_pytree(self):
        params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([-3.0])}
        out = l1_penalty(params, scale=1.0)
        assert float(out) == pytest.approx(6.0)

    def test_weights_only(self):
        params = [{"w": jnp.array([[1.0, -2.0]]), "b": jnp.array([100.0])}]
        out = l1_penalty(params, scale=1.0, weights_only=True)
        assert float(out) == pytest.approx(3.0)

    def test_grad(self):
        params = jnp.array([2.0, -3.0])
        g = jax.grad(lambda p: l1_penalty(p, scale=1.0))(params)
        np.testing.assert_allclose(np.asarray(g), np.array([1.0, -1.0]), atol=1e-9)

    def test_jit(self):
        f = jax.jit(lambda p: l1_penalty(p, scale=2.0))
        out = f(jnp.array([1.0, 1.0]))
        assert float(out) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# l2_penalty
# ---------------------------------------------------------------------------
class TestL2Penalty:
    def test_basic(self):
        params = jnp.array([1.0, 2.0, 3.0])
        out = l2_penalty(params, scale=1.0)
        assert float(out) == pytest.approx(14.0)

    def test_scale(self):
        params = jnp.array([2.0])
        out = l2_penalty(params, scale=0.5)
        assert float(out) == pytest.approx(2.0)

    def test_zero(self):
        out = l2_penalty(jnp.zeros(3), scale=1.0)
        assert float(out) == pytest.approx(0.0)

    def test_weights_only(self):
        params = [{"w": jnp.array([[2.0]]), "b": jnp.array([10.0])}]
        out = l2_penalty(params, scale=1.0, weights_only=True)
        assert float(out) == pytest.approx(4.0)

    def test_grad(self):
        params = jnp.array([2.0, -3.0])
        g = jax.grad(lambda p: l2_penalty(p, scale=1.0))(params)
        np.testing.assert_allclose(np.asarray(g), np.array([4.0, -6.0]), atol=1e-9)

    def test_pytree(self):
        params = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
        out = l2_penalty(params, scale=1.0)
        assert float(out) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# elastic_net_penalty
# ---------------------------------------------------------------------------
class TestElasticNetPenalty:
    def test_basic(self):
        params = jnp.array([1.0, -2.0])
        # l1 = 3, l2 = 5
        out = elastic_net_penalty(params, l1=1.0, l2=1.0)
        assert float(out) == pytest.approx(8.0)

    def test_separate_scales(self):
        params = jnp.array([2.0])
        # l1: 0.5 * 2 = 1, l2: 0.25 * 4 = 1
        out = elastic_net_penalty(params, l1=0.5, l2=0.25)
        assert float(out) == pytest.approx(2.0)

    def test_equals_sum_of_parts(self):
        params = jnp.array([1.5, -2.5, 0.3])
        a = elastic_net_penalty(params, l1=0.1, l2=0.2)
        b = l1_penalty(params, 0.1) + l2_penalty(params, 0.2)
        assert float(a) == pytest.approx(float(b))

    def test_grad(self):
        params = jnp.array([1.0])
        g = jax.grad(lambda p: elastic_net_penalty(p, l1=1.0, l2=1.0))(params)
        # d/dp (|p| + p^2) = sign(p) + 2p = 1 + 2 = 3
        assert float(g[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# quantization_delta_penalty
# ---------------------------------------------------------------------------
class TestQuantizationDeltaPenalty:
    def test_requires_bits_or_step(self):
        with pytest.raises(ValueError):
            quantization_delta_penalty(jnp.array([0.0]))

    def test_zero_on_grid(self):
        # values already on grid -> zero penalty
        params = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        out = quantization_delta_penalty(params, scale=1.0, step=0.5)
        assert float(out) == pytest.approx(0.0, abs=1e-9)

    def test_midpoint_max(self):
        # midpoint 0.25 between 0 and 0.5 -> delta = 0.25
        params = jnp.array([0.25])
        out = quantization_delta_penalty(params, scale=1.0, step=0.5)
        assert float(out) == pytest.approx(0.25, abs=1e-9)

    def test_matches_round_to_grid(self):
        rng = np.random.default_rng(1)
        params = jnp.asarray(rng.uniform(-1.0, 1.0, size=50))
        out = quantization_delta_penalty(params, scale=1.0, bits=4)
        clipped = jnp.clip(params, -1.0, 1.0)
        grid = round_to_grid(params, bits=4)
        expected = float(jnp.sum(jnp.abs(clipped - grid)))
        assert float(out) == pytest.approx(expected, abs=1e-9)

    def test_step_overrides_bits(self):
        params = jnp.array([0.25])
        out_step = quantization_delta_penalty(params, scale=1.0, step=0.5, bits=8)
        out_only = quantization_delta_penalty(params, scale=1.0, step=0.5)
        assert float(out_step) == pytest.approx(float(out_only))

    def test_scale(self):
        params = jnp.array([0.25])
        out = quantization_delta_penalty(params, scale=2.0, step=0.5)
        assert float(out) == pytest.approx(0.5, abs=1e-9)

    def test_clipping(self):
        # value beyond range gets clipped to grid endpoint -> zero delta
        params = jnp.array([5.0])
        out = quantization_delta_penalty(params, scale=1.0, step=0.5)
        assert float(out) == pytest.approx(0.0, abs=1e-9)

    def test_weights_only(self):
        params = [{"w": jnp.array([[0.25]]), "b": jnp.array([0.25])}]
        out = quantization_delta_penalty(params, scale=1.0, step=0.5, weights_only=True)
        assert float(out) == pytest.approx(0.25, abs=1e-9)

    def test_grad_is_finite(self):
        params = jnp.array([0.1, 0.3, -0.2])
        g = jax.grad(lambda p: quantization_delta_penalty(p, scale=1.0, step=0.5))(
            params
        )
        assert np.all(np.isfinite(np.asarray(g)))

    def test_jit(self):
        f = jax.jit(lambda p: quantization_delta_penalty(p, scale=1.0, bits=4))
        out = f(jnp.array([0.3, 0.7]))
        assert jnp.isfinite(out)

    def test_non_symmetric_range(self):
        # range [0, 1], step 0.25
        params = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        out = quantization_delta_penalty(params, scale=1.0, step=0.25, lo=0.0, hi=1.0)
        assert float(out) == pytest.approx(0.0, abs=1e-9)

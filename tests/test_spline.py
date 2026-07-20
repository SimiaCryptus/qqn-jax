"""Comprehensive unit tests for qqn_jax.paths.spline.

These tests cover:
  * The parameter-space geometry (offset / velocity) and its distinction
    from the accumulating Hermite model.
  * Cubic Hermite basis correctness (endpoint interpolation & derivatives).
  * Tangent orientation heuristic.
  * Single-segment evaluation and stationary-point solving (quadratic,
    linear-fallback, and degenerate branches).
  * Multi-point proposal from padded / masked buffers.
  * The stateful SplineState memory (init / observe / propose).
  * The full spline_refine accumulation loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.paths.spline import (
    SPLINE_PATH,
    _orient_tangents,
    _spline_offset,
    _spline_velocity,
    hermite_basis,
    propose_from_points,
    propose_step,
    segment_candidates,
    segment_eval,
    spline_init,
    spline_observe,
    spline_propose,
    spline_refine,
)


def _close(a, b, tol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=tol)


GRAD_DIR = {"w": jnp.array([1.0, -2.0]), "b": jnp.array(3.0)}
DIRECTION = {"w": jnp.array([4.0, 0.5]), "b": jnp.array(-1.0)}


class TestSplineGeometry:
    def test_offset_endpoints(self):

        d0 = _spline_offset(0.0, GRAD_DIR, DIRECTION)
        assert _close(d0["w"], jnp.zeros(2))
        assert _close(d0["b"], 0.0)

        d1 = _spline_offset(1.0, GRAD_DIR, DIRECTION)
        assert _close(d1["w"], DIRECTION["w"])
        assert _close(d1["b"], DIRECTION["b"])

    def test_offset_midpoint(self):
        t = 0.5
        a = t * (1.0 - t)
        b = t * t
        d = _spline_offset(t, GRAD_DIR, DIRECTION)
        expected_w = a * GRAD_DIR["w"] + b * DIRECTION["w"]
        assert _close(d["w"], expected_w)

    def test_velocity_endpoints(self):

        v0 = _spline_velocity(0.0, GRAD_DIR, DIRECTION)
        assert _close(v0["w"], GRAD_DIR["w"])

        v1 = _spline_velocity(1.0, GRAD_DIR, DIRECTION)
        expected = 2.0 * DIRECTION["w"] - GRAD_DIR["w"]
        assert _close(v1["w"], expected)

    def test_velocity_is_offset_derivative(self):

        t = 0.37
        eps = 1e-4
        dp = _spline_offset(t + eps, GRAD_DIR, DIRECTION)
        dm = _spline_offset(t - eps, GRAD_DIR, DIRECTION)
        num = (dp["w"] - dm["w"]) / (2 * eps)
        ana = _spline_velocity(t, GRAD_DIR, DIRECTION)["w"]
        assert _close(num, ana, tol=1e-3)

    def test_spline_path_strategy_stateful(self):
        assert SPLINE_PATH.stateful is True
        assert SPLINE_PATH.init_state is not None
        assert SPLINE_PATH.observe is not None
        assert SPLINE_PATH.propose is not None


class TestHermiteBasis:
    def test_endpoint_values(self):

        h00, h10, h01, h11 = hermite_basis(0.0)
        assert _close(h00, 1.0)
        assert _close(h10, 0.0)
        assert _close(h01, 0.0)
        assert _close(h11, 0.0)

        h00, h10, h01, h11 = hermite_basis(1.0)
        assert _close(h00, 0.0)
        assert _close(h10, 0.0)
        assert _close(h01, 1.0)
        assert _close(h11, 0.0)

    def test_derivative_endpoints(self):

        eps = 1e-5

        def dbasis(s, idx):
            bp = hermite_basis(s + eps)[idx]
            bm = hermite_basis(s - eps)[idx]
            return (bp - bm) / (2 * eps)

        assert _close(dbasis(0.0, 1), 1.0, tol=1e-3)
        assert _close(dbasis(1.0, 3), 1.0, tol=1e-3)
        assert _close(dbasis(0.0, 0), 0.0, tol=1e-3)
        assert _close(dbasis(1.0, 2), 0.0, tol=1e-3)

    def test_partition_of_position_basis(self):

        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            h00, _, h01, _ = hermite_basis(s)
            assert _close(h00 + h01, 1.0)


class TestOrientTangents:
    def test_reflect_against_secant(self):

        m0, m1 = _orient_tangents(-2.0, 3.0, delta=1.0)
        assert _close(m0, 2.0)
        assert _close(m1, 3.0)

    def test_no_reflect_when_aligned(self):
        m0, m1 = _orient_tangents(1.0, 2.0, delta=1.0)
        assert _close(m0, 1.0)
        assert _close(m1, 2.0)

    def test_no_reflect_when_delta_zero(self):

        m0, m1 = _orient_tangents(-5.0, 5.0, delta=0.0)
        assert _close(m0, -5.0)
        assert _close(m1, 5.0)

    def test_negative_delta(self):

        m0, m1 = _orient_tangents(3.0, -1.0, delta=-2.0)
        assert _close(m0, -3.0)
        assert _close(m1, -1.0)


class TestSegmentEval:
    def test_endpoint_interpolation(self):

        t0, f0, m0 = 0.0, 1.0, 1.0
        t1, f1, m1 = 1.0, 5.0, 1.0
        assert _close(segment_eval(t0, t0, f0, m0, t1, f1, m1), f0)
        assert _close(segment_eval(t1, t0, f0, m0, t1, f1, m1), f1)

    def test_matches_manual_hermite(self):

        t0, f0, m0 = 0.0, 0.0, 1.0
        t1, f1, m1 = 2.0, 4.0, 1.0
        h = t1 - t0
        for t in [0.5, 1.0, 1.5]:
            s = (t - t0) / h
            h00, h10, h01, h11 = hermite_basis(s)
            expected = h00 * f0 + h10 * h * m0 + h01 * f1 + h11 * h * m1
            assert _close(segment_eval(t, t0, f0, m0, t1, f1, m1), expected)

    def test_vectorized_eval(self):
        t0, f0, m0 = 0.0, 1.0, 0.5
        t1, f1, m1 = 1.0, 2.0, 0.5
        ts = jnp.array([0.0, 0.5, 1.0])
        out = segment_eval(ts, t0, f0, m0, t1, f1, m1)
        assert out.shape == (3,)
        assert _close(out[0], f0)
        assert _close(out[2], f1)


class TestSegmentCandidates:
    def test_finds_interior_minimum(self):

        t0, f0, m0 = 0.0, 1.0, -1.0
        t1, f1, m1 = 1.0, 1.0, 1.0
        t_cand, f_cand, valid = segment_candidates(t0, f0, m0, t1, f1, m1)
        assert t_cand.shape == (2,)
        assert bool(jnp.any(valid))

        idx = int(jnp.argmin(jnp.where(valid, f_cand, jnp.inf)))
        assert _close(t_cand[idx], 0.5, tol=1e-4)
        assert f_cand[idx] < f0

    def test_candidate_matches_segment_eval(self):
        t0, f0, m0 = 0.0, 2.0, -1.5
        t1, f1, m1 = 1.0, 2.0, 1.5
        t_cand, f_cand, valid = segment_candidates(t0, f0, m0, t1, f1, m1)
        for i in range(2):
            if bool(valid[i]):
                ev = segment_eval(t_cand[i], t0, f0, m0, t1, f1, m1)
                assert _close(ev, f_cand[i], tol=1e-4)

    def test_stationary_point_has_zero_derivative(self):
        t0, f0, m0 = 0.0, 3.0, -2.0
        t1, f1, m1 = 1.0, 3.0, 2.0
        t_cand, f_cand, valid = segment_candidates(t0, f0, m0, t1, f1, m1)
        eps = 1e-5
        for i in range(2):
            if bool(valid[i]):
                tc = float(t_cand[i])
                if eps < tc < 1 - eps:
                    fp = segment_eval(tc + eps, t0, f0, m0, t1, f1, m1)
                    fm = segment_eval(tc - eps, t0, f0, m0, t1, f1, m1)
                    deriv = (fp - fm) / (2 * eps)
                    assert _close(deriv, 0.0, tol=1e-2)

    def test_out_of_range_marked_invalid(self):

        t0, f0, m0 = 0.0, 0.0, 1.0
        t1, f1, m1 = 1.0, 3.0, 1.0
        _, f_cand, valid = segment_candidates(t0, f0, m0, t1, f1, m1)

        for i in range(2):
            if not bool(valid[i]):
                assert jnp.isinf(f_cand[i])

    def test_linear_fallback(self):

        t0, f0, m0 = 0.0, 1.0, 1.0
        t1, f1, m1 = 1.0, 1.0, -1.0
        t_cand, f_cand, valid = segment_candidates(t0, f0, m0, t1, f1, m1)

        assert t_cand.shape == (2,)

        assert not bool(valid[1])

    def test_no_nan_in_output(self):

        t0, f0, m0 = 0.0, 1.0, -0.5
        t1, f1, m1 = 1.0, 1.0, 0.5
        t_cand, f_cand, valid = segment_candidates(t0, f0, m0, t1, f1, m1)
        masked_t = jnp.where(valid, t_cand, 0.0)
        masked_f = jnp.where(valid, f_cand, 0.0)
        assert not bool(jnp.any(jnp.isnan(masked_t)))
        assert not bool(jnp.any(jnp.isnan(masked_f)))


class TestPropose:
    def test_propose_step_simple_bowl(self):

        ts = jnp.array([0.0, 0.5, 1.0])
        fs = jnp.array([1.0, 0.2, 1.0])
        ms = jnp.array([-1.0, 0.0, 1.0])
        t_best, f_best, found = propose_step(ts, fs, ms)
        assert bool(found)
        assert 0.0 <= float(t_best) <= 1.0

    def test_propose_returns_lowest_predicted(self):

        ts = jnp.array([0.0, 1.0, 2.0])
        fs = jnp.array([1.0, 0.5, 1.0])
        ms = jnp.array([-1.0, 0.0, 1.0])
        t_best, f_best, found = propose_step(ts, fs, ms)
        assert bool(found)
        assert float(f_best) < jnp.inf

    def test_padded_buffer_ignores_invalid(self):

        ts = jnp.array([0.0, 1.0, 99.0, 99.0])
        fs = jnp.array([1.0, 1.0, jnp.inf, jnp.inf])
        ms = jnp.array([-1.0, 1.0, 0.0, 0.0])
        valid = jnp.array([True, True, False, False])
        t_best, f_best, found = propose_from_points(ts, fs, ms, valid)
        assert bool(found)

        assert 0.0 <= float(t_best) <= 1.0

    def test_unsorted_input_handled(self):
        ts = jnp.array([1.0, 0.0, 0.5])
        fs = jnp.array([1.0, 1.0, 0.2])
        ms = jnp.array([1.0, -1.0, 0.0])
        t_best, f_best, found = propose_step(ts, fs, ms)
        assert bool(found)
        assert 0.0 <= float(t_best) <= 1.0

    def test_no_valid_segment_returns_found_false(self):

        ts = jnp.array([0.0, 5.0, 5.0])
        fs = jnp.array([1.0, jnp.inf, jnp.inf])
        ms = jnp.array([-1.0, 0.0, 0.0])
        valid = jnp.array([True, False, False])
        _, _, found = propose_from_points(ts, fs, ms, valid)
        assert not bool(found)

    def test_duplicate_t_rejected(self):

        ts = jnp.array([0.0, 0.0])
        fs = jnp.array([1.0, 1.0])
        ms = jnp.array([-1.0, 1.0])
        _, _, found = propose_step(ts, fs, ms)
        assert not bool(found)


class TestSplineState:
    def test_init_shapes(self):
        st = spline_init(GRAD_DIR, DIRECTION, capacity=8)
        assert st.ts.shape == (8,)
        assert st.fs.shape == (8,)
        assert st.ms.shape == (8,)
        assert st.valid.shape == (8,)
        assert int(st.num_points) == 0
        assert not bool(jnp.any(st.valid))
        assert bool(jnp.all(jnp.isinf(st.fs)))

    def test_observe_records_point(self):
        st = spline_init(GRAD_DIR, DIRECTION, capacity=4)
        st = spline_observe(st, 0.5, 0.3, -1.0)
        assert int(st.num_points) == 1
        assert bool(st.valid[0])
        assert _close(st.ts[0], 0.5)
        assert _close(st.fs[0], 0.3)
        assert _close(st.ms[0], -1.0)

    def test_observe_multiple(self):
        st = spline_init(GRAD_DIR, DIRECTION, capacity=4)
        st = spline_observe(st, 0.0, 1.0, -1.0)
        st = spline_observe(st, 1.0, 1.0, 1.0)
        assert int(st.num_points) == 2
        assert bool(st.valid[0]) and bool(st.valid[1])

    def test_observe_saturates_at_capacity(self):
        st = spline_init(GRAD_DIR, DIRECTION, capacity=2)
        st = spline_observe(st, 0.0, 1.0, -1.0)
        st = spline_observe(st, 1.0, 1.0, 1.0)

        st = spline_observe(st, 0.5, 0.1, 0.0)
        assert int(st.num_points) == 3

        assert _close(st.ts[1], 0.5)
        assert _close(st.fs[1], 0.1)

    def test_propose_from_state(self):
        st = spline_init(GRAD_DIR, DIRECTION, capacity=4)
        st = spline_observe(st, 0.0, 1.0, -1.0)
        st = spline_observe(st, 1.0, 1.0, 1.0)
        t_best, f_best, found = spline_propose(st)
        assert bool(found)
        assert 0.0 <= float(t_best) <= 1.0

    def test_propose_empty_state(self):
        st = spline_init(GRAD_DIR, DIRECTION, capacity=4)
        _, _, found = spline_propose(st)
        assert not bool(found)


class TestSplineRefine:
    def _make_inner(self, params, value, grad, step_size, dtype=jnp.float32):
        """Construct a minimal LineSearchResult with a couple of probes."""
        n_probes = 2
        dim = params.shape[0]
        probe_alphas = jnp.array([0.3, 0.7], dtype)

        probe_grads = jnp.stack(
            [
                jnp.full((dim,), 0.5, dtype),
                jnp.full((dim,), -0.5, dtype),
            ]
        )
        probe_values = jnp.array([0.9, 0.8], dtype)
        probe_valid = jnp.array([True, True])
        probe_params = jnp.zeros((n_probes, dim), dtype)
        return LineSearchResult(
            step_size=jnp.asarray(step_size, dtype),
            new_value=jnp.asarray(value, dtype),
            new_grad=grad,
            new_params=params,
            done=jnp.asarray(False),
            probe_params=probe_params,
            probe_grads=probe_grads,
            probe_valid=probe_valid,
            probe_values=probe_values,
            probe_alphas=probe_alphas,
            num_evals=jnp.asarray(3, jnp.int32),
        )

    def test_refine_reduces_value_on_bowl(self):

        grad_dir = jnp.array([1.0])
        direction = jnp.array([1.0])
        dtype = jnp.float32

        def objective(t):
            return (t - 0.5) ** 2 + 0.1

        def eval_at(t):

            p = jnp.array([t])
            v = objective(t)

            slope = 2.0 * (t - 0.5)
            g = jnp.array([slope])
            return p, v, g, slope

        inner = self._make_inner(
            params=jnp.array([1.0]),
            value=objective(1.0),
            grad=jnp.array([1.0]),
            step_size=1.0,
            dtype=dtype,
        )

        result = spline_refine(
            inner=inner,
            eval_at=eval_at,
            path=SPLINE_PATH,
            grad_dir=grad_dir,
            direction=direction,
            f0=objective(0.0),
            slope0=2.0 * (0.0 - 0.5),
            dtype=dtype,
            rounds=4,
        )

        assert float(result.new_value) < float(inner.new_value)
        assert float(result.new_value) <= 0.35
        assert bool(result.done)

        assert int(result.num_evals) == int(inner.num_evals) + 4

    def test_refine_preserves_probes(self):
        grad_dir = jnp.array([1.0])
        direction = jnp.array([1.0])
        dtype = jnp.float32

        def eval_at(t):
            p = jnp.array([t])
            v = (t - 0.5) ** 2
            slope = 2.0 * (t - 0.5)
            g = jnp.array([slope])
            return p, v, g, slope

        inner = self._make_inner(
            params=jnp.array([1.0]),
            value=0.25,
            grad=jnp.array([1.0]),
            step_size=1.0,
        )
        result = spline_refine(
            inner=inner,
            eval_at=eval_at,
            path=SPLINE_PATH,
            grad_dir=grad_dir,
            direction=direction,
            f0=0.25,
            slope0=-1.0,
            dtype=dtype,
            rounds=2,
        )
        assert _close(result.probe_alphas, inner.probe_alphas)
        assert _close(result.probe_values, inner.probe_values)

    def test_refine_no_improvement_keeps_inner(self):

        grad_dir = jnp.array([1.0])
        direction = jnp.array([1.0])
        dtype = jnp.float32

        def eval_at(t):
            p = jnp.array([t])
            v = -t
            slope = -1.0
            g = jnp.array([slope])
            return p, v, g, slope

        inner = self._make_inner(
            params=jnp.array([1.0]),
            value=-1.0,
            grad=jnp.array([-1.0]),
            step_size=1.0,
        )
        result = spline_refine(
            inner=inner,
            eval_at=eval_at,
            path=SPLINE_PATH,
            grad_dir=grad_dir,
            direction=direction,
            f0=0.0,
            slope0=-1.0,
            dtype=dtype,
            rounds=3,
        )

        assert float(result.new_value) <= float(inner.new_value) + 1e-6

    def test_refine_jittable(self):

        grad_dir = jnp.array([1.0])
        direction = jnp.array([1.0])
        dtype = jnp.float32

        def eval_at(t):
            p = jnp.array([t])
            v = (t - 0.5) ** 2 + 0.1
            slope = 2.0 * (t - 0.5)
            g = jnp.array([slope])
            return p, v, g, slope

        inner = self._make_inner(
            params=jnp.array([1.0]),
            value=0.35,
            grad=jnp.array([1.0]),
            step_size=1.0,
        )

        def run(f0, slope0):
            return spline_refine(
                inner=inner,
                eval_at=eval_at,
                path=SPLINE_PATH,
                grad_dir=grad_dir,
                direction=direction,
                f0=f0,
                slope0=slope0,
                dtype=dtype,
                rounds=3,
            )

        jitted = jax.jit(run)
        result = jitted(jnp.asarray(0.35), jnp.asarray(-1.0))
        assert float(result.new_value) < 0.35


class TestJitSafety:
    def test_segment_candidates_jit(self):
        f = jax.jit(segment_candidates)
        t_cand, f_cand, valid = f(0.0, 1.0, -1.0, 1.0, 1.0, 1.0)
        assert t_cand.shape == (2,)

    def test_propose_step_jit(self):
        f = jax.jit(propose_step)
        ts = jnp.array([0.0, 0.5, 1.0])
        fs = jnp.array([1.0, 0.2, 1.0])
        ms = jnp.array([-1.0, 0.0, 1.0])
        t_best, f_best, found = f(ts, fs, ms)
        assert bool(found)

    def test_hermite_basis_vmap(self):
        ss = jnp.linspace(0.0, 1.0, 5)
        out = jax.vmap(hermite_basis)(ss)
        assert len(out) == 4
        assert out[0].shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

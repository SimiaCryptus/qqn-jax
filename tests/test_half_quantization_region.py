"""Unit tests for HalfQuantizationRegion and half-quantization lattice helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax import QQN, HalfQuantizationRegion
from qqn_jax.regions.half_quantization import (
    half_quant_bounds,
    half_quant_cell_width,
    is_on_half_quant_boundary,
    round_to_half_quant,
    round_toward_zero_half_quant,
)


@pytest.mark.parametrize("qtype", ["bfloat16", "float16", "uniform"])
def test_bounds_bracket_and_round_with_zero_error(qtype):
    """Both endpoints [q_lo, q_hi] must bracket x and round with exact 0 error."""
    rng = np.random.default_rng(123)
    x = jnp.asarray(rng.uniform(-0.85, 0.85, size=(256,)), dtype=jnp.float32)
    # Perturb low mantissa bits so values are genuinely 32-bit interior points
    x = x + jnp.asarray(1.234567e-6, dtype=jnp.float32)

    q_lo, q_hi = half_quant_bounds(x, qtype=qtype, bits=16, lo=-1.0, hi=1.0)
    assert jnp.all(q_lo <= x)
    assert jnp.all(x <= q_hi)
    assert jnp.any(q_lo < q_hi)

    # Every boundary point MUST round with exact zero error to target 16-bit grid!
    assert jnp.all(is_on_half_quant_boundary(q_lo, qtype=qtype, bits=16, lo=-1.0, hi=1.0))
    assert jnp.all(is_on_half_quant_boundary(q_hi, qtype=qtype, bits=16, lo=-1.0, hi=1.0))

    lo_round = round_to_half_quant(q_lo, qtype=qtype, bits=16, lo=-1.0, hi=1.0)
    hi_round = round_to_half_quant(q_hi, qtype=qtype, bits=16, lo=-1.0, hi=1.0)
    np.testing.assert_array_equal(np.asarray(q_lo), np.asarray(lo_round))
    np.testing.assert_array_equal(np.asarray(q_hi), np.asarray(hi_round))


@pytest.mark.parametrize("qtype", ["bfloat16", "float16", "uniform"])
def test_clamping_lands_on_exact_16bit_boundary_and_preserves_sign(qtype):
    """Large steps in either direction clamp onto exact 16-bit boundaries without sign flip."""
    x = jnp.array([0.1234567, -0.3456789, 1.2345e-4, -9.8765e-4, 0.0], dtype=jnp.float32)
    region = HalfQuantizationRegion(qtype=qtype, bits=16, lo=-1.0, hi=1.0, boundary_mode="sticky")
    st = region.init(x)

    # Push strongly toward -1 and +1
    out_down = region.project(x, x - 1.0, st)
    out_up = region.project(x, x + 1.0, st)

    assert jnp.all(is_on_half_quant_boundary(out_down, qtype=qtype, bits=16, lo=-1.0, hi=1.0))
    assert jnp.all(is_on_half_quant_boundary(out_up, qtype=qtype, bits=16, lo=-1.0, hi=1.0))

    # Sign orthant refinement: positive stays >= 0, negative stays <= 0, zero stays == 0
    assert float(out_down[0]) > 0.0 and float(out_up[0]) > 0.0
    assert float(out_down[1]) < 0.0 and float(out_up[1]) < 0.0
    assert float(out_down[4]) == 0.0 and float(out_up[4]) == 0.0


def test_sticky_vs_owl_vs_owl_snap_boundary_modes():
    """Verify sticky locks on boundary, owl enters adjacent cell, owl_snap steps lattice-wise."""
    # Uniform grid with step=0.25 -> grid points {..., -0.5, -0.25, 0.0, 0.25, 0.5, ...}
    p_on_grid = jnp.array([0.5, -0.5, 0.0], dtype=jnp.float32)
    cand_small = jnp.array([0.40, -0.40, 0.20], dtype=jnp.float32)  # inside next cell
    cand_large = jnp.array([0.10, -0.10, -0.20], dtype=jnp.float32)  # past next boundary

    r_sticky = HalfQuantizationRegion(
        qtype="uniform", step=0.25, lo=-1.0, hi=1.0, boundary_mode="sticky"
    )
    r_owl = HalfQuantizationRegion(
        qtype="uniform", step=0.25, lo=-1.0, hi=1.0, boundary_mode="owl"
    )
    r_snap = HalfQuantizationRegion(
        qtype="uniform", step=0.25, lo=-1.0, hi=1.0, boundary_mode="owl_snap"
    )

    # Sticky stays locked at [0.5, -0.5, 0.0]
    np.testing.assert_allclose(
        np.asarray(r_sticky.project(p_on_grid, cand_large, ())),
        np.array([0.5, -0.5, 0.0]),
    )

    # OWL enters [0.25, 0.5] and [-0.5, -0.25] (moving smoothly for small step, clamping at 0.25 / -0.25 for large)
    np.testing.assert_allclose(
        np.asarray(r_owl.project(p_on_grid, cand_small, ())),
        np.array([0.40, -0.40, 0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(r_owl.project(p_on_grid, cand_large, ())),
        np.array([0.25, -0.25, 0.0]),
    )

    # OWL-Snap stays on [0.5, -0.5] when step doesn't reach 0.25/-0.25, and snaps to [0.25, -0.25] when it does!
    np.testing.assert_allclose(
        np.asarray(r_snap.project(p_on_grid, cand_small, ())),
        np.array([0.5, -0.5, 0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(r_snap.project(p_on_grid, cand_large, ())),
        np.array([0.25, -0.25, 0.0]),
    )


def test_round_toward_zero_minimizes_l1_on_cell():
    rng = np.random.default_rng(7)
    p = {"w": jnp.asarray(rng.standard_normal((64,)).astype(np.float32) * 0.2)}
    q_rtz = round_toward_zero_half_quant(p, qtype="bfloat16")
    q_rtn = round_to_half_quant(p, qtype="bfloat16")
    assert float(jnp.sum(jnp.abs(q_rtz["w"]))) < float(jnp.sum(jnp.abs(p["w"])))
    assert float(jnp.sum(jnp.abs(q_rtz["w"]))) < float(jnp.sum(jnp.abs(q_rtn["w"])))
    assert jnp.all(is_on_half_quant_boundary(q_rtz["w"], qtype="bfloat16"))


def test_jit_vmap_grad_and_qqn_solver_integration():
    region = HalfQuantizationRegion(qtype="bfloat16", boundary_mode="sticky")
    x = jnp.array([0.1234567, -0.7654321], dtype=jnp.float32)
    w = half_quant_cell_width(x, qtype="bfloat16")
    assert jnp.all(w > 0.0)

    proj_jit = jax.jit(lambda p, c: region.project(p, c, ()))
    out = proj_jit(x, x * 0.5)
    assert jnp.all(is_on_half_quant_boundary(out, qtype="bfloat16"))

    # Differentiate through projection inside cell
    g = jax.grad(lambda c: jnp.sum(region.project(x, c, ()) ** 2))(x)
    assert jnp.all(jnp.isfinite(g))

    # Vmap across batch of parameter vectors
    bx = jnp.stack([x, x * 2.0])
    bout = jax.vmap(lambda p, c: region.project(p, c, ()))(bx, jnp.zeros_like(bx))
    assert bout.shape == (2, 2)
    assert jnp.all(is_on_half_quant_boundary(bout, qtype="bfloat16"))

    # Drive standalone QQN solver on L1 objective inside HalfQuantizationRegion
    solver = QQN(
        lambda p: jnp.sum(jnp.abs(p)),
        maxiter=5,
        oracle="lbfgs",
        line_search="backtracking",
        region=region,
    )
    final_p, final_st = jax.jit(solver.run)(x)
    assert float(final_st.value) < float(jnp.sum(jnp.abs(x)))
    assert jnp.all(is_on_half_quant_boundary(final_p, qtype="bfloat16"))

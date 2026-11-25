"""Half-quantization constraint regions for OWL-like quantization optimization.

Unlike :func:`QuantizationRegion` (which places walls at the *midpoints*
``g_k ± Δ/2`` around each grid value ``g_k``, leaving non-quantized midpoints as
the active boundaries and keeping the ``|x − g_k|`` kink inside the interior),
:func:`HalfQuantizationRegion` partitions parameter space directly at the
**target precision lattice points** ``{q_k}`` (e.g. 16-bit ``bfloat16``,
``float16``, or symmetric ``int16`` / ``b``-bit uniform grid for 32-bit
``float32`` weights).

Key geometric properties:
1. **Exact boundary quantization:** Every region ``[q_k, q_{k+1}]`` has
   representable target-precision values at *both* endpoints. Any coordinate
   clamped onto either boundary rounds to the target bit-width with **exact zero
   error**.
2. **Refinement of sign orthants:** Because ``0`` lies on the target grid
   (``0 ∈ Q``), no open cell ``(q_k, q_{k+1})`` straddles zero. Consequently,
   ``‖θ‖₁ = Σ|θ_i|`` is strictly linear (``C^∞``, zero Hessian kink) inside
   every open cell, generalizing OWL-QN sign orthants from a single hyperplane
   at ``0`` to the full half-precision lattice.
"""

from __future__ import annotations

from typing import Literal, Optional

import jax
from jax import numpy as jnp

from qqn_jax.regions.identity import _identity_init, _identity_update
from qqn_jax.regions.strategy import Region

__all__ = [
    "HalfQuantizationRegion",
    "half_quant_bounds",
    "half_quant_cell_width",
    "is_on_half_quant_boundary",
    "round_to_half_quant",
    "round_toward_zero_half_quant",
]

QType = Literal["bfloat16", "float16", "uniform"]
BoundaryMode = Literal["sticky", "owl", "owl_snap"]

_FP16_MAX = 65504.0


def _resolve_uniform_delta(
    dtype,
    *,
    bits: Optional[int],
    step: Optional[float],
    lo: float,
    hi: float,
) -> jnp.ndarray:
    if step is not None:
        return jnp.asarray(step, dtype=dtype)
    eff_bits = 16 if bits is None else int(bits)
    if eff_bits < 1:
        raise ValueError(f"bits must be >= 1, got {eff_bits}.")
    # When [lo, hi] is symmetric (-M, +M) and bits >= 2, use 2*(2^(bits-1) - 1)
    # intervals (= 2^bits - 2 intervals, 2^bits - 1 levels) so 0.0 is an exact
    # integer grid point (just like signed int16 / int8 symmetric quantization).
    if eff_bits >= 2 and abs(lo + hi) <= 1e-12 * max(1.0, abs(hi)):
        half_levels = (2 ** (eff_bits - 1)) - 1
        return jnp.asarray(hi / float(half_levels), dtype=dtype)
    levels = (2**eff_bits) - 1
    return jnp.asarray((hi - lo) / float(levels), dtype=dtype)


def _bf16_round_f32(x_f32: jnp.ndarray) -> jnp.ndarray:
    """Bit-exact IEEE round-to-nearest-even ``float32 -> bfloat16 -> float32``.

    Uses ``uint32`` bitcast arithmetic rather than ``astype(bfloat16).astype(float32)``
    so XLA's algebraic simplifier can never elide the cast pair inside ``jax.jit``.
    """
    ax = jnp.abs(x_f32)
    u = jax.lax.bitcast_convert_type(ax, jnp.uint32)
    lsb = (u >> jnp.uint32(16)) & jnp.uint32(1)
    u_rne = (u + jnp.uint32(0x00007FFF) + lsb) & jnp.uint32(0xFFFF0000)
    mag = jax.lax.bitcast_convert_type(u_rne, jnp.float32)
    return jnp.where(x_f32 < 0.0, -mag, jnp.where(x_f32 > 0.0, mag, 0.0))


def _fp16_round_f32(x_f32: jnp.ndarray) -> jnp.ndarray:
    """Bit-exact ``float32 -> float16 -> float32`` protected against XLA cast fusion."""
    clipped = jnp.clip(x_f32, -_FP16_MAX, _FP16_MAX)
    u16 = jax.lax.bitcast_convert_type(clipped.astype(jnp.float16), jnp.uint16)
    out = jax.lax.bitcast_convert_type(u16, jnp.float16).astype(jnp.float32)
    return jnp.where(out == 0.0, 0.0, out)


def _float_grid_neighbors(
    x: jnp.ndarray, target_dtype
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(q_lo, q_hi, prev_q, next_q)`` in ``x.dtype`` via exact IEEE bitcast stepping.

    Uses Kahan's monotone integer ordering of IEEE-754 magnitudes rather than
    ``jnp.nextafter`` (which is not lowered for ``bfloat16`` on all CUDA architectures).
    """
    orig_dt = x.dtype
    work = x.astype(jnp.float32)

    if target_dtype == jnp.bfloat16:
        ax = jnp.abs(work)
        u = jax.lax.bitcast_convert_type(ax, jnp.uint32)
        u_trunc = u & jnp.uint32(0xFFFF0000)
        rem = u & jnp.uint32(0x0000FFFF)
        on_grid = rem == jnp.uint32(0)

        # Max finite bf16 magnitude is 0x7F7F0000
        u_max = jnp.uint32(0x7F7F0000)
        u_trunc_c = jnp.minimum(u_trunc, u_max)
        u_next = jnp.minimum(u_trunc_c + jnp.uint32(0x00010000), u_max)
        u_prev = jnp.where(
            u_trunc_c > jnp.uint32(0),
            u_trunc_c - jnp.uint32(0x00010000),
            jnp.uint32(0),
        )
        mag_lo = jax.lax.bitcast_convert_type(u_trunc_c, jnp.float32)
        mag_hi = jax.lax.bitcast_convert_type(u_next, jnp.float32)
        mag_prev = jax.lax.bitcast_convert_type(u_prev, jnp.float32)
        tiny_bf16 = jax.lax.bitcast_convert_type(jnp.uint32(0x00010000), jnp.float32)

        # Off-grid bounds [q_lo, q_hi]; on-grid has q_lo == q_hi == x
        q_lo_pos = jnp.where(on_grid, mag_lo, mag_lo)
        q_hi_pos = jnp.where(on_grid, mag_lo, mag_hi)
        q_lo_neg = jnp.where(on_grid, -mag_lo, -mag_hi)
        q_hi_neg = jnp.where(on_grid, -mag_lo, -mag_lo)

        is_zero = work == 0.0
        q_lo = jnp.where(is_zero, 0.0, jnp.where(work > 0.0, q_lo_pos, q_lo_neg))
        q_hi = jnp.where(is_zero, 0.0, jnp.where(work > 0.0, q_hi_pos, q_hi_neg))

        # Neighbours around round_nearest(work)
        r_near = _bf16_round_f32(work)
        r_abs_u = jnp.minimum(
            jax.lax.bitcast_convert_type(jnp.abs(r_near), jnp.uint32)
            & jnp.uint32(0xFFFF0000),
            u_max,
        )
        r_next_mag = jax.lax.bitcast_convert_type(
            jnp.minimum(r_abs_u + jnp.uint32(0x00010000), u_max), jnp.float32
        )
        r_prev_mag = jax.lax.bitcast_convert_type(
            jnp.where(
                r_abs_u > jnp.uint32(0),
                r_abs_u - jnp.uint32(0x00010000),
                jnp.uint32(0),
            ),
            jnp.float32,
        )
        r_zero = r_near == 0.0
        prev_f32 = jnp.where(
            r_zero,
            -tiny_bf16,
            jnp.where(r_near > 0.0, r_prev_mag, -r_next_mag),
        )
        next_f32 = jnp.where(
            r_zero,
            tiny_bf16,
            jnp.where(r_near > 0.0, r_next_mag, -r_prev_mag),
        )
        return (
            q_lo.astype(orig_dt),
            q_hi.astype(orig_dt),
            prev_f32.astype(orig_dt),
            next_f32.astype(orig_dt),
        )

    # target_dtype == jnp.float16
    work_c = jnp.clip(work, -_FP16_MAX, _FP16_MAX)
    r_f32 = _fp16_round_f32(work_c)
    r_abs_u16 = jnp.minimum(
        jax.lax.bitcast_convert_type(jnp.abs(r_f32).astype(jnp.float16), jnp.uint16),
        jnp.uint16(0x7BFF),
    )
    r_next_mag = jax.lax.bitcast_convert_type(
        jnp.minimum(r_abs_u16 + jnp.uint16(1), jnp.uint16(0x7BFF)), jnp.float16
    ).astype(jnp.float32)
    r_prev_mag = jax.lax.bitcast_convert_type(
        jnp.where(r_abs_u16 > jnp.uint16(0), r_abs_u16 - jnp.uint16(1), jnp.uint16(0)),
        jnp.float16,
    ).astype(jnp.float32)
    tiny_fp16 = jax.lax.bitcast_convert_type(jnp.uint16(1), jnp.float16).astype(
        jnp.float32
    )
    r_zero = r_f32 == 0.0
    prev_f32 = jnp.where(
        r_zero, -tiny_fp16, jnp.where(r_f32 > 0.0, r_prev_mag, -r_next_mag)
    )
    next_f32 = jnp.where(
        r_zero, tiny_fp16, jnp.where(r_f32 > 0.0, r_next_mag, -r_prev_mag)
    )
    on_grid = work_c == r_f32
    q_lo = jnp.where(on_grid, r_f32, jnp.where(r_f32 < work_c, r_f32, prev_f32))
    q_hi = jnp.where(on_grid, r_f32, jnp.where(r_f32 > work_c, r_f32, next_f32))
    return (
        q_lo.astype(orig_dt),
        q_hi.astype(orig_dt),
        prev_f32.astype(orig_dt),
        next_f32.astype(orig_dt),
    )


def _uniform_grid_neighbors(
    x: jnp.ndarray,
    *,
    bits: Optional[int],
    step: Optional[float],
    lo: float,
    hi: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(q_lo, q_hi, prev_q, next_q)`` on a uniform grid over ``[lo, hi]``."""
    dt = x.dtype
    lo_v = jnp.asarray(lo, dtype=dt)
    hi_v = jnp.asarray(hi, dtype=dt)
    delta = _resolve_uniform_delta(dt, bits=bits, step=step, lo=lo, hi=hi)
    x_clip = jnp.clip(x, lo_v, hi_v)

    if abs(lo + hi) <= 1e-12 * max(1.0, abs(hi)):
        k_max = jnp.round(hi_v / delta)
        u = x_clip / delta
        k_lo = jnp.clip(jnp.floor(u), -k_max, k_max)
        k_hi = jnp.clip(jnp.ceil(u), -k_max, k_max)
        k_near = jnp.clip(jnp.round(u), -k_max, k_max)
        q_lo = k_lo * delta
        q_hi = k_hi * delta
        prev_q = jnp.maximum(k_near - 1.0, -k_max) * delta
        next_q = jnp.minimum(k_near + 1.0, k_max) * delta
        return q_lo, q_hi, prev_q, next_q

    k_max = jnp.floor((hi_v - lo_v) / delta)
    u = (x_clip - lo_v) / delta
    k_lo = jnp.clip(jnp.floor(u), 0.0, k_max)
    k_hi = jnp.clip(jnp.ceil(u), 0.0, k_max)
    k_near = jnp.clip(jnp.round(u), 0.0, k_max)
    q_lo = lo_v + k_lo * delta
    q_hi = lo_v + k_hi * delta
    prev_q = lo_v + jnp.maximum(k_near - 1.0, 0.0) * delta
    next_q = lo_v + jnp.minimum(k_near + 1.0, k_max) * delta
    return q_lo, q_hi, prev_q, next_q


def half_quant_bounds(
    x: jnp.ndarray,
    *,
    qtype: QType = "bfloat16",
    bits: Optional[int] = 16,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(q_lo, q_hi)`` bracketing ``x`` on the target quantization lattice."""
    if qtype == "bfloat16":
        q_lo, q_hi, _, _ = _float_grid_neighbors(x, jnp.bfloat16)
        return q_lo, q_hi
    if qtype == "float16":
        q_lo, q_hi, _, _ = _float_grid_neighbors(x, jnp.float16)
        return q_lo, q_hi
    if qtype == "uniform":
        q_lo, q_hi, _, _ = _uniform_grid_neighbors(
            x, bits=bits, step=step, lo=lo, hi=hi
        )
        return q_lo, q_hi
    raise ValueError(f"Unsupported qtype: {qtype!r}")


def half_quant_cell_width(
    x: jnp.ndarray,
    *,
    qtype: QType = "bfloat16",
    bits: Optional[int] = 16,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
) -> jnp.ndarray:
    """Local cell width (ULP) around ``x`` on the target quantization grid."""
    if qtype == "bfloat16":
        q_lo, q_hi, prev_q, next_q = _float_grid_neighbors(x, jnp.bfloat16)
    elif qtype == "float16":
        q_lo, q_hi, prev_q, next_q = _float_grid_neighbors(x, jnp.float16)
    elif qtype == "uniform":
        q_lo, q_hi, prev_q, next_q = _uniform_grid_neighbors(
            x, bits=bits, step=step, lo=lo, hi=hi
        )
    else:
        raise ValueError(f"Unsupported qtype: {qtype!r}")
    interior_w = q_hi - q_lo
    boundary_w = 0.5 * (next_q - prev_q)
    return jnp.maximum(jnp.where(interior_w > 0.0, interior_w, boundary_w), 1e-30)


def round_to_half_quant(
    params,
    *,
    qtype: QType = "bfloat16",
    bits: Optional[int] = 16,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
):
    """Round a PyTree (or array) to nearest value on the target quantization grid."""

    def _round_leaf(x):
        if qtype == "bfloat16":
            return _bf16_round_f32(x.astype(jnp.float32)).astype(x.dtype)
        if qtype == "float16":
            return _fp16_round_f32(x.astype(jnp.float32)).astype(x.dtype)
        if qtype == "uniform":
            q_lo, q_hi, _, _ = _uniform_grid_neighbors(
                x, bits=bits, step=step, lo=lo, hi=hi
            )
            return jnp.where(x - q_lo <= q_hi - x, q_lo, q_hi)
        raise ValueError(f"Unsupported qtype: {qtype!r}")

    return jax.tree_util.tree_map(_round_leaf, params)


def round_toward_zero_half_quant(
    params,
    *,
    qtype: QType = "bfloat16",
    bits: Optional[int] = 16,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
):
    """Round a PyTree toward zero onto its half-quantization cell boundary."""

    def _rtz_leaf(x):
        q_lo, q_hi = half_quant_bounds(
            x, qtype=qtype, bits=bits, step=step, lo=lo, hi=hi
        )
        return jnp.where(x >= 0.0, q_lo, q_hi)

    return jax.tree_util.tree_map(_rtz_leaf, params)


def is_on_half_quant_boundary(
    x: jnp.ndarray,
    *,
    qtype: QType = "bfloat16",
    bits: Optional[int] = 16,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
    atol: float = 0.0,
) -> jnp.ndarray:
    """Boolean mask indicating coordinates lying on the target quantization lattice."""
    q_near = round_to_half_quant(
        x, qtype=qtype, bits=bits, step=step, lo=lo, hi=hi
    )
    return jnp.abs(x.astype(jnp.float32) - q_near.astype(jnp.float32)) <= atol


def HalfQuantizationRegion(
    qtype: QType = "bfloat16",
    *,
    bits: Optional[int] = 16,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
    boundary_mode: BoundaryMode = "sticky",
) -> Region:
    """Constrain steps to half-quantization intervals ``[q_lo(x), q_hi(x)]``."""
    if qtype not in ("bfloat16", "float16", "uniform"):
        raise ValueError(
            f"Unsupported qtype {qtype!r}; choose 'bfloat16', 'float16', or 'uniform'."
        )
    if boundary_mode not in ("sticky", "owl", "owl_snap"):
        raise ValueError(
            f"Unsupported boundary_mode {boundary_mode!r}; "
            "choose 'sticky', 'owl', or 'owl_snap'."
        )
    if qtype == "uniform" and step is not None and float(step) <= 0.0:
        raise ValueError(f"step must be > 0, got {step}.")

    def _neighbors(x):
        if qtype == "bfloat16":
            return _float_grid_neighbors(x, jnp.bfloat16)
        if qtype == "float16":
            return _float_grid_neighbors(x, jnp.float16)
        return _uniform_grid_neighbors(x, bits=bits, step=step, lo=lo, hi=hi)

    def project(params, candidate, state):
        del state

        def proj_leaf(x, c):
            dt = c.dtype
            x_dt = x.astype(dt)
            zero = jnp.zeros((), dtype=dt)
            q_lo, q_hi, prev_q, next_q = _neighbors(x_dt)
            interior_clip = jnp.clip(c, q_lo, q_hi)

            if boundary_mode == "sticky":
                on_b = q_lo == q_hi
                return jnp.where(on_b, q_lo, interior_clip)

            on_b = q_lo == q_hi
            is_zero = x_dt == zero
            adj_lo = jnp.where(x_dt > zero, jnp.maximum(prev_q, zero), prev_q)
            adj_hi = jnp.where(x_dt < zero, jnp.minimum(next_q, zero), next_q)

            if boundary_mode == "owl":
                b_lo = jnp.where(c < x_dt, adj_lo, x_dt)
                b_hi = jnp.where(c > x_dt, adj_hi, x_dt)
                owl_val = jnp.where(is_zero, zero, jnp.clip(c, b_lo, b_hi))
                return jnp.where(on_b, owl_val, interior_clip)

            snap_down = jnp.where(c <= adj_lo, adj_lo, x_dt)
            snap_up = jnp.where(c >= adj_hi, adj_hi, x_dt)
            snap_val = jnp.where(
                is_zero,
                zero,
                jnp.where(c < x_dt, snap_down, jnp.where(c > x_dt, snap_up, x_dt)),
            )
            return jnp.where(on_b, snap_val, interior_clip)

        return jax.tree_util.tree_map(proj_leaf, params, candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )

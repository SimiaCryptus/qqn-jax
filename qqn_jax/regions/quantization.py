from typing import Optional

import jax
from jax import numpy as jnp


from qqn_jax.regions.strategy import (
    Region,
)
from qqn_jax.regions.identity import _identity_init, _identity_update


def QuantizationRegion(
    bits: Optional[int] = None,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
    lock: bool = False,
    window: float = 1.0,
) -> Region:
    """Confine weights to the rounding *cell* of their starting value.

     Quantization with ``bits`` levels over ``[lo, hi]`` defines a uniform grid
     of representable values spaced by ``Δ = (hi − lo) / (2**bits − 1)`` (or by
     an explicit ``step``). The relevant regularizer is the **L1-norm of the
     rounding delta** ``|x − round(x)|``: a sawtooth whose *minima* (zero error)
     sit on the grid points and whose *maxima* (largest error, ``Δ/2``) sit on
     the midpoints *between* grid points.

     These periodic maxima of the rounding delta partition the line into
     hypercubic *cells* ``[g_k − Δ/2, g_k + Δ/2]`` around each grid point
     ``g_k = lo + k·Δ``. This region anchors the cell to the iterate's value
     **at the start of the step** (``params``): a coordinate is free to explore
     its own cell but is walled at the midpoints (the local rounding-delta
     maxima), never crossing into a neighbour's cell.

     The **cell center** is the *grid point* itself — the local *maximum* of the
     rounding-delta penalty's negative, i.e. the natural attractor of minimum
     rounding error. Because the search is projected onto this cell, the
     optimizer is drawn toward the quantized grid value rather than allowed to
     drift toward the high-error midpoints.

    Args:
        bits: number of quantization bits; the grid has ``2**bits`` levels over
            ``[lo, hi]``. Provide either ``bits`` or ``step``.
        step: explicit grid spacing ``Δ`` (overrides ``bits`` when given).
        lo, hi: the quantization range. Values are clipped to ``[lo, hi]``
            before rounding.
         lock: if ``True``, collapse each coordinate to the nearest grid point —
             hard-snap (true quantization, no exploration).
            If ``False`` (default), the coordinate may roam freely within the
             rounding cell around its starting value.
        window: fraction of the half-cell the coordinate may explore when
            ``lock=False``. ``window=1.0`` (default) exposes the full cell
             ``[g_k − Δ/2, g_k + Δ/2]``; smaller values tighten the box
             symmetrically about the grid point.
    """
    if step is None and bits is None:
        raise ValueError("QuantizationRegion requires either `bits` or `step`.")
    # After the guard above, `bits` is non-None whenever `step` is None.
    _bits: int = 0 if bits is None else int(bits)

    def _delta(dtype):
        if step is not None:
            return jnp.asarray(step, dtype=dtype)
        levels = (2**_bits) - 1
        return jnp.asarray((hi - lo) / levels, dtype=dtype)

    def project(params, candidate, state):
        def proj_leaf(x, c):
            dt = c.dtype
            delta = _delta(dt)
            lo_v = jnp.asarray(lo, dtype=dt)
            hi_v = jnp.asarray(hi, dtype=dt)
            # Grid points sit at g_k = lo + k*delta. The rounding-delta penalty
            # |x - round(x)| has its maxima at the midpoints g_k ± delta/2,
            # which form the natural cell walls. The k-th cell is therefore
            # [g_k - delta/2, g_k + delta/2], centered on the grid point g_k
            # (the minimum-rounding-error attractor).
            x_clipped = jnp.clip(x, lo_v, hi_v)
            # Nearest grid index to x: round() snaps x into its own cell.
            k = jnp.round((x_clipped - lo_v) / delta)
            # Clamp k to the valid grid range.
            k_max = jnp.floor((hi_v - lo_v) / delta)
            k = jnp.clip(k, 0.0, k_max)
            # Cell center: the grid point itself — the rounding-error minimum.
            center = lo_v + k * delta
            if lock:
                # Collapse to the grid point (true quantization).
                return center
            # The explorable cell is the midpoint-walled box around the grid
            # point, optionally narrowed by `window`. The walls are the local
            # maxima of the rounding delta (the half-cell midpoints), clipped
            # to the [lo, hi] range.
            half = jnp.asarray(window, dtype=dt) * delta * 0.5
            cell_lo = jnp.maximum(center - half, lo_v)
            cell_hi = jnp.minimum(center + half, hi_v)
            return jnp.clip(c, cell_lo, cell_hi)

        return jax.tree_util.tree_map(proj_leaf, params, candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )

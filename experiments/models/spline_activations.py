from jax import numpy as jnp


def _build_hermite_presets():
    """Build the named Hermite-spline presets.

    Collecting the presets inside a function lets us share common variables
    (e.g. ``pi``/``two_pi``) and use small generating helpers to reduce
    duplication. Returns a ``dict`` mapping name -> ``(xs, ys, ms, period)``.
    """
    pi = float(jnp.pi)
    two_pi = 2.0 * pi
    half_pi = pi / 2.0

    def wave_knots():
        """Symmetric five-knot layout over [-pi, pi]."""
        return [-pi, -half_pi, 0.0, half_pi, pi]

    def symmetric_knots(width):
        """Symmetric five-knot layout over [-width, width]."""
        return [-width, -width / 2.0, 0.0, width / 2.0, width]

    presets = {}

    # Smooth periodic "bump" wave over [-pi, pi], repeating with period 2*pi.
    presets["hermite_wave"] = (
        wave_knots(),
        [0.0, -1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0, -1.0],
        two_pi,
    )
    # Smooth non-periodic step-like squashing function on [-3, 3].
    presets["hermite_step"] = (
        [-3.0, -1.0, 0.0, 1.0, 3.0],
        [-1.0, -0.9, 0.0, 0.9, 1.0],
        [0.0, 0.3, 1.2, 0.3, 0.0],
        None,
    )
    # Periodic asymmetric pulse over [0, 2*pi].
    presets["hermite_pulse"] = (
        [0.0, half_pi, pi, 1.1 * pi, two_pi],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        two_pi,
    )
    # Smooth non-periodic "valley" bump on [-3, 3]: dips down then recovers.
    presets["hermite_valley"] = (
        symmetric_knots(3.0),
        [0.0, -0.5, -1.0, -0.5, 0.0],
        [0.0, -0.8, 0.0, 0.8, 0.0],
        None,
    )
    # Periodic double-bump wave over [-pi, pi] with two peaks per period.
    presets["hermite_double_wave"] = (
        wave_knots(),
        [-1.0, 0.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        two_pi,
    )
    # Non-periodic smooth "S" ramp on [-4, 4], gentler than tanh.
    presets["hermite_sramp"] = (
        symmetric_knots(4.0),
        [-1.0, -0.7, 0.0, 0.7, 1.0],
        [0.0, 0.2, 0.6, 0.2, 0.0],
        None,
    )
    # Periodic sawtooth-like ramp over [0, 2*pi] with a smooth reset.
    presets["hermite_ramp"] = (
        [0.0, 0.95 * two_pi],
        [-1.0, 1.0],
        [0.5, 0.5],
        two_pi,
    )
    return presets


def _cubic_hermite_spline(x, xs, ys, ms, period=None):
    """Evaluate a cubic Hermite spline at ``x``.
    Args:
        x: input array.
        xs: 1-D array of strictly increasing knot positions.
        ys: values at each knot (same length as ``xs``).
        ms: tangents (derivatives) at each knot (same length as ``xs``).
        period: if not ``None``, the domain is treated as periodic with the
            given period; inputs are wrapped into ``[xs[0], xs[0] + period)``
            and a wrap segment links the final knot back to the first for
            C1-continuous periodicity.
    Returns:
        The spline evaluated elementwise at ``x``.
    """
    xs = jnp.asarray(xs, dtype=jnp.float32)
    ys = jnp.asarray(ys, dtype=jnp.float32)
    ms = jnp.asarray(ms, dtype=jnp.float32)
    if period is not None:
        # Wrap x into [xs[0], xs[0] + period).
        x = xs[0] + jnp.mod(x - xs[0], period)
        # Append a virtual knot that closes the loop back to the first knot.
        xs_ext = jnp.concatenate([xs, xs[:1] + period])
        ys_ext = jnp.concatenate([ys, ys[:1]])
        ms_ext = jnp.concatenate([ms, ms[:1]])
    else:
        xs_ext, ys_ext, ms_ext = xs, ys, ms
    # Locate the segment index i such that xs_ext[i] <= x < xs_ext[i+1].
    n_seg = xs_ext.shape[0] - 1
    # searchsorted expects a sorted 1-D array; do the lookup explicitly so we
    # are robust to endpoint handling and array-valued x. For each x we want
    # the last knot i with xs_ext[i] <= x.
    x = jnp.asarray(x, dtype=jnp.float32)
    idx = jnp.sum(xs_ext[:-1] <= x[..., None], axis=-1) - 1
    idx = jnp.clip(idx, 0, n_seg - 1)
    x0 = xs_ext[idx]
    x1 = xs_ext[idx + 1]
    y0 = ys_ext[idx]
    y1 = ys_ext[idx + 1]
    m0 = ms_ext[idx]
    m1 = ms_ext[idx + 1]
    h = x1 - x0
    # Clamp t to [0, 1] so out-of-range (non-periodic) inputs extrapolate
    # flat. NOTE: with t clamped to the endpoints the tangent basis terms
    # (h10, h11) vanish there, so this yields the knot values y0/y1 exactly,
    # independent of the sign of the tangents m0/m1.
    t = jnp.clip((x - x0) / h, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    # Hermite basis functions.
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

"""Canonical activation registry + parsing (the fashion superset).

This consolidates the two divergent ``_ACTIVATIONS`` / ``_parse_activation``
copies. The registry is the superset (fashion's, including ``rolling_*``,
``triangle``, ``sawtooth``, ``logabs``, ``identity``). The default activation
is configurable so the sparse benchmark can keep ``tanh`` while fashion keeps
``tanh,gelu``.
"""

import os

import jax
import jax.numpy as jnp


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


# Preset control points for named Hermite-spline activations. Each preset is
# ``(xs, ys, ms, period)``. These give a variety of smooth, optionally
# periodic shapes.
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


_HERMITE_PRESETS = _build_hermite_presets()

try:
    from rolling_window_activation import (
        rolling_sin_diff,
        rolling_atan2_ramp,
         rolling_euclidean,
         rolling_tan_ratio,
         rolling_xlog_share,
         rolling_harmonic,
         rolling_softmin,
    )

    _ROLLING_ACTIVATIONS = {
        "rolling_sin": rolling_sin_diff,
        "rolling_atan2": rolling_atan2_ramp,
         "rolling_euclidean": rolling_euclidean,
         "rolling_tan_ratio": rolling_tan_ratio,
         "rolling_xlog_share": rolling_xlog_share,
         "rolling_harmonic": rolling_harmonic,
         "rolling_softmin": rolling_softmin,
    }
except Exception:
    _ROLLING_ACTIVATIONS = {}

__all__ = ["ACTIVATIONS", "resolve_activation", "parse_activation"]
ACTIVATIONS = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "sine": jnp.sin,
    "gaussian": lambda x: jnp.exp(-(x ** 2)),
     # --- Continuous wavelets ---
     # Mexican hat / Ricker wavelet: 2nd derivative of a Gaussian.
     "mexican_hat": lambda x: (
             (2.0 / (jnp.sqrt(3.0) * jnp.pi ** 0.25))
             * (1.0 - x ** 2)
             * jnp.exp(-(x ** 2) / 2.0)
     ),
     # Real-valued Morlet wavelet: modulated cosine under a Gaussian envelope.
     "morlet": lambda x: (
             jnp.cos(5.0 * x) * jnp.exp(-(x ** 2) / 2.0)
     ),
     # --- Frequency multiples of the core periodics ---
     "sine2": lambda x: jnp.sin(2.0 * x),
     "sine8": lambda x: jnp.sin(8.0 * x),
     "triangle2": lambda x: (
             2.0 * jnp.abs(2.0 * ((2.0 * x) / (2.0 * jnp.pi)
                                  - jnp.floor((2.0 * x) / (2.0 * jnp.pi) + 0.5)))
             - 1.0
     ),
     "triangle8": lambda x: (
             2.0 * jnp.abs(2.0 * ((8.0 * x) / (2.0 * jnp.pi)
                                  - jnp.floor((8.0 * x) / (2.0 * jnp.pi) + 0.5)))
             - 1.0
     ),
     "sawtooth2": lambda x: (
             2.0 * ((2.0 * x) / (2.0 * jnp.pi)
                    - jnp.floor((2.0 * x) / (2.0 * jnp.pi) + 0.5))
     ),
     "sawtooth8": lambda x: (
             2.0 * ((8.0 * x) / (2.0 * jnp.pi)
                    - jnp.floor((8.0 * x) / (2.0 * jnp.pi) + 0.5))
     ),
     # --- Chirplet: Gaussian-windowed cosine with time-varying frequency. ---
     # Instantaneous frequency grows linearly (chirp rate 1.0) about a base
     # frequency of 3.0, all under a unit-variance Gaussian envelope.
     "chirplet": lambda x: (
             jnp.cos(3.0 * x + 0.5 * x ** 2) * jnp.exp(-(x ** 2) / 2.0)
     ),
    "triangle": lambda x: (
            2.0 * jnp.abs(2.0 * (x / (2.0 * jnp.pi) - jnp.floor(x / (2.0 * jnp.pi) + 0.5)))
            - 1.0
    ),
    "logabs": lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x)),
    "tanh": jnp.tanh,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
    "softplus": jax.nn.softplus,
    "sawtooth": lambda x: (
            2.0 * (x / (2.0 * jnp.pi) - jnp.floor(x / (2.0 * jnp.pi) + 0.5))
    ),
    "abs": jnp.abs,
    "identity": lambda x: x,
}
ACTIVATIONS.update(_ROLLING_ACTIVATIONS)
ACTIVATIONS.update(
    {
        name: (
            lambda x, _xs=xs, _ys=ys, _ms=ms, _p=period: _cubic_hermite_spline(
                x, _xs, _ys, _ms, _p
            )
        )
        for name, (xs, ys, ms, period) in _HERMITE_PRESETS.items()
    }
)


def resolve_activation(name, *, default="sigmoid"):
    """Resolve a single activation name to ``(name, fn)``; fall back to default."""
    name = name.strip().lower()
    if name not in ACTIVATIONS:
        print(
            f"[config] Unknown ACTIVATION={name!r}; falling back to {default!r}. "
            f"Valid values: {', '.join(sorted(ACTIVATIONS))}."
        )
        name = default
    return name, ACTIVATIONS[name]


def parse_activation(n_hidden_layers=None, *, default="tanh,gelu", env="ACTIVATION"):
    """Resolve the hidden-layer activation(s) from the env var.

    The env var accepts either a single name (applied to every hidden layer)
    or a comma-separated list to *mix* activations across hidden layers. When
    a list is shorter than the number of hidden layers it is cycled; when
    longer it is truncated.

    Args:
        n_hidden_layers: number of hidden layers, used to expand/cycle a
            mixed list. If ``None`` the parsed (un-expanded) spec is returned.
        default: the default spec when the env var is unset (``tanh,gelu``
            for fashion; pass ``tanh`` for the sparse benchmark).
        env: the environment variable name to read.

    Returns:
        ``(name, fn)`` for a single activation, or ``(names, fns)`` lists for
        a mixed spec (one entry per hidden layer when ``n_hidden_layers`` is
        given).
    """
    raw = os.environ.get(env, default).strip().lower()
    tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
    if not tokens:
        tokens = ["sigmoid"]

    if len(tokens) == 1:
        return resolve_activation(tokens[0])

    resolved = [resolve_activation(t) for t in tokens]
    names = [n for n, _ in resolved]
    fns = [f for _, f in resolved]

    if n_hidden_layers is not None and n_hidden_layers > 0:
        names = [names[i % len(names)] for i in range(n_hidden_layers)]
        fns = [fns[i % len(fns)] for i in range(n_hidden_layers)]
    return names, fns
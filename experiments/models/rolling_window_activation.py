"""Rolling-window activation functions.

A rolling-window activation treats the per-layer pre-activation vector as a
1D *ring* (circular buffer) of N scalar signals and slides a small window
over it. For a window of width ``w`` and a base function
``g(x_0, x_1, ..., x_{w-1})``, the i-th output is

    y_i = g(h_i, h_{i+1}, ..., h_{i+w-1})        (indices taken mod N)

so for N inputs the activation runs N evaluations of ``g`` and produces N
outputs. This couples neighbouring units (a cheap, weight-free mixing /
convolution-like nonlinearity) while preserving the layer width.

The default 2-input base function is ``sin(x - y)``: each output looks at a
unit and its (circular) right-neighbour and emits the sine of their
difference. This is smooth, bounded in [-1, 1], periodic, and inherently
*relational* (it depends on differences between adjacent signals rather
than their absolute values), making it an interesting non-convex test bed.

The functions here operate on the last axis of a JAX array of arbitrary
leading (batch) shape, so they slot directly into the MLP forward pass used
by the comparison experiments.
"""

import jax.numpy as jnp


def _sin_diff(x, y):
    """Default 2-input base activation: ``sin(x - y)``."""
    return jnp.sin(x - y)
def _euclidean(x, y):
     """Symmetric 2-input base activation: ``sqrt(x^2 + y^2)``.
     The Euclidean norm of a unit and its (circular) right-neighbour. It is
     symmetric in its arguments, smooth (with a soft floor added under the
     root to keep the gradient finite at the origin), and unbounded above,
     giving a radial "magnitude" mixing of adjacent signals.
     """
     return jnp.sqrt(x * x + y * y + 1e-12)
def _tan_ratio(x, y):
     """2-input base activation: ``tan(x / (y + eps))`` squashed by ``tanh``.
     Emits the tangent of the ratio of a unit to its (circular) neighbour.
     Raw ``tan`` is periodic with poles, so we wrap it in ``tanh`` to keep
     the output bounded in ``(-1, 1)`` while preserving the oscillatory,
     ratio-sensitive structure. Not symmetric (ratio is order-dependent).
     """
     return jnp.tanh(jnp.tan(x / (y + jnp.where(y >= 0, 1e-3, -1e-3))))
def _xlog_share(x, y):
     """2-input base activation: ``x * log(|y| / (|x| + |y|)) / (|x| + |y|)``.
     A normalized "information share" style coupling: the log term measures
     the fractional weight of ``y`` within the pair, scaled by ``x`` and the
     total magnitude. Small floors keep the log and division finite. Bounded
     and relational; asymmetric by construction.
     """
     ax = jnp.abs(x)
     ay = jnp.abs(y)
     total = ax + ay + 1e-6
     return x * jnp.log(ay / total + 1e-6) / total
def _harmonic(x, y):
     """Symmetric 2-input base activation: smoothed harmonic-mean style mix.
     ``2 * x * y / (x + y)`` regularized to avoid the pole at ``x + y = 0``.
     Symmetric in its arguments and relational, emphasizing agreement between
     a unit and its (circular) neighbour.
     """
     denom = x + y
     denom = denom + jnp.where(denom >= 0, 1e-3, -1e-3)
     return 2.0 * x * y / denom
def _softmin_diff(x, y):
     """Symmetric 2-input base activation: ``-log(exp(-x) + exp(-y))``.
     A smooth (soft) minimum of a unit and its (circular) neighbour, computed
     via ``logsumexp`` for numerical stability. Symmetric and smooth, giving
     a "soft AND"-like mixing of adjacent signals.
     """
     m = jnp.minimum(x, y)
     return m - jnp.log(jnp.exp(-(x - m)) + jnp.exp(-(y - m)))
def _sum_over_prod(x, y):
    """2-input base activation: ``(x + y) / (x * y)``.
    Equivalent to ``1/x + 1/y`` (the sum of reciprocals). Small signed floors
    on the denominator keep the value finite near ``x*y = 0``. Symmetric and
    relational, emphasizing small-magnitude signals via the reciprocal terms.
    """
    denom = x * y
    denom = denom + jnp.where(denom >= 0, 1e-3, -1e-3)
    return (x + y) / denom
def _parallel(x, y):
    """Symmetric 2-input base activation: ``1 / (1/x + 1/y)``.
    The "parallel resistor" combination of a unit and its (circular)
    neighbour, equal to ``x*y / (x + y)`` (half the harmonic mean). Small
    signed floors keep the reciprocals and final division finite. Symmetric
    and relational.
    """
    rx = 1.0 / (x + jnp.where(x >= 0, 1e-3, -1e-3))
    ry = 1.0 / (y + jnp.where(y >= 0, 1e-3, -1e-3))
    inv = rx + ry
    inv = inv + jnp.where(inv >= 0, 1e-3, -1e-3)
    return 1.0 / inv
def _ratio(x, y):
    """2-input base activation: ``x / y``.
    The raw ratio of a unit to its (circular) right-neighbour. A small signed
    floor on the denominator keeps the value finite near ``y = 0``. Unbounded
    and asymmetric (order-dependent), giving a scale-relative coupling.
    """
    return x / (y + jnp.where(y >= 0, 1e-3, -1e-3))






def _atan2_ramp(a, b, c):
    """3-input base activation: ``atan2(a + b, b + c)``.
    Looks at a unit and its two (circular) right-neighbours and emits the
    angle of the 2D vector ``(a + b, b + c)``. It is smooth, bounded in
    ``(-pi, pi]``, and relational (the middle element ``b`` couples both
    components), making it a richer 3-wide companion to ``sin(x - y)``.
    """
    return jnp.arctan2(a + b, b + c)


def rolling_window(h, base_fn=_sin_diff, window=2):
    """Apply a rolling-window activation over the last axis of ``h``.

    Treats the last axis of ``h`` (length ``N``) as a circular ring and
    computes, for each position ``i``,

        y_i = base_fn(h_i, h_{i+1}, ..., h_{i+window-1})   (mod N)

    Args:
        h: array of shape ``(..., N)`` of pre-activation signals.
        base_fn: callable accepting ``window`` positional array arguments
            (each of shape ``(..., N)``) and returning an array of the same
            shape. Defaults to ``sin(x - y)`` (window=2).
        window: number of consecutive ring elements fed to ``base_fn``.

    Returns:
        Array of the same shape as ``h`` (N evaluations -> N outputs).
    """

    shifted = [jnp.roll(h, shift=-k, axis=-1) for k in range(window)]
    return base_fn(*shifted)


def make_rolling_window(base_fn=_sin_diff, window=2):
    """Return a single-argument activation closure for the MLP forward pass.

    The comparison experiment applies hidden-layer activations as
    ``fn(h)``; this factory binds ``base_fn`` / ``window`` so the resulting
    callable has that single-argument signature.
    """
    return lambda h: rolling_window(h, base_fn=base_fn, window=window)


rolling_sin_diff = make_rolling_window(_sin_diff, window=2)

rolling_atan2_ramp = make_rolling_window(_atan2_ramp, window=3)
rolling_euclidean = make_rolling_window(_euclidean, window=2)
rolling_tan_ratio = make_rolling_window(_tan_ratio, window=2)
rolling_xlog_share = make_rolling_window(_xlog_share, window=2)
rolling_harmonic = make_rolling_window(_harmonic, window=2)
rolling_softmin = make_rolling_window(_softmin_diff, window=2)
rolling_sum_over_prod = make_rolling_window(_sum_over_prod, window=2)
rolling_parallel = make_rolling_window(_parallel, window=2)
rolling_ratio = make_rolling_window(_ratio, window=2)
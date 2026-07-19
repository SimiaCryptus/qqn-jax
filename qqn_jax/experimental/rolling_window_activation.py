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

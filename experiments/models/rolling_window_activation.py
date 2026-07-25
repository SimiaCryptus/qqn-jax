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
import jax


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


def _soft_xor(x, y):
    """Symmetric 2-input base activation: smooth XOR of two "soft bits".
    Squashes a unit and its (circular) right-neighbour to ``(0, 1)`` via
    ``sigmoid`` (treating them as fuzzy truth values ``a``/``b``) and combines
    them with the probabilistic XOR ``a + b - 2*a*b``. Smooth, bounded in
    ``(0, 1)``, symmetric, and relational (peaks when the two disagree).
    """
    a = jax.nn.sigmoid(x)
    b = jax.nn.sigmoid(y)
    return a + b - 2.0 * a * b


def _soft_and(x, y):
    """Symmetric 2-input base activation: smooth AND of two "soft bits".
    Squashes a unit and its (circular) right-neighbour to ``(0, 1)`` via
    ``sigmoid`` and multiplies them (``a * b``), the probabilistic AND. Smooth,
    bounded in ``(0, 1)``, symmetric; high only when both signals are strongly
    positive.
    """
    a = jax.nn.sigmoid(x)
    b = jax.nn.sigmoid(y)
    return a * b


def _soft_or(x, y):
    """Symmetric 2-input base activation: smooth OR of two "soft bits".
    Squashes a unit and its (circular) right-neighbour to ``(0, 1)`` via
    ``sigmoid`` and combines them with the probabilistic OR ``a + b - a*b``.
    Smooth, bounded in ``(0, 1)``, symmetric; low only when both signals are
    strongly negative.
    """
    a = jax.nn.sigmoid(x)
    b = jax.nn.sigmoid(y)
    return a + b - a * b


# ---------------------------------------------------------------------------
# Symmetric g(x, y) = g(y, x): direction-free ring couplings with genuine 2D
# structure (multiple independent terms stay alive).
# ---------------------------------------------------------------------------
def _sym_mixed_poly(x, y):
    """Symmetric mixed polynomial: ``x^2 y + y^2 x``."""
    return x * x * y + y * y * x


def _sym_sig_sum(x, y):
    """Symmetric sigmoid sum: ``sigma(x) + sigma(y)``."""
    return jax.nn.sigmoid(x) + jax.nn.sigmoid(y)


def _sym_sig_prod(x, y):
    """Symmetric sigmoid product: ``sigma(x) * sigma(y)``."""
    return jax.nn.sigmoid(x) * jax.nn.sigmoid(y)


def _sym_sig_hybrid(x, y):
    """Symmetric hybrid: ``sigma(x) + sigma(y) + sigma(x) sigma(y)``."""
    a = jax.nn.sigmoid(x)
    b = jax.nn.sigmoid(y)
    return a + b + a * b


def _sym_interaction_norm(x, y):
    """Symmetric interaction-normalized: ``x y / (1 + x^2 + y^2)``."""
    return (x * y) / (1.0 + x * x + y * y)


def _sym_param_kernel(x, y, w1=1.0, w2=1.0, w3=1.0):
    """Symmetric parametric kernel.
    ``sigma(w1 (x + y) + w2 x y + w3 (x^2 + y^2))`` with fixed weights.
    """
    return jax.nn.sigmoid(w1 * (x + y) + w2 * x * y + w3 * (x * x + y * y))


# ---------------------------------------------------------------------------
# Antisymmetric g(x, y) = -g(y, x): inject orientation into the ring.
# ---------------------------------------------------------------------------
def _anti_diff(x, y):
    """Basic antisymmetric: ``x - y``."""
    return x - y


def _anti_tanh_diff(x, y):
    """Nonlinear antisymmetric: ``tanh(x - y)``."""
    return jnp.tanh(x - y)


def _anti_mixed_poly(x, y):
    """Antisymmetric mixed polynomial: ``x^2 y - y^2 x``."""
    return x * x * y - y * y * x


def _anti_gated(x, y):
    """Antisymmetric gated: ``(x - y) sigma(x + y)``."""
    return (x - y) * jax.nn.sigmoid(x + y)


def _anti_normalized(x, y):
    """Antisymmetric normalized: ``(x - y) / (1 + x^2 + y^2)``."""
    return (x - y) / (1.0 + x * x + y * y)


# ---------------------------------------------------------------------------
# Asymmetric (no constraint): let the ring pick up a directed "flow" flavor.
# ---------------------------------------------------------------------------
def _asym_directional_gate(x, y):
    """Directional gate: ``x * sigma(y)``."""
    return x * jax.nn.sigmoid(y)


def _asym_forward_biased(x, y, alpha=2.0):
    """Forward-biased: ``sigma(x) + sigma(y) + alpha sigma(x) sigma(y)``."""
    a = jax.nn.sigmoid(x)
    b = jax.nn.sigmoid(y)
    return a + b + alpha * a * b


def _asym_source_sink(x, y):
    """Source-sink style: ``sigma(x) - sigma(y)``."""
    return jax.nn.sigmoid(x) - jax.nn.sigmoid(y)


def _asym_tiny_mlp(x, y, w1=1.0, w2=0.5, w3=1.0, w4=0.5):
    """Asymmetric tiny MLP: ``sigma(w1 x + w2 y + w3 x y + w4 x^2)``."""
    return jax.nn.sigmoid(w1 * x + w2 * y + w3 * x * y + w4 * x * x)


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
rolling_soft_xor = make_rolling_window(_soft_xor, window=2)
rolling_soft_and = make_rolling_window(_soft_and, window=2)
rolling_soft_or = make_rolling_window(_soft_or, window=2)
# Symmetric couplings.
rolling_sym_mixed_poly = make_rolling_window(_sym_mixed_poly, window=2)
rolling_sym_sig_sum = make_rolling_window(_sym_sig_sum, window=2)
rolling_sym_sig_prod = make_rolling_window(_sym_sig_prod, window=2)
rolling_sym_sig_hybrid = make_rolling_window(_sym_sig_hybrid, window=2)
rolling_sym_interaction_norm = make_rolling_window(_sym_interaction_norm, window=2)
rolling_sym_param_kernel = make_rolling_window(_sym_param_kernel, window=2)
# Antisymmetric couplings.
rolling_anti_diff = make_rolling_window(_anti_diff, window=2)
rolling_anti_tanh_diff = make_rolling_window(_anti_tanh_diff, window=2)
rolling_anti_mixed_poly = make_rolling_window(_anti_mixed_poly, window=2)
rolling_anti_gated = make_rolling_window(_anti_gated, window=2)
rolling_anti_normalized = make_rolling_window(_anti_normalized, window=2)
# Asymmetric couplings.
rolling_asym_directional_gate = make_rolling_window(_asym_directional_gate, window=2)
rolling_asym_forward_biased = make_rolling_window(_asym_forward_biased, window=2)
rolling_asym_source_sink = make_rolling_window(_asym_source_sink, window=2)
rolling_asym_tiny_mlp = make_rolling_window(_asym_tiny_mlp, window=2)
# Single source of truth: base function + window width, keyed by display
# name. Both the wrapped activations below and the plotting utility derive
# from this so the two never drift out of sync.
_ROLLING_BASE_FUNCTIONS = {
     "sin_diff": (_sin_diff, 2),
     "atan2_ramp": (_atan2_ramp, 3),
     "euclidean": (_euclidean, 2),
     "tan_ratio": (_tan_ratio, 2),
     "xlog_share": (_xlog_share, 2),
     "harmonic": (_harmonic, 2),
     "softmin": (_softmin_diff, 2),
     "sum_over_prod": (_sum_over_prod, 2),
     "parallel": (_parallel, 2),
     "ratio": (_ratio, 2),
     "soft_xor": (_soft_xor, 2),
     "soft_and": (_soft_and, 2),
     "soft_or": (_soft_or, 2),
     # Symmetric couplings g(x, y) = g(y, x).
     "sym_mixed_poly": (_sym_mixed_poly, 2),
     "sym_sig_sum": (_sym_sig_sum, 2),
     "sym_sig_prod": (_sym_sig_prod, 2),
     "sym_sig_hybrid": (_sym_sig_hybrid, 2),
     "sym_interaction_norm": (_sym_interaction_norm, 2),
     "sym_param_kernel": (_sym_param_kernel, 2),
     # Antisymmetric couplings g(x, y) = -g(y, x).
     "anti_diff": (_anti_diff, 2),
     "anti_tanh_diff": (_anti_tanh_diff, 2),
     "anti_mixed_poly": (_anti_mixed_poly, 2),
     "anti_gated": (_anti_gated, 2),
     "anti_normalized": (_anti_normalized, 2),
     # Asymmetric couplings (no constraint).
     "asym_directional_gate": (_asym_directional_gate, 2),
     "asym_forward_biased": (_asym_forward_biased, 2),
     "asym_source_sink": (_asym_source_sink, 2),
     "asym_tiny_mlp": (_asym_tiny_mlp, 2),
}


_ROLLING_ACTIVATIONS = {
     f"rolling_{name}": make_rolling_window(fn, window=window)
     for name, (fn, window) in _ROLLING_BASE_FUNCTIONS.items()
}
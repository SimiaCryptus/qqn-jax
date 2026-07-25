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

from experiments.models.spline_activations import _build_hermite_presets, _cubic_hermite_spline
from experiments.models.rolling_window_activation import _ROLLING_ACTIVATIONS

__all__ = ["ACTIVATIONS","UNIVARIATE_ACTIVATIONS", "resolve_activation", "parse_activation"]
UNIVARIATE_ACTIVATIONS = {
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
UNIVARIATE_ACTIVATIONS.update(
    {
        name: (
            lambda x, _xs=xs, _ys=ys, _ms=ms, _p=period: _cubic_hermite_spline(
                x, _xs, _ys, _ms, _p
            )
        )
        for name, (xs, ys, ms, period) in _build_hermite_presets().items()
    }
)
ACTIVATIONS = dict(UNIVARIATE_ACTIVATIONS)
ACTIVATIONS.update(_ROLLING_ACTIVATIONS)


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
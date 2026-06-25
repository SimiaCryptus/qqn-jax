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

try:  # rolling-window activations are optional (qqn_jax extra)
    from qqn_jax.rolling_window_activation import (
        rolling_sin_diff,
        rolling_atan2_ramp,
    )

    _ROLLING_ACTIVATIONS = {
        "rolling_sin": rolling_sin_diff,
        "rolling_atan2": rolling_atan2_ramp,
    }
except Exception:  # pragma: no cover - graceful degradation
    _ROLLING_ACTIVATIONS = {}

__all__ = ["ACTIVATIONS", "resolve_activation", "parse_activation"]
ACTIVATIONS = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "sine": jnp.sin,
    # Gaussian "bump" activation, exp(-x^2): localized, smooth, RBF-like.
    "gaussian": lambda x: jnp.exp(-(x**2)),
    # Triangle waveform: periodic, piecewise-linear in [-1, 1].
    "triangle": lambda x: (
        2.0 * jnp.abs(2.0 * (x / (2.0 * jnp.pi) - jnp.floor(x / (2.0 * jnp.pi) + 0.5)))
        - 1.0
    ),
    # Symmetric logarithm of |x|: ln(|x|+1) * sign(x).
    "logabs": lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x)),
    "tanh": jnp.tanh,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
    "softplus": jax.nn.softplus,
    # Sawtooth waveform: periodic ramp in [-1, 1).
    "sawtooth": lambda x: (
        2.0 * (x / (2.0 * jnp.pi) - jnp.floor(x / (2.0 * jnp.pi) + 0.5))
    ),
    "abs": jnp.abs,
    "identity": lambda x: x,
}
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

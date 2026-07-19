"""Pytree-based MLP for region-heavy sparse / quantization benchmarks.

Unlike ``FlatMLP`` (which lays parameters out as a single flat vector for
the QQN-vs-Optax comparison drivers), this representation keeps parameters
as a list of ``{"w", "b"}`` dicts. It is flattened to a flat vector only at
the solver boundary via ``jax.flatten_util.ravel_pytree`` so the flat-array
L-BFGS oracle / regions can operate on it, while sparsity and quantization
metrics are computed naturally per-layer on the structured pytree.

Shares the suite-wide init policy (He for relu, Glorot otherwise) and the
per-hidden-layer activation-cycling rule with ``FlatMLP``.
"""

from typing import Any, Dict, List

import jax
import jax.numpy as jnp

__all__ = [
    "init_params",
    "mlp_forward",
    "cross_entropy_loss",
    "test_loss",
    "sparsity",
    "round_params_to_grid",
]


def init_params(key, sizes: List[int], activation: Any = "tanh"):
    """Initialize MLP parameters with scaled Gaussian weights.

    Uses He-style init for ReLU hidden layers and Xavier/Glorot-style init
    otherwise. ``activation`` may be a single name string (applied uniformly)
    or a per-hidden-layer list of names for mixed-activation networks.
    """
    params: List[Dict[str, jnp.ndarray]] = []
    keys = jax.random.split(key, len(sizes) - 1)
    n_layers = len(sizes) - 1
    n_hidden = n_layers - 1
    if isinstance(activation, (list, tuple)):
        hidden_names = [
            activation[i % len(activation)] for i in range(max(n_hidden, 0))
        ]
    else:
        hidden_names = [activation] * max(n_hidden, 0)

    layer_names = hidden_names + ["identity"]
    for li, (k, (n_in, n_out)) in enumerate(zip(keys, zip(sizes[:-1], sizes[1:]))):
        wk, _bk = jax.random.split(k)
        act_name = layer_names[li] if li < len(layer_names) else "identity"
        if act_name == "relu":
            scale = jnp.sqrt(2.0 / n_in)
        else:
            scale = 1.0 / jnp.sqrt(n_in)
        params.append(
            {
                "w": scale * jax.random.normal(wk, (n_in, n_out)),
                "b": jnp.zeros((n_out,)),
            }
        )
    return params


def mlp_forward(params, x, activation=jnp.tanh):
    """Forward pass: configurable activation hidden layers, linear logits.

    ``activation`` may be a single callable (applied to every hidden layer)
    or a list/tuple of callables (one per hidden layer, cycled if short).
    """
    n_hidden = len(params) - 1
    if isinstance(activation, (list, tuple)):
        acts = [activation[i % len(activation)] for i in range(n_hidden)]
    else:
        acts = [activation] * n_hidden
    h = x
    for i, layer in enumerate(params[:-1]):
        h = acts[i](h @ layer["w"] + layer["b"])
    last = params[-1]
    return h @ last["w"] + last["b"]


def cross_entropy_loss(
    params, x, y, l2: float = 1e-4, regularizer=None, activation=jnp.tanh
):
    """Softmax cross-entropy with optional extra regularization.

    ``regularizer`` (optional) is a ``params -> scalar`` penalty added to the
    loss — used to inject L1 sparsity or the quantization-delta penalty that
    turns ordinary training into precision-optimized training.
    """
    logits = mlp_forward(params, x, activation)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.mean(jnp.take_along_axis(log_probs, y[:, None], axis=-1))
    reg = l2 * sum(jnp.sum(layer["w"] ** 2) for layer in params)
    if regularizer is not None:
        reg = reg + regularizer(params)
    return nll + reg


def test_loss(params, x, y, activation=jnp.tanh):
    """Plain softmax cross-entropy (no regularization) on a dataset."""
    logits = mlp_forward(params, x, activation)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, y[:, None], axis=-1))


def sparsity(params, threshold: float = 1e-6) -> float:
    """Fraction of weight entries with magnitude below ``threshold``."""
    total = 0
    zeros = 0
    for layer in params:
        w = layer["w"]
        total += w.size
        zeros += int(jnp.sum((jnp.abs(w) < threshold).astype(jnp.int32)))
    return zeros / max(total, 1)


def round_params_to_grid(params, bits: int = 4, lo: float = -1.0, hi: float = 1.0):
    """Round each layer's weights onto a ``bits``-level grid over ``[lo, hi]``.

    Returns a new parameter pytree with quantized weights (biases kept as-is).
    Used to measure the *post-rounding* loss: a precision-optimized network
    should show (almost) no loss increase after this rounding.
    """
    from qqn_jax.regularizers import round_to_grid

    quantized = []
    for layer in params:
        grid = round_to_grid(layer["w"], bits=bits, lo=lo, hi=hi)
        quantized.append({"w": grid, "b": layer["b"]})
    return quantized

"""Flat-vector multi-layer MLP: init, unpack, forward, loss, accuracy.

This is the canonical model for the QQN-vs-Optax comparison drivers. The
parameter vector is a single flat array laying out W_1, b_1, ..., W_L, b_L,
so it slots directly into QQN / Optax which both operate on flat vectors.

One init policy (He for relu, Glorot otherwise) and one per-hidden-layer
activation-cycling rule, shared by every benchmark.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "FlatMLP",
    "layer_dims",
    "param_layout",
    "partition_sizes",
    "init_params",
    "unpack",
    "forward",
    "make_loss",
    "accuracy",
]


def layer_dims(dim, hidden_sizes, n_classes):
    """Return the full list of layer dimensions [dim, *hidden, n_classes]."""
    return [dim, *list(hidden_sizes), n_classes]


def param_layout(dim, hidden_sizes, n_classes):
    """Cumulative offsets delimiting each W/b block in the flat vector."""
    dims = layer_dims(dim, hidden_sizes, n_classes)
    sizes = [0]
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        sizes.append(fan_in * fan_out)  # W block
        sizes.append(fan_out)  # b block
    return np.cumsum(sizes)


def partition_sizes(dim, hidden_sizes, n_classes):
    """Per-block segment sizes of the flat vector (W_1, b_1, ..., W_L, b_L).
    Returns a tuple of contiguous segment lengths suitable for QQN's
    ``partition_sizes`` (one oracle history per weight/bias block). The sum
    equals the total flat parameter count.
    """
    dims = layer_dims(dim, hidden_sizes, n_classes)
    sizes = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        sizes.append(fan_in * fan_out)  # W block
        sizes.append(fan_out)  # b block
    return tuple(int(s) for s in sizes)


def init_params(dim, hidden_sizes, n_classes, key, activation: Any = "sigmoid"):
    """Flat parameter vector for a multi-layer MLP.

    Uses He-style init for ReLU and Xavier/Glorot-style init otherwise, which
    keeps activations well-scaled at init. ``activation`` may be a single name
    string (applied uniformly) or a list of per-hidden-layer name strings.
    """
    dims = layer_dims(dim, hidden_sizes, n_classes)
    keys = jax.random.split(key, len(dims) - 1)
    n_weight_layers = len(dims) - 1
    n_hidden = n_weight_layers - 1
    if isinstance(activation, (list, tuple)):
        hidden_names = [
            activation[i % len(activation)] for i in range(max(n_hidden, 0))
        ]
    else:
        hidden_names = [activation] * max(n_hidden, 0)
    layer_names = hidden_names + ["identity"]  # output layer is linear
    blocks = []
    for li, (k, fan_in, fan_out) in enumerate(zip(keys, dims[:-1], dims[1:])):
        act_name = layer_names[li] if li < len(layer_names) else "identity"
        if act_name == "relu":
            scale = jnp.sqrt(2.0 / fan_in)  # He
        else:
            scale = jnp.sqrt(1.0 / fan_in)  # Glorot/Xavier-style
        w = jax.random.normal(k, (fan_in * fan_out,)) * scale
        b = jnp.zeros((fan_out,))
        blocks.append(w)
        blocks.append(b)
    return jnp.concatenate(blocks)


def unpack(params, dim, hidden_sizes, n_classes):
    """Split the flat vector into a list of (W, b) tuples, one per layer."""
    dims = layer_dims(dim, hidden_sizes, n_classes)
    o = param_layout(dim, hidden_sizes, n_classes)
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        w_start, w_end = o[2 * i], o[2 * i + 1]
        b_start, b_end = o[2 * i + 1], o[2 * i + 2]
        w = params[w_start:w_end].reshape(fan_in, fan_out)
        b = params[b_start:b_end]
        layers.append((w, b))
    return layers


def forward(params, X, dim, hidden_sizes, n_classes, activation=jax.nn.sigmoid):
    """Forward pass: activation after every layer except the output layer."""
    layers = unpack(params, dim, hidden_sizes, n_classes)
    h = X
    n_hidden = len(layers) - 1
    if isinstance(activation, (list, tuple)):
        acts = [activation[i % len(activation)] for i in range(n_hidden)]
    else:
        acts = [activation] * n_hidden
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < len(layers) - 1:
            h = acts[i](h)
    return h


def make_loss(X, y, dim, hidden_sizes, n_classes, l2=1e-4, activation=jax.nn.sigmoid):
    """Build a full-batch cross-entropy loss ``f(params) -> scalar``."""
    Y = jax.nn.one_hot(y, n_classes)

    def loss(params):
        logits = forward(params, X, dim, hidden_sizes, n_classes, activation)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.mean(jnp.sum(Y * log_probs, axis=-1))
        reg = 0.5 * l2 * jnp.sum(params**2)
        return ce + reg

    return loss


def accuracy(params, X, y, dim, hidden_sizes, n_classes, activation=jax.nn.sigmoid):
    """Classification accuracy of the flat-vector MLP."""
    logits = forward(params, X, dim, hidden_sizes, n_classes, activation)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == y).astype(jnp.float32))


class FlatMLP:
    """Thin object wrapper bundling the flat-vector MLP for the driver.

    Carries the geometry (``dim``, ``hidden_sizes``, ``n_classes``) and the
    resolved activation callable(s) so the driver can build the loss, init
    params, and compute accuracy without re-threading the geometry args.
    """

    def __init__(self, dim, hidden_sizes, n_classes, activation_fn, activation_name):
        self.dim = dim
        self.hidden_sizes = list(hidden_sizes)
        self.n_classes = n_classes
        self.activation_fn = activation_fn
        self.activation_name = activation_name

    @property
    def n_hidden_layers(self):
        return len(self.hidden_sizes)

    @property
    def partition_sizes(self):
        """Per-layer (W/b block) flat-vector segment sizes for QQN."""
        return partition_sizes(self.dim, self.hidden_sizes, self.n_classes)

    def init_params(self, key):
        return init_params(
            self.dim,
            self.hidden_sizes,
            self.n_classes,
            key,
            activation=self.activation_name,
        )

    def make_loss(self, X, y, l2=1e-4):
        return make_loss(
            X,
            y,
            self.dim,
            self.hidden_sizes,
            self.n_classes,
            l2=l2,
            activation=self.activation_fn,
        )

    def accuracy(self, params, X, y):
        return accuracy(
            params,
            X,
            y,
            self.dim,
            self.hidden_sizes,
            self.n_classes,
            self.activation_fn,
        )

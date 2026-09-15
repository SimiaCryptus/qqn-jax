"""Model package: flat-vector MLP, activations, and topology parsing."""

from experiments.models import mlp, pytree_mlp
from experiments.models.activations import (
    ACTIVATIONS,
    parse_activation,
    resolve_activation,
)
from experiments.models.topology import parse_hidden_sizes

__all__ = [
    "ACTIVATIONS",
    "mlp",
    "parse_activation",
    "parse_hidden_sizes",
    "pytree_mlp",
    "resolve_activation",
]

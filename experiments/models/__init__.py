"""Model package: flat-vector MLP, activations, and topology parsing."""

from experiments.models.activations import (
    ACTIVATIONS,
    parse_activation,
    resolve_activation,
)
from experiments.models.topology import parse_hidden_sizes
from experiments.models import mlp

__all__ = [
    "ACTIVATIONS",
    "parse_activation",
    "resolve_activation",
    "parse_hidden_sizes",
    "mlp",
]

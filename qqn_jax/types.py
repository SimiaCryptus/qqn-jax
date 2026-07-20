"""Typed interfaces for QQN-JAX.

All array types are annotated using ``chex.Array`` / ``jaxtyping`` so that
shapes and dtypes are documented and (optionally) runtime-checkable.
"""

from typing import Any, Callable, Tuple

import chex
from jaxtyping import Array, Float

Scalar = Float[Array, ""]
Params = Float[Array, " n"]
Grad = Float[Array, " n"]
Direction = Float[Array, " n"]
Value = Float[Array, ""]
ObjectiveFn = Callable[..., Scalar]
ValueAndGradFn = Callable[..., Tuple[Value, Grad]]

__all__ = [
    "Params",
    "Grad",
    "Direction",
    "Value",
    "ObjectiveFn",
    "ValueAndGradFn",
    "Any",
    "chex",
]
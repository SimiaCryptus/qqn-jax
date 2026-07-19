"""Path-strategy abstractions for QQN line-search augmentations.

A *path strategy* (``PathStrategy``) is the shared component that remaps
the scalar line-search parameter ``t`` into the probe point ``x + d(t)``
and its velocity ``d'(t)``. ``qqn_jax.paths.linear``,
``qqn_jax.paths.quadratic`` and ``qqn_jax.paths.spline`` all build their
probes through this shared interface (``qqn_jax.paths.base.make_evaluator``),
so every consumer — regardless of which curve it traverses or which line
search wraps it — constructs probe points identically.
"""

from qqn_jax.paths.base import PathStrategy, make_evaluator, path_search
from qqn_jax.paths.linear import LINEAR_PATH
from qqn_jax.paths.quadratic import (
    QUADRATIC_PATH,
    quadratic_path,
    quadratic_path_derivative,
)

__all__ = [
    "PathStrategy",
    "make_evaluator",
     "path_search",
    "LINEAR_PATH",
    "QUADRATIC_PATH",
    "quadratic_path",
    "quadratic_path_derivative",
]
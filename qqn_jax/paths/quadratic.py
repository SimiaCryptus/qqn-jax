"""Quadratic (parabolic) path — QQN's canonical curve.

    d(t) = t(1-t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]

blending the steepest-descent tangent with the oracle endpoint at ``t = 1``.
See ``docs/paper/draft.md`` §4.2 and ``qqn_jax.solver`` for the full
derivation.

``quadratic_path`` / ``quadratic_path_derivative`` are re-exported here
(from ``qqn_jax.utils``, where they remain defined for backward
compatibility) and packaged as ``QUADRATIC_PATH``, the canonical
``PathStrategy`` instance. ``qqn_jax.paths.spline`` uses ``QUADRATIC_PATH``
by default so that every spline probe stays on the exact curve traversed
by the wrapped inner line search.
"""

from qqn_jax.utils import quadratic_path, quadratic_path_derivative
from qqn_jax.paths.base import PathStrategy


QUADRATIC_PATH = PathStrategy(offset=quadratic_path, velocity=quadratic_path_derivative)

__all__ = ["quadratic_path", "quadratic_path_derivative", "QUADRATIC_PATH"]

"""Quadratic (parabolic) path — QQN's canonical curve.

    d(t) = t(1-t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]

blending the steepest-descent tangent with the oracle endpoint at ``t = 1``.
See ``docs/paper/draft.md`` §4.2 and ``qqn_jax.solver`` for the full
derivation.

``quadratic_path`` / ``quadratic_path_derivative`` are packaged as
``QUADRATIC_PATH``, the canonical ``PathStrategy`` instance.
``qqn_jax.paths.spline`` uses ``QUADRATIC_PATH`` by default so that every
spline probe stays on the exact curve traversed by the wrapped inner line
search.
"""

import jax

from qqn_jax.paths.base import PathStrategy


def _quadratic_path(t, grad_dir, qn_dir):
    """Construct the QQN quadratic path direction.

        d(t) = t(1-t)(-∇f) + t²(-H∇f)

    Args:
        t: interpolation parameter in [0, 1].
        grad_dir: steepest descent direction ``-∇f``.
        qn_dir: L-BFGS direction ``-H∇f``.

    Returns:
        The blended direction ``d(t)`` as a pytree.
    """
    a = t * (1.0 - t)
    b = t * t
    return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, qn_dir)


def quadratic_path_derivative(t, grad_dir, qn_dir):
    """Derivative of the quadratic path w.r.t. ``t``.

    d'(t) = (1 - 2t)(-∇f) + 2t(-H∇f)
    """
    a = 1.0 - 2.0 * t
    b = 2.0 * t
    return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, qn_dir)


QUADRATIC_PATH = PathStrategy(
    offset=_quadratic_path, velocity=quadratic_path_derivative
)

__all__ = ["quadratic_path_derivative", "QUADRATIC_PATH"]

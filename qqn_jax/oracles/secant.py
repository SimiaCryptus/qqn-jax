import jax
from jax import numpy as jnp

from qqn_jax.utils import tree_negative
from typing import NamedTuple
from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.point_history import publish, secant_view


class SecantState(NamedTuple):
    prev_params: jnp.ndarray
    prev_grad: jnp.ndarray
    alpha: jnp.ndarray
    step_count: jnp.ndarray


def SecantOracle(alpha0: float = 1.0, alpha_max: float = 1e3) -> Oracle:
    """Barzilai-Borwein curvature oracle (matrix-free, O(n) memory).
    The ``t = 1`` endpoint is the gradient scaled by an inverse-curvature
    estimate inferred from the *realized* secant of the previous step::
        s = x      - x_prev
        y = ∇f     - ∇f_prev
        α = ⟨s, s⟩ / ⟨s, y⟩        (BB1 step; the Rayleigh quotient's inverse)
        direction = -α · ∇f
    This reuses the curvature signal that the path *already measured* —
    no Hessian, no history buffers. It is a featherweight companion for a
    ``Fallback`` and a probe of how much curvature lives in a single step.
    The very first step (no secant yet) falls back to ``-alpha0 · ∇f``,
    i.e. plain scaled steepest descent, preserving the ``d'(0)`` anchor.
    """
    eps = 1e-12

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return SecantState(
            prev_params=params,
            prev_grad=zeros,
            alpha=jnp.asarray(alpha0, dtype=params.dtype),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        d = tree_negative(jax.tree_util.tree_map(lambda g: state.alpha * g, grad))
        return d, state

    def update(state, info):

        points = publish(info)
        if points is None:
            s = info.new_params - info.params
            y = info.new_grad - info.grad
        else:
            s, y = secant_view(points).newest_secant()
        ss = jnp.vdot(s, s)
        sy = jnp.vdot(s, y)

        curvature_ok = sy > eps
        bb = ss / jnp.where(curvature_ok, sy, 1.0)
        new_alpha = jnp.where(curvature_ok, jnp.clip(bb, eps, alpha_max), state.alpha)
        return SecantState(
            prev_params=info.new_params,
            prev_grad=info.new_grad,
            alpha=new_alpha.astype(state.alpha.dtype),
            step_count=state.step_count + 1,
        )

    return Oracle(init=init, direction=direction, update=update)


def _ordered_probe_secants(info, max_replay=None):
    """Backward-compatible shim over the point-history store.

    Deprecated: prefer :func:`qqn_jax.oracles.point_history.publish` /
    :func:`~qqn_jax.oracles.point_history.secant_view`. Retained so existing
    oracles keep working while they migrate to the store view.

    Returns ``(params_seq, grad_seq, valid_seq)`` oldest-first (increasing α),
    terminating with the accepted point, or ``None`` when unavailable.
    """
    points = publish(info, max_replay=max_replay)
    if points is None:
        return None
    return points.params_seq, points.grad_seq, points.valid_seq

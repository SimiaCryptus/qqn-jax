import jax
from jax import numpy as jnp

from qqn_jax.utils import tree_negative
from typing import NamedTuple
from qqn_jax.oracles.oracle import Oracle


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

        ordered = _ordered_probe_secants(info)
        if ordered is None:
            s = info.new_params - info.params
            y = info.new_grad - info.grad
        else:
            params_seq, grad_seq, valid_seq = ordered

            anchor_p = jnp.concatenate([info.params[None, :], params_seq[:-1]], axis=0)
            anchor_g = jnp.concatenate([info.grad[None, :], grad_seq[:-1]], axis=0)

            prev_valid = valid_seq[:-1]
            idx = jnp.max(jnp.where(prev_valid, jnp.arange(prev_valid.shape[0]), 0))
            p_prev = anchor_p[idx]
            g_prev = anchor_g[idx]
            s = info.new_params - p_prev
            y = info.new_grad - g_prev
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
    """Extract probe points ordered by increasing α, masked by validity.
    Returns ``(params_seq, grad_seq, valid_seq)`` where the sequence runs
    oldest-first (increasing α) and terminates with the accepted point as the
    newest (always-valid) entry. When ``max_replay`` is given only the probes
    CLOSEST to the accepted step (largest α among valid, descent-gated probes)
    are retained, capping how many collinear probes are folded in.
    Returns ``None`` when the probe buffers are not populated (no alphas /
    valid mask / params), signalling the caller to fall back to a single-pair
    (accepted-point-only) update.
    """
    if (
        info.probe_params is None
        or info.probe_alphas is None
        or info.probe_valid is None
    ):
        return None
    k = info.probe_alphas.shape[0]
    if max_replay is not None:
        n_keep = min(max_replay, k)

        ranked_alpha = jnp.where(info.probe_valid, info.probe_alphas, -jnp.inf)
        keep_order = jnp.argsort(-ranked_alpha)[:n_keep]
        kept_params = info.probe_params[keep_order]
        kept_grads = info.probe_grads[keep_order]
        kept_valid = info.probe_valid[keep_order]
        kept_alphas = info.probe_alphas[keep_order]
    else:
        kept_params = info.probe_params
        kept_grads = info.probe_grads
        kept_valid = info.probe_valid
        kept_alphas = info.probe_alphas

    inner = jnp.argsort(jnp.where(kept_valid, kept_alphas, jnp.inf))
    probe_params = kept_params[inner]
    probe_grads = kept_grads[inner]
    probe_valid = kept_valid[inner]

    params_seq = jnp.concatenate([probe_params, info.new_params[None, :]], axis=0)
    grad_seq = jnp.concatenate([probe_grads, info.new_grad[None, :]], axis=0)
    valid_seq = jnp.concatenate([probe_valid, jnp.asarray([True])], axis=0)
    return params_seq, grad_seq, valid_seq

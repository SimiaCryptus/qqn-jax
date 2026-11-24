"""Adam-preconditioned L-BFGS hybrid oracle for QQN.

Replaces standard L-BFGS's scalar initial inverse-Hessian ``H0 = γI`` in the
middle of Nocedal & Wright Algorithm 7.4 with Adam's per-parameter diagonal
preconditioner ``H0 = diag(lr / (sqrt(v_hat) + eps))``.

This combines per-parameter diagonal conditioning (essential for anisotropic
architectures like Transformers with mixed embedding, norm, and MLP scales)
with L-BFGS's rank-2m off-diagonal curvature corrections.
"""

from typing import Any, NamedTuple

import jax
from jax import numpy as jnp

from qqn_jax.lbfgs import (
    LBFGSState,
    init_lbfgs_state,
    update_lbfgs_history,
    update_lbfgs_history_batch,
)
from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.point_history import publish
from qqn_jax.utils import tree_scalar_dtype, tree_vdot, tree_where


class AdamPreconditionedLBFGSState(NamedTuple):
    """State for the Adam-preconditioned L-BFGS hybrid oracle.

    Attributes:
        lbfgs: underlying L-BFGS curvature history ``(s, y, rho)``.
        v: exponential moving average of squared gradients (``float32``).
        step: iteration counter for bias correction of ``v``.
    """

    lbfgs: LBFGSState
    v: Any
    step: jnp.ndarray


def AdamPreconditionedLBFGSOracle(
    history_size: int = 10,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    learning_rate: float = 1e-3,
    max_probe_replay: int = 2,
) -> Oracle:
    """Hybrid L-BFGS oracle with Adam diagonal initial Hessian ``H0``.

    In the two-loop recursion:
      1. First loop (newest -> oldest): ``α_i = ρ_i ⟨s_i, q⟩``, ``q ← q - α_i y_i``.
      2. Diagonal scaling: ``r = H0 · q`` with ``H0 = diag(lr / (√v̂ + ε))``.
      3. Second loop (oldest -> newest): ``β_i = ρ_i ⟨y_i, r⟩``, ``r ← r + (α_i - β_i) s_i``.
      4. Direction: ``d = -r`` (with descent safeguard falling back to ``-H0 · ∇f``).

    Because ``H0`` is strictly positive definite and admitted pairs satisfy
    ``yᵀs > 0``, the resulting inverse-Hessian approximation is positive
    definite by construction.
    """

    def init(params) -> AdamPreconditionedLBFGSState:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        v_zeros = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, dtype=jnp.float32), params
        )
        return AdamPreconditionedLBFGSState(
            lbfgs=init_lbfgs_state(params, zeros, history_size),
            v=v_zeros,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def _apply_h0(q, v, step_t, scalar_dtype):
        bc2 = jnp.asarray(1.0, dtype=scalar_dtype) - jnp.asarray(
            beta2, dtype=scalar_dtype
        ) ** step_t.astype(scalar_dtype)
        lr = jnp.asarray(learning_rate, dtype=scalar_dtype)
        eps = jnp.asarray(epsilon, dtype=scalar_dtype)
        return jax.tree_util.tree_map(
            lambda qi, vi: (
                (lr / (jnp.sqrt(vi.astype(scalar_dtype) / bc2) + eps))
                * qi.astype(scalar_dtype)
            ).astype(qi.dtype),
            q,
            v,
        )

    def direction(params, grad, state: AdamPreconditionedLBFGSState):
        del params
        scalar_dtype = tree_scalar_dtype(grad)
        t = state.step + jnp.asarray(1, dtype=jnp.int32)
        v_eff = jax.tree_util.tree_map(
            lambda vi, g: beta2 * vi
            + (1.0 - beta2) * (g.astype(vi.dtype) * g.astype(vi.dtype)),
            state.v,
            grad,
        )

        s_hist = state.lbfgs.s_history
        y_hist = state.lbfgs.y_history
        rho_hist = state.lbfgs.rho_history

        def first_loop(carry, inputs):
            q = carry
            s_i, y_i, rho_i = inputs
            alpha_i = rho_i * tree_vdot(s_i, q, accum_dtype=scalar_dtype)
            q = jax.tree_util.tree_map(
                lambda qi, yi: qi - alpha_i.astype(qi.dtype) * yi.astype(qi.dtype),
                q,
                y_i,
            )
            return q, alpha_i

        q, alphas = jax.lax.scan(first_loop, grad, (s_hist, y_hist, rho_hist))
        r = _apply_h0(q, v_eff, t, scalar_dtype)

        def second_loop(carry, inputs):
            r = carry
            s_i, y_i, rho_i, alpha_i = inputs
            beta_i = rho_i * tree_vdot(y_i, r, accum_dtype=scalar_dtype)
            coeff = alpha_i - beta_i
            r = jax.tree_util.tree_map(
                lambda ri, si: ri + coeff.astype(ri.dtype) * si.astype(ri.dtype),
                r,
                s_i,
            )
            return r, None

        r, _ = jax.lax.scan(
            second_loop,
            r,
            (s_hist, y_hist, rho_hist, alphas),
            reverse=True,
        )

        d = jax.tree_util.tree_map(lambda ri: -ri, r)
        # Descent safeguard: fall back to -H0 * grad if numerical drift breaks descent
        gd = tree_vdot(grad, d, accum_dtype=scalar_dtype)
        d_fb = jax.tree_util.tree_map(
            lambda ri: -ri, _apply_h0(grad, v_eff, t, scalar_dtype)
        )
        d = tree_where(gd < jnp.asarray(0.0, dtype=scalar_dtype), d, d_fb)
        return d, state

    def update(
        state: AdamPreconditionedLBFGSState, info
    ) -> AdamPreconditionedLBFGSState:
        def fold_v(v, g):
            return jax.tree_util.tree_map(
                lambda vi, gi: beta2 * vi
                + (1.0 - beta2) * (gi.astype(vi.dtype) * gi.astype(vi.dtype)),
                v,
                g,
            )

        points = publish(info, max_replay=max_probe_replay)
        if points is None:
            new_lbfgs = update_lbfgs_history(
                state.lbfgs, info.new_params, info.new_grad, history_size
            )
            new_v = fold_v(state.v, info.grad)
            return AdamPreconditionedLBFGSState(
                lbfgs=new_lbfgs, v=new_v, step=state.step + 1
            )

        new_lbfgs = update_lbfgs_history_batch(
            state.lbfgs,
            points.params_seq,
            points.grad_seq,
            points.valid_seq,
            history_size,
        )

        def body(v, elem):
            g, valid = elem
            v_new = fold_v(v, g)
            return tree_where(valid, v_new, v), None

        new_v, _ = jax.lax.scan(
            body, state.v, (points.grad_seq, points.valid_seq)
        )
        return AdamPreconditionedLBFGSState(
            lbfgs=new_lbfgs, v=new_v, step=state.step + 1
        )

    return Oracle(init=init, direction=direction, update=update)


__all__ = ["AdamPreconditionedLBFGSOracle", "AdamPreconditionedLBFGSState"]

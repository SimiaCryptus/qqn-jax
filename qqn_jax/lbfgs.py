"""L-BFGS oracle wrapper.

This module wraps a limited-memory BFGS two-loop recursion as a single-step
*oracle*. Given the current gradient and a history of (s, y) pairs, it
produces the quasi-Newton direction ``-H∇f`` where ``H`` is the implicit
inverse-Hessian approximation.

The state is stored in a JIT-compatible NamedTuple with fixed-size circular
buffers so the whole thing can live inside ``lax`` control flow.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_vdot


class LBFGSState(NamedTuple):
    """State for the L-BFGS oracle.

    Attributes:
        s_history: buffer of parameter differences, shape (history_size, n).
        y_history: buffer of gradient differences, shape (history_size, n).
        rho_history: buffer of 1 / (yᵀs), shape (history_size,).
        count: number of valid entries currently stored.
        gamma: scaling factor for the initial Hessian H0 = gamma * I.
        prev_params: previous parameters (for computing s).
        prev_grad: previous gradient (for computing y).
    """

    s_history: jnp.ndarray
    y_history: jnp.ndarray
    rho_history: jnp.ndarray
    count: jnp.ndarray
    gamma: jnp.ndarray
    prev_params: jnp.ndarray
    prev_grad: jnp.ndarray


def init_lbfgs_state(params, grad, history_size: int) -> LBFGSState:
    """Initialize an empty L-BFGS state for the given parameter shape."""
    n = params.shape[0]
    return LBFGSState(
        s_history=jnp.zeros((history_size, n), dtype=params.dtype),
        y_history=jnp.zeros((history_size, n), dtype=params.dtype),
        rho_history=jnp.zeros((history_size,), dtype=params.dtype),
        count=jnp.asarray(0, dtype=jnp.int32),
        gamma=jnp.asarray(1.0, dtype=params.dtype),
        prev_params=params,
        prev_grad=grad,
    )


def update_lbfgs_history(
    state: LBFGSState, params, grad, history_size: int
) -> LBFGSState:
    """Push a new (s, y) pair into the circular history buffer.

    The update is only applied if the curvature condition ``yᵀs > eps`` is
    satisfied; otherwise the history is left unchanged (a standard L-BFGS
    safeguard for non-convex problems).
    """
    s = params - state.prev_params
    y = grad - state.prev_grad
    ys = jnp.vdot(y, s)
    yy = jnp.vdot(y, y)

    eps = jnp.asarray(1e-10, dtype=params.dtype)
    valid = ys > eps

    # Roll buffers to make room at index 0 (most recent first).
    new_s = jnp.where(
        valid,
        jnp.roll(state.s_history, shift=1, axis=0).at[0].set(s),
        state.s_history,
    )
    new_y = jnp.where(
        valid,
        jnp.roll(state.y_history, shift=1, axis=0).at[0].set(y),
        state.y_history,
    )
    rho = jnp.where(valid, 1.0 / ys, 0.0)
    new_rho = jnp.where(
        valid,
        jnp.roll(state.rho_history, shift=1, axis=0).at[0].set(rho),
        state.rho_history,
    )
    new_count = jnp.where(
        valid,
        jnp.minimum(state.count + 1, history_size),
        state.count,
    )
    new_gamma = jnp.where(valid, ys / yy, state.gamma)

    return LBFGSState(
        s_history=new_s,
        y_history=new_y,
        rho_history=new_rho,
        count=new_count,
        gamma=new_gamma,
        prev_params=params,
        prev_grad=grad,
    )


def lbfgs_direction(state: LBFGSState, grad) -> jnp.ndarray:
    """Compute the L-BFGS direction ``-H∇f`` via the two-loop recursion.

    Inactive history slots (index >= count) are masked out so the result is
    correct even before the buffer has filled up.
    """
    history_size = state.s_history.shape[0]
    idx = jnp.arange(history_size)
    active = idx < state.count  # shape (history_size,)

    q = grad

    # First loop (recent -> old).
    def first_loop(carry, i):
        q = carry
        s_i = state.s_history[i]
        y_i = state.y_history[i]
        rho_i = state.rho_history[i]
        is_active = active[i]
        alpha_i = jnp.where(is_active, rho_i * jnp.vdot(s_i, q), 0.0)
        q = q - jnp.where(is_active, alpha_i, 0.0) * y_i
        return q, alpha_i

    q, alphas = jax.lax.scan(first_loop, q, idx)

    # Initial Hessian scaling: H0 = gamma * I.
    r = state.gamma * q

    # Second loop (old -> recent).
    def second_loop(carry, i):
        r = carry
        s_i = state.s_history[i]
        y_i = state.y_history[i]
        rho_i = state.rho_history[i]
        alpha_i = alphas[i]
        is_active = active[i]
        beta = jnp.where(is_active, rho_i * jnp.vdot(y_i, r), 0.0)
        r = r + jnp.where(is_active, alpha_i - beta, 0.0) * s_i
        return r, None

    r, _ = jax.lax.scan(second_loop, r, idx, reverse=True)

    # Direction is -H∇f.
    return -r
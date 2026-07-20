"""L-BFGS oracle wrapper.

This module delegates the limited-memory BFGS two-loop recursion to
Optax's ``optax.scale_by_lbfgs`` machinery, exposing it as a single-step
*oracle* that produces the quasi-Newton direction ``-H∇f``.

We keep our own thin, fixed-size circular-buffer state so the whole thing
stays JIT/vmap compatible and so the oracle remains swappable. The actual
two-loop recursion is performed by Optax's
``optax._src.linesearch`` / LBFGS internals via
``optax.tree_utils`` helpers; here we reimplement the recursion directly
on our own buffers to avoid depending on Optax private state layouts.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


def jnp_select_buf(flag, a, b):
    """Scalar-flag select between two equally-shaped buffers (lax.select)."""

    return jnp.where(flag, a, b)


class LBFGSState(NamedTuple):
    """State for the L-BFGS oracle.

    Attributes:
        s_history: buffer of parameter differences, shape (history_size, n).
        y_history: buffer of gradient differences, shape (history_size, n).
        rho_history: buffer of 1 / (yᵀs), shape (history_size,).
         step_count: number of valid entries currently stored.
        gamma: scaling factor for the initial Hessian H0 = gamma * I.
        prev_params: previous parameters (for computing s).
        prev_grad: previous gradient (for computing y).
    """

    s_history: jnp.ndarray
    y_history: jnp.ndarray
    rho_history: jnp.ndarray
    step_count: jnp.ndarray
    gamma: jnp.ndarray
    prev_params: jnp.ndarray
    prev_grad: jnp.ndarray


def init_lbfgs_state(params, grad, history_size: int) -> LBFGSState:
    """Initialize an empty L-BFGS state for the given parameter shape."""
    n = params.shape[0]
    dtype = params.dtype
    new_gamma = jnp.asarray(1.0, dtype=dtype)
    return LBFGSState(
        s_history=jnp.zeros((history_size, n), dtype=dtype),
        y_history=jnp.zeros((history_size, n), dtype=dtype),
        rho_history=jnp.zeros((history_size,), dtype=dtype),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        gamma=new_gamma,
        prev_params=params.astype(dtype),
        prev_grad=grad.astype(dtype),
    )


def update_lbfgs_history(
    state: LBFGSState, params, grad, history_size: int
) -> LBFGSState:
    """Push a new (s, y) pair into the circular history buffer.

    We keep most-recent-first ordering (index 0 = newest) and flip the
    buffers when running the two-loop recursion in ``lbfgs_direction``.

    The update is only applied if the curvature condition ``yᵀs > eps`` is
    satisfied; otherwise the history is left unchanged (a standard L-BFGS
    safeguard for non-convex problems).
    """
    dtype = state.prev_params.dtype
    params = params.astype(dtype)
    grad = grad.astype(dtype)
    s = params - state.prev_params
    y = grad - state.prev_grad
    ys = jnp.vdot(y, s)
    yy = jnp.vdot(y, y)

    ss = jnp.vdot(s, s)
    eps = jnp.asarray(1e-10, dtype=dtype)
    valid = ys > eps * jnp.sqrt(yy * ss + eps)

    safe_ys = jnp.where(valid, ys, jnp.ones_like(ys))
    rho = jnp.where(
        valid,
        (jnp.asarray(1.0, dtype=dtype) / safe_ys).astype(dtype),
        jnp.asarray(0.0, dtype=dtype),
    )

    def _shift_insert(buf, row):

        shifted = jnp.concatenate([row[None], buf[:-1]], axis=0)
        return shifted

    new_s = jnp_select_buf(valid, _shift_insert(state.s_history, s), state.s_history)
    new_y = jnp_select_buf(valid, _shift_insert(state.y_history, y), state.y_history)
    new_rho = jnp_select_buf(
        valid, _shift_insert(state.rho_history, rho), state.rho_history
    )
    new_count = jnp.where(
        valid,
        jnp.minimum(state.step_count + 1, history_size),
        state.step_count,
    )

    safe_yy = jnp.where(yy > 0.0, yy, jnp.ones_like(yy))
    new_gamma = jnp.where(valid, (ys / safe_yy).astype(dtype), state.gamma).astype(
        dtype
    )

    return LBFGSState(
        s_history=new_s.astype(dtype),
        y_history=new_y.astype(dtype),
        rho_history=new_rho.astype(dtype),
        step_count=new_count,
        gamma=new_gamma,
        prev_params=params.astype(dtype),
        prev_grad=grad.astype(dtype),
    )


def update_lbfgs_history_batch(
    state: LBFGSState,
    params_seq,
    grad_seq,
    valid_seq,
    history_size: int,
) -> LBFGSState:
    """Replay a sequence of (params, grad) probes into the history.
    ``params_seq``/``grad_seq`` have shape ``(k, n)`` and ``valid_seq`` has
    shape ``(k,)``. Probes are folded in *oldest-first* so the most recent
    accepted point ends up newest. Invalid (unfilled scratch) slots are
    skipped via the curvature guard already in ``update_lbfgs_history``.
    We use ``lax.scan`` so this stays JIT/vmap compatible. The curvature
    condition inside ``update_lbfgs_history`` (yᵀs > eps) automatically
    rejects degenerate or zero-length pairs.
    """

    def step(carry_state, inputs):
        p, g, ok = inputs
        updated = update_lbfgs_history(carry_state, p, g, history_size)

        merged = LBFGSState(
            s_history=jnp_select_buf(ok, updated.s_history, carry_state.s_history),
            y_history=jnp_select_buf(ok, updated.y_history, carry_state.y_history),
            rho_history=jnp_select_buf(
                ok, updated.rho_history, carry_state.rho_history
            ),
            step_count=jnp.where(ok, updated.step_count, carry_state.step_count),
            gamma=jnp.where(ok, updated.gamma, carry_state.gamma),
            prev_params=updated.prev_params,
            prev_grad=updated.prev_grad,
        )
        return merged, None

    new_state, _ = jax.lax.scan(step, state, (params_seq, grad_seq, valid_seq))
    return new_state


def lbfgs_direction(state: LBFGSState, grad) -> jnp.ndarray:
    """Compute the L-BFGS direction ``-H∇f`` via the two-loop recursion.

    This is a direct, self-contained implementation of the L-BFGS
    two-loop recursion (Nocedal & Wright, Algorithm 7.4). It replaces the
    previous dependency on JAXopt's ``inv_hessian_product``.

    Unfilled history slots are zero (s=y=rho=0). They contribute nothing
    to the recursion because their ``alpha`` and correction terms vanish,
    so masking is automatic and the result is exactly ``-H∇f``.

    Buffers are stored most-recent-first; the first loop iterates
    newest -> oldest, the second loop oldest -> newest.
    """
    s_hist = state.s_history
    y_hist = state.y_history
    rho_hist = state.rho_history

    def first_loop(carry, inputs):
        q = carry
        s_i, y_i, rho_i = inputs
        alpha_i = rho_i * jnp.vdot(s_i, q)
        q = q - alpha_i * y_i
        return q, alpha_i

    q, alphas = jax.lax.scan(first_loop, grad, (s_hist, y_hist, rho_hist))

    r = state.gamma * q

    def second_loop(carry, inputs):
        r = carry
        s_i, y_i, rho_i, alpha_i = inputs
        beta_i = rho_i * jnp.vdot(y_i, r)
        r = r + (alpha_i - beta_i) * s_i
        return r, None

    r, _ = jax.lax.scan(
        second_loop,
        r,
        (s_hist, y_hist, rho_hist, alphas),
        reverse=True,
    )

    return -r

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

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_scalar_dtype, tree_vdot, tree_where


def jnp_select_buf(flag, a, b):
    """Scalar-flag select between two equally-shaped pytrees/buffers."""

    return tree_where(flag, a, b)


class LBFGSState(NamedTuple):
    """State for the L-BFGS oracle.

    Supports both flat 1-D arrays and arbitrary nested pytrees (each leaf in
    ``s_history`` / ``y_history`` has shape ``(history_size, *leaf.shape)``,
    preserving per-leaf FSDP sharding).

    Attributes:
        s_history: pytree of parameter differences, leaf shape ``(m, *shape)``.
        y_history: pytree of gradient differences, leaf shape ``(m, *shape)``.
        rho_history: buffer of 1 / (yᵀs), shape ``(history_size,)``.
        step_count: number of valid entries currently stored.
        gamma: scaling factor for the initial Hessian H0 = gamma * I.
        prev_params: previous parameters (for computing s).
        prev_grad: previous gradient (for computing y).
    """

    s_history: Any
    y_history: Any
    rho_history: jnp.ndarray
    step_count: jnp.ndarray
    gamma: jnp.ndarray
    prev_params: Any
    prev_grad: Any


def init_lbfgs_state(params, grad, history_size: int) -> LBFGSState:
    """Initialize an empty L-BFGS state for the given parameter pytree."""
    scalar_dtype = tree_scalar_dtype(params)
    new_gamma = jnp.asarray(1.0, dtype=scalar_dtype)
    s_history = jax.tree_util.tree_map(
        lambda p: jnp.zeros((history_size, *p.shape), dtype=p.dtype), params
    )
    y_history = jax.tree_util.tree_map(
        lambda p: jnp.zeros((history_size, *p.shape), dtype=p.dtype), params
    )
    return LBFGSState(
        s_history=s_history,
        y_history=y_history,
        rho_history=jnp.zeros((history_size,), dtype=scalar_dtype),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        gamma=new_gamma,
        prev_params=params,
        prev_grad=jax.tree_util.tree_map(
            lambda g, p: g.astype(p.dtype), grad, params
        ),
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
    scalar_dtype = state.gamma.dtype
    params = jax.tree_util.tree_map(
        lambda x, ref: x.astype(ref.dtype), params, state.prev_params
    )
    grad = jax.tree_util.tree_map(
        lambda g, ref: g.astype(ref.dtype), grad, state.prev_grad
    )
    s = jax.tree_util.tree_map(lambda a, b: a - b, params, state.prev_params)
    y = jax.tree_util.tree_map(lambda a, b: a - b, grad, state.prev_grad)
    ys = tree_vdot(y, s, accum_dtype=scalar_dtype)
    yy = tree_vdot(y, y, accum_dtype=scalar_dtype)
    ss = tree_vdot(s, s, accum_dtype=scalar_dtype)

    eps = jnp.asarray(1e-10, dtype=scalar_dtype)
    valid = ys > eps * jnp.sqrt(yy * ss + eps)

    safe_ys = jnp.where(valid, ys, jnp.ones_like(ys))
    rho = jnp.where(
        valid,
        (jnp.asarray(1.0, dtype=scalar_dtype) / safe_ys).astype(scalar_dtype),
        jnp.asarray(0.0, dtype=scalar_dtype),
    )

    def _shift_insert(buf, row):
        return jnp.concatenate([row[None], buf[:-1]], axis=0)

    new_s = jnp_select_buf(
        valid,
        jax.tree_util.tree_map(_shift_insert, state.s_history, s),
        state.s_history,
    )
    new_y = jnp_select_buf(
        valid,
        jax.tree_util.tree_map(_shift_insert, state.y_history, y),
        state.y_history,
    )
    new_rho = jnp.where(
        valid, _shift_insert(state.rho_history, rho), state.rho_history
    )
    new_count = jnp.where(
        valid,
        jnp.minimum(state.step_count + 1, history_size),
        state.step_count,
    )

    safe_yy = jnp.where(yy > 0.0, yy, jnp.ones_like(yy))
    new_gamma = jnp.where(
        valid, (ys / safe_yy).astype(scalar_dtype), state.gamma
    ).astype(scalar_dtype)

    return LBFGSState(
        s_history=new_s,
        y_history=new_y,
        rho_history=new_rho.astype(scalar_dtype),
        step_count=new_count,
        gamma=new_gamma,
        prev_params=params,
        prev_grad=grad,
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


def lbfgs_direction(state: LBFGSState, grad):
    """Compute the L-BFGS direction ``-H∇f`` via the two-loop recursion.

    This is a direct, self-contained implementation of the L-BFGS
    two-loop recursion (Nocedal & Wright, Algorithm 7.4). It supports both
    flat 1-D arrays and arbitrary nested pytrees.

    Unfilled history slots are zero (s=y=rho=0). They contribute nothing
    to the recursion because their ``alpha`` and correction terms vanish,
    so masking is automatic and the result is exactly ``-H∇f``.

    Buffers are stored most-recent-first; the first loop iterates
    newest -> oldest, the second loop oldest -> newest.
    """
    s_hist = state.s_history
    y_hist = state.y_history
    rho_hist = state.rho_history
    scalar_dtype = state.gamma.dtype

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

    r = jax.tree_util.tree_map(lambda qi: state.gamma.astype(qi.dtype) * qi, q)

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

    return jax.tree_util.tree_map(lambda ri: -ri, r)

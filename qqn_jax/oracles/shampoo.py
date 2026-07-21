import jax
from jax import numpy as jnp
from typing import NamedTuple
from qqn_jax.oracles.oracle import Oracle


class ShampooState(NamedTuple):
    L: jnp.ndarray
    R: jnp.ndarray
    step: jnp.ndarray


def _matrix_inverse_pth_root(mat, p, epsilon):
    """Compute ``mat^{-1/p}`` for a symmetric PSD matrix via eigh."""
    n = mat.shape[0]
    dtype = mat.dtype
    mat = mat + epsilon * jnp.eye(n, dtype=mat.dtype)
    w, v = jnp.linalg.eigh(mat)
    w = jnp.maximum(w, epsilon)
    inv_root = w ** (-1.0 / p)
    return ((v * inv_root) @ v.T).astype(dtype)


def ShampooOracle(
    update_freq: int = 20,
    epsilon: float = 1e-6,
) -> Oracle:
    """Structure-aware preconditioned oracle (Shampoo).

    Operates on the flat parameter vector by reshaping it into a single
    matrix block. For the flat-vector setting used throughout
    ``qqn-jax`` the gradient ``g`` (shape ``(n,)``) is treated as a column
    and preconditioned via accumulated second-moment statistics.

    The inverse roots are recomputed on a fixed static cadence
    (``update_freq``) so the per-step cost stays amortized and the whole
    computation remains ``jit``-friendly.
    """

    def init(params):
        n = params.shape[0]
        return ShampooState(
            L=jnp.zeros((n, n), dtype=params.dtype),
            R=jnp.zeros((1, 1), dtype=params.dtype),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        dtype = state.L.dtype
        grad = grad.astype(dtype)
        g = grad.reshape(-1, 1)

        do_refresh = (state.step % update_freq) == 0

        R_new = state.R + jnp.vdot(grad, grad).reshape(1, 1).astype(dtype)

        def refresh(_):
            L_new = state.L + g @ g.T
            Lr = _matrix_inverse_pth_root(L_new, 4.0, epsilon)
            Rr = _matrix_inverse_pth_root(R_new, 4.0, epsilon)
            precond = (Lr @ g) @ Rr
            return precond.reshape(-1).astype(dtype), L_new.astype(dtype)

        def keep(_):

            return grad, state.L

        precond, L_new = jax.lax.cond(do_refresh, refresh, keep, operand=None)
        d = -precond
        new_state = ShampooState(L=L_new, R=R_new, step=state.step + 1)
        return d, new_state

    def update(state, info):
        return state

    return Oracle(init=init, direction=direction, update=update)

from typing import NamedTuple

import jax
from jax import numpy as jnp

from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.point_history import publish


class AndersonState(NamedTuple):
    """Residual/iterate windows for Anderson (Type-II) acceleration.

    Attributes:
        g_history: window of recent gradients (residuals), (m, n).
        x_history: window of recent iterates,             (m, n).
         step_count:     number of valid columns currently stored.
    """

    g_history: jnp.ndarray
    x_history: jnp.ndarray
    step_count: jnp.ndarray


def AndersonOracle(window: int = 5, reg: float = 1e-8, beta: float = 1.0) -> Oracle:
    """Anderson-accelerated (Type-II) oracle — the variational ideal that
    L-BFGS approximates.

    The ``t = 1`` endpoint is formed by solving a tiny constrained
    least-squares problem over recent gradient *differences*::

        min_θ ‖ ∇f − ΔG θ ‖²  (+ reg·‖θ‖²)
         direction = −β·( ∇f − ΔG θ )  −  ΔX θ

    where ΔG, ΔX are first-differences of the stored gradient/iterate
    windows. With ``window=1`` this reduces to a secant step; with a deep
    window it captures multi-step curvature the single-secant cannot. No
    Hessian is ever formed; the only solve is an ``(m × m)`` system.
     The *coupling constant* ``β`` (the mixing parameter of the classical
     Anderson scheme) rescales the accelerated residual toward the
     gradient's natural magnitude. ``β = 1`` recovers the pure Type-II
     update; ``β > 1`` lets the deep-residual descent stretch, converting
     this oracle's leading trajectory-AUC into a leading *iteration* count.
    """

    def init(params):
        n = params.shape[0]
        zeros = jnp.zeros((window, n), dtype=params.dtype)
        return AndersonState(
            g_history=zeros,
            x_history=zeros,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):

        g_hist = state.g_history
        x_hist = state.x_history

        dG_first = (grad - g_hist[0])[:, None]
        dX_first = (params - x_hist[0])[:, None]
        dG_rest = (g_hist[:-1] - g_hist[1:]).T
        dX_rest = (x_hist[:-1] - x_hist[1:]).T
        dG = jnp.concatenate([dG_first, dG_rest], axis=1)
        dX = jnp.concatenate([dX_first, dX_rest], axis=1)

        m = dG.shape[1]
        gram = dG.T @ dG

        trace = jnp.trace(gram)
        scale = jnp.where(trace > 0.0, trace / m, 1.0)
        eye_m = jnp.eye(m, dtype=grad.dtype)
        A = gram + reg * scale * eye_m
        b = dG.T @ grad

        active = jnp.arange(m) < state.step_count
        b = jnp.where(active, b, 0.0)

        active_mask = active[:, None] & active[None, :]
        A = jnp.where(active_mask, A, eye_m) + (
            jnp.asarray(1e-12, dtype=grad.dtype) * eye_m
        )
        theta = jnp.linalg.solve(A, b)
        theta = jnp.where(active, theta, 0.0)

        residual = grad - dG @ theta
        d = -beta * residual - dX @ theta

        ok = jnp.all(jnp.isfinite(d)) & (state.step_count > 0)
        d = jnp.where(ok, d, -grad)
        return d, state

    def update(state, info):

        points = publish(info)
        if points is None:
            new_x = (
                jnp.roll(state.x_history, shift=1, axis=0).at[0].set(info.new_params)
            )
            new_g = jnp.roll(state.g_history, shift=1, axis=0).at[0].set(info.new_grad)
            new_count = jnp.minimum(state.step_count + 1, window)
            return AndersonState(g_history=new_g, x_history=new_x, step_count=new_count)

        params_seq = points.params_seq
        grad_seq = points.grad_seq
        valid_seq = points.valid_seq

        def body(carry, elem):
            x_hist, g_hist, count = carry
            x, g, valid = elem
            rolled_x = jnp.roll(x_hist, shift=1, axis=0).at[0].set(x)
            rolled_g = jnp.roll(g_hist, shift=1, axis=0).at[0].set(g)
            new_x_hist = jnp.where(valid, rolled_x, x_hist)
            new_g_hist = jnp.where(valid, rolled_g, g_hist)
            new_count = jnp.where(valid, jnp.minimum(count + 1, window), count)
            return (new_x_hist, new_g_hist, new_count), None

        (new_x, new_g, new_count), _ = jax.lax.scan(
            body,
            (state.x_history, state.g_history, state.step_count),
            (params_seq, grad_seq, valid_seq),
        )
        return AndersonState(g_history=new_g, x_history=new_x, step_count=new_count)

    return Oracle(init=init, direction=direction, update=update)

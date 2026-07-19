from typing import NamedTuple
from qqn_jax.oracles.strategy import Oracle

import jax
from jax import numpy as jnp

from qqn_jax.oracles.secant import _ordered_probe_secants


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
        # Build first-difference matrices from the window (newest-first).
        # ΔG[:, k] = g_k − g_{k+1}, ΔX[:, k] = x_k − x_{k+1}. Unfilled
        # slots are zero and contribute nothing to the normal equations.
        g_hist = state.g_history
        x_hist = state.x_history
        # Differences anchored on the present iterate, computed without the
        # extra (window+1, n) concat allocations: the first column is
        # (current - newest_stored), the rest are stored[k] - stored[k+1].
        # dG[:, 0] = grad - g_hist[0]; dG[:, k>=1] = g_hist[k-1] - g_hist[k].
        dG_first = (grad - g_hist[0])[:, None]
        dX_first = (params - x_hist[0])[:, None]
        dG_rest = (g_hist[:-1] - g_hist[1:]).T  # (n, window-1)
        dX_rest = (x_hist[:-1] - x_hist[1:]).T
        dG = jnp.concatenate([dG_first, dG_rest], axis=1)  # (n, window)
        dX = jnp.concatenate([dX_first, dX_rest], axis=1)

        # Solve (dGᵀ dG + reg·I) θ = dGᵀ ∇f  — an (m × m) system.
        m = dG.shape[1]
        gram = dG.T @ dG
        # Scale-aware Tikhonov: anchor the regularizer to the Gram trace so
        # conditioning is invariant to the magnitude of the residual window.
        trace = jnp.trace(gram)
        scale = jnp.where(trace > 0.0, trace / m, 1.0)
        eye_m = jnp.eye(m, dtype=grad.dtype)
        A = gram + reg * scale * eye_m
        b = dG.T @ grad
        # Mask columns with no stored history so empty windows are inert.
        active = jnp.arange(m) < state.step_count
        b = jnp.where(active, b, 0.0)
        # Mask inactive rows/cols to the identity and add an absolute diagonal
        # ridge in one fused step: a degenerate window can otherwise leave A
        # near-singular, making solve() emit NaN that backprops through the
        # downstream safeguard. The ridge guarantees SPD-ness.
        active_mask = active[:, None] & active[None, :]
        A = jnp.where(active_mask, A, eye_m) + (
            jnp.asarray(1e-12, dtype=grad.dtype) * eye_m
        )
        theta = jnp.linalg.solve(A, b)
        theta = jnp.where(active, theta, 0.0)

        # Accelerated residual and the corresponding iterate correction.
        residual = grad - dG @ theta
        d = -beta * residual - dX @ theta
        # Safeguard: fall back to steepest descent if the solve degenerates.
        ok = jnp.all(jnp.isfinite(d)) & (state.step_count > 0)
        d = jnp.where(ok, d, -grad)
        return d, state

    def update(state, info):
        # Roll the windows, inserting the freshly-accepted (x, g).
        # When line-search probes are populated, roll each valid probe
        # (oldest-first) into the windows before the accepted point. Every
        # probed (x, g) enriches the first-difference matrices ΔG/ΔX with an
        # extra residual observation, deepening the least-squares window
        # without extra accepted iterations. Absent probes we roll only the
        # accepted point.
        ordered = _ordered_probe_secants(info)
        if ordered is None:
            new_x = (
                jnp.roll(state.x_history, shift=1, axis=0).at[0].set(info.new_params)
            )
            new_g = jnp.roll(state.g_history, shift=1, axis=0).at[0].set(info.new_grad)
            new_count = jnp.minimum(state.step_count + 1, window)
            return AndersonState(g_history=new_g, x_history=new_x, step_count=new_count)

        params_seq, grad_seq, valid_seq = ordered

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

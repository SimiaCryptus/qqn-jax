from typing import NamedTuple

import jax
from jax import numpy as jnp

from qqn_jax.oracles.oracle import Oracle
from qqn_jax.oracles.point_history import publish


class AnchoredMultiSecantState(NamedTuple):
    """Residual/iterate windows for the anchored multi-secant oracle.

    Attributes:
        g_history: window of recent gradients, shape ``(m, n)``.
        x_history: window of recent iterates, shape ``(m, n)``.
        step_count: number of valid columns currently stored.
    """

    g_history: jnp.ndarray
    x_history: jnp.ndarray
    step_count: jnp.ndarray


def AnchoredMultiSecantOracle(
    window: int = 10,
    reg: float = 1e-8,
    beta: float = 1.0,
    kernel: str = "rational",
    sigma: float = 1.0,
) -> Oracle:
    """Anchored Multi-Secant quasi-Newton oracle (AMS-QN / TK-QN).

    Standard quasi-Newton curvature pairs are *pairwise-local* secants::

        s_{t-1} = x_t   − x_{t-1}
        y_{t-1} = ∇f_t  − ∇f_{t-1}

    chained across adjacent iterates. This oracle instead re-anchors *every*
    stored history pair ``(x_i, ∇f_i)`` at the *current* iterate ``x_t``::

        Δx_i = x_t  − x_i
        Δg_i = ∇f_t − ∇f_i          for every stored i in the window

    pulling all curvature information back into the tangent space at the
    current point — a Euclidean analogue of Riemannian vector transport. When
    the trajectory oscillates, adjacent secants can cancel to a noisy,
    near-zero chord; anchored secants instead accumulate the *full* distance
    to each past point, turning oscillation into a longer, cleaner curvature
    sample along the stiff direction.

    Each anchored pair is additionally weighted by a kernel over the anchored
    displacement magnitude ``‖Δx_i‖``::

        rational:  w_i = 1 / (1 + ‖Δx_i‖ / σ)
        gaussian:  w_i = exp(−‖Δx_i‖² / σ²)

    so very-near (redundant) or very-far (stale/unreliable) samples are
    naturally down-weighted relative to the current tangent space.

    The multi-secant condition ``H Δg_i ≈ Δx_i`` (for all valid ``i``,
    weighted by ``w_i``) is solved as a small ``(m × m)`` weighted
    least-squares system — no ``(n × n)`` curvature matrix is ever formed —
    mirroring the two-loop-free small-system solve used by
    :func:`~qqn_jax.oracles.anderson.AndersonOracle`, but built from anchored
    rather than first-differenced pairs::

        θ = argmin_θ  Σ_i w_i ‖∇f − ΔG θ‖²  (+ reg·‖θ‖²)
        direction = −β·(∇f − ΔG θ) − ΔX θ

    With an empty window the endpoint reduces to plain steepest descent,
    preserving the ``d'(0)`` anchor.
    """

    def init(params):
        n = params.shape[0]
        zeros = jnp.zeros((window, n), dtype=params.dtype)
        return AnchoredMultiSecantState(
            g_history=zeros,
            x_history=zeros,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        g_hist = state.g_history
        x_hist = state.x_history

        dX = (params[None, :] - x_hist).T
        dG = (grad[None, :] - g_hist).T

        m = dX.shape[1]
        active = jnp.arange(m) < state.step_count

        dx_norm2 = jnp.sum(dX * dX, axis=0)
        if kernel == "gaussian":
            w = jnp.exp(-dx_norm2 / (sigma**2 + 1e-12))
        elif kernel == "rational":
            w = 1.0 / (1.0 + jnp.sqrt(dx_norm2) / (sigma + 1e-12))
        else:
            raise ValueError(f"Unknown kernel: {kernel!r}")
        w = jnp.where(active, w, 0.0)
        sqrt_w = jnp.sqrt(w)

        dG_w = dG * sqrt_w[None, :]
        gram = dG_w.T @ dG_w
        trace = jnp.trace(gram)
        scale = jnp.where(trace > 0.0, trace / jnp.maximum(m, 1), 1.0)
        eye_m = jnp.eye(m, dtype=grad.dtype)
        A = gram + reg * scale * eye_m
        b = (dG * w[None, :]).T @ grad

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
            return AnchoredMultiSecantState(
                g_history=new_g, x_history=new_x, step_count=new_count
            )

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
        return AnchoredMultiSecantState(
            g_history=new_g, x_history=new_x, step_count=new_count
        )

    return Oracle(init=init, direction=direction, update=update)

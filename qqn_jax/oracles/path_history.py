import jax
from jax import numpy as jnp
from typing import NamedTuple
from qqn_jax.oracles.strategy import Oracle
from qqn_jax.oracles.secant import _ordered_probe_secants


class PathHistoryMomentumState(NamedTuple):
    """State for the path-history-momentum oracle.
    Attributes:
        delta_history: buffer of realized per-iteration deltas Δx = x_new − x,
            shape ``(history_size, n)``, most-recent-first (index 0 = newest).
        step_count: number of valid entries currently stored.
    """

    delta_history: jnp.ndarray
    step_count: jnp.ndarray


def PathHistoryMomentumOracle(history_size: int = 10, beta: float = 0.9) -> Oracle:
    """Momentum oracle that integrates the *actual accepted iteration history*.
    Unlike :func:`MomentumOracle`, which keeps a single running decaying
    average of the realized deltas, this oracle stores an explicit buffer of
    the last ``history_size`` accepted parameter deltas Δx = x_new − x and
    reconstructs the momentum by re-weighting the *whole* stored path::
        v = Σ_k  β^k · Δx_k                 (k = 0 newest .. m-1 oldest)
        direction = -∇f + v
    Because the momentum is recomputed from the genuine accepted trajectory
    (not folded destructively into a scalar EMA), the oracle can weight the
    real path geometry directly — recent steps dominate while older steps
    decay, but every stored step contributes exactly its geometric weight.
    On the very first step the buffer is empty and the endpoint reduces to
    plain steepest descent, preserving the ``d'(0)`` anchor.
    """

    def init(params):
        n = params.shape[0]
        return PathHistoryMomentumState(
            delta_history=jnp.zeros((history_size, n), dtype=params.dtype),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        # Reconstruct the momentum from the stored path history: weight each
        # realized delta by β^k (newest first). Unfilled slots are zero and
        # contribute nothing. Do NOT mutate state here.
        m = state.delta_history.shape[0]
        weights = beta ** jnp.arange(m, dtype=grad.dtype)  # (m,)
        # Mask slots beyond the currently-filled history.
        active = jnp.arange(m) < state.step_count
        weights = jnp.where(active, weights, 0.0)
        v = jnp.tensordot(weights, state.delta_history, axes=(0, 0))  # (n,)
        d = -grad + v
        return d, state

    def update(state, info):
        # Push the freshly-realized delta into the front of the circular
        # buffer (most-recent-first), dropping the oldest.
        # When line-search probes are populated, push each incremental probe
        # delta (oldest-first) into the buffer so the reconstructed momentum
        # weights the genuine intermediate path geometry, finishing with the
        # accepted point. Absent probes we push the single accepted delta.
        ordered = _ordered_probe_secants(info)
        if ordered is None:
            delta = info.new_params - info.params
            shifted = jnp.concatenate([delta[None], state.delta_history[:-1]], axis=0)
            new_count = jnp.minimum(state.step_count + 1, history_size)
            return PathHistoryMomentumState(delta_history=shifted, step_count=new_count)

        params_seq, _, valid_seq = ordered
        anchored = jnp.concatenate([info.params[None, :], params_seq], axis=0)
        deltas = anchored[1:] - anchored[:-1]  # (k+1, n) incremental deltas

        def body(carry, elem):
            hist, count = carry
            dx, valid = elem
            pushed = jnp.concatenate([dx[None], hist[:-1]], axis=0)
            new_hist = jnp.where(valid, pushed, hist)
            new_count = jnp.where(valid, jnp.minimum(count + 1, history_size), count)
            return (new_hist, new_count), None

        (hist, count), _ = jax.lax.scan(
            body, (state.delta_history, state.step_count), (deltas, valid_seq)
        )
        return PathHistoryMomentumState(delta_history=hist, step_count=count)

    return Oracle(init=init, direction=direction, update=update)

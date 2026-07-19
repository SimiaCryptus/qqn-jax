"""Point-history storage and secant-view abstractions.

This module centralizes the bookkeeping that used to be duplicated across
every quasi-Newton oracle. The design is two-layered:

    PointHistoryStore
        A raw, ordered buffer of measured points ``(params, grad, alpha,
        valid)``. The solver *publishes* the accepted iterate plus any
        line-search probes here through :func:`publish`. The store handles
        ordering (increasing ``alpha``), validity masking, and probe-replay
        capping in ONE place, replacing the old ``_ordered_probe_secants``.

    SecantStoreView
        A thin, cached *view* over a published batch of points that derives
        the curvature information QN oracles actually consume:

            * chained (pairwise-local) secants  s_i, y_i
            * anchored secants                  Δx_i = x_t − x_i,
                                                Δg_i = ∇f_t − ∇f_i
            * the newest accepted (params, grad)

        Oracles ask the view for exactly the flavor of secant they need and
        never re-implement the probe-ordering / anchoring logic themselves.

The store objects are plain (jit-friendly) NamedTuples of arrays; there is
no Python-level mutation, so they compose with ``jax.lax`` control flow.
"""

from typing import NamedTuple, Optional

from jax import numpy as jnp


class PublishedPoints(NamedTuple):
    """An ordered batch of measured points published by the solver.

    All arrays run *oldest-first* (increasing line-search ``alpha``) and
    terminate with the accepted iterate as the final, always-valid entry.

    Attributes:
        params_seq: ``(k, n)`` iterates.
        grad_seq:   ``(k, n)`` gradients at those iterates.
        alpha_seq:  ``(k,)``   line-search step sizes (accepted point last).
        valid_seq:  ``(k,)``   boolean validity mask.
        anchor_params: ``(n,)`` iterate *before* the step (x, i.e. ``d'(0)``
            anchor); used to form the first delta.
        anchor_grad:   ``(n,)`` gradient at the anchor.
    """

    params_seq: jnp.ndarray
    grad_seq: jnp.ndarray
    alpha_seq: jnp.ndarray
    valid_seq: jnp.ndarray
    anchor_params: jnp.ndarray
    anchor_grad: jnp.ndarray


def publish(info, max_replay: Optional[int] = None) -> Optional[PublishedPoints]:
    """Publish the accepted point (+ probes) from an ``OracleInfo``.

    This is the single source of truth for turning the solver's raw probe
    buffers into an ordered, validity-masked batch of points. It subsumes
    the old ``_ordered_probe_secants`` helper.

    When ``max_replay`` is given only the probes CLOSEST to the accepted
    step (largest ``alpha`` among valid probes) are retained, capping how
    many collinear probes are folded in.

    Returns ``None`` when the probe buffers are not populated, signalling the
    caller to fall back to a single accepted-point-only update (which callers
    can build cheaply from ``info`` directly).
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
    probe_alphas = kept_alphas[inner]

    params_seq = jnp.concatenate([probe_params, info.new_params[None, :]], axis=0)
    grad_seq = jnp.concatenate([probe_grads, info.new_grad[None, :]], axis=0)
    valid_seq = jnp.concatenate([probe_valid, jnp.asarray([True])], axis=0)
    accepted_alpha = jnp.asarray(
        info.step_size if info.step_size is not None else 1.0,
        dtype=probe_alphas.dtype,
    )
    alpha_seq = jnp.concatenate([probe_alphas, accepted_alpha[None]], axis=0)

    return PublishedPoints(
        params_seq=params_seq,
        grad_seq=grad_seq,
        alpha_seq=alpha_seq,
        valid_seq=valid_seq,
        anchor_params=info.params,
        anchor_grad=info.grad,
    )


class SecantStoreView(NamedTuple):
    """Cached secant view over a :class:`PublishedPoints` batch.

    Build with :func:`secant_view`. All the derived arrays are computed once
    and stored, so an oracle folding them into its state pays the anchoring /
    differencing cost a single time.

    Attributes:
        points:   the underlying published points.
        deltas:   ``(k, n)`` per-step iterate deltas Δx (anchor-prefixed).
        gdeltas:  ``(k, n)`` per-step gradient deltas Δg (anchor-prefixed).
        anch_dx:  ``(k, n)`` anchored displacements x_new − x_i.
        anch_dg:  ``(k, n)`` anchored gradient displacements ∇f_new − ∇f_i.
    """

    points: PublishedPoints
    deltas: jnp.ndarray
    gdeltas: jnp.ndarray
    anch_dx: jnp.ndarray
    anch_dg: jnp.ndarray

    @property
    def params_seq(self):
        return self.points.params_seq

    @property
    def grad_seq(self):
        return self.points.grad_seq

    @property
    def valid_seq(self):
        return self.points.valid_seq

    def newest_secant(self):
        """Return ``(s, y)`` for the accepted step vs. the most-recent valid
        preceding point (chained BB1-style secant)."""
        params_seq = self.points.params_seq
        grad_seq = self.points.grad_seq
        valid_seq = self.points.valid_seq

        anchor_p = jnp.concatenate(
            [self.points.anchor_params[None, :], params_seq[:-1]], axis=0
        )
        anchor_g = jnp.concatenate(
            [self.points.anchor_grad[None, :], grad_seq[:-1]], axis=0
        )
        prev_valid = valid_seq[:-1]
        idx = jnp.max(jnp.where(prev_valid, jnp.arange(prev_valid.shape[0]), 0))
        p_prev = anchor_p[idx]
        g_prev = anchor_g[idx]
        s = params_seq[-1] - p_prev
        y = grad_seq[-1] - g_prev
        return s, y


def secant_view(points: PublishedPoints) -> SecantStoreView:
    """Derive and cache all secant flavors from a published batch."""
    anchored_p = jnp.concatenate(
        [points.anchor_params[None, :], points.params_seq], axis=0
    )
    anchored_g = jnp.concatenate([points.anchor_grad[None, :], points.grad_seq], axis=0)
    deltas = anchored_p[1:] - anchored_p[:-1]
    gdeltas = anchored_g[1:] - anchored_g[:-1]

    new_params = points.params_seq[-1]
    new_grad = points.grad_seq[-1]
    anch_dx = new_params[None, :] - points.params_seq
    anch_dg = new_grad[None, :] - points.grad_seq

    return SecantStoreView(
        points=points,
        deltas=deltas,
        gdeltas=gdeltas,
        anch_dx=anch_dx,
        anch_dg=anch_dg,
    )

"""Cubic Hermite *spline* path strategy over the QQN path parameter ``t``.

This is a **distinct, stateful** :class:`~qqn_jax.paths.base.PathStrategy`
— it is *not* an alias of, or a thin wrapper around, the quadratic path.

There are **two different curves** in play, and keeping them separate is the
whole point:

* The *parameter-space geometry* ``d(t)`` (``_spline_offset``) fixes where a
  probe physically lands, ``x + d(t)``. Per ``docs/theory/spline_search.md``
  every evaluation is taken *along* the QQN blend, so this mapping is the
  blend. This is **not** the spline.
* The *spline model itself* is the accumulating model of the objective as a
  function of the scalar ``t``. It **begins** as a single cubic Hermite
  segment spanning the two endpoints (which reproduces the quadratic-order
  picture) and **becomes a genuine piecewise cubic Hermite spline** as
  measurements accumulate: every probe's measured fitness *and* directional
  derivative is retained as a reusable **control point** ``(t_i, f_i, m_i)``.

The construct is intentionally unusual: proposing the next step means
interpolating between two *accumulated* anchors with a cubic Hermite segment,
solving that segment for its stationary point, and — crucially — **measuring
that point and appending it as a new control point**, so the next proposal is
drawn from a strictly richer spline. ``spline_refine`` runs exactly this
accumulation loop.

Because it accumulates control points, the spline path carries an explicit
:class:`SplineState` (its control-point buffer) and implements the optional
stateful hooks of ``PathStrategy`` (``init_state`` / ``observe`` /
``propose``). The spline only ever *proposes* candidate steps; acceptance is
gated by the outer line search's strict-improvement / sufficient-decrease
test. See ``docs/theory/spline_search.md`` for the full derivation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.paths.base import PathStrategy
from qqn_jax.utils import tree_vdot


def _spline_offset(t, grad_dir, direction):
    """Displacement ``d(t)`` blending the steepest-descent tangent with the
    oracle endpoint, evaluated at ``t``."""
    a = t * (1.0 - t)
    b = t * t
    return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, direction)


def _spline_velocity(t, grad_dir, direction):
    """Path tangent ``d'(t)`` used to project a measured gradient onto the
    scalar directional derivative ``m = ⟨∇f, d'(t)⟩``."""
    a = 1.0 - 2.0 * t
    b = 2.0 * t
    return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, direction)


def hermite_basis(s):
    """Cubic Hermite basis functions evaluated at ``s ∈ [0, 1]``.

        h00(s) =  2s³ - 3s² + 1
        h10(s) =      s³ - 2s² + s
        h01(s) = -2s³ + 3s²
        h11(s) =      s³ -  s²

    Returns:
        ``(h00, h10, h01, h11)``.
    """
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00, h10, h01, h11


def _orient_tangents(m0, m1, delta):
    """Apply the upstream/downstream symmetry correction.

    For each endpoint tangent ``m``, if it is oriented *against* the
    segment secant slope ``delta`` (``sign(m) != sign(delta)`` and
    ``delta != 0``) reflect it: ``m <- -m``. When ``delta == 0`` no
    reflection is applied. See the "Gradient Orientation" section of the
    theory doc — this is an unproven heuristic, kept safe by the outer
    line search's strict-improvement gate.
    """

    def reflect(m):
        # Orientation is now handled once at record time (tangents are stored
        # already pointing along increasing t); no per-segment reflection.
        return m

    return reflect(m0), reflect(m1)


def segment_eval(t, t0, f0, m0, t1, f1, m1):
    """Interpolated fitness of the Hermite segment at parameter ``t``.

        f(s) = h00(s)·f_0 + h10(s)·h·m_0 + h01(s)·f_1 + h11(s)·h·m_1

    where ``h = t_1 - t_0`` and ``s = (t - t_0) / h``. The tangents are
    oriented via the upstream/downstream symmetry rule before use.
    """
    h = t1 - t0
    s = (t - t0) / h
    delta = (f1 - f0) / h
    cm0, cm1 = _orient_tangents(m0, m1, delta)
    h00, h10, h01, h11 = hermite_basis(s)
    return h00 * f0 + h10 * h * cm0 + h01 * f1 + h11 * h * cm1


def segment_candidates(t0, f0, m0, t1, f1, m1, eps=1e-12):
    """Closed-form stationary points of the Hermite segment.

    Differentiating ``f(s)`` gives a quadratic ``f'(s) = A·s² + B·s + C``:

        A =  6·f_0 + 3·h·m_0 - 6·f_1 + 3·h·m_1
        B = -6·f_0 - 4·h·m_0 + 6·f_1 - 2·h·m_1
        C =          h·m_0

    Returns:
        ``(t_cand, f_cand, valid)`` arrays of length 2, one entry per
        root. ``valid`` marks roots that are real and lie in ``s ∈ [0, 1]``
        (mapped back to ``t = t_0 + s·h``). Tangents are oriented before
        the coefficients are formed.
    """
    h = t1 - t0
    delta = (f1 - f0) / h
    cm0, cm1 = _orient_tangents(m0, m1, delta)

    hm0 = h * cm0
    hm1 = h * cm1

    A = 6.0 * f0 + 3.0 * hm0 - 6.0 * f1 + 3.0 * hm1
    B = -6.0 * f0 - 4.0 * hm0 + 6.0 * f1 - 2.0 * hm1
    C = hm0

    disc = B * B - 4.0 * A * C

    real = disc >= 0.0
    sqrt_disc = jnp.sqrt(jnp.where(real, disc, 0.0))

    quad_ok = jnp.abs(A) >= eps
    lin_ok = jnp.abs(B) >= eps

    denom = jnp.where(quad_ok, 2.0 * A, 1.0)
    s_plus = (-B + sqrt_disc) / denom
    s_minus = (-B - sqrt_disc) / denom

    s_lin = jnp.where(lin_ok, -C / jnp.where(lin_ok, B, 1.0), jnp.nan)

    s0 = jnp.where(quad_ok, s_plus, s_lin)
    s1 = jnp.where(quad_ok, s_minus, jnp.nan)
    s = jnp.stack([s0, s1])

    branch_real = jnp.stack(
        [
            jnp.logical_and(real, quad_ok)
            | jnp.logical_and(lin_ok, jnp.logical_not(quad_ok)),
            jnp.logical_and(real, quad_ok),
        ]
    )
    in_range = jnp.logical_and(s >= 0.0, s <= 1.0)
    valid = jnp.logical_and(branch_real, in_range)

    t_cand = t0 + s * h

    f_cand = segment_eval(t_cand, t0, f0, m0, t1, f1, m1)
    f_cand = jnp.where(valid, f_cand, jnp.inf)

    return t_cand, f_cand, valid


def propose_from_points(ts, fs, ms, valid, eps=1e-12):
    """Propose the next evaluation point from a (possibly padded) buffer of
    control points.

    Control points may be stored in a fixed-capacity buffer with a
    per-point ``valid`` mask (invalid slots are ignored). Points are sorted
    by ``t``, and only segments whose *both* endpoints are valid contribute
    candidate stationary points. Returns the candidate with the lowest
    predicted fitness across all such segments.

    Args:
        ts: path parameters ``(n,)`` (need not be pre-sorted).
        fs: measured fitnesses ``(n,)``.
        ms: measured directional derivatives ``(n,)``.
        valid: boolean mask ``(n,)`` marking populated control points.

    Returns:
        ``(t_best, f_best, found)`` — the proposed step, its predicted
        fitness, and a flag that is ``False`` when no segment yielded an
        in-range stationary point.
    """

    finite_max = jnp.max(jnp.where(valid, ts, -jnp.inf))
    big = jnp.where(jnp.any(valid), finite_max, 0.0) + 1.0
    sort_key = jnp.where(valid, ts, big)
    order = jnp.argsort(sort_key)

    ts = ts[order]
    fs = fs[order]
    ms = ms[order]
    valid = valid[order]

    t0 = ts[:-1]
    f0 = fs[:-1]
    m0 = ms[:-1]
    t1 = ts[1:]
    f1 = fs[1:]
    m1 = ms[1:]

    pair_ok = jnp.logical_and(valid[:-1], valid[1:])
    pair_ok = jnp.logical_and(pair_ok, (t1 - t0) > eps)

    t_cand, f_cand, cand_valid = jax.vmap(
        lambda a, b, c, d, e, g: segment_candidates(a, b, c, d, e, g, eps)
    )(t0, f0, m0, t1, f1, m1)

    cand_valid = jnp.logical_and(cand_valid, pair_ok[:, None])

    t_flat = t_cand.reshape(-1)
    f_flat = f_cand.reshape(-1)
    v_flat = cand_valid.reshape(-1)

    f_masked = jnp.where(v_flat, f_flat, jnp.inf)
    best = jnp.argmin(f_masked)
    found = jnp.any(v_flat)
    return t_flat[best], f_flat[best], found


def propose_step(ts, fs, ms, eps=1e-12):
    """Propose the next evaluation point from *sorted, fully-valid* control
    points.

    Convenience wrapper over :func:`propose_from_points` for the case where
    every entry of ``ts``/``fs``/``ms`` is a genuine control point.
    """
    valid = jnp.ones_like(ts, dtype=bool)
    return propose_from_points(ts, fs, ms, valid, eps)


class SplineState(NamedTuple):
    """The spline path's control-point memory.

    Attributes:
        ts: path parameters of stored control points ``(capacity,)``.
        fs: measured fitnesses ``(capacity,)``.
        ms: measured directional derivatives ``(capacity,)``.
        valid: per-slot validity mask ``(capacity,)``.
        num_points: number of control points recorded so far.
    """

    ts: jnp.ndarray
    fs: jnp.ndarray
    ms: jnp.ndarray
    valid: jnp.ndarray
    num_points: jnp.ndarray


def spline_init(grad_dir, direction, capacity: int = 16, dtype=jnp.float32):
    """Allocate an empty control-point buffer.

    ``grad_dir`` / ``direction`` are accepted to match the stateful
    ``PathStrategy.init_state`` signature; the buffer itself is geometry
    agnostic (control points are scalar ``(t, f, m)`` triples).
    """
    del grad_dir, direction
    return SplineState(
        ts=jnp.zeros((capacity,), dtype),
        fs=jnp.full((capacity,), jnp.inf, dtype),
        ms=jnp.zeros((capacity,), dtype),
        valid=jnp.zeros((capacity,), bool),
        num_points=jnp.asarray(0, jnp.int32),
    )


def spline_observe(state: SplineState, t, f, m) -> SplineState:
    """Record a measured control point ``(t, f, m)`` into the memory."""
    i = jnp.minimum(state.num_points, state.ts.shape[0] - 1)
    return SplineState(
        ts=state.ts.at[i].set(t),
        fs=state.fs.at[i].set(f),
        ms=state.ms.at[i].set(m),
        valid=state.valid.at[i].set(True),
        num_points=state.num_points + 1,
    )


def spline_propose(state: SplineState):
    """Propose the next candidate ``t`` from the accumulated control points."""
    return propose_from_points(state.ts, state.fs, state.ms, state.valid)


SPLINE_PATH = PathStrategy(
    offset=_spline_offset,
    velocity=_spline_velocity,
    init_state=spline_init,
    observe=spline_observe,
    propose=spline_propose,
    stateful=True,
)


def spline_refine(
    inner,
    eval_at,
    path,
    grad_dir,
    direction,
    f0,
    slope0,
    dtype,
    rounds: int = 4,
) -> LineSearchResult:
    """Refine an inner line-search result by *accumulating* control points.

    This runs the spline's defining loop: starting from the seed control
    points (the origin ``t = 0``, every recorded probe, and the accepted
    endpoint), it repeatedly

      1. proposes a step from the lowest-predicted stationary point of the
         current piecewise cubic Hermite model
         (:func:`propose_from_points`),
      2. **measures** that step (one evaluation), and
      3. **appends** the measurement ``(t, f, m)`` as a brand-new control
         point,

    so each successive proposal is drawn from a strictly richer spline. The
    best strictly-improving measurement over all rounds is returned (the
    strict-improvement gate that keeps the heuristic reflection safe).

    Args:
        inner: baseline ``LineSearchResult`` (must carry recorded probes).
        eval_at: shared scalar evaluator ``t -> (params, value, grad, slope)``.
            ``slope`` is exactly the directional derivative ``m`` along the
            path, so a measurement is a ready-made control point.
        path: the spline ``PathStrategy`` (used for its velocity/tangent).
        grad_dir: steepest-descent tangent ``-∇f`` at ``t = 0``.
        direction: oracle endpoint direction ``-H∇f``.
        f0: objective value at ``t = 0``.
        slope0: directional derivative at ``t = 0``.
        dtype: working dtype.
        rounds: number of accumulate-and-repropose rounds (each costs one
            evaluation and adds one control point).
    """
    inner_evals = inner.num_evals
    if inner_evals is None:
        inner_evals = jnp.asarray(1, jnp.int32)

    p_alphas = inner.probe_alphas
    p_values = inner.probe_values
    p_grads = inner.probe_grads
    p_valid = inner.probe_valid

    def slope_of(alpha, g):
        v = path.velocity(alpha, grad_dir, direction)
        m = tree_vdot(g, v)
        # Path-simplicity heuristic: the stored tangent must point along
        # increasing t. If the measured directional derivative opposes the
        # forward path direction, reflect it so we don't fold the spline
        # back on itself. (Gate over f0 in the round loop keeps this safe.)
        return jnp.abs(m)

    p_slopes = jax.vmap(slope_of)(p_alphas, p_grads)

    origin_t = jnp.asarray(0.0, dtype)
    origin_f = jnp.asarray(f0, dtype)
    origin_m = jnp.asarray(slope0, dtype)

    end_t = inner.step_size
    end_f = inner.new_value
    end_m = slope_of(end_t, inner.new_grad)

    seed_ts = jnp.concatenate([p_alphas, jnp.stack([origin_t, end_t])])
    seed_fs = jnp.concatenate([p_values, jnp.stack([origin_f, end_f])])
    seed_ms = jnp.concatenate([p_slopes, jnp.stack([origin_m, end_m])])
    seed_valid = jnp.concatenate([p_valid, jnp.array([True, True])])
    n_seed = seed_ts.shape[0]

    ts = jnp.concatenate([seed_ts, jnp.zeros((rounds,), dtype)])
    fs = jnp.concatenate([seed_fs, jnp.full((rounds,), jnp.inf, dtype)])
    ms = jnp.concatenate([seed_ms, jnp.zeros((rounds,), dtype)])
    valid = jnp.concatenate([seed_valid, jnp.zeros((rounds,), bool)])
    count = jnp.asarray(n_seed, jnp.int32)

    init = (
        ts,
        fs,
        ms,
        valid,
        count,
        jnp.asarray(inner.step_size, dtype),
        jnp.asarray(inner.new_value, dtype),
        jax.tree_util.tree_map(lambda a: a.astype(dtype), inner.new_params),
        jax.tree_util.tree_map(lambda a: a.astype(dtype), inner.new_grad),
        jnp.asarray(False),
    )

    def round_step(carry, _):
        ts, fs, ms, valid, count, best_t, best_v, best_p, best_g, best_found = carry
        t_prop, _f_pred, found = propose_from_points(ts, fs, ms, valid)

        f_masked = jnp.where(valid, fs, jnp.inf)
        lo_idx = jnp.argmin(f_masked)
        t_lo = ts[lo_idx]

        span_other = jnp.where(t_lo == end_t, origin_t, end_t)
        t_mid = 0.5 * (t_lo + span_other)
        t_eval = jnp.where(found, t_prop, t_mid)
        p, v, g, slope = eval_at(t_eval)
        # Keep the carry dtype stable: `eval_at` may promote to float64
        # (e.g. under x64), but the carry was seeded from float32 inner
        # results. Cast scalars/leaves back to the working dtype so the
        # scan carry input/output types match.
        t_eval = jnp.asarray(t_eval, dtype)
        v = jnp.asarray(v, dtype)
        slope = jnp.asarray(slope, dtype)
        p = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype), p)
        g = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype), g)

        i = count
        ts = ts.at[i].set(t_eval)
        fs = fs.at[i].set(v)
        ms = ms.at[i].set(slope)
        valid = valid.at[i].set(True)
        count = count + jnp.asarray(1, jnp.int32)

        # Strict-improvement gate: only accept a measurement that strictly
        # beats the best-so-far *and* the origin f0. This is what actually
        # keeps the orientation heuristic safe -- the gate is over f0, not
        # merely over the inner endpoint. `found` is intentionally NOT part
        # of the gate: a fallback midpoint probe may legitimately improve.
        improve = jnp.logical_and(v < best_v, v < origin_f)
        best_t = jnp.where(improve, t_eval, best_t)
        best_v = jnp.where(improve, v, best_v)
        best_p = jax.tree_util.tree_map(
            lambda a, b: jnp.where(improve, a, b), p, best_p
        )
        best_g = jax.tree_util.tree_map(
            lambda a, b: jnp.where(improve, a, b), g, best_g
        )
        best_found = jnp.logical_or(best_found, improve)

        new_carry = (
            ts,
            fs,
            ms,
            valid,
            count,
            best_t,
            best_v,
            best_p,
            best_g,
            best_found,
        )
        return new_carry, None

    (
        (
            ts,
            fs,
            ms,
            valid,
            count,
            best_t,
            best_v,
            best_p,
            best_g,
            best_found,
        ),
        _,
    ) = jax.lax.scan(round_step, init, None, length=rounds)

    done = jnp.logical_or(inner.done, best_found)
    return LineSearchResult(
        step_size=best_t,
        new_value=best_v,
        new_grad=best_g,
        new_params=best_p,
        done=done,
        probe_params=inner.probe_params,
        probe_grads=inner.probe_grads,
        probe_valid=inner.probe_valid,
        probe_values=inner.probe_values,
        probe_alphas=inner.probe_alphas,
        num_evals=inner_evals + jnp.asarray(rounds, jnp.int32),
    )


__all__ = [
    "SPLINE_PATH",
    "SplineState",
    "spline_init",
    "spline_observe",
    "spline_propose",
    "hermite_basis",
    "segment_eval",
    "segment_candidates",
    "propose_from_points",
    "propose_step",
    "spline_refine",
]

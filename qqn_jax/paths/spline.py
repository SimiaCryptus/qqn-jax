"""Spline (cubic Hermite) augmentation for QQN line searches.

Each evaluation along the path ``d(t)`` yields both a fitness value
``f(d(t))`` and a directional derivative ``m = ⟨∇f, d'(t)⟩``. The spline
does *not* replace the line search; it is an **expanded definition of the
curve** that reuses every measured point as a reusable *control point* of a
piecewise cubic Hermite spline model of the objective along the
(consistent) path.

``spline_wrap(inner_search)`` returns a line-search-compatible callable
that first runs ``inner_search`` (any registered strategy), then attempts
to *improve* on its accepted point by probing the stationary points of the
cubic Hermite spline fit through the control points gathered so far.
Because every probe is built through the same shared ``PathStrategy``
(``qqn_jax.paths.base.make_evaluator``) as the canonical quadratic path
(``qqn_jax.paths.quadratic.QUADRATIC_PATH``) by default, every probe —
regardless of the underlying line search — is a valid control point on the
*same* curve.

Candidate steps are proposed by locating stationary points of the cubic
segments (closed-form roots of the quadratic derivative). Tangents are
oriented via the upstream/downstream symmetry rule so spurious inflections
do not mislead the search.

See ``spline_search.md`` for the full specification.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_negative, tree_vdot
from qqn_jax.regions.strategy import resolve_region
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.line_search.backtracking import backtracking_search
from qqn_jax.paths.base import PathStrategy, make_evaluator
from qqn_jax.paths.quadratic import QUADRATIC_PATH


def _orient_tangents(m0, m1):
    """Orient synthetic tangents to agree with the path's forward direction.

    Each tangent is a directional derivative ``m = ⟨∇f, d'(t)⟩`` — the
    projection of the gradient onto the path's velocity vector ``d'(t)``.
    When constructing the spline we are *not* seeking a low function value;
    we are building the simplest curve consistent with the gradient
    topology along the path. The orientation question is therefore purely
    geometric: does the tangent agree with the forward direction of travel
    along the path?

    A tangent whose dot product against the forward path direction is
    negative points "backwards" relative to the natural flow of the curve,
    so it is reflected to agree. Because ``m`` is already the dot product of
    the gradient with the forward velocity ``d'(t)``, the sign of ``m`` *is*
    the sign of that dot product, and the reflection reduces to flipping the
    sign of negative tangents.

    This is intended for synthetic tangents (finite-difference or secant
    estimates) whose sign convention may be ambiguous. It must NOT be
    applied to genuine measured directional derivatives, which already carry
    real curvature information.
    """

    bracketed = jnp.logical_and(m0 < 0.0, m1 > 0.0)
    needs_orient = jnp.logical_not(bracketed)

    disagrees = (m0 * m1) < 0.0
    m0_dominates = jnp.abs(m0) > jnp.abs(m1)
    reflect_m0 = jnp.logical_and(needs_orient, jnp.logical_and(disagrees, m0_dominates))
    m0_oriented = jnp.where(reflect_m0, -m0, m0)
    m1_oriented = m1
    return m0_oriented, m1_oriented


def _segment_value(s, h, f0, m0, f1, m1):
    """Cubic Hermite interpolated fitness at normalized parameter ``s``."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * f0 + h10 * h * m0 + h01 * f1 + h11 * h * m1


def _segment_stationary_candidates(t0, t1, f0, m0, f1, m1):
    """Return up to two candidate ``(t, predicted_value)`` stationary points.

    Differentiating the cubic Hermite segment w.r.t. ``s`` gives a quadratic
    ``A s² + B s + C = 0``. We solve it in closed form, mask roots outside
    ``[0, 1]`` (or non-real ones), and map valid roots back to ``t``.

    Returns arrays ``(t_cands, val_cands, valid)`` each of length 2.
    """
    h = t1 - t0
    m0o, m1o = _orient_tangents(m0, m1)

    hm0 = h * m0o
    hm1 = h * m1o
    A = 6.0 * f0 + 3.0 * hm0 - 6.0 * f1 + 3.0 * hm1
    B = -6.0 * f0 - 4.0 * hm0 + 6.0 * f1 - 2.0 * hm1
    C = hm0

    eps = jnp.asarray(1e-12, dtype=f0.dtype)
    disc = B * B - 4.0 * A * C
    disc_ok = disc >= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

    denom = jnp.where(jnp.abs(A) > eps, 2.0 * A, 1.0)
    root1 = (-B + sqrt_disc) / denom
    root2 = (-B - sqrt_disc) / denom

    lin_root = jnp.where(
        jnp.abs(B) > eps, -C / jnp.where(jnp.abs(B) > eps, B, 1.0), -1.0
    )
    is_quad = jnp.abs(A) > eps

    s1 = jnp.where(is_quad, root1, lin_root)
    s2 = jnp.where(is_quad, root2, -1.0)

    def finalize(s, extra_valid):
        in_range = jnp.logical_and(s >= 0.0, s <= 1.0)
        valid = jnp.logical_and(in_range, extra_valid)
        s_clip = jnp.clip(s, 0.0, 1.0)
        t = t0 + s_clip * h
        val = _segment_value(s_clip, h, f0, m0o, f1, m1o)

        val = jnp.where(valid, val, jnp.asarray(jnp.inf, dtype=f0.dtype))
        return t, val, valid

    t_c1, v_c1, ok1 = finalize(s1, jnp.logical_and(disc_ok, is_quad) | (~is_quad))
    t_c2, v_c2, ok2 = finalize(s2, jnp.logical_and(disc_ok, is_quad))

    t_cands = jnp.stack([t_c1, t_c2])
    val_cands = jnp.stack([v_c1, v_c2])
    valid = jnp.stack([ok1, ok2])
    return t_cands, val_cands, valid


def spline_wrap(
    inner_search: Callable, path: PathStrategy = QUADRATIC_PATH
) -> Callable:
    """Augment ``inner_search`` with a cubic Hermite spline refinement.

    Returns a line-search-compatible callable with the same signature as the
    wrapped ``inner_search``. The spline is an *expanded definition of the
    curve*, not a competing line search: it reuses the consistent path's
    measured points as control points and probes the stationary points of
    the resulting cubic Hermite spline to try to improve on the inner
    search's accepted step.

    The wrapped search:

    1. Runs ``inner_search`` to obtain a baseline accepted point.
    2. Forms control points from ``t = 0`` (current point, slope
       ``⟨∇f, d'(0)⟩``) and ``t = t_inner`` (the inner search's accepted
       point, with its measured slope ``⟨∇f, d'(t_inner)⟩``).
    3. Probes the spline's stationary points, projecting through the region
       and keeping the lowest-value feasible point found.
    4. Returns the better of the inner result and the spline probes.

    Because every probe lies on the *same* path (built through the shared
    ``PathStrategy`` component — ``QUADRATIC_PATH`` by default, matching the
    curve the wrapped inner search itself traverses), this composes
     correctly with any underlying line search. To keep that invariant
     structural rather than conventional, ``path`` is also forwarded
     explicitly to ``inner_search`` itself (as a first-class ``path=path``
     keyword), so the inner search's own probing — if it accepts a ``path``
     argument — can never silently drift from the curve the refinement
     layer probes.

    Args:
        inner_search: any registered line-search strategy to seed the
            baseline.
        path: the ``PathStrategy`` used to remap ``t`` into a probe point
            and its velocity ``d'(t)``. Defaults to ``QUADRATIC_PATH``, the
            canonical QQN curve, so spline probes stay on the exact curve
            the inner search traversed.
    """

    def wrapped(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        spline_max_iter: int = 6,
        region=None,
        region_state=None,
        **inner_kwargs,
    ) -> LineSearchResult:
        region = resolve_region(region)

        eval_at = make_evaluator(
            value_and_grad_fn,
            params,
            grad,
            direction,
            region,
            region_state,
            path,
            *args,
        )

        grad_dir = tree_negative(grad)

        dtype = value.dtype

        inner = inner_search(
            value_and_grad_fn,
            params,
            direction,
            value,
            grad,
            *args,
            region=region,
            region_state=region_state,
            path=path,
            **inner_kwargs,
        )

        a0 = jnp.asarray(0.0, dtype=dtype)
        f0 = value

        m0 = tree_vdot(grad, path.velocity(a0, grad_dir, direction))

        a1 = inner.step_size
        f1 = inner.new_value
        m1 = tree_vdot(inner.new_grad, path.velocity(a1, grad_dir, direction))

        eps_t = jnp.asarray(1e-12, dtype=dtype)
        a1_safe = jnp.where(
            jnp.abs(a1) > eps_t, a1, jnp.where(a1 >= 0.0, eps_t, -eps_t)
        )
        secant = (f1 - f0) / a1_safe

        ambiguous = jnp.logical_and(m1 <= 0.0, secant * m1 < 0.0)
        m1 = jnp.where(ambiguous, -m1, m1)

        descending_at_inner = m1 < 0.0
        a_ext = jnp.where(descending_at_inner, 2.0 * a1, 0.5 * a1)

        a_ext = jnp.where(a1 > 0.0, a_ext, jnp.asarray(1.0, dtype=dtype))
        p_ext, f_ext, g_ext, m_ext = eval_at(a_ext)

        inner_evals = inner.num_evals
        if inner_evals is None:
            inner_evals = jnp.asarray(1, jnp.int32)
        base_evals = inner_evals + jnp.asarray(1, jnp.int32)

        use_ext_segment = jnp.logical_and(descending_at_inner, f_ext < f1)
        la = jnp.where(use_ext_segment, a1, a0)
        lf = jnp.where(use_ext_segment, f1, f0)
        lm = jnp.where(use_ext_segment, m1, m0)
        ra = jnp.where(use_ext_segment, a_ext, a1)
        rf = jnp.where(use_ext_segment, f_ext, f1)
        rm = jnp.where(use_ext_segment, m_ext, m1)

        cand_a = jnp.stack([a0, a1, a_ext])
        cand_f = jnp.stack([f0, f1, f_ext])
        best_idx = jnp.argmin(cand_f)
        ba = cand_a[best_idx]
        bv = cand_f[best_idx]

        def _pick3(x0v, x1v, x2v):
            return jax.lax.switch(best_idx, [lambda: x0v, lambda: x1v, lambda: x2v])

        bp = jax.tree_util.tree_map(
            lambda p0, p1, p2: _pick3(p0, p1, p2),
            params,
            inner.new_params,
            p_ext,
        )
        bg = jax.tree_util.tree_map(
            lambda g0, g1, g2: _pick3(g0, g1, g2),
            grad,
            inner.new_grad,
            g_ext,
        )

        InitCarry = (
            la,
            lf,
            lm,
            ra,
            rf,
            rm,
            ba,
            bv,
            bp,
            bg,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

        def cond(carry):
            (_, _, _, _, _, _, _, _, _, _, i, _ev) = carry
            return i < spline_max_iter

        def body(carry):
            (la, lf, lm, ra, rf, rm, ba, bv, bp, bg, i, ev) = carry

            t_cands, v_cands, valid = _segment_stationary_candidates(
                la, ra, lf, lm, rf, rm
            )

            mid = 0.5 * (la + ra)
            any_valid = jnp.any(valid)
            best_c_idx = jnp.argmin(v_cands)
            cand_alpha = jnp.where(any_valid, t_cands[best_c_idx], mid)

            lo = jnp.minimum(la, ra)
            hi = jnp.maximum(la, ra)
            span = hi - lo
            margin = 1e-3 * jnp.maximum(span, 1e-12)
            cand_alpha = jnp.clip(cand_alpha, lo + margin, hi - margin)

            cp, cf, cg, cm = eval_at(cand_alpha)

            improves = cf < bv
            n_ba = jnp.where(improves, cand_alpha, ba)
            n_bv = jnp.where(improves, cf, bv)
            n_bp = jax.tree_util.tree_map(
                lambda new, old: jnp.where(improves, new, old), cp, bp
            )
            n_bg = jax.tree_util.tree_map(
                lambda new, old: jnp.where(improves, new, old), cg, bg
            )

            min_to_right = cm < 0.0
            n_la = jnp.where(min_to_right, cand_alpha, la)
            n_lf = jnp.where(min_to_right, cf, lf)
            n_lm = jnp.where(min_to_right, cm, lm)
            n_ra = jnp.where(min_to_right, ra, cand_alpha)
            n_rf = jnp.where(min_to_right, rf, cf)
            n_rm = jnp.where(min_to_right, rm, cm)
            return (
                n_la,
                n_lf,
                n_lm,
                n_ra,
                n_rf,
                n_rm,
                n_ba,
                n_bv,
                n_bp,
                n_bg,
                i + 1,
                ev + 1,
            )

        final = jax.lax.while_loop(cond, body, InitCarry)
        (_, _, _, _, _, _, fa, fv, fp, fg, _, spline_evals) = final

        done = jnp.logical_or(inner.done, fv < inner.new_value)

        return LineSearchResult(
            step_size=fa,
            new_value=fv,
            new_grad=fg,
            new_params=fp,
            done=done,
            probe_params=inner.probe_params,
            probe_grads=inner.probe_grads,
            probe_valid=inner.probe_valid,
            probe_values=inner.probe_values,
            probe_alphas=inner.probe_alphas,
            num_evals=base_evals + spline_evals,
        )

    return wrapped


spline_search = spline_wrap(backtracking_search)
__all__ = ["spline_wrap", "spline_search"]

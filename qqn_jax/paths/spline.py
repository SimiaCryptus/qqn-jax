"""Cubic Hermite *spline* model over the QQN path parameter ``t``.

Unlike ``qqn_jax.paths.linear`` (which discards gradient information)
this module reuses every probe's measured fitness *and* directional
derivative as reusable **control points** ``(t_i, f_i, m_i)``, building
a piecewise cubic Hermite spline of the objective along the path. The
spline is only ever used to *propose* candidate steps; acceptance is
gated by the outer line search's sufficient-decrease test.

The probes themselves are built on the shared ``PathStrategy`` remapping
(defaulting to ``QUADRATIC_PATH``) so every spline probe stays on the
exact curve traversed by the wrapped inner line search. See
``docs/theory/spline_search.md`` for the full derivation.
"""

import jax
import jax.numpy as jnp

from qqn_jax.paths.quadratic import QUADRATIC_PATH

# Re-export the canonical curve the spline model lives on: the spline
# reuses the *quadratic* path geometry and layers a Hermite model on top.
SPLINE_PATH = QUADRATIC_PATH


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
        against = jnp.sign(m) != jnp.sign(delta)
        flat = delta == 0.0
        do_reflect = jnp.logical_and(against, jnp.logical_not(flat))
        return jnp.where(do_reflect, -m, m)

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
    # Guard: negative discriminant -> no real roots.
    real = disc >= 0.0
    sqrt_disc = jnp.sqrt(jnp.where(real, disc, 0.0))

    quad_ok = jnp.abs(A) >= eps
    lin_ok = jnp.abs(B) >= eps

    # Quadratic roots (jit-safe: denominator guarded).
    denom = jnp.where(quad_ok, 2.0 * A, 1.0)
    s_plus = (-B + sqrt_disc) / denom
    s_minus = (-B - sqrt_disc) / denom

    # Linear fallback ``s = -C / B`` when |A| < eps.
    s_lin = jnp.where(lin_ok, -C / jnp.where(lin_ok, B, 1.0), jnp.nan)

    s0 = jnp.where(quad_ok, s_plus, s_lin)
    s1 = jnp.where(quad_ok, s_minus, jnp.nan)
    s = jnp.stack([s0, s1])

    # A root is usable when it is real (quadratic branch) and within [0, 1].
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
    # Evaluate predicted fitness at each candidate (garbage where invalid).
    f_cand = segment_eval(t_cand, t0, f0, m0, t1, f1, m1)
    f_cand = jnp.where(valid, f_cand, jnp.inf)

    return t_cand, f_cand, valid


def propose_step(ts, fs, ms, eps=1e-12):
    """Propose the next evaluation point from sorted control points.

    Scans every adjacent bracketing pair of control points, collects the
    cubic stationary points, and returns the candidate ``t`` with the
    lowest predicted fitness across all segments.

    Args:
        ts: sorted path parameters ``(n,)``.
        fs: measured fitnesses ``(n,)``.
        ms: measured directional derivatives ``(n,)``.

    Returns:
        ``(t_best, f_best, found)`` — the proposed step, its predicted
        fitness, and a boolean flag that is ``False`` when no segment
        yielded an in-range stationary point.
    """
    t0 = ts[:-1]
    f0 = fs[:-1]
    m0 = ms[:-1]
    t1 = ts[1:]
    f1 = fs[1:]
    m1 = ms[1:]

    t_cand, f_cand, valid = jax.vmap(
        lambda a, b, c, d, e, g: segment_candidates(a, b, c, d, e, g, eps)
    )(t0, f0, m0, t1, f1, m1)

    # Flatten (num_segments, 2) -> (num_segments*2,).
    t_flat = t_cand.reshape(-1)
    f_flat = f_cand.reshape(-1)
    v_flat = valid.reshape(-1)

    f_masked = jnp.where(v_flat, f_flat, jnp.inf)
    best = jnp.argmin(f_masked)
    found = jnp.any(v_flat)
    return t_flat[best], f_flat[best], found


__all__ = [
    "SPLINE_PATH",
    "hermite_basis",
    "segment_eval",
    "segment_candidates",
    "propose_step",
]

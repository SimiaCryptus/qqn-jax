"""Cubic Hermite *spline* line search over the scalar step ``t``.

This is a first-class :mod:`qqn_jax.line_search` strategy. It is closely
related to the cubic Hermite *spline path*
(:mod:`qqn_jax.paths.spline`), but the goal is different: rather than
reusing probes to enrich a path model, this line search builds a
piecewise cubic Hermite model of the scalar reduction

    phi(t)  = f(x + d(t)),  phi'(t) = <grad f, d'(t)>

from every measured ``(t, phi, phi')`` pair (a control point) and solves
that model *for an actual minimum* — the in-range stationary point whose
second derivative is positive and whose predicted value is lowest.

Unlike the path variant, no gradient-orientation reflection heuristic is
applied: the slopes returned by ``eval_at`` are the genuine directional
derivatives, so the interpolant is exact to cubic order and its
stationary points are meaningful. Each round measures the proposed point
and appends it as a control point, so successive proposals are drawn from
a strictly richer spline that tightens around the minimizer.

The search is fully JAX-traceable: fixed-capacity control-point buffers
(``2 + max_iter`` slots) and a ``lax.scan`` accumulation loop keep every
shape static.
"""

from typing import Callable

import jax
from jax import numpy as jnp

from qqn_jax.line_search.util import (
    _empty_probes,
    _record_probe,
    _metropolis_accept,
)
from qqn_jax.line_search.result import LineSearchResult


def _hermite_basis(s):
    """Cubic Hermite basis functions at ``s in [0, 1]``."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00, h10, h01, h11


def _segment_eval(t, t0, f0, m0, t1, f1, m1):
    """Interpolated fitness of one Hermite segment (real slopes)."""
    h = t1 - t0
    s = (t - t0) / h
    h00, h10, h01, h11 = _hermite_basis(s)
    return h00 * f0 + h10 * h * m0 + h01 * f1 + h11 * h * m1


def _segment_candidates(t0, f0, m0, t1, f1, m1, eps=1e-12):
    """Closed-form *minimizing* stationary points of a Hermite segment.

    Differentiating ``f(s)`` gives ``f'(s) = A s^2 + B s + C`` with

        A =  6 f0 + 3 h m0 - 6 f1 + 3 h m1
        B = -6 f0 - 4 h m0 + 6 f1 - 2 h m1
        C =         h m0

    A root is retained only when it is real, lies in ``s in [0, 1]``, and
    is a genuine minimum (``f''(s) = 2 A s + B > 0``). No orientation
    heuristic is applied — the measured slopes are used verbatim.
    """
    h = t1 - t0
    hm0 = h * m0
    hm1 = h * m1

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
    # Second derivative test: keep only local minima.
    fpp = 2.0 * A * s + B
    is_min = fpp > 0.0
    valid = jnp.logical_and(jnp.logical_and(branch_real, in_range), is_min)

    t_cand = t0 + s * h
    f_cand = _segment_eval(t_cand, t0, f0, m0, t1, f1, m1)
    f_cand = jnp.where(valid, f_cand, jnp.inf)
    return t_cand, f_cand, valid


def _propose(ts, fs, ms, valid, eps=1e-12):
    """Best minimizing proposal across all segments of a padded buffer."""
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
        lambda a, b, c, d, e, g: _segment_candidates(a, b, c, d, e, g, eps)
    )(t0, f0, m0, t1, f1, m1)

    cand_valid = jnp.logical_and(cand_valid, pair_ok[:, None])

    t_flat = t_cand.reshape(-1)
    f_flat = f_cand.reshape(-1)
    v_flat = cand_valid.reshape(-1)

    f_masked = jnp.where(v_flat, f_flat, jnp.inf)
    best = jnp.argmin(f_masked)
    found = jnp.any(v_flat)
    return t_flat[best], f_flat[best], found


def spline_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
    init_step: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 20,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Cubic Hermite spline line search seeking a true 1-D minimum.

    Seeds two control points — the origin ``(0, phi(0), phi'(0))`` and an
    initial probe at ``init_step`` — then runs ``max_iter`` accumulate-and-
    repropose rounds. Each round proposes the lowest-value minimizing
    stationary point of the current piecewise cubic model (falling back to
    a midpoint bisection of the best span when the model yields no interior
    minimum), measures it, and appends the measurement as a new control
    point. The best strictly-improving measurement is returned.
    """
    dtype = value.dtype
    dg = slope0
    max_alpha = jnp.asarray(max_step, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    temp0 = jnp.asarray(temperature, dtype=dtype)
    key0 = jax.random.PRNGKey(seed)

    eff_probes = max_probes if record_probes else 1
    pp, pg, pv, pval, pa = _empty_probes(params, eff_probes)

    capacity = 2 + int(max_iter)
    ts = jnp.zeros((capacity,), dtype)
    fs = jnp.full((capacity,), jnp.inf, dtype)
    ms = jnp.zeros((capacity,), dtype)
    valid = jnp.zeros((capacity,), bool)

    # Origin control point (no evaluation needed).
    ts = ts.at[0].set(zero)
    fs = fs.at[0].set(value)
    ms = ms.at[0].set(dg)
    valid = valid.at[0].set(True)

    # Initial probe.
    a0 = jnp.minimum(jnp.asarray(init_step, dtype=dtype), max_alpha)
    p0, v0, g0, s0 = eval_at(a0)
    pp, pg, pv, pval, pa = _record_probe(
        pp, pg, pv, pval, pa, 0, p0, g0, v0, a0, eff_probes
    )
    ts = ts.at[1].set(a0)
    fs = fs.at[1].set(v0)
    ms = ms.at[1].set(s0)
    valid = valid.at[1].set(True)
    count = jnp.asarray(2, jnp.int32)

    improved0 = v0 < value
    best_a = jnp.where(improved0, a0, zero)
    best_v = jnp.where(improved0, v0, value)
    best_p = jnp.where(improved0, p0, params)
    best_g = jnp.where(improved0, g0, grad)

    stoch0, key0 = _metropolis_accept(v0 - value, temp0, key0, dtype)
    temp0 = temp0 * cooling
    best_a = jnp.where(stoch0, a0, best_a)
    best_v = jnp.where(stoch0, v0, best_v)
    best_p = jnp.where(stoch0, p0, best_p)
    best_g = jnp.where(stoch0, g0, best_g)
    accepted0 = stoch0

    def round_step(carry, _):
        (
            ts,
            fs,
            ms,
            valid,
            count,
            best_a,
            best_v,
            best_p,
            best_g,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ) = carry

        t_prop, _f_pred, found = _propose(ts, fs, ms, valid)

        f_masked = jnp.where(valid, fs, jnp.inf)
        lo_idx = jnp.argmin(f_masked)
        t_lo = ts[lo_idx]
        t_min = jnp.min(jnp.where(valid, ts, jnp.inf))
        t_max = jnp.max(jnp.where(valid, ts, -jnp.inf))
        span_other = jnp.where(t_lo == t_max, t_min, t_max)
        t_mid = 0.5 * (t_lo + span_other)

        t_eval = jnp.where(found, t_prop, t_mid)
        t_eval = jnp.clip(t_eval, zero, max_alpha)

        p, v, g, s = eval_at(t_eval)
        t_eval = jnp.asarray(t_eval, dtype)
        v = jnp.asarray(v, dtype)
        s = jnp.asarray(s, dtype)

        probe_slot = count - 1
        pp, pg, pv, pval, pa = _record_probe(
            pp, pg, pv, pval, pa, probe_slot, p, g, v, t_eval, eff_probes
        )

        i = count
        ts = ts.at[i].set(t_eval)
        fs = fs.at[i].set(v)
        ms = ms.at[i].set(s)
        valid = valid.at[i].set(True)
        count = count + jnp.asarray(1, jnp.int32)

        improved = v < best_v
        best_a = jnp.where(improved, t_eval, best_a)
        best_v = jnp.where(improved, v, best_v)
        best_p = jnp.where(improved, p, best_p)
        best_g = jnp.where(improved, g, best_g)

        stoch, key = _metropolis_accept(v - value, temp, key, dtype)
        temp = temp * cooling
        best_a = jnp.where(stoch, t_eval, best_a)
        best_v = jnp.where(stoch, v, best_v)
        best_p = jnp.where(stoch, p, best_p)
        best_g = jnp.where(stoch, g, best_g)
        accepted = jnp.logical_or(accepted, stoch)

        new_carry = (
            ts,
            fs,
            ms,
            valid,
            count,
            best_a,
            best_v,
            best_p,
            best_g,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        )
        return new_carry, None

    init_carry = (
        ts,
        fs,
        ms,
        valid,
        count,
        best_a,
        best_v,
        best_p,
        best_g,
        accepted0,
        temp0,
        key0,
        pp,
        pg,
        pv,
        pval,
        pa,
    )

    (
        (
            ts,
            fs,
            ms,
            valid,
            count,
            best_a,
            best_v,
            best_p,
            best_g,
            accepted,
            temp,
            key,
            pp,
            pg,
            pv,
            pval,
            pa,
        ),
        _,
    ) = jax.lax.scan(round_step, init_carry, None, length=int(max_iter))

    improved = best_v < value
    armijo = best_v <= value + c1 * best_a * dg
    done = jnp.logical_or(jnp.logical_or(armijo, accepted), improved)

    total_evals = jnp.asarray(1 + int(max_iter), jnp.int32)
    return LineSearchResult(
        step_size=best_a,
        new_value=best_v,
        new_grad=best_g,
        new_params=best_p,
        done=done,
        probe_params=pp,
        probe_grads=pg,
        probe_valid=pv,
        probe_values=pval,
        probe_alphas=pa,
        num_evals=total_evals,
    )


__all__ = [
    "spline_search",
    "hermite_basis",
    "segment_eval",
    "segment_candidates",
]

# Public aliases (documented helper names).
hermite_basis = _hermite_basis
segment_eval = _segment_eval
segment_candidates = _segment_candidates
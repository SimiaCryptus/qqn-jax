# Cubic Hermite Spline Path Search

A stateful line-search *path strategy* for QQN (Quadratic-Quadratic Newton) that builds and refines a **piecewise cubic
Hermite spline model**
of the objective along the QQN blend, using every probe as a reusable control point.

---

## 1. Overview

The spline path is a **distinct, stateful** `PathStrategy`. It is *not* a wrapper around the quadratic path. Its central
idea is that the objective can be modeled as a scalar function `f(t)` of the QQN path parameter `t`, and that this model
can be *accumulated and refined* as measurements come in.

Two curves must be kept carefully distinct:

| Curve                        | Symbol | Role                                                                                       |
|------------------------------|--------|--------------------------------------------------------------------------------------------|
| **Parameter-space geometry** | `d(t)` | Where a probe physically lands: `x + d(t)`. This is the QQN blend, *not* the spline.       |
| **Spline model**             | `f(t)` | The accumulating scalar model of the objective as a function of `t`. This *is* the spline. |

The model **begins** as a single cubic Hermite segment spanning two endpoints (reproducing the quadratic-order picture)
and **becomes a genuine piecewise cubic Hermite spline** as measurements accumulate. Each probe contributes both its
measured fitness `f_i` *and* directional derivative
`m_i`, retained as a control point `(t_i, f_i, m_i)`.

---

## 2. Path Geometry `d(t)`

The displacement blends the steepest-descent tangent `grad_dir` (`-∇f`)
with the oracle endpoint direction `direction` (`-H∇f`):

```
d(t)  = t(1 - t)·grad_dir + t²·direction
d'(t) = (1 - 2t)·grad_dir + 2t·direction
```

- `_spline_offset(t, ...)` returns `d(t)` — the actual probe displacement.
- `_spline_velocity(t, ...)` returns `d'(t)` — used to project a measured gradient into a scalar directional derivative
  `m = ⟨∇f, d'(t)⟩`.

Every evaluation is taken *along* this blend, so the scalar `m` returned by the shared evaluator is a ready-made spline
control-point slope.

---

## 3. Cubic Hermite Model

### 3.1 Basis functions

On a normalized segment coordinate `s ∈ [0, 1]`:

```
h00(s) =  2s³ - 3s² + 1
h10(s) =      s³ - 2s² + s
h01(s) = -2s³ + 3s²
h11(s) =      s³ -  s²
```

(`hermite_basis(s)`.)

### 3.2 Segment interpolation

For a segment between control points `(t0, f0, m0)` and `(t1, f1, m1)`, with `h = t1 - t0` and `s = (t - t0)/h`:

```
f(s) = h00(s)·f0 + h10(s)·h·m0 + h01(s)·f1 + h11(s)·h·m1
```

(`segment_eval`.)

### 3.3 Gradient orientation heuristic

Before use, endpoint tangents are passed through `_orient_tangents`. If a tangent `m` opposes the segment secant slope
`delta = (f1 - f0)/h`
(`sign(m) ≠ sign(delta)` and `delta ≠ 0`), it is reflected: `m ← -m`. When `delta == 0`, no reflection is applied.

This is an **unproven heuristic** for enforcing upstream/downstream symmetry. It is kept safe by the outer line search's
strict-improvement gate — a bad reflection can only fail to improve, never corrupt the iterate.

---

## 4. Stationary Points (Proposal Core)

Differentiating the Hermite segment gives a quadratic
`f'(s) = A·s² + B·s + C`:

```
A =  6·f0 + 3·h·m0 - 6·f1 + 3·h·m1
B = -6·f0 - 4·h·m0 + 6·f1 - 2·h·m1
C =         h·m0
```

`segment_candidates` solves this closed form:

- Uses the discriminant `disc = B² - 4AC` to detect real roots.
- Falls back to the linear root `-C/B` when `|A| < eps` (degenerate quadratic).
- Returns two candidate roots, each with a `valid` flag requiring the root be **real** and lie in `s ∈ [0, 1]` (mapped
  back to `t = t0 + s·h`).
- Invalid candidates get fitness `+∞`.

Numerical guards (`eps`, `jnp.where` masking of divisors and sqrt arguments) keep the computation NaN-free and `jit`/
`vmap`-safe.

---

## 5. Proposing From a Control-Point Buffer

`propose_from_points(ts, fs, ms, valid, eps)`:

1. **Sort** control points by `t`, pushing invalid slots (masked by
   `valid`) to the end via a large sort key.
2. Form consecutive segment pairs; a pair contributes only if **both**
   endpoints are valid and `t1 - t0 > eps`.
3. `vmap` `segment_candidates` over all segments.
4. Return `(t_best, f_best, found)` — the candidate with the lowest predicted fitness across all segments;
   `found = False` if no segment produced an in-range stationary point.

`propose_step(ts, fs, ms)` is a convenience wrapper for the fully-valid, unpadded case.

---

## 6. Stateful Interface (`SplineState`)

Because it accumulates control points, the spline carries explicit state and implements the optional stateful
`PathStrategy` hooks.

`SplineState` fields (fixed-capacity buffers):

| Field        | Shape              | Meaning                          |
|--------------|--------------------|----------------------------------|
| `ts`         | `(capacity,)`      | control-point parameters         |
| `fs`         | `(capacity,)`      | measured fitnesses (init `+∞`)   |
| `ms`         | `(capacity,)`      | measured directional derivatives |
| `valid`      | `(capacity,)` bool | per-slot validity                |
| `num_points` | scalar int32       | count recorded so far            |

- `spline_init(grad_dir, direction, capacity=16, dtype)` — allocate empty buffer (geometry args accepted only to match
  the hook signature).
- `spline_observe(state, t, f, m)` — write a control point at the next slot (clamped to capacity).
- `spline_propose(state)` — delegate to `propose_from_points`.

These are wired into `SPLINE_PATH`, a `PathStrategy` with
`stateful=True` and the offset/velocity/init/observe/propose callbacks.

---

## 7. Refinement Loop (`spline_refine`)

This is the algorithm's defining behavior: turning an inner line-search result into a *strictly richer* spline via
accumulate-and-repropose.

### 7.1 Seeding

Control points are seeded from:

- the origin `t = 0` → `(0, f0, slope0)`;
- every recorded inner probe `(alpha_i, value_i, |m_i|)`, where
  `m_i = ⟨g_i, d'(alpha_i)⟩`;
- the accepted endpoint `(step_size, new_value, |m_end|)`.

Slopes are stored as **absolute values** (`slope_of` returns `|m|`), consistent with the orientation heuristic that
reflects tangents by secant sign.

Buffers are pre-padded with `rounds` extra slots for the accumulation loop.

### 7.2 Per-round step (`lax.scan`, `rounds` iterations)

Each round costs exactly **one** evaluation:

1. **Propose**: `propose_from_points` over the current buffer.
2. **Fallback**: if no valid stationary point (`found = False`), evaluate a midpoint between the current lowest-fitness
   control point and the opposite span endpoint (origin or accepted endpoint). This guarantees progress even on a
   degenerate model.
3. **Measure**: call `eval_at(t_eval)` → `(params, value, grad, slope)`.
4. **Append**: store the new `(t, f, m)` as a control point at index
   `count`, incrementing `count`.
5. **Track best**: update the running best iterate only on **strict improvement** over both the current best *and* the
   origin fitness (`v < best_v ∧ v < origin_f`).

### 7.3 Result

Returns a `LineSearchResult` carrying the best strictly-improving measurement over all rounds. `done` is
`inner.done ∨ best_found`. Probe bookkeeping from the inner result is preserved; `num_evals` is bumped by
`rounds`.

---

## 8. Design Discussion

### Why accumulate?

A single Hermite segment captures curvature to quadratic-plus order. By *retaining* each measured `(t, f, m)`,
subsequent proposals interpolate between richer anchors — the model literally becomes a piecewise cubic spline that
tightens around the minimizer with each evaluation. This is unusual: rather than a fixed interpolation, the search is a
*self-refining*
model.

### Why is the reflection heuristic safe?

The orientation rule (`_orient_tangents`) is not proven optimal. It is a symmetry correction that can only change
*which* candidate the model proposes. Because acceptance is gated by strict improvement, an inaccurate reflection wastes
at most one evaluation; it can never worsen the returned iterate.

### JAX-friendliness

All routines are shape-static and branch-free (`jnp.where`, masking, fixed
`rounds` in `lax.scan`), so the whole strategy is `jit`- and
`vmap`-compatible. Fixed-capacity buffers with a `valid` mask replace dynamic growth.

### Cost model

- Seeding: no extra evaluations (reuses inner probes).
- Refinement: exactly `rounds` evaluations, one control point added each.
- Proposal: closed-form; no inner optimization loop.

---

## 9. Public API

| Symbol                                              | Purpose                                      |
|-----------------------------------------------------|----------------------------------------------|
| `SPLINE_PATH`                                       | ready-to-use stateful `PathStrategy`         |
| `SplineState`                                       | control-point memory container               |
| `spline_init` / `spline_observe` / `spline_propose` | stateful hooks                               |
| `hermite_basis`                                     | cubic Hermite basis at `s`                   |
| `segment_eval`                                      | interpolated fitness of one segment          |
| `segment_candidates`                                | closed-form stationary points of one segment |
| `propose_from_points`                               | best proposal from a padded buffer           |
| `propose_step`                                      | proposal from fully-valid points             |
| `spline_refine`                                     | accumulate-and-repropose refinement loop     |

See `docs/theory/spline_search.md` for the full derivation.
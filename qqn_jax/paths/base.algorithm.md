# Path-Strategy Abstraction (`base.py`)

## Overview

This module defines the **shared path-strategy abstraction** used throughout `qqn_jax.paths`. Its job is to decouple the
*one-dimensional*
line-search problem from the *multidimensional* geometry of the curve being traversed. Every path module — `linear`,
`quadratic`, and
`spline` — implements the same interface so that line searches (and their augmentations) never need to know which curve
is in play.

## Core Concept: The Path

A **path** remaps the scalar line-search parameter `t` into a probe point in parameter space:

```
    probe(t) = x + d(t)
```

where:

- `x` is the current parameter pytree (`params`),
- `d(t)` is the **offset** (displacement) produced by the path,
- `d'(t)` is the **velocity** (tangent) of the path at `t`.

The velocity is used to project a measured gradient `∇f` onto the directional derivative along the curve:

```
    slope(t) = ⟨∇f(probe(t)), d'(t)⟩
```

This projection is what turns the multidimensional optimization step into a scalar 1-D subproblem that the line searches
can consume.

## `PathStrategy`

`PathStrategy` is a `NamedTuple` bundling the callables that define a path. It supports both **stateless** and
**stateful** paths.

| Field        | Signature                                     | Purpose                                                           |
|--------------|-----------------------------------------------|-------------------------------------------------------------------|
| `offset`     | `(t, grad_dir, direction) -> d(t)`            | Displacement from `params` at `t`.                                |
| `velocity`   | `(t, grad_dir, direction) -> d'(t)`           | Path tangent at `t`, used for slope projection.                   |
| `init_state` | `(grad_dir, direction, ...) -> path_state`    | Allocate control-point memory (stateful paths). Optional.         |
| `observe`    | `(path_state, t, value, slope) -> path_state` | Record a measured control point `(t, f, m)`. Optional.            |
| `propose`    | `(path_state) -> (t, found)`                  | Propose the next candidate `t` from accumulated points. Optional. |
| `stateful`   | `bool`                                        | `True` when the path carries/updates its own state.               |

### Stateless vs. Stateful Paths

- **Stateless paths** (e.g. `linear`, `quadratic`) implement only
  `offset` and `velocity`. Their `init_state`, `observe`, and `propose`
  fields are `None`, and `stateful` is `False`. The curve is fully determined analytically by `grad_dir` and
  `direction`.

- **Stateful paths** (e.g. `spline`) additionally implement
  `init_state`, `observe`, and `propose`. They accumulate measured control points `(t, value, slope)` in a `path_state`
  and use them to construct and refine the curve on the fly. `stateful` is `True`.

### Common Arguments

Every `offset`/`velocity` call receives:

- `t` — the scalar path parameter,
- `grad_dir` — the steepest-descent tangent `-grad` measured at `t = 0`,
- `direction` — the oracle-supplied search direction.

This uniform signature is what allows all paths to be substituted interchangeably at every call site.

## `make_evaluator`

`make_evaluator` is the **single place** where the scalar parameter `t`
is remapped into the multidimensional probe point. It builds a closure:

```
    eval_at(t) -> (projected_params, value, grad, slope)
```

which is exactly the scalar 1-D problem handed to the (path-unaware)
line searches.

### Algorithm

Given `value_and_grad_fn`, `params`, `grad`, `direction`, a `region`
(with its `region_state`), and a `PathStrategy`:

1. **Fix the descent tangent.** Compute `grad_dir = -grad`, where
   `grad` is the gradient measured at `t = 0` (i.e. at `params`). This is the steepest-descent tangent every path
   receives.

2. **Define projection.** `project(candidate)` maps a raw probe back into the feasible region via `region.project(params, candidate,
   region_state)`.

3. **Define `eval_at(t)`:**
    1. Compute the offset `d = path.offset(t, grad_dir, direction)`.
    2. Form the raw probe `raw = params + 1.0 * d` (via
       `tree_add_scaled`).
    3. Project it: `projected = project(raw)`.
    4. Evaluate the oracle: `val, g = value_and_grad_fn(projected,
      *args)`.
    5. Compute the velocity `v = path.velocity(t, grad_dir,
      direction)`.
    6. Project the gradient onto the tangent:
       `slope = ⟨g, v⟩` (via `tree_vdot`).
    7. Return `(projected, val, g, slope)`.

### Region Projection

Every probe passes through `region.project` before evaluation. This keeps candidates feasible (e.g. inside a trust
region) regardless of the path being traversed, and ensures line searches operate on projected points consistently.

## Discussion

### Why a Shared Abstraction?

The key design goal is **path-agnostic line searches**. By funneling every probe construction through `make_evaluator`
and `PathStrategy`, the codebase guarantees that:

- `linear`, `quadratic`, and `spline` build candidates identically, modulo the strategy they are given;
- line searches see only the scalar `eval_at(t)` interface and remain entirely unaware of curve geometry;
- gradient-to-slope projection is derived from the path's own analytic
  `velocity`, so no consumer re-derives the curve's derivative.

### The Velocity/Slope Contract

A subtle but important detail: the slope returned to the line search is the *directional derivative along the curve*,
`⟨∇f, d'(t)⟩`, not the plain directional derivative along a fixed direction. This is what lets curved paths (quadratic,
spline) satisfy Wolfe-style conditions correctly — the line search's notion of "slope" tracks the actual tangent of the
curve at the probe point.

### Statefulness and Adaptive Paths

The optional `init_state`/`observe`/`propose` trio enables paths that *learn* from the measurements taken during a line
search. A stateful path can record each `(t, value, slope)` control point and use the accumulated data to propose better
candidates (e.g. spline interpolation through observed points). Stateless analytic paths simply ignore this machinery.

## Dependencies

- `qqn_jax.utils.tree_add_scaled` — pytree scaled addition (`params + 1.0 * d`).
- `qqn_jax.utils.tree_negative` — pytree negation (`grad_dir = -grad`).
- `qqn_jax.utils.tree_vdot` — pytree inner product (slope projection).

## Public API

```
    __all__ = ["PathStrategy", "make_evaluator"]
```
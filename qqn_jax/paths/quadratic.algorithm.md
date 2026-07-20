# Quadratic Path Algorithm

## Overview

The **quadratic path** is QQN's canonical parametric curve. It blends the
steepest-descent direction with the quasi-Newton (L-BFGS) direction along a
single scalar parameter `t ∈ [0, 1]`, producing a smooth parabolic
trajectory in parameter space.

This document describes the algorithm as implemented in
`qqn_jax/paths/quadratic.py`.

## Mathematical Definition

Given:

- `grad_dir = -∇f` — the steepest-descent direction (negative gradient),
- `qn_dir  = -H∇f` — the L-BFGS / quasi-Newton direction,

the quadratic path is defined as

```
d(t) = t(1 - t) · (-∇f)  +  t² · (-H∇f),      t ∈ [0, 1]
```

Expanding the scalar coefficients:

- `a(t) = t(1 - t)` weights the gradient tangent,
- `b(t) = t²`       weights the quasi-Newton endpoint.

### Boundary Behaviour

| `t`   | `a(t)` | `b(t)` | `d(t)`             | Interpretation                        |
|-------|--------|--------|--------------------|---------------------------------------|
| `0`   | `0`    | `0`    | `0`                | Origin (current iterate).             |
| `1`   | `0`    | `1`    | `-H∇f`             | Exact quasi-Newton (oracle) endpoint. |

Near the origin (`t → 0`), the path is tangent to the steepest-descent
direction because `d'(0) = -∇f` (see below). This guarantees an initial
descent direction whenever the gradient is nonzero, while smoothly steering
toward the quasi-Newton step as `t → 1`.

## Derivative

The velocity (derivative with respect to `t`) is

```
d'(t) = (1 - 2t) · (-∇f)  +  2t · (-H∇f)
```

with coefficients:

- `a'(t) = 1 - 2t`,
- `b'(t) = 2t`.

Key values:

- `d'(0) = -∇f` — the initial tangent is exactly steepest descent.
- `d'(1) = -∇f + 2(-H∇f)` — the exit velocity.

## Implementation

The module exposes two pure functions and one packaged strategy instance.

### `_quadratic_path(t, grad_dir, qn_dir)`

Computes `d(t)`. Coefficients `a = t(1 - t)` and `b = t²` are combined via
`jax.tree_util.tree_map`, so `grad_dir` and `qn_dir` may be arbitrary
(matching) pytrees:

```python
a = t * (1.0 - t)
b = t * t
return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, qn_dir)
```

### `quadratic_path_derivative(t, grad_dir, qn_dir)`

Computes `d'(t)` using coefficients `a = 1 - 2t` and `b = 2t`:

```python
a = 1.0 - 2.0 * t
b = 2.0 * t
return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, qn_dir)
```

### `QUADRATIC_PATH`

A `PathStrategy` instance bundling the offset and velocity functions:

```python
QUADRATIC_PATH = PathStrategy(
    offset=_quadratic_path, velocity=quadratic_path_derivative
)
```

This is the canonical strategy used throughout the solver. In particular,
`qqn_jax.paths.spline` uses `QUADRATIC_PATH` by default, ensuring every
spline probe stays on the exact curve traversed by the wrapped inner line
search.

## Design Notes and Discussion

- **Pytree generality.** Both functions rely on `tree_map`, so the path
  works transparently for scalar, array, or nested-structure parameter
  representations without special-casing.

- **Purity / JAX-compatibility.** The functions are side-effect free and
  composed of primitive arithmetic, making them fully compatible with
  `jit`, `grad`, `vmap`, and other JAX transformations.

- **Why parabolic?** The `t(1 - t)` gradient weight vanishes at both
  endpoints, while the `t²` quasi-Newton weight grows monotonically. This
  yields the two crucial QQN guarantees: a guaranteed descent tangent at
  the start and an exact quasi-Newton step at the end, with a single smooth
  interpolation in between. The one-dimensional parameterisation reduces the
  multi-dimensional step-selection problem to a scalar line search over
  `t`.

## References

- `docs/paper/draft.md` §4.2 — full derivation.
- `qqn_jax.solver` — how the path is consumed by the optimizer.
- `qqn_jax.paths.base.PathStrategy` — the offset/velocity container type.
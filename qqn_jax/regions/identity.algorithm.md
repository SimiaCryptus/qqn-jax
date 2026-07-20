# IdentityRegion Algorithm

## Overview

The `IdentityRegion` implements the **trivial region** within the
`qqn_jax.regions` framework. A "region" in this context represents a
constraint set (or trust region) onto which candidate parameter updates
are projected during optimization. The identity region imposes **no
constraints** whatsoever: every candidate point is accepted as-is.

This is the simplest possible `Region` implementation and serves several
purposes:

- **Baseline / no-op behavior**: Use it when you want to disable region
  constraints entirely while still conforming to the `Region` interface.
- **Testing and composition**: It acts as a neutral element for algorithms
  that expect a `Region` object but where constraint handling is optional.
- **Reference implementation**: It documents the minimal contract every
  `Region` must satisfy.

## The `Region` Interface

A `Region` is a triple of pure functions:

| Function | Signature | Responsibility |
|----------|-----------|----------------|
| `init`    | `init(params) -> state`                    | Produce the initial region state given the model parameters. |
| `project` | `project(params, candidate, state) -> point` | Map a candidate point into the feasible region. |
| `update`  | `update(state, info) -> state`             | Evolve the region state using step feedback (`info`). |

All functions are intended to be side-effect free (JAX-friendly), so the
region can be embedded inside `jit`/`scan` transformations.

## Algorithm Description

The identity region defines each of the three functions as a no-op:

### `_identity_init(params) -> ()`

Returns an empty tuple `()` as the region state. The identity region is
**stateless** — there is nothing to track between iterations, so an empty
tuple is used as a trivial, immutable placeholder state.

### `_identity_project(params, candidate, state) -> candidate`

Returns the `candidate` unchanged. Because the identity region represents
the entire (unconstrained) space, every candidate point is already
feasible, so projection is the identity map:

```
project(params, x, state) = x
```

### `_identity_update(state, info) -> state`

Returns the incoming `state` unchanged. Since the state is empty and
carries no information, there is nothing to update in response to step
feedback.

### `IdentityRegion() -> Region`

A factory function that assembles the three no-op functions into a
`Region` instance:

```python
Region(
    init=_identity_init,
    project=_identity_project,
    update=_identity_update,
)
```

## Mathematical Interpretation

Formally, the identity region corresponds to the feasible set
`C = ℝⁿ` (the whole parameter space). The Euclidean projection onto ℝⁿ is
the identity operator:

```
P_C(x) = argmin_{y ∈ ℝⁿ} ‖y − x‖²  =  x
```

Because the constraint set never changes, the state update is likewise a
fixed point.

## Complexity

- **Time**: O(1) per call for all three functions (no computation beyond
  returning the argument).
- **Space**: O(1); the state is an empty tuple.

## Usage Example

```python
from qqn_jax.regions.identity import IdentityRegion

region = IdentityRegion()

state = region.init(params)                       # ()
x_next = region.project(params, candidate, state) # == candidate
state = region.update(state, info)                # == state (unchanged)
```

## Discussion

The `IdentityRegion` embodies the *null object* design pattern applied to
optimization constraints. Rather than sprinkling optional/`None` checks
throughout the optimizer, callers can always operate on a valid `Region`;
when no constraint is desired, they simply supply the identity region.

Key properties:

- **Neutral element**: Composing (or chaining) any region with the
  identity region leaves behavior unchanged, making it convenient for
  generic pipelines.
- **Statelessness**: The empty-tuple state guarantees that the region
  introduces no hidden dynamics and is trivially compatible with JAX
  functional transformations (`jit`, `vmap`, `scan`), which require pure,
  traceable state.
- **Zero overhead**: The functions perform no arithmetic, so wrapping an
  optimizer with `IdentityRegion` incurs negligible runtime cost.

### When to use

Prefer `IdentityRegion` for:

- Unconstrained optimization within a region-aware framework.
- Ablation studies where the effect of a constraint is measured against a
  no-constraint baseline.
- Unit tests that need a valid but inert `Region`.

### When not to use

If actual constraints (box bounds, trust-region radii, simplex
projections, etc.) are required, use the corresponding specialized region
implementation instead — the identity region will silently accept
infeasible candidates.
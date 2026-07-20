# Projective Regions for QQN

## Overview

This document describes the *projective region* abstraction used by the
QQN (Quadratic Quasi-Newton) optimizer, as implemented in `strategy.py`
and the `qqn_jax.regions` package. It explains the underlying algorithm,
the design rationale, and how regions integrate with QQN's line search.

## Motivation

QQN searches a single continuous **quadratic path** `d(t)` in parameter
space rather than a straight line. A *projective region* constrains or
reshapes a proposed update so that the applied step always lands in a
feasible or preferred set. Examples of such sets include:

- Box / bound constraints on parameters.
- Trust regions that limit the magnitude of a step.
- The identity ("no constraint") region, used as the default.

By expressing the constraint as a **projection operator**, the region
composes cleanly with the existing quadratic path search — no bespoke
constrained-optimization machinery is required.

## Core Algorithm

A region `R` supplies a projection

    project_R(x, y) -> y'

that maps a proposed point `y` back onto the feasible set relative to the
current iterate `x`. Given QQN's quadratic path `d(t)`, the region defines
the **projected path**

    d_R(t) = project_R(x, x + d(t)) - x

The line search then operates on `d_R(t)` instead of `d(t)`. Because the
projection is folded into the path itself, every candidate step evaluated
during the line search is already feasible.

### Step-by-step

1. QQN constructs the quadratic path `d(t)` from the gradient and the
   quasi-Newton direction.
2. For each trial value of `t`, the raw candidate point `x + d(t)` is
   computed.
3. The region projects this candidate: `project_R(x, x + d(t))`.
4. The projected displacement `d_R(t)` is used for evaluation and,
   ultimately, the accepted update.

## Functional / JAX Design

All regions are implemented as **pure, functional JAX** so that they
compose with the standard transformations:

- `jit`   — for compilation.
- `vmap`  — for batched / vectorized application.
- `pmap`  — for multi-device parallelism.
- `grad`  — for differentiating through the projection.

Purity (no in-place mutation, no hidden state) is essential to preserve
these properties. State that a region needs to carry (e.g. a trust-region
radius) is threaded explicitly through a `RegionState` value rather than
stored on the region object.

## Public Interface (`strategy.py`)

The `strategy` module re-exports the region API and provides a small
resolution helper.

### Types

- `Region` — the abstract region protocol (from `regions.types`).
- `RegionInfo` — auxiliary information returned alongside a projection.
- `RegionState` — an alias for `Any`; the opaque per-region state.
- `TrustRegionState` — state for the trust-region variant.

### `resolve_region`

```py
def resolve_region(region: Optional[Region]) -> Region:
    """Return ``region`` or the identity region when ``None``."""
    return IdentityRegion() if region is None else region
```

This normalizes the caller-supplied region. Passing `None` (or an
explicit `IdentityRegion()`) selects the no-op projection.

### Tree helpers

- `_tree_add(a, b)` — pytree-wise addition (`x + d`).
- `_tree_sub(a, b)` — pytree-wise subtraction (`y' - x`).

These operate over arbitrarily nested parameter pytrees, allowing the
projected-path arithmetic (`project_R(x, x + d) - x`) to be expressed
structurally.

## The Identity Region

When `region is None` or `region` is an `IdentityRegion`, the projection
is the identity map:

    project_R(x, y) = y  =>  d_R(t) = d(t)

In this case the optimizer is **byte-for-byte equivalent** to the
un-regioned QQN. This guarantee is important: adding the region
abstraction imposes zero behavioral change when no region is requested,
which makes it safe to layer regions onto existing code and to validate
them against a known-good baseline.

## Discussion

### Why projection instead of penalty methods?

Penalty and barrier methods perturb the objective and require careful
tuning of penalty weights, and they only approximately enforce
constraints. Projection enforces feasibility **exactly** at every trial
point and keeps the objective unmodified, so the line search retains its
usual acceptance criteria.

### Why fold projection into the path?

Projecting the entire path (rather than only the final accepted step)
ensures the line search "sees" the true feasible geometry. This avoids
accepting a step length based on infeasible candidate values that would
later be clipped, which can otherwise cause poor step selection and
stalling near constraint boundaries.

### Composability

Because regions are pure JAX and state is explicit, they can be:

- composed with one another (chained projections),
- batched across problem instances with `vmap`,
- distributed with `pmap`,
- and differentiated through when the projection is smooth.

### Extending with new regions

A new region implements the `Region` protocol: a projection function and,
optionally, a `RegionState` for carried state (e.g. an adaptive
trust-region radius updated between iterations). The trust-region variant,
represented by `TrustRegionState`, is the canonical stateful example.

## Summary

Projective regions give QQN a clean, functional mechanism for constrained
and preference-shaped optimization. By remapping the quadratic search path
through a pure projection operator, they integrate transparently with the
line search, preserve full JAX transformability, and default to a no-op
identity that leaves the base optimizer unchanged.
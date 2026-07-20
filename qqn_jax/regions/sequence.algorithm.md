# Sequential Region Composition

## Overview

The `Sequential` combinator composes a collection of *regions* into a
single region by applying their projection operators one after another.
A region encapsulates the notion of a feasible set together with a
projection onto that set and an internal, updatable state. Composing
regions lets us express the intersection of several constraints and
reduce a candidate point onto all of them in sequence.

## The `Region` Abstraction

A `Region` (defined in `qqn_jax.regions.strategy`) is a triple of pure
functions:

- `init(params) -> state` — build the initial internal state from the
  problem parameters.
- `project(params, candidate, state) -> candidate` — map an arbitrary
  candidate point onto (or toward) the feasible set represented by the
  region.
- `update(state, info) -> state` — evolve the internal state given new
  runtime information (for example, iteration statistics).

Keeping these as plain functions makes regions composable and friendly to
JAX transformations (`jit`, `vmap`, `grad`), since there is no hidden
mutable object state.

## Algorithm

Given an ordered sequence of child regions `[R_1, R_2, ..., R_k]`,
`Sequential` produces a new region whose three operations are defined as
follows.

### Initialization

```
init(params) = (R_1.init(params), ..., R_k.init(params))
```

The composite state is simply a tuple holding the initial state of each
child, in order.

### Projection

The projection is the **left-to-right function composition** of the child
projections:

```
project = R_k ∘ R_{k-1} ∘ ... ∘ R_1
```

Concretely, the candidate is threaded through each child projection using
that child's slice of the composite state:

```
c_0 = candidate
c_i = R_i.project(params, c_{i-1}, state_i)   for i = 1..k
project(...) = c_k
```

### Update

The update **fans out** to every child independently, pairing each child
region with its corresponding state entry and threading the same `info`:

```
update(state, info) = (R_1.update(state_1, info), ..., R_k.update(state_k, info))
```

## Discussion

### Ordering matters

Because `project` is a composition, the order of the regions is
significant. Projecting onto `R_1` and then `R_2` is not, in general, the
same as projecting onto `R_2` and then `R_1`. This is the classic
behavior of alternating projections: a single pass does not necessarily
land on the exact intersection of all constraint sets unless the sets are
compatible (e.g., they commute, or the projections are onto affine
subspaces that are handled in an idempotent way). Callers that need
convergence onto the intersection typically iterate the sequential
projection repeatedly (a Kaczmarz / POCS-style scheme).

### State is per-child and parallel

Even though projection is inherently sequential, the state carried by the
composite is structurally parallel — a tuple mirroring the children. The
`update` step does not depend on projection results; it broadcasts the
same `info` to every child, letting each region maintain its own local
bookkeeping independently.

### Purity and composability

All three operations are pure and closed over the immutable `regions`
tuple (frozen at construction time via `tuple(regions)`). This makes the
resulting `Region` itself a first-class value that can be nested inside
further `Sequential` compositions or other combinators without surprises.

### Edge cases

- **Empty sequence**: with no children, `init` returns `()`, `project`
  returns the candidate unchanged (identity projection), and `update`
  returns `()`. This is a well-defined no-op region.
- **Single region**: `Sequential([R])` behaves identically to `R`, aside
  from the extra one-element tuple wrapping its state.

## Complexity

Let `k` be the number of regions. Both `project` and `update` perform `k`
child operations, so the per-call overhead is `O(k)` on top of the cost of
the individual child projections/updates. Memory for the composite state
is the sum of the children's state sizes.

## Example

```python
composite = Sequential([box_region, ball_region])
state = composite.init(params)

# Reduce a candidate onto the box, then onto the ball.
x = composite.project(params, x0, state)

# Advance every child's state with the latest iteration info.
state = composite.update(state, info)
```
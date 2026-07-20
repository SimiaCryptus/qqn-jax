# BoxRegion Algorithm

## Overview

`BoxRegion` implements a simple **box-constraint projection** for
optimization trust/feasible regions. It enforces elementwise lower and
upper bounds on candidate parameter updates, ensuring that every proposed
point satisfies:

```
lo ≤ x_new ≤ hi
```

This is one of the simplest feasible-region strategies: it constrains the
parameter space to an axis-aligned hyperrectangle (a "box").

## API

```python
def BoxRegion(lo=None, hi=None) -> Region
```

### Parameters

- **`lo`** — the elementwise lower bound. May be:
  - a scalar (broadcast to all parameters),
  - a pytree broadcastable to the parameter structure,
  - or `None`, in which case it is mapped to `-inf` (no lower bound).
- **`hi`** — the elementwise upper bound, following the same rules as `lo`,
  with `None` mapped to `+inf` (no upper bound).

### Returns

A `Region` instance composed of three functions:

- `init`  — reuses `_identity_init` (the region carries no persistent state).
- `project` — clips candidate values to `[lo, hi]`.
- `update` — reuses `_identity_update` (no state to update).

## Algorithm

The core logic lives in the `project` function:

```python
def project(params, candidate, state):
    return jax.tree_util.tree_map(
        lambda c: jnp.clip(c, lo_val, hi_val), candidate
    )
```

### Steps

1. **Bound resolution.** When `BoxRegion` is constructed, `None` bounds are
   replaced with `±jnp.inf`:
   - `lo_val = -jnp.inf if lo is None else lo`
   - `hi_val =  jnp.inf if hi is None else hi`

2. **Projection.** During each optimization step, the candidate update is
   passed through `project`. Using `jax.tree_util.tree_map`, every leaf of
   the candidate pytree is independently clipped with `jnp.clip`, so that
   each element is constrained to `[lo_val, hi_val]`.

3. **Statelessness.** Because a box constraint depends only on the fixed
   bounds (not on optimization history), the `init` and `update` steps are
   delegated to the identity implementations `_identity_init` and
   `_identity_update`. The region therefore holds no mutable state.

## Discussion

### Why projection?

Box projection is the Euclidean projection onto the feasible box: for each
coordinate, the closest feasible value to an infeasible candidate is simply
the nearest bound. `jnp.clip` performs exactly this operation, making the
projection both correct and cheap (`O(n)` in the number of parameters).

### Broadcasting and pytrees

By using `tree_map`, `BoxRegion` transparently supports arbitrarily nested
parameter structures (e.g., neural-network parameter trees). The bounds
`lo`/`hi` are broadcast against each leaf, so scalars apply uniformly while
matching pytrees allow per-parameter bounds.

### Edge cases

- **Unbounded on one side.** Passing `None` for `lo` or `hi` disables that
  bound via `±inf`, which `jnp.clip` handles naturally.
- **Both `None`.** The region degenerates to an identity/no-op projection
  (clipping to `[-inf, +inf]`), functionally equivalent to `IdentityRegion`.
- **Infeasible bounds.** If `lo > hi`, `jnp.clip` follows NumPy/JAX
  semantics and effectively pins values to `hi`; callers are responsible for
  supplying consistent bounds.

### Relationship to other regions

`BoxRegion` shares the `init`/`update` hooks with `IdentityRegion`, differing
only in the `project` step. This composition-by-reuse pattern keeps stateless
regions minimal and consistent across the `qqn_jax.regions` package.

## Example

```python
from qqn_jax.regions.box import BoxRegion

# Constrain all parameters to the unit interval [0, 1].
region = BoxRegion(lo=0.0, hi=1.0)

state = region.init(params)
safe_candidate = region.project(params, candidate, state)
```
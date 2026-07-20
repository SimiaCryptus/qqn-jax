# Orthant Region

## Overview

The `OrthantRegion` implements a **pure geometric projection** that
constrains an optimization step so that no coordinate crosses zero. Each
coordinate is kept within the *orthant* (the sign region) defined by the
current point. This is useful for optimization problems where sign changes
of variables are meaningless, forbidden, or where a sparsity-inducing
boundary at zero is desired.

Unlike many region strategies, `OrthantRegion` does **not** modify the
fitness/objective and does **not** add any penalty term. Despite the
presence of an `l1` parameter in the signature, the current implementation
performs no L1 regularization — it is a purely projective operation.

## Mathematical Description

Given a coordinate with current value `x` and a proposed candidate value
`c`, the projection produces a clamped value `c'`:

- If `x > 0`: `c' = max(c, 0)` — the coordinate may not become negative.
- If `x < 0`: `c' = min(c, 0)` — the coordinate may not become positive.
- If `x == 0`: `c' = 0` — a coordinate at the wall stays at zero.

In all cases, `sign(c')` equals `sign(x)` (with zero acting as a wall),
ensuring the projected candidate remains within the current orthant.

## API

```python
def OrthantRegion(l1: float = 0.0) -> Region
```

### Parameters

- `l1` (`float`, default `0.0`): Present for interface compatibility with
  other region strategies. **Currently unused** — no L1 term is applied.

### Returns

A `Region` named tuple with three members:

- `init`: `_identity_init` — the region maintains no internal state.
- `project`: the orthant projection function described above.
- `update`: `_identity_update` — no state is updated between iterations.

## Implementation Details

The projection is applied leaf-wise across the parameter pytree using
`jax.tree_util.tree_map`, matching each parameter leaf `x` against the
corresponding candidate leaf `c`. The clamping is expressed with
`jnp.where` so the operation is fully vectorized and differentiable /
JIT-compatible.

```python
def proj_leaf(x, c):
    zero = jnp.zeros((), dtype=c.dtype)
    c = jnp.where(x > 0.0, jnp.maximum(c, zero), c)
    c = jnp.where(x < 0.0, jnp.minimum(c, zero), c)
    c = jnp.where(x == 0.0, zero, c)
    return c
```

The `zero` constant is created with the candidate's dtype to avoid dtype
promotion issues.

### State Management

Because the projection depends only on the current point and the candidate,
the region is stateless. It reuses the shared identity helpers
(`_identity_init`, `_identity_update`) so that its `init` and `update`
behaviors integrate seamlessly with the generic region strategy protocol.

## Discussion

### Behavior at the boundary

A key characteristic of this region is that zero acts as an absorbing wall:
once a coordinate reaches exactly zero, the `x == 0` branch permanently
pins it there. This can induce sparsity — coordinates that are driven to
the boundary will remain fixed at zero in subsequent iterations unless the
surrounding algorithm explicitly reinitializes them.

### Sign preservation

The region guarantees the sign of every coordinate is preserved across a
step. This can be valuable when variables represent physically or
semantically constrained quantities (e.g., strictly non-negative rates,
probabilities before normalization, or masses).

### The unused `l1` parameter

The docstring explicitly notes that there is no `l1` term. The parameter is
retained to keep the constructor signature uniform with related region
factories. If future work adds an L1 penalty or a soft-thresholding step,
this parameter provides a natural hook. Until then, callers should treat it
as a no-op.

## Example

```python
from qqn_jax.regions.orthant import OrthantRegion

region = OrthantRegion()
state = region.init(params)

# candidate produced by an optimizer step
projected = region.project(params, candidate, state)
```

## Limitations & Future Work

- No L1 regularization is currently implemented despite the parameter.
- Coordinates pinned at zero cannot escape the boundary within the region
  itself.
- The projection is coordinate-wise and does not account for correlations
  between variables.
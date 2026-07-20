# Momentum (Heavy-Ball) Oracle

## Overview

`MomentumOracle` is a first-order **accelerated** oracle implementing the
classic *heavy-ball* momentum method. It augments plain steepest descent by
biasing the search direction toward the trajectory the optimizer has
actually been travelling. The key design decision — and what distinguishes
this implementation from naive gradient-averaging momentum — is that the
momentum accumulator tracks the **realized per-iteration steps**
`Δx = x_new − x`, not the raw gradients.

## Mathematical Formulation

The oracle maintains a single piece of state: a velocity vector `v`
(matching the structure of the parameters), initialized to zero.

Two quantities are computed:

### Direction (queried at the current iterate)

```
direction = -∇f + β · v
```

This is the descent direction returned to the solver. It blends the current
steepest-descent move `-∇f` with the accumulated momentum `β · v`.

### Velocity update (committed after an accepted step)

```
v_new = β · v + (1 − β) · Δx        where   Δx = x_new − x
```

The velocity is an exponential moving average of the *actual* deltas that
the solver has realized. `β ∈ [0, 1)` (default `0.9`) controls how much
history is retained.

### First-step behavior

On the very first iteration `v = 0`, so:

```
direction = -∇f
```

This reduces exactly to plain steepest descent, which preserves the
`d'(0)` directional-derivative anchor expected by downstream line-search /
quadratic-quotient machinery.

## Why Realized Deltas Rather Than Gradients?

A common momentum formulation accumulates gradients:
`v = β·v + (1−β)·∇f`. This implementation instead accumulates the realized
parameter changes `Δx = x_new − x`. The consequences:

- The momentum term `β · v` nudges the `t = 1` endpoint along the direction
  the optimizer has genuinely been moving, giving **true heavy-ball
  momentum** rather than an average of raw gradient signals.
- Step-size effects (from line search or trust regions) are naturally
  folded into the accumulated velocity, since `Δx` already incorporates the
  accepted step length.

## Implementation Details

The oracle conforms to the `Oracle` interface with three functions:

### `init(params)`

Creates a `MomentumState` with `velocity` set to a zero pytree matching the
parameter structure via `jax.tree_util.tree_map(jnp.zeros_like, params)`.

### `direction(params, grad, state)`

Returns `-g + β·v` computed elementwise over the gradient and velocity
pytrees. The state is returned unchanged (direction is a pure query).

### `update(state, info)`

Advances the velocity. Two code paths handle the point-history abstraction:

1. **No published points** (`publish(info)` returns `None`): a single delta
   `Δx = new_params − params` is computed directly from `info`, and the EMA
   update is applied once.

2. **Published points available**: the `secant_view` provides a sequence of
   `deltas` together with a `valid_seq` mask. A `jax.lax.scan` folds the EMA
   update over the sequence, applying it only where `valid` is true
   (`jnp.where(valid, v_new, v)`), so masked/invalid entries leave the
   velocity untouched.

Both paths return a fresh `MomentumState`.

## State

```python
class MomentumState(NamedTuple):
    velocity: jnp.ndarray   # EMA of realized steps Δx, matching params structure
```

## Parameters

| Parameter | Default | Description                                            |
|-----------|---------|--------------------------------------------------------|
| `beta`    | `0.9`   | Momentum decay factor in `[0, 1)`. Higher = more inertia. |

## Discussion

- **Acceleration**: Heavy-ball momentum can substantially accelerate
  convergence on ill-conditioned problems by damping oscillation across
  steep directions and reinforcing progress along shallow ones.
- **Stability**: Because the endpoint reduces to steepest descent when
  `v = 0` and the direction always contains the `-∇f` component, the method
  degrades gracefully and preserves the descent anchor at `t = 0`.
- **JAX compatibility**: All operations are pytree- and `jit`-friendly; the
  `lax.scan` path handles variable-length published histories in a
  trace-static manner using the validity mask.
- **Tuning**: Larger `β` retains more history (more inertia, slower
  response to gradient changes); smaller `β` behaves closer to plain
  gradient descent.
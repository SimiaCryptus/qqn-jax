# Path-History Momentum Oracle

## Overview

The **Path-History Momentum Oracle** (`PathHistoryMomentumOracle`) is a search-direction
oracle that constructs a momentum term from the *actual accepted optimization
trajectory*. Rather than folding realized parameter deltas into a single scalar
exponential moving average (EMA), it retains an explicit ring buffer of the last
`history_size` accepted parameter deltas and reconstructs the momentum by geometrically
re-weighting the *entire* stored path on every direction query.

## Motivation

A classical momentum oracle maintains one decaying vector:

```
    v ← β · v + Δx
```

This destructively compresses all past steps into a single accumulator. Once a step is
absorbed, its individual geometry can no longer be inspected or re-weighted.

The path-history variant keeps the raw trajectory instead. This allows the momentum to be
recomputed *from scratch* each time from genuine accepted steps, so:

- Recent steps dominate (largest geometric weight).
- Older steps decay smoothly but still contribute exactly their geometric weight.
- The real path geometry — not a lossy scalar summary — drives the momentum.

## Algorithm

### State

The oracle state is a `PathHistoryMomentumState` `NamedTuple`:

| Field           | Shape                  | Description                                              |
| --------------- | ---------------------- | -------------------------------------------------------- |
| `delta_history` | `(history_size, n)`    | Buffer of accepted deltas Δx = x_new − x, newest first.  |
| `step_count`    | scalar `int32`         | Number of valid entries currently stored (≤ history).    |

The buffer is **most-recent-first**: index `0` is the newest accepted delta,
index `m-1` the oldest.

### Initialization (`init`)

Given initial `params` of dimension `n`:

```
    delta_history = zeros((history_size, n))
    step_count    = 0
```

### Direction (`direction`)

Given the current `grad` (∇f) and state, the momentum vector is reconstructed by
geometric re-weighting of the stored path:

```
    weights_k = β^k          for k = 0 .. m-1   (k = 0 newest)
    weights_k = 0            where k ≥ step_count   (inactive slots)
    v         = Σ_k weights_k · Δx_k
    d         = -grad + v
```

- The `active` mask (`arange(m) < step_count`) zeroes the weights of buffer slots that
  have not yet been filled with a genuine accepted delta.
- On the very first step the buffer is empty (`step_count = 0`), so `v = 0` and the
  direction reduces to plain steepest descent `d = -grad`. This preserves the `d'(0)`
  descent anchor.

The direction function returns `(d, state)` unchanged — it is a pure read of the state.

### Update (`update`)

Updates fold newly accepted iterations into the buffer. Two paths are supported,
depending on whether the oracle info carries a batch of published points.

**Single-step path** (`publish(info)` returns `None`):

```
    delta   = new_params - params
    shifted = concat([delta[None], delta_history[:-1]])   # push front, drop oldest
    count   = min(step_count + 1, history_size)
```

**Batched path** (`publish(info)` returns points):

The `secant_view` provides a sequence of `deltas` and a `valid_seq` mask. A `lax.scan`
replays them in order, pushing each *valid* delta onto the front of the buffer:

```
    for (dx, valid) in (deltas, valid_seq):
        pushed   = concat([dx[None], hist[:-1]])
        hist     = where(valid, pushed, hist)
        count    = where(valid, min(count + 1, history_size), count)
```

Invalid entries leave the buffer and count untouched, so only genuinely accepted steps
enter the trajectory.

## Parameters

| Parameter      | Type    | Default | Description                                                  |
| -------------- | ------- | ------- | ------------------------------------------------------------ |
| `history_size` | `int`   | `10`    | Number of accepted deltas retained in the buffer.            |
| `beta`         | `float` | `0.9`   | Geometric decay base; higher values give longer memory.      |

## Properties and Discussion

- **Faithful trajectory weighting.** Because momentum is recomputed from the raw stored
  deltas each query, the geometric weight `β^k` is applied exactly and consistently. There
  is no drift from repeated in-place multiplication of a scalar accumulator.

- **Steepest-descent anchor.** With an empty buffer the direction is exactly `-grad`,
  guaranteeing a valid descent direction on the first step regardless of `beta`.

- **JAX-friendly, fixed-shape.** The ring buffer has a static shape and updates use
  masking (`jnp.where`) and `lax.scan` rather than data-dependent control flow, so the
  oracle is fully `jit`/`vmap` compatible.

- **Memory vs. EMA trade-off.** The explicit buffer costs `O(history_size · n)` memory
  versus `O(n)` for a scalar EMA, but exposes the full recent geometry. Effective memory
  length is governed jointly by `history_size` (hard cutoff) and `beta` (soft decay). A
  too-small `history_size` truncates the geometric tail; a very large one wastes memory on
  steps whose `β^k` weight is negligible.

- **Robustness to rejected steps.** Only accepted (valid) deltas populate the buffer, so
  the momentum reflects the true optimization path rather than trial line-search points.

## Complexity

| Operation   | Time                        | Memory                    |
| ----------- | --------------------------- | ------------------------- |
| `direction` | `O(history_size · n)`       | `O(n)` scratch            |
| `update`    | `O(history_size · n)`       | `O(history_size · n)`     |

## Relationship to Other Oracles

Compared with `MomentumOracle` (single decaying EMA), this oracle preserves the same
conceptual `direction = -∇f + v` form but replaces the lossy scalar accumulator with a
lossless, explicitly re-weighted path history.
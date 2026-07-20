# ADAM Oracle

## Overview

The **ADAM oracle** (`AdamOracle`) implements the *adaptive moment
estimation* algorithm as a search-direction *oracle*. Rather than being a
stand-alone optimizer loop, it exposes the three-function `Oracle`
interface (`init`, `direction`, `update`) so that the ADAM update can be
embedded inside a broader optimization framework (e.g. one driven by a line
search along a quadratic path).

ADAM combines two ideas:

1. **Momentum** — an exponentially decaying running mean of the gradient
   (the *first moment* `m`).
2. **Adaptive per-parameter scaling** — an exponentially decaying running
   mean of the squared gradient (the *second moment* `v`), used to
   normalize each coordinate of the step.

Both moving averages are *bias-corrected* to counteract their
zero-initialization.

## Mathematical Formulation

Given a gradient `∇f` at iteration `t`, the update proceeds as:

```
m ← β1·m + (1 − β1)·∇f            (first moment / momentum)
v ← β2·v + (1 − β2)·∇f²           (second moment / energy)
m̂ ← m / (1 − β1^t)               (bias-corrected first moment)
v̂ ← v / (1 − β2^t)               (bias-corrected second moment)
direction ← − learning_rate · m̂ / (√v̂ + ε)
```

The returned `direction` is the classical ADAM step, **scaled by
`learning_rate`**. This scaling is deliberate: it makes the `t = 1`
endpoint of the oracle a *genuine* ADAM step. Consequently, taking a fixed
unit step along the resulting quadratic path (`line_search="fixed"`)
reproduces plain ADAM's per-iteration behavior, instead of an unscaled —
and typically catastrophically large — update.

## Hyperparameters

| Parameter        | Default | Meaning                                              |
|------------------|---------|------------------------------------------------------|
| `beta1`          | `0.9`   | Decay rate for the first moment (momentum).          |
| `beta2`          | `0.999` | Decay rate for the second moment (energy).           |
| `epsilon`        | `1e-8`  | Numerical-stability term added to `√v̂`.             |
| `learning_rate`  | `1e-3`  | Overall step scaling applied to the ADAM direction.  |

## State

The oracle carries an `AdamState` `NamedTuple`:

| Field  | Type          | Description                                        |
|--------|---------------|----------------------------------------------------|
| `m`    | `jnp.ndarray` | First-moment (momentum) estimate of the gradient.  |
| `v`    | `jnp.ndarray` | Second-moment (energy) estimate of the gradient.   |
| `step` | `jnp.ndarray` | Iteration counter (int32) used for bias correction.|

`init` allocates zero-filled `m` and `v` matching the shape of `params`
(via `jax.tree_util.tree_map`), and sets `step = 0`.

## Interface Functions

### `init(params) -> AdamState`

Creates the zero-initialized state. Because `m` and `v` start at zero, the
bias-correction terms `(1 − β^t)` are essential for the early iterations to
avoid a strongly biased (near-zero) direction.

### `direction(params, grad, state) -> (direction, state)`

Computes the ADAM step **without mutating the persisted moments**. It uses
a *provisional* `t = state.step + 1` together with locally folded `m` and
`v` to build the direction. The state is returned unchanged; committing the
moments is deferred to `update`. This separation lets a line search probe
the direction repeatedly without corrupting the accumulated moments.

### `update(state, info) -> AdamState`

Commits the moment integration once a step outcome is known. It supports
two modes depending on whether a *point history* is published:

- **No history** (`publish(info) is None`): a single fold using
  `info.grad` is applied, and `step` is incremented by one.

- **With history**: a sequence of gradients (`grad_seq`) and a validity
  mask (`valid_seq`) is scanned with `jax.lax.scan`. Each *valid* gradient
  folds into `m` and `v`; invalid entries leave the moments unchanged
  (`jnp.where(valid, new, old)`). This allows batching several candidate
  gradients while only accepting the ones marked valid. `step` is
  incremented by one for the accepted step.

The internal `fold` helper implements a single moment update:

```
m ← β1·m + (1 − β1)·g
v ← β2·v + (1 − β2)·g²
```

## Design Notes & Discussion

- **Anchor preservation (`d'(0)`).** Because the moments are integrated in
  `update` (only after a step is accepted) rather than in `direction`, the
  very first step — before any accepted gradient — reduces to plain
  (scaled) steepest descent. This preserves the `d'(0)` anchor expected by
  the surrounding quadratic-path machinery.

- **Bias correction.** Since `m` and `v` are zero-initialized, their raw
  values underestimate the true moments early on. Dividing by
  `(1 − β1^t)` and `(1 − β2^t)` corrects this. As `t → ∞`, both terms
  approach `1` and the correction fades.

- **Numerical stability.** `epsilon` prevents division by zero and damps
  updates in flat regions where `v̂` is tiny.

- **Separation of concerns.** Keeping `direction` side-effect-free is what
  makes the oracle safe to compose with line searches: candidate steps can
  be evaluated any number of times, and only the final accepted step feeds
  the moment accumulators through `update`.

- **`learning_rate` semantics.** Unlike a raw normalized ADAM direction,
  this oracle folds `learning_rate` into the direction itself, so a unit
  step is a full ADAM step. This is critical for `line_search="fixed"`
  correctness.
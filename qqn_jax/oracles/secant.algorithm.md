# Secant Oracle (Barzilai–Borwein Curvature)

## Overview

The **Secant Oracle** is a matrix-free, `O(n)`-memory curvature oracle
that produces a search direction by scaling the gradient with an
inverse-curvature estimate. That estimate is inferred from the
*realized secant* of the previous optimization step — the pair of
changes in position and gradient that the optimizer already measured.

Unlike quasi-Newton methods that maintain an explicit (or implicitly
represented) approximation of the Hessian, the Secant Oracle reuses a
single step's curvature signal. This makes it a lightweight companion
for a `Fallback` strategy and a useful probe of how much curvature is
captured by one step.

## Mathematical Background

Given the current iterate `x` with gradient `∇f`, and the previous
iterate `x_prev` with gradient `∇f_prev`, define the secant pair:

```
    s = x    - x_prev        (change in parameters)
    y = ∇f   - ∇f_prev       (change in gradient)
```

The **BB1** (Barzilai–Borwein) step length is the reciprocal Rayleigh
quotient:

```
    α = ⟨s, s⟩ / ⟨s, y⟩
```

which approximates the inverse curvature along the most recent step.
The resulting descent direction is:

```
    direction = -α · ∇f
```

### Endpoint Semantics

- **`t = 1` endpoint**: the gradient scaled by the inverse-curvature
  estimate `α`.
- **First step (no secant yet)**: falls back to `-alpha0 · ∇f`, i.e.
  plain scaled steepest descent. This preserves the `d'(0)` anchor.

## Algorithm

### State

The oracle carries a `SecantState`:

| Field         | Type        | Description                              |
|---------------|-------------|------------------------------------------|
| `prev_params` | `ndarray`   | Parameters from the previous step        |
| `prev_grad`   | `ndarray`   | Gradient from the previous step          |
| `alpha`       | `ndarray`   | Current inverse-curvature estimate       |
| `step_count`  | `int32`     | Number of completed updates              |

### Steps

1. **`init(params)`** — initialize `alpha = alpha0`, `step_count = 0`,
   and seed `prev_params`/`prev_grad` (gradient seeded with zeros).

2. **`direction(params, grad, state)`** — return the scaled steepest
   descent direction `d = -alpha · grad`. The state is returned
   unchanged.

3. **`update(state, info)`** — compute the new curvature estimate:
   - Obtain the secant pair `(s, y)`. Prefer the point-history store
     via `publish(info)` / `secant_view(...).newest_secant()`; if the
     store is unavailable, fall back to raw differences
     `s = new_params - params`, `y = new_grad - grad`.
   - Compute `ss = ⟨s, s⟩` and `sy = ⟨s, y⟩`.
   - **Curvature guard**: accept the BB step only when `sy > eps`
     (positive curvature). Otherwise retain the previous `alpha`.
   - **Clipping**: clamp the accepted step to `[eps, alpha_max]` to
     avoid degenerate or unbounded step sizes.

## Parameters

| Parameter    | Default | Description                                        |
|--------------|---------|----------------------------------------------------|
| `alpha0`     | `1.0`   | Initial inverse-curvature estimate (first step).   |
| `alpha_max`  | `1e3`   | Upper clip on the BB step length.                  |
| `eps`        | `1e-12` | Numerical floor / curvature positivity threshold.  |

## Numerical Safeguards

- **Positive curvature check** (`sy > eps`): a non-positive `sy`
  indicates non-convex or noisy curvature; the previous `alpha` is
  kept rather than producing a bad (possibly negative) step.
- **Safe division**: the denominator uses `jnp.where(curvature_ok, sy,
  1.0)` so the division never divides by ~0 even in the rejected
  branch.
- **Clipping**: `jnp.clip(bb, eps, alpha_max)` bounds the step size.

## Point-History Integration

The oracle integrates with the point-history store:

- `publish(info)` extracts the recorded points for the current step.
- `secant_view(points).newest_secant()` returns the most recent
  `(s, y)` secant pair.

A backward-compatible shim, `_ordered_probe_secants(info, max_replay)`,
is retained for oracles migrating to the store view. It returns
`(params_seq, grad_seq, valid_seq)` ordered oldest-first (increasing
`α`), terminating at the accepted point, or `None` when unavailable.

> **Deprecated**: prefer `publish` / `secant_view` directly.

## Complexity

- **Memory**: `O(n)` — only the previous params/grad and scalar state.
- **Compute per step**: two inner products (`⟨s, s⟩`, `⟨s, y⟩`) plus a
  scaling of the gradient.

## Usage

```python
from qqn_jax.oracles.secant import SecantOracle

oracle = SecantOracle(alpha0=1.0, alpha_max=1e3)
state = oracle.init(params)
direction, state = oracle.direction(params, grad, state)
# ... after taking a step and evaluating new_params/new_grad ...
state = oracle.update(state, info)
```

## Discussion

### Strengths

- **Extremely cheap**: no Hessian, no history buffers, `O(n)` memory.
- **Adaptive**: automatically adjusts scale using measured curvature.
- **Robust fallback**: degrades gracefully to scaled steepest descent
  on the first step or under non-positive curvature.

### Limitations

- Uses only a **single** secant pair, so it captures far less curvature
  information than multi-memory methods (e.g. L-BFGS).
- Sensitive to noisy gradients; the curvature guard mitigates but does
  not eliminate this.
- The BB step is a scalar — it cannot represent anisotropic curvature.

### When to Use

Use the Secant Oracle as a low-cost curvature probe or as a component
within a `Fallback` line-search strategy, particularly when memory is
constrained or when a lightweight adaptive step size is preferable to
full quasi-Newton machinery.
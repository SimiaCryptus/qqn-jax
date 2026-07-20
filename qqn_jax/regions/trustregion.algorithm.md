# Trust-Region Step Constraint

## Overview

The `TrustRegion` region enforces a bound on the length of an optimization
step: given the current parameters `x` and a candidate `x_new`, it guarantees
that

    ‖x_new − x‖₂ ≤ Δ

where `Δ` (the *trust-region radius*) is a scalar state variable. When a
proposed step exceeds the radius, it is **radially clipped** back onto the
boundary of the trust region rather than rejected outright. Optionally, the
radius adapts over iterations based on how well a local model predicts the
observed change in the objective.

This component implements the `Region` protocol (`init`, `project`,
`update`) and integrates with a JAX-based optimizer via `qqn_jax`.

## State

```python
class TrustRegionState(NamedTuple):
    radius: jnp.ndarray
```

The only mutable state is the scalar `radius` (`Δ`), stored with the same
dtype as the parameters.

## Parameters

| Name         | Default | Meaning                                                        |
|--------------|---------|----------------------------------------------------------------|
| `radius`     | `1.0`   | Initial trust-region radius `Δ₀`.                              |
| `radius_max` | `1e3`   | Upper bound on the radius when expanding.                      |
| `adaptive`   | `True`  | If `False`, the radius is held fixed and `update` is a no-op.  |
| `shrink`     | `0.5`   | Multiplicative factor applied to shrink the radius.            |
| `expand`     | `2.0`   | Multiplicative factor applied to expand the radius.            |
| `eta_lo`     | `0.1`   | Lower acceptance threshold for the reduction ratio `ρ`.        |
| `eta_hi`     | `0.75`  | Upper threshold above which expansion is considered.           |

A small constant `eps = 1e-12` guards against division by zero.

## Algorithm

### Initialization (`init`)

Reads the dtype of the first parameter leaf and returns a
`TrustRegionState` whose `radius` is `radius` cast to that dtype.

### Projection (`project`)

Given `params`, a `candidate`, and the current `state`:

1. Compute the step `s = candidate − params` (tree-structured subtraction).
2. Compute its Euclidean norm `n = ‖s‖₂`.
3. Compute a clipping scale

       scale = min(1, Δ / (n + eps))

4. Return `params + scale · s`.

If the step already lies within the radius (`n ≤ Δ`), `scale = 1` and the
candidate is returned unchanged. Otherwise the step is scaled so its length
equals `Δ`, i.e. the candidate is projected onto the trust-region boundary
along the same direction.

### Radius Update (`update`)

When `adaptive=False`, `update` returns the state unchanged.

Otherwise, given the previous `state` and an `info` object carrying:

- `pred_reduction` — the reduction in objective predicted by the local model,
- `actual_reduction` — the observed reduction in objective,
- `params`, `new_params` — parameters before and after the accepted step,

the update proceeds:

1. Compute the **reduction ratio**

       ρ = ared / (pred + eps)

2. Compute the actual step length `n = ‖new_params − params‖₂` and determine
   whether the step landed on the boundary:

       at_boundary = n ≥ Δ − 1e-6

3. Adjust the radius:

   - If `ρ < eta_lo` (poor agreement): **shrink**, `Δ ← shrink · Δ`.
   - Else if `ρ > eta_hi` **and** `at_boundary` (excellent agreement while
     constrained): **expand**, `Δ ← min(expand · Δ, radius_max)`.
   - Otherwise (acceptable band `[eta_lo, eta_hi]` or good ρ but not at the
     boundary): **hold**, `Δ` unchanged.

4. Enforce a floor: if progress was made (`ared > 0`), the radius is never
   shrunk below the current step length `n`; otherwise the floor is `eps`.

       Δ ← max(Δ, floor)

## Design Rationale — Chord vs. Arc

A conventional trust-region method shrinks aggressively (factor `0.25`) on
the common threshold `ρ < 0.25`. The docstring records an important subtlety
("Andromeda gradient-clusters"): on a *curved* optimization path the radius
constrains the **chord length** (straight-line distance) while the predicted
reduction is integrated along the **arc length**. These are different
coordinates, so a naive `ρ < 0.25` shrink rule over-reacts to the mismatch
and can stall progress.

This implementation therefore:

- **(a)** only shrinks on a genuinely poor ratio, `ρ < eta_lo` (`0.1`),
- **(b)** shrinks *gently* using `shrink = 0.5` rather than `0.25`, and
- **(c)** holds the radius across the wide acceptable band `[eta_lo, eta_hi]`
  so the adaptive feedback does not react to the chord/arc discrepancy.

## Discussion

- **Radial clipping vs. rejection.** By projecting onto the boundary,
  `project` always returns a usable candidate. This keeps the optimizer
  moving even when the raw step is too long, and it makes the component
  composable with other regions/projections.

- **Boundary-gated expansion.** Expansion only happens when the step is at
  the boundary (`at_boundary`). If the step was short (interior), the radius
  is not the binding constraint, so growing it would be pointless.

- **Progress-aware floor.** The floor rule prevents the radius from
  collapsing below a step that actually decreased the objective, protecting
  against pathological over-shrinking while still allowing shrinkage down to
  `eps` when no progress was made.

- **JAX compatibility.** All branching uses `jnp.where` / `jnp.logical_and`
  rather than Python control flow (except the static `adaptive` flag), making
  `update` and `project` traceable and `jit`-friendly.

## Example

```python
from qqn_jax.regions.trustregion import TrustRegion

region = TrustRegion(radius=1.0, adaptive=True)
state = region.init(params)

# Clip a proposed candidate to the trust region:
projected = region.project(params, candidate, state)

# After evaluating the accepted step, adapt the radius:
state = region.update(state, info)
```
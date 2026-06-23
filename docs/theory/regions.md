# Projective Regions Support

## Overview

QQN searches along a continuous quadratic path

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

A **projective region** is a strategy that *remaps* a proposed parameter
update onto a feasible (or otherwise preferred) set before it is applied.
Because QQN already exposes a single continuous path, regions integrate
cleanly: rather than searching the raw path, the line search navigates the
**projected path**

```
d_R(t) = project_R(x, d(t)) - x
```

where `project_R` maps the candidate point `x + d(t)` onto the region `R`.
This keeps descent/Wolfe guarantees meaningful on the *feasible* path and
preserves QQN's adaptive blend of gradient and oracle directions.

This document specifies the region abstraction and the concrete regions to be
implemented as pure, functional JAX so they compose with `jit`, `vmap`,
`pmap`, and `grad`, consistent with the rest of `qqn-jax`.

---

## Goals

* Add an optional, composable `region` configuration to `qqn(...)` and `QQN(...)`.
* Keep every region a pure function of `(params, candidate)` (and optional
  state), with no host-side control flow.
* Preserve QQN's convergence behavior: when `region = None`, behavior is
  identical to the current implementation.
* Make regions independent of the gradient/oracle/search components so they
  can be combined and substituted freely.

## Non-Goals

* General constrained optimization with Lagrange multipliers.
* Regions that require solving an inner optimization problem per step
  (beyond cheap closed-form projections), except where explicitly noted.

---

## Core Abstraction

A region is described by a small, pure interface. All functions are JAX-traceable
and operate on pytrees of parameters.

```python
class Region(NamedTuple):
    # Optional per-region state (e.g. trust-region radius). Use an empty
    # pytree () when no state is needed.
    init: Callable[[Params], RegionState]

    # Project a candidate point onto the region.
    #   params:    current iterate x (pytree)
    #   candidate: proposed point x + d(t) (pytree)
    #   state:     region state
    # returns the projected point (pytree), same structure as candidate.
    project: Callable[[Params, Params, RegionState], Params]

    # Optional update of region state after a step is accepted/rejected.
    #   Used by adaptive regions (e.g. trust region radius adjustment).
    update: Callable[[RegionState, RegionInfo], RegionState]
```

`RegionState` and `RegionInfo` are region-specific pytrees. `RegionInfo`
carries quantities the outer loop already computes (e.g. predicted vs. actual
reduction, accepted step, `t`, `α`).

### Integration with the QQN path

The line search evaluates the objective at `x + d(t)`. With a region, it
instead evaluates at the **projected candidate**:

```python
def projected_point(region, state, params, t):
    candidate = tree_add(params, path_d(params, t))   # x + d(t)
    return region.project(params, candidate, state)

# effective update returned to optax.apply_updates:
updates = tree_sub(projected_point(...), params)
```

Projection happens *inside* the line search loop so that the `t`/`α` search
operates on the feasible path. The default (no region) uses the identity
projection `project(params, candidate, state) = candidate`.

---

## Regions to Implement

### 1. Orthant Region (sparsity, OWL-QN style)

Encourages sparsity by constraining each step to remain within the orthant
defined by the current point's signs, zeroing out coordinates that would
cross zero.

* **State**: none (`()`), unless an L1 weight is configured.
* **Projection** (per-coordinate):

  ```
  ξ = sign(xᵢ) if xᵢ ≠ 0 else sign(-∇fᵢ)     # chosen orthant
  yᵢ = candidateᵢ if sign(candidateᵢ) == ξ else 0
  ```

* **Config**: `OrthantRegion(l1: float = 0.0)`. When `l1 > 0`, the effective
  gradient used to choose the orthant for zero coordinates is the
  pseudo-gradient `∇f + l1·sign(x)` (OWL-QN convention).
* **Notes**: Pure elementwise; trivially `vmap`/`jit`-able.

### 2. Trust-Region Sphere (maximum step size)

Enforces `‖x_new − x‖ ≤ Δ` by radially clipping the step.

* **State**: `radius: float = Δ₀`.
* **Projection**:

  ```
  step = candidate − x
  n    = ‖step‖₂                      # global L2 over the pytree
  scale = minimum(1.0, radius / (n + eps))
  y     = x + scale · step
  ```

* **Update (adaptive radius)**: Using `RegionInfo` with predicted reduction
  `pred` (from the quadratic model along `d(t)`) and actual reduction `ared`:

  ```
  ρ = ared / (pred + eps)
  radius = where(ρ < 0.25, 0.25 · radius,
            where(ρ > 0.75 & at_boundary, minimum(2 · radius, radius_max),
                  radius))
  ```

  All branches via `jnp.where`/`lax.select`; no Python conditionals.
* **Config**: `TrustRegion(radius=1.0, radius_max=1e3, adaptive=True)`.

### 3. Box / Min-Max Region (valid parameter ranges)

Enforces elementwise bounds `lo ≤ x_new ≤ hi`.

* **State**: none; `lo`/`hi` are static config (scalars or pytrees broadcast
  to the parameter structure).
* **Projection**:

  ```
  y = clip(candidate, lo, hi)
  ```

* **Config**: `BoxRegion(lo=-inf, hi=+inf)`.
* **Notes**: Bounds may be `None` on either side (mapped to ±inf).

### 4. Combinator Regions

Compose multiple regions into one.

* **`Sequential([R1, R2, ...])`**: Apply projections in order,
  `project = Rk ∘ ... ∘ R1`. State is a tuple of child states; `update`
  fans out to children. Intended for stacking independent constraints
  (e.g. box ∩ trust-region).
* **`Intersection([...])`** *(stretch)*: For regions admitting cheap
  alternating projection (Dykstra-style), iterate a fixed, static number of
  passes to approximate projection onto the intersection. Default 1 pass
  reduces to `Sequential`.

Combinators must preserve the pure-function contract and a fixed (static)
number of inner iterations so they remain `jit`-friendly.

---

## Experimental Region (future work)

### No-Decrease Region (multi-objective guard)

Constrains the search direction so it does **not** increase the loss on a
secondary dataset/objective `g`, helping preserve fitness on a main dataset
while fine-tuning on a specialized one.

* **Idea**: Given the secondary gradient `∇g(x)`, project the candidate step
  onto the half-space `{ s : ⟨∇g, s⟩ ≤ 0 }`:

  ```
  step = candidate − x
  c    = ⟨∇g, step⟩
   # Remove only the g-increasing (positive) component of the step.
   # relu(c) gates on c > 0, so descent on g passes through untouched.
   y    = x + step − (relu(c) / (‖∇g‖² + eps)) · ∇g
  ```

  i.e. remove only the component of the step that would increase `g`.
* **Cost**: Requires one extra gradient evaluation of `g` per projection;
  gate behind explicit opt-in config.
* **Status**: experimental; not part of the initial release surface.

---

## Public API

```python
qqn(
    history_size=10,
     line_search="armijo",        # default; see results.md
    region=None,                 # Region | None
)

QQN(
    fun,
    maxiter=100,
    tol=1e-5,
    history_size=10,
     line_search="armijo",        # default; see results.md
    has_aux=False,
    region=None,                 # Region | None
)
```

Convenience constructors:

```python
from qqn_jax.regions import (
    OrthantRegion, TrustRegion, BoxRegion, Sequential,
)

region = Sequential([
    BoxRegion(lo=0.0, hi=1.0),
    TrustRegion(radius=0.5),
])

solver = QQN(fun, region=region)
```

When `region=None`, the optimizer is byte-for-byte equivalent to the current
behavior — numerically equivalent up to floating-point reordering — using the
identity projection with no extra state.

---

## Implementation Plan

1. **`regions.py`**: Define the `Region` NamedTuple, the identity region, and
   `project`/`init`/`update` helpers operating on pytrees
   (`jax.tree_util`).
2. **Wire into the line search** (`line_search.py`): replace the raw path
   evaluation `x + d(t)` with the projected candidate. Keep the projection
   call optional and zero-overhead when the region is the identity.
3. **Thread `RegionState`** through `QQNState`/`solver.py` so adaptive regions
   (trust region) can update their state via `region.update` after each
   accepted step.
4. **Implement concrete regions**: Box, Orthant, Trust-Region, then
   `Sequential`.
5. **Combinators and (optional) Intersection**.
6. **Experimental `NoDecreaseRegion`** behind a feature flag.

### State threading

`QQNState` gains an optional `region_state` field. `opt.init` calls
`region.init(params)`; `opt.update` calls `region.update(...)` with a
`RegionInfo` assembled from line-search results. Default region uses `()` and
a no-op update, so existing state layouts are unaffected when regions are off.

---

## Testing Strategy

* **Projection correctness** (unit, per region):
  * Box: outputs within `[lo, hi]`; idempotence `project ∘ project == project`.
  * Trust: `‖y − x‖ ≤ radius + eps`; no-op when step already inside.
  * Orthant: sign preservation; zeros where sign flips.
* **Identity equivalence**: `region=None` reproduces baseline trajectories
   on Rosenbrock (numerically equivalent up to floating-point reordering).
* **Descent preservation**: with a region, the line search still returns a
  step with `f(y) ≤ f(x)` (or rejects), verified on convex quadratics.
* **Trust-region adaptation**: radius grows/shrinks with `ρ` on a known
  quadratic; converges to unconstrained optimum when `Δ` is large.
* **Combinator**: `Sequential([Box, Trust])` output satisfies both
  constraints.
* **Transform compatibility**: every region passes through `jit`, `vmap`
  (batched starting points), and `grad` (differentiate through `solver.run`).

---

## Open Questions

* Should the predicted reduction for trust-region `ρ` use the QQN quadratic
  path model directly, or a separate local quadratic? (Initial: use the
  along-path directional model.)
* For `Orthant`, how should `t`-search interact with sign-flips that
  discontinuously change `d_R(t)`? (Initial: project pointwise and rely on the
  line search's sufficient-decrease check; revisit if non-smoothness hurts.)
* Intersection projection iteration count: fixed static `k` vs. tolerance —
  fixed `k` chosen for `jit` friendliness.
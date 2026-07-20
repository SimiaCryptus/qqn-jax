# PSD Secant Region

## Overview

The **PSD Secant Region** reshapes the *feasible set* of an optimization
step using curvature information harvested from realized secant pairs. It
is not an oracle (it does not choose a search direction); instead it
constrains a proposed step to lie within an anisotropic ellipsoid whose
shape is inferred from recently observed curvature.

This makes it a drop-in *trust-region-style* projection that behaves
generously along soft (low-curvature) directions and restrictively along
stiff (high-curvature) directions.

## Background: Secant Pairs and Curvature

Quasi-Newton methods measure curvature through **secant pairs** collected
across iterates:

    s_i = x_{i+1} − x_i           (iterate delta)
    y_i = ∇f_{i+1} − ∇f_i         (gradient delta)

When the observed curvature along a step is positive,

    ⟨s_i, y_i⟩ > 0,

the pair `(s_i, y_i)` samples the Hessian along the direction `s_i`. This
is exactly the curvature information whose inverse the BFGS / L-BFGS
updates approximate.

## The PSD Metric

Rather than approximating the inverse Hessian to steer the search, this
region *reuses* the measured curvature geometrically. It maintains a
bounded window of the most recent `window` accepted secant pairs and forms
a low-rank, positive semi-definite metric:

    M = γ·I + Σ_i (y_i y_iᵀ) / ⟨s_i, y_i⟩

- `γ·I` is an isotropic floor that keeps `M` positive definite even when
  the history is empty or curvature-degenerate.
- Each rank-one term `(y_i y_iᵀ) / ⟨s_i, y_i⟩` is the BFGS-flavored
  curvature accumulation. Only pairs with positive curvature
  (`⟨s_i, y_i⟩ > eps`) contribute; the rest are masked out.

The metric is never materialized as a dense matrix. Instead, `M · v` is
computed implicitly from the stored histories:

    M · v = γ·v + Σ_i [ (y_iᵀ v) / ⟨s_i, y_i⟩ ] · y_i

which is efficient (linear in the window size and dimension).

## The Projection

Given the current iterate `x` (`params`) and a proposed `candidate`, the
region forms the step

    step = candidate − x

and confines it to the `M`-ellipsoid of squared radius `radius²`:

    q      = ⟨step, M · step⟩
    scale  = min(1, radius / √(q + reg))
    s_proj = scale · step

The projected step is then added back to `params`. Because `M` is
anisotropic, the step shrinks *hardest* along the stiff directions exposed
by the secants and *gently* along the flat ones.

### Limiting Behaviors

- **Empty history** (`step_count = 0`): `M = γ·I`, so the region reduces to
  an isotropic trust-region-style clip of radius `radius / √γ`.
- **radius → ∞**: `scale = 1`, so the region is the identity and preserves
  the un-regioned optimizer exactly.

## State

The region carries a `PSDSecantState`:

| Field         | Shape     | Meaning                                       |
|---------------|-----------|-----------------------------------------------|
| `s_history`   | `(m, n)`  | window of iterate deltas `s`                  |
| `y_history`   | `(m, n)`  | window of gradient deltas `y`                 |
| `prev_params` | `(n,)`    | previous accepted iterate (flat)              |
| `prev_grad`   | `(n,)`    | previous accepted gradient (flat)             |
| `step_count`  | scalar    | number of valid columns currently stored      |
| `initialized` | scalar    | whether the state has been initialized         |

Here `m = window` (buffer depth) and `n` is the total flattened parameter
dimension.

## Lifecycle

### `init(params)`
Flattens the parameter pytree to obtain `n`, allocates zeroed `s_history`
and `y_history` buffers of shape `(window, n)`, seeds `prev_params` with
the flattened parameters, zeros `prev_grad`, and sets `step_count = 0`.

### `project(params, candidate, state)`
Flattens `params` and `candidate`, computes `step`, applies the implicit
metric to get `M · step`, computes the ellipsoid scale factor, scales the
step, reshapes it back into the original pytree structure, and adds it to
`params`.

### `update(state, info)`
Forms the new secant pair from `info.new_params` and `info.new_grad`:

    s = new_params − prev_params
    y = new_grad   − prev_grad

The pair is accepted only if the state is initialized **and** the
curvature is positive (`⟨s, y⟩ > eps`). On acceptance, the histories are
rolled by one and the new pair inserted at index 0, and `step_count` is
incremented (capped at `window`). Otherwise the histories are left
unchanged. `prev_params` and `prev_grad` are always refreshed.

## Parameters

| Parameter | Default | Description                                             |
|-----------|---------|---------------------------------------------------------|
| `window`  | `10`    | number of secant pairs retained (buffer depth `m`)      |
| `gamma`   | `1.0`   | isotropic floor `γ` keeping `M` positive definite       |
| `radius`  | `1.0`   | ellipsoid radius bounding the `M`-norm of the step      |
| `reg`     | `1e-8`  | small stabilizer added under the square root            |

Two internal constants guard numerical stability:
- `eps = 1e-12`: curvature threshold for accepting/using secant pairs.

## Discussion

### Why reshape the feasible set instead of the search direction?

Traditional quasi-Newton methods invert curvature to accelerate along
flat directions. This region takes the dual, geometric view: it leaves the
optimizer's step direction untouched but *bounds* how far the step may
travel, using the same curvature information. This has several advantages:

- **Composability**: it can wrap any optimizer that produces a candidate
  step, acting purely as a projection.
- **Safety**: high-curvature directions (where a large step is most
  dangerous) are penalized most, providing implicit damping.
- **Graceful degradation**: with no reliable curvature it becomes an
  isotropic trust region; with a huge radius it disappears entirely.

### Numerical considerations

- Only positive-curvature pairs contribute to `M`, matching the
  positive-definiteness requirement of BFGS-style curvature.
- The `reg` term under the square root prevents division by zero when the
  metric evaluates to zero on a step.
- The implicit metric application avoids forming an `n × n` matrix,
  keeping the cost `O(window · n)` per projection.

### Impacts and follow-up

- This region assumes `info` exposes `new_params` and `new_grad`; ensure
  the driving loop supplies gradient deltas so that curvature can be
  measured.
- The flattening/unflattening logic supports arbitrary parameter pytrees.
- Choice of `gamma` and `radius` interact: with an empty history the
  effective isotropic radius is `radius / √γ`.
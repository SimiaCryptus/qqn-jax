# Anchored Multi-Secant Quasi-Newton Oracle (AMS-QN / TK-QN)

## Overview

The **Anchored Multi-Secant Quasi-Newton** oracle is a matrix-free
quasi-Newton direction generator. Instead of building curvature
information from *pairwise-adjacent* secant pairs, it re-anchors the
entire stored history at the **current iterate** and solves a small
weighted least-squares system for the search direction.

It is implemented in `ams_qn.py` as the factory function
`AnchoredMultiSecantOracle`, returning a standard
`qqn_jax.oracles.oracle.Oracle` triple `(init, direction, update)`.

## Motivation

Classical quasi-Newton methods (BFGS, L-BFGS, Anderson) chain curvature
pairs across adjacent iterates:

```
s_{t-1} = x_t  − x_{t-1}
y_{t-1} = ∇f_t − ∇f_{t-1}
```

When the optimization trajectory oscillates (e.g. along a stiff/ill
conditioned direction), adjacent secants tend to cancel, producing a
noisy, near-zero chord that carries little usable curvature.

**Anchored secants** instead measure the *full* displacement from every
stored past point to the current iterate:

```
Δx_i = x_t  − x_i
Δg_i = ∇f_t − ∇f_i      for every stored i in the window
```

This pulls all curvature samples into the tangent space at the current
point — a Euclidean analogue of Riemannian vector transport — turning
oscillation into a longer, cleaner curvature sample along stiff
directions.

## Kernel Weighting

Each anchored pair is weighted by a kernel over the anchored displacement
magnitude `‖Δx_i‖`:

| kernel      | weight formula                          |
|-------------|-----------------------------------------|
| `rational`  | `w_i = 1 / (1 + ‖Δx_i‖ / σ)`            |
| `gaussian`  | `w_i = exp(−‖Δx_i‖² / σ²)`              |

- **Near samples** (redundant, `‖Δx_i‖ → 0`) receive high weight but add
  little new information.
- **Far samples** (stale/unreliable) are down-weighted, so the estimate
  stays anchored to the local tangent space.

Any kernel string other than `rational` or `gaussian` raises
`ValueError`.

## Mathematical Formulation

The multi-secant condition `H Δg_i ≈ Δx_i` (for all valid `i`, weighted
by `w_i`) is expressed as a small `(m × m)` weighted least-squares
problem — **no `(n × n)` curvature matrix is ever formed**:

```
θ = argmin_θ  Σ_i w_i ‖∇f − ΔG θ‖²  (+ reg · ‖θ‖²)
direction = −β · (∇f − ΔG θ) − ΔX θ
```

where `ΔX` and `ΔG` are the `(n × m)` matrices whose columns are the
anchored displacements `Δx_i` and `Δg_i`.

This mirrors the two-loop-free small-system solve used by
`AndersonOracle`, but is built from **anchored** rather than
**first-differenced** pairs.

With an empty window the endpoint reduces to plain steepest descent
(`d = −∇f`), preserving the descent anchor `d'(0) < 0`.

## State

`AnchoredMultiSecantState` is a `NamedTuple` holding sliding windows:

| field         | shape      | description                              |
|---------------|------------|------------------------------------------|
| `g_history`   | `(m, n)`   | window of recent gradients               |
| `x_history`   | `(m, n)`   | window of recent iterates                |
| `step_count`  | scalar int | number of valid columns currently stored |

## API

### `AnchoredMultiSecantOracle(window=10, reg=1e-8, beta=1.0, kernel="rational", sigma=1.0)`

| parameter | default      | meaning                                             |
|-----------|--------------|-----------------------------------------------------|
| `window`  | `10`         | maximum number of stored history pairs `m`          |
| `reg`     | `1e-8`       | Tikhonov regularization scale (relative to trace)   |
| `beta`    | `1.0`        | scaling of the residual (steepest-descent) term     |
| `kernel`  | `"rational"` | anchored-displacement weighting kernel              |
| `sigma`   | `1.0`        | kernel length scale `σ`                             |

Returns an `Oracle(init, direction, update)`.

#### `init(params) -> AnchoredMultiSecantState`

Allocates zeroed `(window, n)` history buffers and a zero step count.

#### `direction(params, grad, state) -> (d, state)`

1. Builds anchored displacement matrices `dX`, `dG` (shape `(n, m)`).
2. Masks columns beyond `step_count` as inactive.
3. Computes kernel weights `w` from `‖Δx_i‖`, zeroed on inactive
   columns; `sqrt_w` is used to form the weighted Gram matrix.
4. Assembles `A = ΔGᵀ W ΔG + reg · scale · I`, where `scale` is the mean
   diagonal (`trace / m`) for scale-invariant regularization.
5. Applies the active mask so inactive rows/columns behave as identity,
   adds a tiny `1e-12·I` jitter, and solves `A θ = b` with
   `b = (ΔG W)ᵀ ∇f`.
6. Forms `residual = ∇f − ΔG θ` and
   `d = −β · residual − ΔX θ`.
7. **Fallback:** if `d` is non-finite or `step_count == 0`, returns
   `−∇f` (steepest descent).

The state is returned unchanged from `direction`.

#### `update(state, info) -> AnchoredMultiSecantState`

Two paths, driven by `publish(info)`:

- **No published points** (`None`): roll the windows and insert the
  single new `(new_params, new_grad)` pair, incrementing `step_count`
  up to `window`.
- **Published point history**: `jax.lax.scan` over
  `(params_seq, grad_seq, valid_seq)`, rolling and inserting each valid
  point while leaving the history untouched on invalid entries.

## Implementation Notes

- **Matrix-free:** cost per direction is dominated by the `(m × m)`
  solve and the `(n × m)` matmuls, never an `(n × n)` operation. This
  keeps the method scalable in the problem dimension `n`.
- **JAX-friendly:** fixed-shape buffers, masking instead of dynamic
  slicing, and `lax.scan` make `direction`/`update` fully
  `jit`/`vmap` compatible.
- **Numerical safety:** trace-relative regularization, active masking to
  identity, and a `1e-12` jitter keep the linear solve well posed even
  with a partially filled or degenerate window.
- **Descent guarantee:** the empty-window / non-finite fallback to
  `−grad` preserves the sufficient-decrease anchor required by line
  searches.

## Discussion

AMS-QN trades the strict recursive structure of L-BFGS for a fresh,
globally re-anchored least-squares fit at every step. This makes it
especially robust on **oscillatory or non-monotone trajectories**, where
first-differenced curvature estimates degrade. The kernel weighting adds
a principled way to discount both redundant near-duplicates and stale
far-away samples, keeping the estimated curvature representative of the
*current* tangent space.

Compared with `AndersonOracle`, which shares the small-system,
matrix-free philosophy, AMS-QN differs primarily in how the columns of
its regression matrices are built (anchored vs. adjacent differences)
and in its explicit kernel reweighting.

### Limitations
- Quality depends on window contents; a poorly conditioned or
  redundant window falls back toward steepest descent.
- Kernel choice and `sigma` require tuning for the problem scale.
- Re-anchoring recomputes displacements each step, so history must store
  full iterates/gradients (`O(window · n)` memory).
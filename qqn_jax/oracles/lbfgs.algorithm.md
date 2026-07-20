# L-BFGS Oracle Algorithm

## Overview

This document describes the limited-memory BFGS (L-BFGS) oracle
implemented in `lbfgs.py`. The oracle produces a quasi-Newton search
direction `-H∇f`, where `H` is an implicit approximation to the inverse
Hessian built from a fixed-size window of the most recent gradient and
parameter differences.

The implementation is fully self-contained and JAX-friendly: all state
lives in fixed-shape arrays and every update uses `lax.scan` / `where`
so the code is safe to `jit` and `vmap`.

## State

The oracle carries an immutable `LBFGSState` (a `NamedTuple`) with the
following fields:

| Field         | Shape                | Meaning                                        |
|---------------|----------------------|------------------------------------------------|
| `s_history`   | `(history_size, n)`  | parameter differences `sₖ = xₖ₊₁ - xₖ`         |
| `y_history`   | `(history_size, n)`  | gradient differences `yₖ = ∇fₖ₊₁ - ∇fₖ`        |
| `rho_history` | `(history_size,)`    | curvature scalars `ρₖ = 1 / (yₖᵀsₖ)`           |
| `step_count`  | scalar `int32`       | number of valid entries currently stored       |
| `gamma`       | scalar               | initial Hessian scaling `H₀ = γ·I`             |
| `prev_params` | `(n,)`               | previous parameters (for computing `s`)        |
| `prev_grad`   | `(n,)`               | previous gradient (for computing `y`)          |

### Buffer ordering

History buffers are stored **most-recent-first** (index `0` is the
newest pair). Unfilled slots are left as zeros; because their `s`, `y`,
and `ρ` are all zero, they contribute nothing to the recursion and no
explicit masking is needed.

## Initialization — `init_lbfgs_state`

Creates an empty state for a given parameter vector:

- all history buffers are zero-filled,
- `step_count = 0`,
- `gamma = 1.0`,
- `prev_params` / `prev_grad` seeded with the supplied values.

Dtype is inherited from `params` to keep the whole state consistent.

## History update — `update_lbfgs_history`

Given new `params` and `grad`, the update computes:

```
s  = params - prev_params
y  = grad   - prev_grad
ys = yᵀs
```

### Curvature safeguard

The pair is only accepted when the curvature condition

```
yᵀs > eps · sqrt(yᵀy · sᵀs + eps)
```

holds (with `eps = 1e-10`). This is a standard L-BFGS safeguard for
non-convex problems: it ensures the maintained inverse-Hessian
approximation stays positive definite. If the condition fails, the
history, `step_count`, and `gamma` are left unchanged (only
`prev_params` / `prev_grad` advance).

### Acceptance mechanics

When accepted:

- `ρ = 1 / (yᵀs)` is inserted (otherwise `ρ = 0`),
- each buffer is shifted right by one (`_shift_insert`) and the new row
  is placed at index `0`,
- `step_count` increments up to `history_size`,
- the initial-Hessian scale is refreshed as `γ = (yᵀs) / (yᵀy)`.

All conditional writes use `jnp.where` / `jnp_select_buf`, so the
operation is branch-free and JIT-compatible.

## Batched history update — `update_lbfgs_history_batch`

Replays a sequence of probes `(params_seq, grad_seq, valid_seq)` of
shape `(k, n)`, `(k, n)`, `(k,)` into the history using `lax.scan`.
Probes are folded **oldest-first** so the most recent accepted point
ends up newest. Each step:

1. computes a candidate update via `update_lbfgs_history`,
2. merges it into the carry only where the per-probe `valid` flag is
   set (`prev_params`/`prev_grad` always advance).

Degenerate or zero-length pairs are additionally rejected by the
intrinsic curvature guard.

## Direction — `lbfgs_direction`

Implements the classic two-loop recursion (Nocedal & Wright,
Algorithm 7.4) to compute `-H∇f`.

### First loop (newest → oldest)

```
q = ∇f
for i in newest..oldest:
    αᵢ = ρᵢ · sᵢᵀq
    q  = q - αᵢ · yᵢ
```

Implemented as a forward `lax.scan` over the most-recent-first buffers,
collecting the `αᵢ` values.

### Initial scaling

```
r = γ · q
```

### Second loop (oldest → newest)

```
for i in oldest..newest:
    βᵢ = ρᵢ · yᵢᵀr
    r  = r + (αᵢ - βᵢ) · sᵢ
```

Implemented as a `reverse=True` `lax.scan` reusing the stored `αᵢ`.

The oracle returns `-r = -H∇f`, the quasi-Newton descent direction.

Because empty slots have `s = y = ρ = 0`, both `αᵢ` and the correction
term vanish for them, so the recursion yields exactly the result of the
`step_count` valid pairs without any explicit masking.

## Design notes

- **Self-contained**: the two-loop recursion runs directly on our own
  buffers rather than Optax/JAXopt private state, keeping the oracle
  swappable and dependency-light.
- **JIT / vmap safe**: fixed shapes plus `lax.scan` and `jnp.where`
  throughout, with no data-dependent control flow.
- **Numerically guarded**: curvature and division safeguards prevent
  NaNs from degenerate updates.

## Complexity

For history size `m` and dimension `n`:

- update: `O(n)` per accepted pair,
- direction: `O(m·n)` per call,
- memory: `O(m·n)`.
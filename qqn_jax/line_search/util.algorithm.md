# QQN Line Search Utilities — Algorithm Documentation

This document describes the helper algorithms implemented in `util.py`.
These utilities support the **line search** component of the
Quadratic-Quasi-Newton (QQN) optimizer. They are written to be
JIT- and `vmap`-compatible (pure functions, fixed-shape buffers,
explicit PRNG key threading).

---

## 1. Overview

The line search in QQN operates over a precomputed quadratic path
direction `d`, selecting a step size `α` that satisfies a sufficient
decrease (Armijo) condition and optionally a curvature (strong Wolfe)
condition.

`util.py` provides supporting primitives:

1. **Metropolis-style stochastic acceptance** — a meta-rule that can
   accept an otherwise-rejected step with a temperature-controlled
   probability.
2. **Probe buffer management** — fixed-size, JIT-safe buffers used to
   record candidate points (`params`, `grads`, values, step sizes)
   evaluated during the search.

---

## 2. Metropolis Acceptance — `_metropolis_accept`

### Purpose
Provides a simulated-annealing-inspired escape mechanism. When a
candidate step increases the objective (`ΔE > 0`), it may still be
accepted with probability `exp(−ΔE / T)`, allowing the optimizer to
escape shallow local minima.

### Algorithm
Given the energy change `ΔE`, temperature `T`, PRNG `key`, and `dtype`:

```
if T <= 0:            # annealing disabled
    return False, key
p  = clip(exp(-ΔE / T), 0, 1)
key, subkey = split(key)
u  = uniform(subkey)
accepted = (T > 0) and (u < p)
return accepted, key
```

### Implementation Notes
- **Guarded division:** `safe_t = max(T, 1e-12)` prevents division by
  zero while remaining differentiable-friendly and branch-free.
- **Branch-free selection:** `use_temp = T > 0` gates acceptance via
  `logical_and`, avoiding Python-level control flow so the function is
  traceable under `jit`/`vmap`.
- **Explicit key threading:** the split key is returned to the caller
  so randomness remains reproducible and functionally pure.
- **Probability clipping:** `p ∈ [0, 1]` handles the `ΔE < 0`
  (improvement) case, where `exp` may exceed 1.

### Behavior Summary
| Condition        | Result                                   |
|------------------|------------------------------------------|
| `T <= 0`         | Never accepts (`accepted = False`)       |
| `ΔE <= 0`        | High acceptance probability (`p → 1`)    |
| `ΔE > 0`, `T` big| Moderate/high acceptance probability     |
| `ΔE > 0`, `T` small | Low acceptance probability            |

---

## 3. Probe Buffers

During a line search, several trial step sizes are evaluated. Because
JIT compilation requires static shapes, probe results are stored in
**preallocated fixed-size buffers** of capacity `max_probes`.

### 3.1 Allocation — `_empty_probes`

Given a flat parameter vector `params` (length `n`) and capacity
`max_probes`, returns a 5-tuple of buffers:

| Buffer         | Shape                | dtype           | Init    |
|----------------|----------------------|-----------------|---------|
| `probe_params` | `(max_probes, n)`    | `params.dtype`  | `0`     |
| `probe_grads`  | `(max_probes, n)`    | `params.dtype`  | `0`     |
| `probe_valid`  | `(max_probes,)`      | `bool`          | `False` |
| `probe_values` | `(max_probes,)`      | `params.dtype`  | `+inf`  |
| `probe_alphas` | `(max_probes,)`      | `params.dtype`  | `0`     |

Initializing values to `+inf` guarantees that unfilled slots are never
mistakenly selected as the "best" candidate during a subsequent
minimization/argmin over `probe_values`.

### 3.2 Recording — `_record_probe`

Writes a single probe `(p, g, v, a)` into a given `slot` of the
buffers, then returns updated buffers.

```
in_range = (slot >= 0) and (slot < max_probes)
idx      = clip(slot, 0, max_probes - 1)
for each buffer B with value x:
    B = where(in_range, B.at[idx].set(x), B)
```

### Implementation Notes
- **Out-of-range safety:** `idx` is always clipped to a valid index so
  the functional `.at[idx].set(...)` update never traces an
  out-of-bounds access. The `in_range` mask then discards the write if
  the original `slot` was invalid, leaving buffers unchanged.
- **Functional updates:** JAX arrays are immutable; `.at[idx].set(...)`
  produces new arrays, and `jnp.where` selects between old and new
  buffers based on `in_range`. This keeps the function pure and safe
  under transformation.
- **Uniform masking:** every buffer (params, grads, valid flag, value,
  alpha) is updated atomically with the same `in_range` guard, keeping
  them mutually consistent.

---

## 4. Design Principles

These utilities embody the constraints of a JIT/`vmap`-first codebase:

- **No data-dependent control flow.** Conditional behavior is expressed
  with `jnp.where`, `logical_and`, `clip`, and `maximum` rather than
  Python `if` statements over traced values.
- **Static shapes.** Buffers are allocated once with a fixed
  `max_probes` capacity; slot writes are index-guarded rather than
  shape-changing.
- **Explicit randomness.** PRNG keys are split and returned so callers
  retain reproducibility.
- **Swappable strategies.** The metropolis meta-rule and probe buffers
  are decoupled from the specific search (Armijo backtracking or the
  Optax strong-Wolfe zoom line search), allowing strategies to remain
  interchangeable behind the QQN line-search interface.

---

## 5. Discussion

The combination of a **deterministic Armijo/Wolfe search** with an
optional **stochastic Metropolis acceptance** gives QQN a tunable
exploration–exploitation trade-off. Setting the temperature `T = 0`
recovers a purely greedy line search; raising `T` permits occasional
uphill steps that can help escape poor basins in non-convex problems.

The probe buffers serve two roles: they provide a diagnostic record of
the search trajectory and enable "best-of-probes" fallback selection
when the primary condition is not met within the iteration budget. The
`+inf` initialization and index-guarded writes ensure this selection is
robust regardless of how many probes were actually recorded.
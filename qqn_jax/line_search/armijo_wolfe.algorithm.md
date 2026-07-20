# Armijo–Wolfe Line Search

## Overview

`armijo_wolfe_search` implements a classical two-phase (bracketing +
zoom) line search on a one-dimensional scalar reduction of a
multidimensional optimization problem. It enforces the **Armijo
sufficient-decrease** condition together with the **strong Wolfe
curvature** condition on the univariate function

```
φ(t)  = f(x + d(t))
φ'(t) = ∇f(x + d(t)) · d'(t)
```

The search direction / path `d(t)` and any region projection are
pre-baked into the `eval_at` callback by the calling solver, so this
search is **fully path-agnostic**: it never inspects the geometry of
the problem directly. This makes it reusable for line, quadratic, or
arbitrarily curved paths.

The implementation is written in JAX using `jax.lax.while_loop`, so it
is fully traceable/JIT-compatible and differentiable-friendly (no
Python-level control flow depends on traced values).

## Reference

Nocedal & Wright, *Numerical Optimization*, Algorithms 3.5 and 3.6
(line search satisfying the strong Wolfe conditions, plus the `zoom`
subroutine).

## Signature

```python
armijo_wolfe_search(
    eval_at, params, value, grad, slope0, *,
    init_step=1.0, c1=1e-4, c2=0.9, max_iter=20,
    temperature=0.0, cooling=0.95, seed=0,
    max_probes=32, record_probes=True, max_step=1.0,
) -> LineSearchResult
```

### Parameters

| Name           | Meaning                                                                 |
|----------------|-------------------------------------------------------------------------|
| `eval_at`      | Callable `t -> (params, value, grad, slope)` evaluating φ and φ' at `t`.|
| `params`       | Current parameters (used only to shape the probe buffers).             |
| `value`        | φ(0), the value at the current point.                                  |
| `grad`         | Gradient at the current point (bookkeeping).                           |
| `slope0`       | φ'(0), the directional derivative at `t = 0`. Should be negative.      |
| `init_step`    | Initial trial step length (clamped to `max_step`).                     |
| `c1`           | Armijo sufficient-decrease constant (typically `1e-4`).                |
| `c2`           | Wolfe curvature constant (typically `0.9`). Requires `0 < c1 < c2 < 1`.|
| `max_iter`     | Maximum iterations for **each** of the bracket and zoom phases.        |
| `temperature`  | Initial temperature for optional stochastic (Metropolis) acceptance.   |
| `cooling`      | Multiplicative cooling factor applied to the temperature.              |
| `seed`         | PRNG seed for the stochastic acceptance.                               |
| `max_probes`   | Capacity of the probe-recording buffers.                               |
| `record_probes`| If `False`, disables probe recording (buffer size collapses to 1).     |
| `max_step`     | Upper bound on the step length.                                        |

### Conditions

- **Armijo (sufficient decrease):**
  `φ(t) ≤ φ(0) + c1 · t · φ'(0)`
- **Strong Wolfe (curvature):**
  `|φ'(t)| ≤ c2 · |φ'(0)|`

A point is *accepted* when both conditions hold, or (optionally) when a
Metropolis criterion accepts an increase given the current temperature.

## Algorithm

### Phase 0 — Initialization

1. Clamp `init_step` to `max_step` to get `a0`.
2. Evaluate `eval_at(a0)` giving `(p0, v0, g0, s0)` and record it as the
   first probe.
3. Classify the initial point:
   - **Armijo violated** → bracket found on side A (`[0, a0]`).
   - **Armijo satisfied, curvature satisfied (or stochastic accept)** →
     *found*, terminate early.
   - **Armijo satisfied, positive slope `s0 ≥ 0`** → bracket found on
     side C (`[a0, 0]`, i.e. minimum lies below `a0`).
   - Otherwise → keep growing.
4. Seed the `(lo, hi)` bracket endpoints and associated φ/φ' values
   accordingly.

### Phase 1 — Bracketing (`bracket_cond` / `bracket_body`)

Repeatedly doubles the step (`a_cur * 2`, clamped to `max_step`) until a
bracket enclosing an acceptable point is found or a stopping condition
triggers. On each iteration it:

- Tracks the best-so-far point (`best_a/v/p/g`).
- Checks the bracket-forming conditions:
  - **cond_a**: Armijo violated, or φ increased relative to the previous
    step (`φ_cur ≥ φ_prev` for `i > 0`) → bracket `[a_prev, a_cur]`.
  - **cond_found**: curvature (strong Wolfe) satisfied, or stochastic
    acceptance → terminate.
  - **cond_c**: positive slope `s_cur ≥ 0` → bracket `[a_cur, a_prev]`.
- Updates the low/high bracket endpoints with `jnp.where` selectors.
- Evaluates the next (doubled) step and records the probe.
- Uses `stop_now` masks so that once stopped, the carried state freezes.

The loop continues while **not** stopped **and** `i < max_iter` **and**
`a_cur < max_step`.

### Phase 2 — Zoom (`zoom_cond` / `zoom_body`)

Runs only when a bracket was produced (`bracketed`). It performs
**bisection** of the `[lo, hi]` interval:

1. Evaluate at the midpoint `mid = 0.5 * (lo + hi)` and record probe.
2. Update best-so-far.
3. **shrink_hi** if Armijo fails at `mid` or `φ(mid) ≥ φ(lo)` → move the
   high endpoint to `mid`.
4. Otherwise check strong Wolfe / stochastic acceptance → *found*.
5. Otherwise `flip` the interval if the slope sign indicates the
   minimum is on the other side (`s·(hi-lo) ≥ 0`), then advance `lo` to
   `mid`.

The loop continues while not found and `i < max_iter`.

### Result Selection

Priority order:

1. `use_found` — a bracket-phase acceptance,
2. `use_zoom` — a zoom-phase acceptance,
3. **fallback** — the best-improving point found (`best_v ≤ value`),
   otherwise step `0` (i.e. reject the move).

A final Metropolis test on `out_v - value` can still mark the result as
`done` even without a strict Wolfe acceptance.

## Stochastic Acceptance (Metropolis)

When `temperature > 0`, the search may accept a step that increases the
objective with probability `exp(-(Δφ)/T)` (via `_metropolis_accept`),
with `T` cooled by `cooling` after each use. Setting `temperature = 0`
(the default) yields the deterministic, textbook behavior.

This is useful for stochastic / non-convex problems where escaping a
local trap by occasionally accepting an increase can improve global
behavior.

## Probe Recording

Every function evaluation can be stored in fixed-size probe buffers
(`probe_params`, `probe_grads`, `probe_valid`, `probe_values`,
`probe_alphas`). This provides an audit trail of all trial points and
is used by higher-level solvers for diagnostics or surrogate modeling.
Setting `record_probes=False` disables recording (buffer size 1) to save
memory.

## Return Value

Returns a `LineSearchResult` with:

- `step_size` — accepted step length `t`,
- `new_value` — φ at the accepted step,
- `new_grad`, `new_params` — gradient/params at the accepted step,
- `done` — whether an acceptable/accepted step was produced,
- `probe_*` — recorded probe buffers,
- `num_evals` — total number of `eval_at` calls.

## Design Notes & Discussion

- **Path-agnostic contract.** All geometry lives in `eval_at`. This
  decoupling keeps the search reusable and lets the solver own the path
  parameterization (linear, quadratic/QQN, projected, etc.).
- **Branch-free control flow.** All decisions are expressed via masks
  (`jnp.where`, `jnp.logical_*`) so both phases run inside
  `jax.lax.while_loop`. State that should be "frozen" after a stop is
  preserved via `stop_now`/selector patterns rather than early `break`.
- **Two independent iteration budgets.** `max_iter` bounds bracketing and
  zoom separately, so the worst case is roughly `2 * max_iter + 1`
  evaluations.
- **Fallback safety.** If neither phase yields a Wolfe point, the best
  improving probe is returned; if nothing improved, the step collapses to
  zero so the caller does not move to a worse point.
- **dtype discipline.** Scalars are cast to `value.dtype` throughout to
  avoid unintended float64/float32 promotion under JAX.

## Potential Follow-ups

- Consider cubic/quadratic interpolation in the zoom phase instead of
  pure bisection for faster convergence.
- Expose the number of bracket vs. zoom iterations in the result for
  finer diagnostics.
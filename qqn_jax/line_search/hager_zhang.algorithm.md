# Hager-Zhang Line Search

## Overview

This module implements a **Hager-Zhang style line search** built on top of
Optax's `scale_by_backtracking_linesearch`. The search operates on the scalar
one-dimensional problem `φ(t)` (the objective restricted to a search
direction/path), and is intended to be used as a pluggable line-search
component inside a larger optimization solver.

The public entry point is:

```python
hager_zhang_search(eval_at, params, value, grad, slope0, *, ...) -> LineSearchResult
```

## Design Philosophy: Path-Agnostic Search

The function is **path-agnostic**. It does not know or care about the geometry
of the search direction or the trust region. Instead, the solver pre-bakes the
path into the `eval_at` callable. From the perspective of this routine, the
problem is purely scalar:

- Given a step `t`, `eval_at(t)` returns `(new_params, value, grad, slope)`
  corresponding to moving a distance `t` along the pre-baked path.
- The scalar objective is `φ(t) = value` returned by `eval_at`.

This separation of concerns keeps the line search reusable across different
path parameterizations (straight-line, curved, quadratic-quotient paths, etc.).

## Parameters

| Parameter        | Type       | Meaning                                                        |
|------------------|------------|----------------------------------------------------------------|
| `eval_at`        | `Callable` | Evaluates the real path at scalar step `t`.                    |
| `params`         | array      | Current parameters (used for probe bookkeeping shapes).        |
| `value`          | scalar     | Objective value at the current point (`φ(0)`).                 |
| `grad`           | array      | Gradient at current point (not directly used by the scalar LS).|
| `slope0`         | scalar     | Directional derivative `φ'(0)` along the path.                 |
| `init_step`      | float      | **Deprecated / ignored** (see notes below).                    |
| `c1`             | float      | Armijo sufficient-decrease relative tolerance (`slope_rtol`).  |
| `max_iter`       | int        | Maximum backtracking steps.                                    |
| `temperature`    | float      | Temperature for optional stochastic (Metropolis) acceptance.   |
| `cooling`        | float      | Cooling factor for the temperature schedule.                   |
| `seed`           | int        | PRNG seed for the Metropolis acceptance draw.                  |
| `max_probes`     | int        | Maximum number of recorded probes.                             |
| `record_probes`  | bool       | Whether to record probe telemetry.                             |
| `max_step`       | float      | Upper bound used to cap the increase factor.                   |

## Algorithm

The implementation reduces the Hager-Zhang search to a single Optax
backtracking-line-search update on the scalar problem:

1. **Set up the scalar problem.**
   - `dtype` is taken from `value`.
   - `t0` is the scalar starting step, initialized to zero (`[0.0]`).
   - `unit` is the unit update direction `[1.0]` in step-space.
   - `phi(tvec)` returns the objective value from `eval_at(tvec[0])`.
   - `scalar_grad = [slope0]` provides the scalar directional derivative.

2. **Configure Optax backtracking line search.**
   - `max_backtracking_steps = max_iter`
   - `slope_rtol = c1` (Armijo condition)
   - `decrease_factor = 0.8` (step shrink per backtrack)
   - `increase_factor = min(1.0, max_step)` (step-growth cap)
   - `store_grad = False`

3. **Run one update step.**
   - Initialize the line-search state with `t0`.
   - Call `ls.update(...)` with `updates=unit`, `params=t0`, the current
     `value`, the scalar `grad`, and `value_fn=phi`.
   - Extract the resulting `step_size` from the scaled updates.

4. **Recompute along the real path.**
   - `eval_at(step_size)` yields `new_params, new_value, new_grad, slope`.

5. **Acceptance decision.**
   - Compute a Metropolis acceptance flag using the change in value,
     `temperature`, and the PRNG key.
   - `done = (new_value < value) OR stochastic_accept`, allowing either a
     deterministic sufficient-decrease acceptance or a probabilistic one at
     nonzero temperature.

6. **Record probes.**
   - Allocate probe buffers of size `max_probes` (or 1 if not recording).
   - Record the single evaluated probe at index 0.

7. **Return** a `LineSearchResult` with the step, new state, done flag,
   probe telemetry, and `num_evals = max_iter + 1`.

## Return Value

A `LineSearchResult` containing:

- `step_size` — the discovered scalar step `t`.
- `new_value`, `new_grad`, `new_params` — state recomputed along the real path.
- `done` — whether the step is accepted.
- `probe_*` — probe telemetry buffers.
- `num_evals` — reported as `max_iter + 1`.

## Discussion and Notes

- **`init_step` is ignored.** The parameter is accepted for API compatibility
  but immediately `del`-eted. The Optax backtracking search always begins from
  the `unit` step and shrinks by `decrease_factor`, so an explicit initial step
  is not honored. Callers should not rely on `init_step` affecting behavior.

- **`increase_factor` capping.** The increase factor is `min(1.0, max_step)`.
  With `max_step <= 1.0` this disables step growth (the search only ever
  backtracks). This is a conservative choice that guarantees the step never
  exceeds `max_step`, at the cost of not being able to expand beyond the unit
  step.

- **`cooling` is currently unused** in the acceptance logic here; the
  temperature passed to `_metropolis_accept` is not decayed within this
  function. If a cooling schedule is desired, it must be applied by the caller
  across outer iterations.

- **`num_evals` is an upper bound**, not an exact count. It reports
  `max_iter + 1` regardless of how many backtracking steps Optax actually
  performed, since the underlying number of `phi` evaluations is not surfaced.

- **Scalar gradient fidelity.** Only `slope0` (the directional derivative at
  `t = 0`) is provided to the line search. The Armijo condition is therefore
  evaluated against the slope at the origin, consistent with a standard
  sufficient-decrease test.

## Potential Impacts / Follow-ups

- If exact evaluation counts are needed for benchmarking, thread the actual
  backtracking count out of the Optax state.
- If step expansion beyond the unit step is desired, revisit the
  `increase_factor` cap and the `max_step` semantics.
- Consider wiring the `cooling` schedule into the temperature used for
  Metropolis acceptance if stochastic search behavior is intended.
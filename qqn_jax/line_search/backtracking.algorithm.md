# Backtracking (Armijo) Line Search

## Overview

`backtracking_search` implements a classic **backtracking line search**
with an optional **extrapolation** phase and an optional **Metropolis
stochastic acceptance** layer. It operates purely on a 1-D scalar
reduction `φ(t)` of an optimization problem, exposed through a
caller-supplied `eval_at` callback. The search is fully JAX-traceable —
it uses `jax.lax.while_loop` and `jax.lax.cond` so that it can be
`jit`-compiled and differentiated through the surrounding solver.

## Mathematical Background

Given a current point, a search direction, and the scalar problem

    φ(t) = f(x + t·d)

the search seeks a step size `t > 0` that produces "sufficient
decrease". The **Armijo condition** is

    φ(t) ≤ φ(0) + c1·t·φ'(0)

where:

- `φ(0)` is the current objective value (`value`),
- `φ'(0)` is the directional derivative `slope0` (denoted `dg`),
- `c1 ∈ (0, 1)` is the sufficient-decrease coefficient (`c1`, default `1e-2`).

Because `φ'(0) < 0` for a valid descent direction, the right-hand side
decreases as `t` grows, requiring progressively larger reductions in the
objective for larger steps.

## Interface

```python
backtracking_search(
    eval_at,        # Callable: t -> (params, value, grad, slope)
    params,         # current parameters (used for probe buffers)
    value,          # φ(0)
    grad,           # gradient at t=0 (unused directly here)
    slope0,         # φ'(0) directional derivative
    *,
    init_step=1.0,  # initial trial step t0
    c1=1e-2,        # Armijo coefficient
    shrink=0.5,     # backtracking shrink factor (0<shrink<1)
    max_iter=5,     # maximum backtracking iterations
    temperature=0.0,# Metropolis temperature (0 disables)
    cooling=0.95,   # temperature decay per evaluation
    seed=0,         # PRNG seed for stochastic acceptance
    max_probes=32,  # probe buffer capacity
    record_probes=True,
    max_step=1.0,   # upper bound enabling extrapolation
) -> LineSearchResult
```

The callback `eval_at(t)` returns `(params, value, grad, slope)`; the
search internally wraps it as `eval_pvg` that discards the slope.
Crucially, the line search has **no knowledge** of the path, direction,
or trust region — those concerns are baked into `eval_at` by the caller.

## Algorithm

The procedure has three phases.

### 1. Initial Evaluation

Evaluate `φ(init_step)`, test acceptance, and record the first probe.
The initial temperature `temp0` is created with the same dtype as
`value`, and the PRNG key is derived from `seed`.

### 2. Extrapolation Phase (optional)

Runs only when `max_step > init_step` **and** the initial step was
accepted. It grows the step by `grow = 1/shrink` while both:

- the Armijo condition still holds, and
- the objective keeps improving (`φ(new) < φ(prev)`),

and while the next step stays `≤ max_step` and `i < max_iter`. Each
candidate is conditionally kept via `jnp.where(keep, ...)`, so the
loop carries the best-so-far step. This phase is wrapped in
`jax.lax.cond` so it is skipped entirely (no evaluations) when
extrapolation does not apply.

### 3. Backtracking Phase

Starting from the (possibly extrapolated) step, repeatedly multiply the
step by `shrink` and re-evaluate until acceptance or `max_iter` is
reached. The `cond` predicate stops as soon as a step is accepted.

### Acceptance Test

The `accept` helper combines two criteria via logical OR:

1. **Armijo**: `val ≤ value + c1·alpha·dg`.
2. **Metropolis** (only meaningful when `temperature > 0`): accepts an
   increase probabilistically based on `Δe = val − value` and the
   current temperature, using `_metropolis_accept`. The temperature is
   multiplied by `cooling` after each evaluation, implementing simulated
   annealing–style cooling.

## Probes

When `record_probes` is `True`, every evaluated candidate `(params,
grad, value, alpha)` is stored into fixed-size probe buffers (allocated
by `_empty_probes`, written by `_record_probe`). This lets the caller
inspect every point the line search touched — useful for surrogate
models, diagnostics, or reusing evaluations. When `record_probes` is
`False`, the effective buffer size collapses to `1` to save memory.

## Return Value

A `LineSearchResult` containing:

- `step_size` — accepted (or final) step,
- `new_value`, `new_grad`, `new_params` — state at that step,
- `done` — whether an acceptable step was found,
- `probe_*` — recorded probe buffers,
- `num_evals` — total number of `eval_at` evaluations performed.

## Design Notes and Discussion

- **JAX-friendliness.** All control flow uses `lax.while_loop`/`lax.cond`
  with fixed-shape carries, so the routine compiles to a static graph.
  The carry tuples are long (14 elements in backtracking, 12 in
  extrapolation) because every piece of loop state — including probe
  buffers, temperature, and PRNG key — must be threaded explicitly.

- **Cost accounting.** `num_evals` accumulates evaluations across both
  phases, since the initial eval counts as one and extrapolation/
  backtracking each increment `evals`.

- **Descent assumption.** Correct Armijo behavior assumes `slope0 < 0`.
  The routine does not verify this; callers are responsible for
  supplying a descent direction.

- **Stochastic layer.** Setting `temperature = 0` disables Metropolis
  acceptance, recovering deterministic backtracking. Positive
  temperatures allow occasional uphill steps, which can help escape
  poor local geometry at the cost of monotone-decrease guarantees.

- **Extrapolation trade-off.** Extrapolation can find larger useful
  steps than `init_step`, but adds evaluations. It only triggers when
  `max_step > init_step`, giving the caller explicit control.
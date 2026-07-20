# Strong Wolfe Line Search

## Overview

`strong_wolfe_search` performs a **Strong Wolfe** line search by
delegating the heavy lifting to Optax's
[`scale_by_zoom_linesearch`](https://optax.readthedocs.io/).
The routine finds a step size `t ≥ 0` along a pre-baked search path
such that the candidate point satisfies both the **Armijo (sufficient
decrease)** condition and the **strong curvature** condition.

A key design decision is that the search is **path-agnostic**: the full,
possibly multidimensional, path or trust region is encapsulated inside
the `eval_at` callable. The optimizer therefore only ever sees a
*scalar* one-dimensional problem — a single-element parameter `t` with a
unit update direction and value function `φ(t)`.

## The Strong Wolfe Conditions

Given a scalar merit function `φ(t) = f(x + t·d)` with initial slope
`φ'(0) = slope0`, a step `t` satisfies the Strong Wolfe conditions when:

1. **Armijo / sufficient decrease** (controlled by `c1`):

   ```
   φ(t) ≤ φ(0) + c1 · t · φ'(0)
   ```

2. **Strong curvature** (controlled by `c2`):

   ```
   |φ'(t)| ≤ c2 · |φ'(0)|
   ```

Typical requirements are `0 < c1 < c2 < 1`. The defaults used here are
`c1 = 1e-3` and `c2 = 0.7`.

## Algorithm Steps

1. **Set up the scalar problem.**
   - `t0 = [0]` is the starting step (a length-1 vector).
   - `unit = [1]` is the search direction in step-space.
   - `phi(tvec)` evaluates the path at `t` and returns only the value.
   - `scalar_grad = [slope0]` supplies the directional derivative.

2. **Configure Optax zoom line search.** The parameters map as:

   | Optax argument            | Source          | Meaning                        |
   |---------------------------|-----------------|--------------------------------|
   | `max_linesearch_steps`    | `max_iter`      | zoom iteration budget          |
   | `curv_rtol`               | `c2`            | strong curvature tolerance     |
   | `slope_rtol` / `tol`      | `c1`            | Armijo / sufficient decrease   |
   | `initial_guess_strategy`  | `"one"`         | start trials from `t = 1`      |
   | `max_learning_rate`       | `max_step`      | upper bound on the step        |

3. **Run one update.** `ls.update` returns `scaled_updates`, whose single
   component is the accepted step `t`.

4. **Re-evaluate along the true path.** `eval_at(step_size)` recomputes
   the real (possibly multidimensional) `new_params`, `new_value`,
   `new_grad`, and slope at the found step.

5. **Acceptance / (optional) Metropolis.** The energy change
   `delta_e = new_value - value` is passed through `_metropolis_accept`.
   When `temperature > 0`, an uphill move may still be accepted
   stochastically; otherwise acceptance reduces to strict decrease.
   `done` is `True` if either the value decreased or the stochastic
   acceptance fired.

6. **Record probes.** A single probe (the accepted point) is stored in
   the probe buffers so the result matches the common
   `LineSearchResult` interface used by other line searches.

## Parameters

- `eval_at`: callable `t -> (params, value, grad, slope)` describing the
  path. All geometry lives here.
- `params`, `value`, `grad`: current iterate state.
- `slope0`: directional derivative `φ'(0)`; must be negative for a
  descent direction.
- `init_step`: **ignored** (`del init_step`); Optax's
  `initial_guess_strategy="one"` determines the initial trial.
- `c1`, `c2`: Wolfe constants (Armijo and curvature respectively).
- `max_iter`: maximum zoom iterations.
- `temperature`, `cooling`, `seed`: control the optional Metropolis
  acceptance step.
- `max_probes`, `record_probes`: probe-buffer sizing/recording.
- `max_step`: caps the returned step size.

## Return Value

A `LineSearchResult` containing:
- `step_size`: the accepted step `t`.
- `new_value`, `new_grad`, `new_params`: state at the accepted step.
- `done`: acceptance flag.
- `probe_*`: probe buffers (single recorded probe).
- `num_evals`: reported as `max_iter + 1` (upper-bound estimate).

## Discussion

**Strengths**

- **Robustness.** Optax's zoom implementation is battle-tested and
  handles bracketing/zooming edge cases that are easy to get wrong in a
  hand-rolled Wolfe search.
- **Separation of concerns.** By collapsing the geometry into `eval_at`,
  the search stays purely one-dimensional regardless of the underlying
  problem dimension or curvature of the path.
- **JAX-friendly.** The whole routine is expressed with `jnp` and
  Optax primitives, making it amenable to `jit`/`vmap`.

**Caveats**

- `init_step` is accepted for interface compatibility but discarded; the
  initial trial is always `t = 1` via `initial_guess_strategy="one"`.
- `num_evals` is an over-estimate rather than the exact evaluation count,
  since Optax may terminate early.
- The optional Metropolis acceptance is a non-standard extension to the
  classical deterministic Wolfe search; with `temperature = 0` it has no
  effect and behaviour is the usual descent check.
- Only the final accepted point is recorded as a probe, so intermediate
  zoom trials are not surfaced.
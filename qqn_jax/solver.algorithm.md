# QQN — Quadratic Quasi-Newton Solver

This document describes the algorithm implemented in `solver.py`, explains the design choices, and discusses its
behaviour, trade-offs, and extension points.

## 1. Overview

QQN (Quadratic Quasi-Newton) is a first/second-order hybrid optimizer. Instead of choosing a *single* search direction
and then performing a 1-D line search along that ray (as in classical steepest descent or quasi-Newton methods), QQN
constructs a **curved path** that smoothly blends the steepest-descent direction with a quasi-Newton direction, and then
performs the line search *along that curve*.

The canonical quadratic path is

```
d(t) = t(1 - t)·(-∇f)  +  t²·(-H∇f)
```

where:

- `-∇f` is the steepest-descent direction,
- `-H∇f` is the quasi-Newton (e.g. L-BFGS) direction, supplied by an oracle,
- `t ∈ [0, 1]` is the path parameter (extended up to `max_t` by the line search).

Key properties of this path:

- **At `t = 0`:** `d(0) = 0` (the current iterate).
- **Initial tangent:** `d'(0) = -∇f`, so the path *starts* moving in the steepest-descent direction. This guarantees an
  initial descent direction whenever `∇f ≠ 0`, regardless of the quality of `H`.
- **At `t = 1`:** `d(1) = -H∇f`, the full quasi-Newton step.

The curve therefore behaves like gradient descent for small `t` (robust, guaranteed descent) and transitions toward the
Newton-like step as `t → 1`
(fast local convergence). A single scalar line search over `t` selects the accepted iterate.

## 2. State

All solver state is kept in the JIT-compatible `QQNState` NamedTuple so that
`run` can be wrapped in `jax.lax.while_loop` and remain `jit`/`vmap`
compatible. Notable fields:

| Field            | Meaning                                                           |
|------------------|-------------------------------------------------------------------|
| `iter`           | iteration counter                                                 |
| `value`, `grad`  | current objective value and gradient                              |
| `oracle_state`   | quasi-Newton oracle history (e.g. L-BFGS), possibly per-partition |
| `step_size`      | last accepted path parameter `t` (`0` if the step was rejected)   |
| `error`          | gradient L2 norm (convergence metric)                             |
| `done`           | convergence / termination flag                                    |
| `aux`            | optional auxiliary output of `fun`                                |
| `region_state`   | optional trust-/projective-region state                           |
| `num_evals`      | cumulative objective/gradient evaluation count                    |
| `qn_slope`       | directional derivative `∇f·(-H∇f)` at the current iterate         |
| `ls_success`     | whether the last line search reported success                     |
| `last_reduction` | objective reduction of the last accepted step                     |

## 3. The interface

Following the JAXopt convention, the solver exposes:

- `init_state(params, *args) -> QQNState` — evaluate `f`, initialize the oracle and region, compute the initial gradient
  norm, and set `done` if already converged.
- `update(params, state, *args) -> (new_params, new_state)` — perform one iteration.
- `run(init_params, *args) -> (final_params, final_state)` — iterate to convergence or `maxiter` using `lax.while_loop`.

## 4. One iteration (`update`)

Each iteration performs the following steps.

1. **Query the oracle.** The oracle produces the quasi-Newton endpoint
   `qn_dir = -H∇f` (the `t=1` point of the path). The directional slope
   `qn_slope = ∇f·qn_dir` is recorded for diagnostics.

2. **Build the 1-D subproblem.** `make_evaluator` produces a closure
   `φ(t) = f(x + region(d(t)))` from the configured path, the steepest-descent endpoint `grad_dir = -∇f`, the oracle
   endpoint `qn_dir`, and the optional region projection. The initial slope `slope0 = ∇f · d'(0)` is computed via the
   path's `velocity` method.

3. **Line search over `t`.** The inner line search (`self._inner_search`)
   selects a step along the curve. In QQN the line search is intentionally *permissive*: because the curvature is
   already encoded in the path, the search only needs to make sufficient progress rather than solve the 1-D problem
   exactly.

   Depending on `path_strategy`, an additional refinement is applied:
    - `"linear"` — value-only chord refinement (`linear_refine`).
    - `"spline"` — cubic-Hermite spline refinement (`spline_refine`), reusing probe points as control points; iterated
      `spline_refine_rounds` times.
    - `"quadratic"` — no extra refinement (the default).

4. **Acceptance and fallback.** The proposed step is accepted only if the new value and gradient are finite and
   `new_value ≤ state.value`. If not, a **safeguarded gradient-descent fallback** is tried:

   ```
   fb_params = params - (1 / (1 + ||∇f||²)) · ∇f
   ```

   This scaled step is accepted when it strictly reduces the objective. If neither the line-search step nor the fallback
   is acceptable, the iterate is left unchanged and `step_size` is set to `0` (a *stall*). All of this is expressed with
   `jnp.where`/boolean masks so the branch structure is JIT-friendly.

5. **Oracle update.** The oracle history is updated from the accepted step. When `feed_probes_to_oracle` is enabled, the
   line-search probe points and their gradients are also fed back to enrich the curvature history; only probes on the
   accepted side (`alpha ≤ step_size`) are marked valid.

6. **Region update.** The predicted reduction `-∇f·d(t)` and the actual reduction `state.value - new_value` are passed
   to the region strategy to rescale/adapt the region for the next iteration. The predicted reduction is floored by a
   small epsilon to avoid division issues downstream.

7. **Convergence / termination.** `error = ||new_grad||₂` is recomputed. Termination (`done`) is triggered when any of
   the following holds:
    - `error ≤ tol` (converged),
    - `new_value` or `error` is non-finite,
    - the step stalled (nothing was accepted).

8. **Evaluation accounting.** `num_evals` accumulates the line-search evaluations, the fallback evaluation, an optional
   aux evaluation, and the per-iteration gradient evaluation.

## 5. Partitioning (block-diagonal quasi-Newton)

When `partition_sizes` is provided, the flat parameter vector is split into contiguous segments with statically-known
offsets. The oracle is then driven *independently* on each segment, forming a **block-diagonal** quasi-Newton
approximation, and the per-segment endpoints are concatenated back into the full direction. `OracleInfo` is sliced
per-segment (`_slice_oracle_info`), with probe buffers sliced along the parameter axis and scalar/mask fields shared
verbatim. This can reduce memory/coupling for problems with natural block structure.

## 6. Configuration summary

- `maxiter`, `tol` — iteration budget and gradient-norm tolerance.
- `history_size` — L-BFGS memory `m`.
- `line_search`, `line_search_options` — inner search strategy and its keyword overrides (e.g. `c1`, `c2`, `max_iter`, `init_step`). `max_step` defaults to `max_t`.
- `path_strategy` — `"quadratic"` (default), `"linear"`, or `"spline"`.
- `region` — optional trust-/projective-region selector (identity when `None`).
- `oracle` — quasi-Newton oracle selector (default `"lbfgs"`).
- `feed_probes_to_oracle`, `max_probes` — probe-recycling into the oracle.
- `max_t` — upper bound on `t` (forwarded as `max_step`).
- `partition_sizes` — block-diagonal segmentation.
- `spline_refine_rounds` — refinement iterations for the spline path.

## 7. Discussion

### Why a curved path?

Classical line searches commit to a fixed direction and then search a ray. If the quasi-Newton direction is poor (e.g.
an inaccurate Hessian approximation early on, or in nonconvex regions), the ray may point uphill or waste iterations.
QQN's path guarantees an initial gradient-descent tangent while still reaching the full quasi-Newton step at `t=1`, so a
single line search can smoothly interpolate between "safe" and "fast" regimes. This yields robustness close to gradient
descent with the asymptotic speed of quasi-Newton methods.

### Permissive line search

Because curvature is baked into the geometry of `d(t)`, the line search does not need strong Wolfe accuracy. A
permissive search reduces the number of function/gradient evaluations per iteration.

### Robustness safeguards

The finite-value checks, the scaled gradient fallback, and the stall detection make the update resilient to non-finite
evaluations and to line searches that fail to make progress — all without Python-level control flow, preserving JIT/vmap
compatibility.

### Trade-offs

- The quadratic path requires one oracle direction and a line search per iteration, comparable to L-BFGS costs.
- Probe recycling and the spline path can improve the effective use of evaluations but increase per-iteration
  bookkeeping and memory (`max_probes`).
- Block-diagonal partitioning trades approximation fidelity for reduced coupling and memory.

## 8. Extension points

- **New paths** implement the path interface (`offset`, `velocity`, and use in
  `make_evaluator`); register a strategy string in `__init__`.
- **New oracles** implement `init`/`direction`/`update` and are resolved via
  `qqn_jax.oracles.strategy.resolve_oracle`.
- **New regions** implement `init`/`update` and are resolved via
  `qqn_jax.regions.strategy.resolve_region`.
- **New line searches** are registered in `qqn_jax.line_search.LINE_SEARCHES`.
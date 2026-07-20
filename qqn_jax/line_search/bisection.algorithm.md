# Bisection Line Search Algorithm

## Overview

The `bisection_search` routine implements a one-dimensional line search
that seeks a **true stationary point** of the objective along a
pre-baked search path. Unlike a classical backtracking search (which
only tries to satisfy a sufficient-decrease condition), this method
drives the directional slope `φ'(t)` toward zero using a bracket-and-
bisect strategy, thereby locating a genuine one-dimensional minimum.

The algorithm operates **purely on the scalar problem** `φ(t)` and its
slope `φ'(t)` exposed by the `eval_at` callback. It has no knowledge of
the higher-dimensional geometry of the underlying optimization path — it
only sees a scalar function of the step length `t`.

## Scalar Model

Along the path we define:

- `φ(t)  = f(params + path(t))` — the objective value at step `t`
- `φ'(t) = <∇f, dpath/dt>`      — the directional slope at step `t`

`eval_at(t)` returns the tuple `(params, value, grad, slope)` describing
the point reached at step `t`.

## Inputs

| Argument        | Meaning                                                        |
|-----------------|----------------------------------------------------------------|
| `eval_at`       | Callable mapping `t -> (params, value, grad, slope)`.          |
| `params`        | Current parameters (start of the path, `t = 0`).               |
| `value`         | Objective value at `t = 0`, i.e. `φ(0)`.                        |
| `grad`          | Gradient at `t = 0`.                                           |
| `slope0`        | Directional slope `φ'(0)` (should be negative for descent).    |
| `init_step`     | Initial trial step `hi0` used to start bracketing.             |
| `c1`            | Armijo sufficient-decrease constant.                           |
| `max_iter`      | Maximum iterations for *each* phase (bracket / bisect).        |
| `temperature`   | Temperature for optional stochastic (Metropolis) acceptance.   |
| `cooling`       | Cooling factor (reserved; not used in the core loop here).     |
| `seed`          | PRNG seed for the Metropolis acceptance draw.                  |
| `max_probes`    | Capacity of the probe-recording buffers.                       |
| `record_probes` | Whether to record every probe (else only a 1-slot buffer).     |
| `max_step`      | Upper bound on the step size; bracketing never exceeds this.   |

## Algorithm

The procedure has two phases, both expressed as `jax.lax.while_loop`s so
the whole search is JIT- and vmap-compatible.

### Phase 1 — Bracketing

Starting from `hi0 = init_step`, the search doubles the upper bound while:

1. the slope at `hi` is still negative (`s_hi < 0`) — meaning the minimum
   lies further along, and
2. the iteration count is below `max_iter`, and
3. `hi` has not yet reached `max_alpha = max_step`.

Each doubling clamps to `max_alpha`:

```
    new_hi = min(hi * 2, max_alpha)
```

The loop terminates when the slope becomes non-negative
(`s_hi >= 0`), at which point `[0, hi]` **brackets** a stationary point
(the slope changes sign across the interval). The boolean `bracketed`
records whether a sign change was actually found.

### Phase 2 — Bisection

With the interval `[lo = 0, hi]`, the search repeatedly evaluates the
midpoint `mid = 0.5 * (lo + hi)` and inspects its slope `s`:

- If `s < 0` (still descending), move `lo` up to `mid` (go right).
- Otherwise, move `hi` down to `mid` (go left).

This halves the bracket each iteration, converging on the point where
`φ'(t) = 0`. Throughout, the algorithm tracks the **best point seen so
far** (lowest objective value) rather than simply the final midpoint,
guarding against non-monotone behaviour:

```
    improved = v < best_v
    best_* = where(improved, current_*, best_*)
```

The bisection runs for `max_iter` iterations.

### Probe Recording

Every evaluation (the initial `hi0` probe, each bracketing step, and each
bisection midpoint) is recorded into the probe buffers via
`_record_probe`, subject to the buffer capacity `eff_probes`. When
`record_probes` is `False`, only a single slot is retained. These probes
are returned in the result and are useful for diagnostics, visualization,
and downstream surrogate modelling.

## Acceptance Criteria

After bisection, the best candidate is accepted if **any** of the
following hold:

1. **Armijo sufficient decrease**:
   `best_value <= value + c1 * best_alpha * slope0`.
2. **Bracketed**: a true sign change was located during Phase 1
   (`s_hi_final >= 0`), indicating a genuine interior minimum was
   bracketed.
3. **Stochastic (Metropolis) acceptance**: with probability governed by
   `temperature`, a non-improving or marginally-improving step may still
   be accepted. This enables escaping poor local structure when the
   search is run at nonzero temperature. At `temperature = 0` this path
   reduces to strict acceptance of improvements only.

The combined `done` flag is the logical OR of these three conditions.

## Output

Returns a `LineSearchResult` containing:

- `step_size`      — best step `best_alpha`.
- `new_value`      — objective at the best step.
- `new_grad`       — gradient at the best step.
- `new_params`     — parameters at the best step.
- `done`           — acceptance flag (see above).
- `probe_*`        — recorded probe params/grads/values/alphas/validity.
- `num_evals`      — total number of `eval_at` calls performed.

## Discussion

### Why bisect on the slope?

Backtracking Armijo searches are cheap but only guarantee *decrease*, not
optimality along the ray. By bisecting on the sign of `φ'(t)`, this
method converges to a stationary point of the 1-D problem, which can
yield substantially better step sizes when the path is well-conditioned
(e.g. quasi-Newton or curvature-informed directions). This can improve
per-step progress at the cost of extra function/gradient evaluations.

### Bracketing robustness

If the slope never turns non-negative before `hi` reaches `max_alpha`
(or `max_iter` is exhausted), no true bracket is found (`bracketed`
is `False`). In that regime, acceptance falls back to the Armijo and
stochastic criteria, and the returned step is the best point observed —
the algorithm degrades gracefully rather than failing.

### JAX considerations

- The two phases use `jax.lax.while_loop`, so all carried state has
  static shape and dtype. Probe buffers are pre-allocated to fixed size
  (`eff_probes`) to satisfy this constraint.
- Branching uses `jnp.where` rather than Python `if`, keeping the code
  traceable and differentiable-friendly.
- `dtype` consistency is maintained by casting constants (`zero`,
  `max_alpha`, `hi0`) to `value.dtype`.

### Complexity

Each phase performs at most `max_iter` evaluations, so the total number
of `eval_at` calls is bounded by roughly `1 + 2 * max_iter`. Bisection
converges linearly, halving the bracket width every iteration.

### Notes and Caveats

- `slope0` should be negative (a descent direction); otherwise the
  bracketing loop terminates immediately and the search reduces to
  evaluating the initial trial step.
- The `cooling` parameter is accepted for interface compatibility with
  other line searches but is not applied to the temperature within the
  core loops here.
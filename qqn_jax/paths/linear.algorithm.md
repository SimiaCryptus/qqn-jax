# Linear (Chord) Path Augmentation

## Overview

The `linear` module implements the **linear (chord) path** augmentation for
QQN line searches. It is deliberately the *control* / *baseline* strategy:
where the quadratic and spline paths exploit curvature and gradient
information, the linear path discards all of it and simply samples the
objective along the straight chord connecting the origin to the oracle
endpoint.

This document describes the algorithm as implemented in `linear.py`,
explains the design rationale, and discusses its role relative to the
other path strategies.

## The Chord Path

Given a starting point `params` (at `t = 0`) and a search `direction`, the
oracle endpoint sits at `t = 1`, i.e. `params + direction`. The linear
path is the straight segment between these two points:

```
d(t) = t · direction        (offset)
d'(t) = direction           (velocity, constant)
```

### `_linear_offset(t, grad_dir, direction)`

Returns `t · direction` via `tree_scale`. The `grad_dir` argument is
**intentionally ignored** (`del grad_dir`) — this is the essence of the
linear control: no curvature/gradient blending is performed.

### `_linear_velocity(t, grad_dir, direction)`

Returns the constant tangent `direction`. Both `t` and `grad_dir` are
unused (`del t, grad_dir`).

### `LINEAR_PATH`

These two functions are packaged into a `PathStrategy`:

```python
LINEAR_PATH = PathStrategy(offset=_linear_offset, velocity=_linear_velocity)
```

Because it is a standard `PathStrategy`, the chord's probes flow through
the same shared `t -> point` remapping component used by
`qqn_jax.paths.quadratic` and `qqn_jax.paths.spline`. This keeps all path
strategies interchangeable within the line-search machinery.

## `linear_refine`

```python
linear_refine(inner, eval_at, dtype, num_samples: int = 8) -> LineSearchResult
```

Performs a **value-only** refinement of an already-computed inner
line-search result by sampling interior points of the chord and keeping the
best.

### Arguments

| Name          | Description                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `inner`       | The baseline `LineSearchResult` to refine.                                  |
| `eval_at`     | Shared scalar evaluator `t -> (params, value, grad, slope)` built from `LINEAR_PATH`. |
| `dtype`       | dtype for the sampling grid.                                                |
| `num_samples` | Number of interior samples of `t ∈ (0, 1]` to probe (default `8`).          |

### Algorithm

1. **Evaluation count seeding.** `inner.num_evals` is read; if `None`, it
   defaults to `1`.

2. **Sampling grid.** A uniform grid of `n = num_samples` step sizes is
   built over `(0, 1]`:

   ```
   alphas = [1/n, 2/n, ..., n/n]
   ```

   Note the grid excludes `0` (the origin) and includes `1` (the oracle
   endpoint).

3. **Vectorized probing.** `eval_at` is `vmap`-ed across the grid,
   producing per-sample `(alpha, value, params, grad)` tuples. This is a
   value-driven search; the returned gradient is carried along only so the
   chosen result is fully populated.

4. **Best-sample selection.** `argmin` over the sampled values picks the
   lowest-value feasible sample.

5. **Acceptance test.** The best sample replaces `inner` **only if** its
   value strictly improves on `inner.new_value`:

   ```
   use_sample = best_sample_val < inner.new_value
   ```

   `jnp.where` / `tree_map` merge the fields so the function stays
   JIT/`vmap`-friendly (no Python-level branching on traced values).

6. **Done flag.** The result is marked done if the inner search was done
   or if a sample was accepted (`inner.done OR use_sample`).

7. **Result assembly.** A new `LineSearchResult` is returned. The
   `step_size`, `new_value`, `new_params`, and `new_grad` fields reflect
   the accepted candidate, while all `probe_*` fields are passed through
   unchanged from `inner`. `num_evals` is incremented by `n`.

## Design Rationale

- **Deliberate control.** By ignoring gradients and curvature, the linear
  path isolates the value of the more sophisticated quadratic/spline
  strategies. Comparisons against it quantify how much curvature/gradient
  reuse actually buys.

- **Shared infrastructure.** Expressing the chord as a `PathStrategy`
  guarantees identical probe remapping semantics across all strategies,
  so any performance difference is attributable to the path shape, not to
  differing plumbing.

- **Value-only, JIT-safe.** The refinement avoids gradient-based decisions
  and Python branching, making it fully compatible with `jax.jit` and
  `jax.vmap`.

## Discussion

- **Monotone improvement.** `linear_refine` can never worsen the result:
  the strict `<` acceptance test only replaces `inner` when a strictly
  better value is found.

- **Endpoint coverage.** Because the grid includes `t = 1`, the oracle
  endpoint itself is always among the candidates.

- **Resolution vs. cost.** Larger `num_samples` gives finer coverage of the
  chord at the cost of `n` additional evaluations. The default of `8`
  balances coverage against evaluation budget.

- **Limitations.** With no gradient/curvature use, the linear path can miss
  minima that lie off the straight chord or between grid points. This is by
  design: it is the baseline against which richer paths are measured.

## Public API

```python
__all__ = ["LINEAR_PATH", "linear_refine"]
```
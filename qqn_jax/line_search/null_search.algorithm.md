# Null Line Search Algorithm

## Overview

The `null_search` algorithm implements a "null" (no-op) line search
strategy. Unlike conventional line search methods (e.g. backtracking,
Wolfe-condition searches, or bracketing/zoom methods), it performs **no
acceptance test** whatsoever. It simply evaluates the objective at a
single, predetermined step size and unconditionally accepts that point.

This is useful as:

- A **baseline / control** for comparing against real line searches.
- A way to run an optimizer with a **fixed step size** (constant or
  externally scheduled) while still using the common `LineSearchResult`
  interface.
- A **path-agnostic** primitive: all direction / region / path handling is
  delegated to the `eval_at` callable supplied by the solver, so this
  routine makes no assumptions about the geometry of the search.

## Signature

```python
null_search(
    eval_at: Callable,
    params,
    value,
    grad,
    slope0,
    *,
    step_size: float = 1.0,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult
```

### Parameters

- **`eval_at`** — Callable mapping a scalar step `t` to the tuple
  `(new_params, new_value, new_grad, slope)`. The solver folds all
  path/region/direction logic into this closure, keeping the search
  fully path-agnostic.
- **`params`** — Current parameters, used only to shape the probe buffers.
- **`value`** — Current objective value; its dtype is used to cast the
  step size.
- **`grad`** — Current gradient (unused by the algorithm itself).
- **`slope0`** — Directional slope at `t = 0`. Explicitly discarded
  (`del slope0`) since no acceptance test is performed.
- **`step_size`** *(keyword)* — The step to accept. Defaults to `1.0`.
- **`temperature`, `cooling`, `seed`** *(keyword)* — Accepted for
  interface compatibility with stochastic line searches; unused here.
- **`max_probes`** *(keyword)* — Capacity of the probe-record buffers.
- **`record_probes`** *(keyword)* — Whether to allocate full-size probe
  buffers (`max_probes`) or a minimal single-slot buffer.
- **`max_step`** *(keyword)* — Upper bound applied to `step_size`.

## Algorithm

The algorithm is deliberately trivial and consists of the following steps:

1. **Discard slope.** `slope0` is dropped since no sufficient-decrease or
   curvature condition is evaluated.

2. **Clip the step size.** Compute
   `alpha = min(step_size, max_step)`, cast to `value.dtype`. This is the
   only bounding applied.

3. **Evaluate once.** Call `eval_at(alpha)` to obtain
   `(new_params, new_val, new_g, slope)`. The returned slope is ignored.

4. **Record the probe.** Allocate probe buffers sized to
   `max_probes` (or `1` when `record_probes` is `False`) via
   `_empty_probes`, then store the single evaluated point at index `0`
   via `_record_probe`.

5. **Return.** Build and return a `LineSearchResult` reporting the
   accepted step, always with `done=True` and `num_evals=1`.

### Pseudocode

```text
alpha  ← min(step_size, max_step)         # cast to value dtype
(p, v, g, _) ← eval_at(alpha)
n      ← max_probes if record_probes else 1
probes ← empty_probes(params, n)
probes ← record_probe(probes, 0, p, g, v, alpha, n)
return LineSearchResult(
    step_size = alpha,
    new_value = v,
    new_grad  = g,
    new_params = p,
    done      = True,
    probes...,
    num_evals = 1,
)
```

## Complexity

- **Function evaluations:** exactly **1** (`eval_at` called once).
- **Time:** O(1) in the number of line-search iterations.
- **Memory:** O(`max_probes`) for the probe buffers (or O(1) when
  `record_probes=False`).

## Return Value

A `LineSearchResult` with:

- `step_size` — the clipped `alpha`.
- `new_value`, `new_grad`, `new_params` — the single evaluated point.
- `done` — always `True` (the search terminates immediately).
- `probe_params`, `probe_grads`, `probe_valid`, `probe_values`,
  `probe_alphas` — buffers holding the single recorded probe.
- `num_evals` — always `1`.

## Discussion

### Design rationale

Because `null_search` conforms to the standard `LineSearchResult`
interface and the `eval_at` calling convention, it can be dropped into
any solver expecting a line search. This makes it an ideal **fixed-step
optimizer driver**: pair it with a solver that computes a search
direction, and the effective update becomes `params + step_size * dir`
(as encoded by `eval_at`).

### JAX considerations

- The routine is written for compatibility with JAX transformations
  (`jit`, `vmap`, `grad`). It uses `jnp` operations and static-shaped
  probe buffers so the computation graph is fully traceable.
- `done` is returned as `jnp.asarray(True)` rather than a Python bool to
  remain a proper array under tracing.
- The step dtype is derived from `value.dtype` to avoid dtype-promotion
  surprises inside `jit`-compiled code.

### Unused parameters

`temperature`, `cooling`, and `seed` are part of the shared line-search
API used by stochastic variants. They are accepted here purely so that
callers can swap searches without changing call sites; the null search
ignores them entirely.

### Caveats

- There is **no guarantee of decrease** in the objective — the step is
  accepted blindly. Convergence of the outer optimizer depends entirely
  on `step_size`/`max_step` being appropriate for the problem.
- Setting `record_probes=False` reduces memory but still records the one
  evaluated point (buffer size `1`).
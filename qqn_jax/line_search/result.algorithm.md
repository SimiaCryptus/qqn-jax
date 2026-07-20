# `LineSearchResult` — Line Search Return Structure

## Overview

`LineSearchResult` is an immutable [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)
that packages the outcome of a **line search** step. A line search seeks a
scalar step size `α` along a search direction `d` such that the update

```
new_params = params + α · d
```

sufficiently reduces the objective function (e.g. satisfying the Armijo /
Wolfe conditions). This structure carries both the *primary* result of the
search and *auxiliary probe data* collected along the search path, which can
be fed back into an optimizer's curvature memory (for example, an oracle or
quasi-Newton estimator).

Being a `NamedTuple` of JAX arrays, the structure is a valid **JAX pytree**:
it can be passed through `jit`, `vmap`, `grad`, and `lax` control-flow
primitives, and its leaves are traced/transformed transparently.

## Fields

### Primary results

| Field        | Type          | Description                                                        |
|--------------|---------------|--------------------------------------------------------------------|
| `step_size`  | `jnp.ndarray` | The chosen step size `α`.                                          |
| `new_value`  | `jnp.ndarray` | Objective value at `params + α·d`.                                 |
| `new_grad`   | `jnp.ndarray` | Gradient of the objective at `params + α·d`.                       |
| `new_params` | `jnp.ndarray` | The updated parameters `params + α·d`.                             |
| `done`       | `jnp.ndarray` | Boolean flag: whether the search satisfied its stopping conditions.|

### Probe buffers (auxiliary, optional)

The probe fields form a **fixed-size** record of the points evaluated during
the search. Fixed sizing is essential in JAX: shapes must be statically known
for `jit` compilation and `lax` loops, so probe storage is pre-allocated to a
maximum count (`max_probes`) rather than grown dynamically. Unused slots are
masked out via `probe_valid`.

| Field          | Shape                | Description                                                              |
|----------------|----------------------|--------------------------------------------------------------------------|
| `probe_params` | `(max_probes, n)`    | Parameter points evaluated along the search path.                        |
| `probe_grads`  | `(max_probes, n)`    | Gradients at each probe point.                                           |
| `probe_valid`  | `(max_probes,)`      | Boolean mask marking which probe slots are filled/valid.                 |
| `probe_values` | `(max_probes,)`      | Objective values at each probe point.                                    |
| `probe_alphas` | `(max_probes,)`      | Step sizes `α` corresponding to each probe.                              |
| `num_evals`    | scalar               | Number of function/gradient evaluations performed during the search.     |

Here `n` is the dimensionality of the parameter vector.

All probe fields default to `None`, allowing the structure to be used for a
minimal result (just the primary fields) when probe collection is not needed.

## Design Rationale

### Why a `NamedTuple`?

- **Pytree compatibility.** JAX treats `NamedTuple` subclasses as pytrees,
  so the whole result flows through transformations and control flow without
  manual flattening.
- **Immutability.** Results cannot be accidentally mutated after creation,
  matching JAX's functional style.
- **Readability.** Named field access (`result.step_size`) is clearer than
  positional tuple indexing.

### Why fixed-size probe buffers?

JAX's tracing model requires static shapes. A line search may probe a
variable number of candidate step sizes, but the *storage* must be allocated
to a compile-time constant `max_probes`. The `probe_valid` mask distinguishes
genuinely filled slots from padding, so downstream consumers (e.g. curvature
memory updates) can ignore stale/unused entries.

### Feeding curvature memory

The `(probe_params, probe_grads)` pairs — together with `probe_values` and
`probe_alphas` — describe the local geometry of the objective along `d`.
Optimizers can reuse these evaluations to build secant/curvature estimates
(as in L-BFGS-style memory) without re-evaluating the function, improving
sample efficiency.

## Usage Notes

- When constructing a `LineSearchResult`, populate at least the primary
  fields. Probe fields may be left as `None` if unused.
- Consumers should always gate probe data on `probe_valid` to avoid reading
  padded slots.
- Because leaves are JAX arrays, avoid Python-level branching on `done`
  inside `jit`-compiled code; use `lax.cond` / `jnp.where` instead.

## Example

```python
from jax import numpy as jnp
from result import LineSearchResult

result = LineSearchResult(
    step_size=jnp.asarray(0.5),
    new_value=jnp.asarray(1.23),
    new_grad=jnp.array([0.1, -0.2]),
    new_params=jnp.array([0.9, 1.1]),
    done=jnp.asarray(True),
)

if bool(result.done):
    params = result.new_params
```
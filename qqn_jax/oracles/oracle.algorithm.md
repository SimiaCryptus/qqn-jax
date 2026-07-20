# Oracle Interface

## Overview

The `oracle.py` module defines a **pure, swappable oracle interface** used
to abstract the computation of a search direction in an optimization
routine. By separating *what direction to move* from *how far to move*
(the line search) and *how the iterate is updated*, the oracle becomes a
plug-in component that can implement anything from plain gradient descent to
quasi-Newton methods (e.g. L-BFGS) without changing the surrounding solver.

The module is intentionally minimal and functional: it contains only two
`NamedTuple` definitions and no concrete implementations. All behavior is
supplied by the callables the user places into an `Oracle` instance. This
keeps the interface **pure** (state is passed explicitly, never hidden) and
**swappable** (any object matching the tuple shape works).

## The `Oracle` Type

```python
class Oracle(NamedTuple):
    init: Callable[[Any], Any]
    direction: Callable[[Any, Any, Any], Tuple[Any, Any]]
    update: Callable[[Any, Any], Any]
```

An `Oracle` bundles three pure functions that together describe a
direction-generating strategy.

### `init(params) -> oracle_state`

Constructs the initial oracle state given the starting `params`. For a
**stateless** oracle (such as steepest descent) this should simply return
`()`. For a **stateful** oracle (such as L-BFGS, which stores a history of
curvature pairs) this returns whatever data structure the oracle needs to
carry between iterations.

### `direction(params, grad, state) -> (direction, new_state)`

Computes the search **direction** at the current iterate. The returned
`direction` is the `t = 1` endpoint of the step, i.e. it represents
`-H∇f`, where `H` is the (implicit or explicit) inverse-Hessian
approximation:

- For steepest descent, `H = I`, so `direction = -grad`.
- For quasi-Newton methods, `H` approximates the inverse Hessian, so the
  returned direction incorporates curvature information.

The function is pure: it takes the current `state` and returns a
(potentially) updated `new_state` alongside the direction. This allows the
oracle to record intermediate computations without mutation.

### `update(state, info) -> state`

Called **after a step has been accepted** by the solver. It receives the
existing `state` and an `OracleInfo` record describing the accepted step,
and returns the updated state. For stateless oracles this is a **no-op**
(return `state` unchanged). For stateful oracles this is where new curvature
information (e.g. the `(s, y)` pair in L-BFGS) is folded into the history.

## The `OracleInfo` Type

```python
class OracleInfo(NamedTuple):
    params: Any = None
    new_params: Any = None
    grad: Any = None
    new_grad: Any = None
    t: Any = None
    step_size: Any = None
    probe_params: Any = None
    probe_grads: Any = None
    probe_valid: Any = None
    probe_alphas: Any = None
```

`OracleInfo` is the payload passed to `Oracle.update` once the solver has
committed to a step. Every field defaults to `None`, so a caller only needs
to populate the fields relevant to a given oracle.

### Core step fields

| Field        | Meaning                                            |
|--------------|----------------------------------------------------|
| `params`     | Iterate `x` **before** the step.                   |
| `new_params` | Accepted iterate `x_new`.                          |
| `grad`       | Gradient `∇f(x)` **before** the step.              |
| `new_grad`   | Gradient `∇f(x_new)` **after** the step.           |
| `t`          | Chosen interpolation parameter along the direction.|
| `step_size`  | Accepted step size `α`.                            |

From these, a quasi-Newton oracle can form the standard curvature pair:

- `s = new_params - params`
- `y = new_grad - grad`

### Probe buffers

The `probe_*` fields optionally carry information gathered by the line
search while probing candidate points along the direction. These enable
richer oracles that learn from *all* function/gradient evaluations, not just
the two endpoints.

| Field          | Shape / Type      | Meaning                              |
|----------------|-------------------|--------------------------------------|
| `probe_params` | `(k, n)` buffer   | Line-search probe points.            |
| `probe_grads`  | `(k, n)` buffer   | Gradients at those probe points.     |
| `probe_valid`  | `(k,)` bool mask  | Which probe slots are filled.        |
| `probe_alphas` | `(k,)`            | Step sizes associated with probes.   |

Here `k` is the maximum number of probe slots and `n` is the parameter
dimensionality. The `probe_valid` mask lets consumers ignore unused slots in
a fixed-size (e.g. JIT-friendly) buffer.

## Design Rationale

- **Purity.** State is threaded explicitly through `init`, `direction`, and
  `update`. No global or hidden mutable state exists, which makes the
  interface deterministic, easy to test, and compatible with functional
  transformation frameworks (such as JAX's `jit`/`vmap`).
- **Swappability.** Because `Oracle` is just a tuple of callables, different
  strategies can be constructed and substituted freely. A solver written
  against this interface never needs to know which concrete oracle it uses.
- **Fixed-size buffers.** The probe fields are described as fixed-shape
  buffers with a validity mask rather than variable-length lists. This is a
  deliberate accommodation for accelerator/compiler backends that require
  static shapes.

## Typical Usage Pattern

A solver consuming an oracle follows this loop:

1. `state = oracle.init(params)` — build initial oracle state.
2. Compute `grad = ∇f(params)`.
3. `direction, state = oracle.direction(params, grad, state)`.
4. Run a line search along `direction` to choose `t` / `step_size`,
   optionally recording probe points.
5. Form `new_params` and `new_grad` for the accepted step.
6. Assemble an `OracleInfo` and call
   `state = oracle.update(state, info)`.
7. Repeat from step 2 until convergence.

## Example: Stateless Steepest Descent

```python
steepest_descent = Oracle(
    init=lambda params: (),
    direction=lambda params, grad, state: (-grad, state),
    update=lambda state, info: state,
)
```

## Example: Stateful Skeleton (e.g. L-BFGS)

```python
def make_lbfgs(history_size):
    def init(params):
        return empty_history(history_size)

    def direction(params, grad, state):
        d = two_loop_recursion(state, grad)  # -H∇f
        return d, state

    def update(state, info):
        s = info.new_params - info.params
        y = info.new_grad - info.grad
        return push_pair(state, s, y)

    return Oracle(init=init, direction=direction, update=update)
```

## Summary

`oracle.py` provides a compact, functional contract for pluggable search
direction strategies. `Oracle` defines the three lifecycle functions
(`init`, `direction`, `update`), and `OracleInfo` carries the post-step
information — including optional line-search probe buffers — needed by
stateful oracles to refine their curvature approximations.
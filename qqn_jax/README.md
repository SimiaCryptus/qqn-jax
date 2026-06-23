# QQN-JAX

**Quasi-Quadratic-Newton (QQN)** optimizer for [JAX](https://github.com/google/jax).

QQN blends the steepest-descent direction (`-∇f`) and a curvature-aware
quasi-Newton direction (`-H∇f`, by default from L-BFGS) along a single
**quadratic interpolation path**:

```
d(t) = t(1 - t)·(-∇f) + t²·(-H∇f)
```

A line search then traverses this path over the parameter `t ∈ [0, 1]`,
selecting one point `x + d(t)` as the next iterate. At `t = 0` the path's
tangent is the steepest-descent direction; at `t = 1` the endpoint is the
pure quasi-Newton step. The quadratic blend gives a smooth, robust
interpolation between first- and second-order behavior.

Everything in QQN-JAX is written as pure, functional JAX so the entire
optimization composes with `jit`, `vmap`, `pmap`, and `grad`. The solver
uses `lax.while_loop` internally, so a full optimization run is itself a
single traceable, differentiable, vmappable operation.

---

## Features

- **JIT / vmap / pmap / grad compatible** — the whole solver is traceable.
- **JAXopt-style interface** — `init_state` / `update` / `run`.
- **Swappable oracles** — choose how the `t = 1` endpoint `-H∇f` is built:
  - `"lbfgs"` (default) — limited-memory BFGS two-loop recursion.
  - `"momentum"` — heavy-ball / exponentially-weighted gradient.
  - `"secant"` — Barzilai-Borwein step (matrix-free, O(n) memory).
  - `"shampoo"` — structure-aware preconditioning.
  - `"anderson"` — Anderson (Type-II) acceleration.
  - `"anderson+secant"`, `"lbfgs+secant"` — safeguarded fallbacks.
  - or any custom `Oracle` instance.
- **Pluggable line searches** — `"armijo"` (default), `"backtracking"`,
`"strong_wolfe"`, `"hager_zhang"`, `"fixed"`, `"spline"`.
- **Cubic Hermite spline refinement** (`spline=True`) — reuses every probe
on the consistent path as a control point and improves on the inner
search's accepted step. Composes with any line search.
- **Projective regions** — constrain / remap each step:
  - `IdentityRegion` (default, zero overhead),
  - `BoxRegion` (elementwise bounds),
  - `OrthantRegion` (OWL-QN-style sparsity),
  - `TrustRegion` (adaptive `‖x_new − x‖₂ ≤ Δ`),
  - `NoDecreaseRegion` (protect a secondary objective),
  - `Sequential` (compose regions).
- **Probe feeding** — optionally fold every gradient evaluated *during* the
line search into the oracle's curvature memory (gated on genuine descent).

---

## Profiling (Perfetto / JAX Profiler / Scalene)
QQN-JAX ships a light profiling facade in `qqn_jax/profiling.py` that wires
three complementary backends into any benchmark via environment variables:

| Backend  | What it captures                          | View with                          |
|----------|-------------------------------------------|------------------------------------|
| JAX      | Device + host op traces (`jax.profiler`)  | TensorBoard Trace Viewer           |
| Perfetto | Same trace in Perfetto protobuf form      | https://ui.perfetto.dev            |
| Scalene  | Whole-process CPU + GPU + memory sampling | `scalene` HTML/CLI report          |

Enable via the `PROFILE` env var (`jax`, `perfetto`, `scalene`, or `all`):

```bash
# Capture a Perfetto-loadable trace into ./profiles:
PROFILE=jax,perfetto PROFILE_DIR=profiles \
     python examples/fashion_mnist_mlp_comparison.py
# Whole-process CPU+GPU+memory profile (Scalene wraps the interpreter):
scalene examples/fashion_mnist_mlp_comparison.py
# All backends at once:
PROFILE=all python examples/fashion_mnist_mlp_comparison.py
```

In code:
```python
from qqn_jax.profiling import profile_session, profile_region
with profile_session("my_run"):          # whole-run device+host trace
     with profile_region("QQN-L80"):      # named span in the Perfetto timeline
         solver.run(x0)
```
Each optimizer variant in the comparison benchmark is automatically wrapped
in a `profile_region`, so the Perfetto timeline cleanly separates the work
done by each method.

---


## Installation

```bash
pip install qqn-jax        # (or install from source)
```

Requires `jax`, `jaxlib`, `optax`, `chex`, and `jaxtyping`.

---

## Quick start

```python
import jax.numpy as jnp
from qqn_jax import QQN

# A simple quadratic objective.
def rosenbrock(x):
  return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

x0 = jnp.zeros(10)

solver = QQN(rosenbrock, maxiter=200, tol=1e-6)
x_opt, state = solver.run(x0)

print("solution:", x_opt)
print("final value:", state.value)
print("iterations:", state.iter)
print("grad norm:", state.error)
```

---

## Usage

### Choosing an oracle

```python
# Default L-BFGS with custom memory size.
QQN(fun, history_size=20)

# A different oracle by name.
QQN(fun, oracle="secant")
QQN(fun, oracle="anderson")

# Safeguarded fallback: deep curvature, with a featherweight backup
# for when the primary curvature estimate degenerates.
QQN(fun, oracle="lbfgs+secant")

# A custom Oracle instance.
from qqn_jax import AndersonOracle
QQN(fun, oracle=AndersonOracle(window=8, beta=1.5))
```

### Choosing a line search

```python
QQN(fun, line_search="armijo")        # default; robust efficiency winner
QQN(fun, line_search="backtracking")
QQN(fun, line_search="strong_wolfe")
QQN(fun, line_search="hager_zhang")
QQN(fun, line_search="fixed")

# Forward extra keyword arguments to the inner line search.
QQN(fun, line_search="backtracking",
  line_search_options={"c1": 1e-3, "shrink": 0.6, "max_iter": 10})
```

> **Note:** `"strong_wolfe"` can over-restrict the quadratic-path step and
> fail to converge on some problems; the Armijo / backtracking family is the
> recommended default for smooth, full-batch objectives.

### Spline refinement

Orthogonal to the line search: each probe along the consistent path is
reused as a control point of a cubic Hermite spline, whose stationary
points are probed to improve on the inner search's accepted step.

```python
QQN(fun, line_search="backtracking", spline=True)
# Equivalent shorthand:
QQN(fun, line_search="spline")
```

### Constrained / projected optimization with regions

```python
from qqn_jax import BoxRegion, TrustRegion, Sequential

# Bound the iterate elementwise to [-1, 1].
QQN(fun, region=BoxRegion(lo=-1.0, hi=1.0))

# Adaptive trust region on the step length.
QQN(fun, region=TrustRegion(radius=1.0, adaptive=True))

# Compose multiple regions (applied in order).
QQN(fun, region=Sequential([TrustRegion(radius=2.0),
                          BoxRegion(lo=0.0)]))
```

### Auxiliary outputs

If your objective returns `(value, aux)`:

```python
def fun_with_aux(x):
  loss = ...
  aux = {"some_metric": ...}
  return loss, aux

solver = QQN(fun_with_aux, has_aux=True)
x_opt, state = solver.run(x0)
print(state.aux)
```

### Feeding line-search probes to the oracle

Every gradient evaluated during the line search can enrich the oracle's
curvature memory — gated, by default, on genuine objective decrease so that
non-representative probes never pollute the history:

```python
QQN(fun, oracle="lbfgs",
  feed_probes_to_oracle=True,
  probe_descent_gate=True,   # only admit strictly-improving probes
  max_probes=32)
```

### vmap over a batch of initializations

Because the whole solver is traceable, you can solve a batch of problems at
once:

```python
import jax

batch_x0 = jnp.stack([x0_a, x0_b, x0_c])      # (B, n)
solver = QQN(fun, maxiter=100)
run = jax.jit(jax.vmap(solver.run))
xs, states = run(batch_x0)
```

---

## The `QQN` interface

| Method                       | Description                                              |
|------------------------------|----------------------------------------------------------|
| `init_state(params, *args)`  | Build the initial `QQNState` at `params`.                |
| `update(params, state, *args)` | Perform one QQN iteration → `(new_params, new_state)`. |
| `run(init_params, *args)`    | Run to convergence (or `maxiter`) → `(params, state)`.   |

### Constructor arguments

| Argument                 | Default     | Description                                              |
|--------------------------|-------------|----------------------------------------------------------|
| `fun`                    | —           | Objective `f(params, *args) -> scalar` (or `(scalar, aux)`). |
| `maxiter`                | `100`       | Maximum number of iterations.                            |
| `tol`                    | `1e-5`      | Convergence tolerance on the gradient L2 norm.           |
| `history_size`           | `10`        | L-BFGS memory size `m`.                                  |
| `line_search`            | `"armijo"`  | Line-search strategy name.                               |
| `line_search_options`    | `None`      | Dict of kwargs forwarded to the line search.             |
| `spline`                 | `False`     | Enable cubic Hermite spline refinement.                  |
| `has_aux`                | `False`     | Whether `fun` returns auxiliary data.                    |
| `region`                 | `None`      | A `Region` (or `None` for the identity region).          |
| `oracle`                 | `"lbfgs"`   | Oracle name or `Oracle` instance.                        |
| `feed_probes_to_oracle`  | `False`     | Feed line-search probes into oracle curvature memory.    |
| `probe_descent_gate`     | `True`      | Only admit strictly-improving probes when feeding.       |
| `max_probes`             | `32`        | Probe-buffer capacity.                                   |

### `QQNState`

| Field          | Description                                       |
|----------------|---------------------------------------------------|
| `iter`         | Iteration counter.                                |
| `value`        | Current objective value.                          |
| `grad`         | Current gradient.                                 |
| `oracle_state` | Oracle state (e.g. L-BFGS history).               |
| `step_size`    | Last accepted step size `α` (path parameter `t`). |
| `error`        | Gradient L2 norm (convergence metric).            |
| `done`         | Whether convergence (or divergence) was reached.  |
| `aux`          | Optional auxiliary output of the objective.       |
| `region_state` | Optional projective-region state.                 |

---

## Project layout

| Module               | Purpose                                                     |
|----------------------|-------------------------------------------------------------|
| `solver.py`          | The `QQN` optimizer and `QQNState`.                         |
| `oracles.py`         | Oracle abstraction and implementations (`-H∇f` endpoint).   |
| `lbfgs.py`           | L-BFGS two-loop recursion and curvature-history buffers.    |
| `line_search.py`     | Line-search strategies (Armijo, Wolfe, Hager-Zhang, …).     |
| `spline_search.py`   | Cubic Hermite spline refinement wrapper.                    |
| `regions.py`         | Projective regions (box, orthant, trust region, …).         |
| `utils.py`           | Pytree / tree-math helpers.                                 |
| `types.py`           | Typed interfaces (`chex` / `jaxtyping`).                    |

---

## Design notes

- **The path is the search space.** The line search traverses the path
parameter `t` directly. Each evaluated `x + d(t)` is a *state*, not a
direction to be independently re-scaled. Rescaling the gradient (or the
oracle direction) does **not** change the geometric path traced by `d(t)`
— it only reparameterizes how `t` maps onto arc length.
- **Honest predicted reduction.** The along-path quadratic model has a
closed-form reduction `pred(t) = -⟨∇f, d(t)⟩`, used directly for the
trust-region acceptance ratio `ρ = ared / pred` with no double-counted
curvature term.
- **NaN-safety everywhere.** Curvature reciprocals and matrix solves are
guarded so that masked-out branches never backpropagate NaNs under
`jax.grad`.
- **Divergence termination.** A run terminates early if an iterate becomes
non-finite, so a single bad start in a vmapped batch does not waste the
rest of the batch's iterations on NaN arithmetic.
- **Honest eval counting.** ``QQNState.num_evals`` accumulates every
value-and-grad evaluation (line-search probes, spline probes, aux
recomputes, fallback recoveries) so benchmarks compare *work done*, not
just iteration counts — QQN performs several evaluations per iteration.
Strong-Wolfe / Hager-Zhang counts are conservative upper bounds because
Optax does not expose its internal probe count.

---

## License

See the repository for license details.
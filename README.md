# qqn-jax

**Quadratic Quasi-Newton (QQN)** — a JAX/Optax optimizer that blends steepest descent with a quasi-Newton oracle (L-BFGS
by default) along a smooth quadratic path, navigated by a robust line search. **No learning rate to tune.**

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

- `d(0) = 0` — path starts at the current point.
- `d'(0) = -∇f` — path begins tangent to gradient descent (globalization).
- `d(1) = -H∇f` — path ends at the pure oracle step (speed).

A single line search picks the interpolation `t` and step size `α`
**together**, discovering the right first-/second-order blend at every iteration.

---

## Contents

- [Install & Import](#install--import)
- [Quick Start](#quick-start)
- [The API](#the-api)
- [JAX Transforms](#jax-transforms)
- [Configuration](#configuration)
- [Theory in Brief](#theory-in-brief)
- [When to Use QQN](#when-to-use-qqn)
- [License](#license)

---

## Install & Import

```python
from qqn_jax import QQN
```

QQN is built on **JAX** and **Optax** (plus `chex`, `jaxtyping`). For GPU, install the matching CUDA wheel of `jaxlib`.

---

## Quick Start

```python
import jax.numpy as jnp
from qqn_jax import QQN


# Rosenbrock function
def fun(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


solver = QQN(fun, maxiter=100, tol=1e-6)
init = jnp.array([-1.2, 1.0])
params, state = solver.run(init)

print(params)  # ~ [1.0, 1.0]
print(state.value)  # ~ 0.0
print(state.iter)  # iterations taken
print(state.error)  # final gradient L2 norm
```

---

## The API

QQN follows a JAXopt-style `init_state` / `update` / `run` interface:

| Method                         | Description                                            |
|--------------------------------|--------------------------------------------------------|
| `init_state(params, *args)`    | Build the initial `QQNState` at `params`.              |
| `update(params, state, *args)` | One QQN iteration → `(new_params, new_state)`.         |
| `run(init_params, *args)`      | Run to convergence (or `maxiter`) → `(params, state)`. |

Manual loop (equivalent to `run`):

```python
solver = QQN(fun, maxiter=100, tol=1e-6)
state = solver.init_state(init)
params = init
for _ in range(solver.maxiter):
    params, state = solver.update(params, state)
    if state.error < solver.tol:
        break
```

`*args` are extra positional arguments forwarded to `fun`. Use
`has_aux=True` if `fun` returns `(value, aux)`.

---

## JAX Transforms

The whole solver is functional JAX (`lax.while_loop` internally), so a full run is a single traceable, differentiable,
vmappable operation:

```python
import jax

# JIT-compiled solve (XLA + GPU/TPU dispatch)
run_jit = jax.jit(QQN(fun).run)
params, state = run_jit(init)

# Batched over many starting points — solve a whole batch at once.
batched = jax.vmap(QQN(fun).run, in_axes=(0,))
params_batch, states = batched(init_batch)
```

A run terminates early if an iterate becomes non-finite, so one bad start in a `vmap` batch does not waste the rest of
the batch's iterations.

---

## Configuration

```python
QQN(
    fun,
    maxiter=100,
    tol=1e-5,
    history_size=10,  # L-BFGS memory size m
    line_search="armijo",  # "armijo" | "backtracking" | "strong_wolfe"
    # | "hager_zhang" | "fixed" | "spline"
    line_search_options=None,  # dict of kwargs for the line search
    spline=False,  # cubic-Hermite spline refinement
    has_aux=False,
    oracle="lbfgs",  # "lbfgs" | "momentum" | "secant"
    # | "shampoo" | "anderson" | ... | Oracle
    region=None,  # Region | None
)
```

With all defaults, QQN is a tightly-coupled gradient + L-BFGS optimizer with an Armijo backtracking line search.

### Oracles — the `t = 1` endpoint `-H∇f`

| Name                | Endpoint                                           |
|---------------------|----------------------------------------------------|
| `"lbfgs"` (default) | limited-memory BFGS two-loop recursion             |
| `"momentum"`        | heavy-ball / exponentially-weighted gradient       |
| `"secant"`          | Barzilai-Borwein step (matrix-free, `O(n)` memory) |
| `"shampoo"`         | structure-aware preconditioning                    |
| `"anderson"`        | Anderson (Type-II) acceleration                    |

Oracles compose. `Fallback` uses the first valid (descending) direction and otherwise falls back to the next:

```python
from qqn_jax.oracles import LBFGSOracle, MomentumOracle, Fallback

oracle = Fallback([
    LBFGSOracle(history_size=10),
    MomentumOracle(beta=0.9),
])
solver = QQN(fun, oracle=oracle)
```

### Line searches

```python
QQN(fun, line_search="armijo")  # default; robust efficiency winner
QQN(fun, line_search="backtracking")
QQN(fun, line_search="strong_wolfe")
QQN(fun, line_search="hager_zhang")
QQN(fun, line_search="fixed")

# Forward extra kwargs to the inner search:
QQN(fun, line_search="backtracking",
    line_search_options={"c1": 1e-3, "shrink": 0.6, "max_iter": 10})
```

> `"strong_wolfe"` can over-restrict the quadratic-path step; the Armijo /
> backtracking family is the recommended default for smooth, full-batch
> objectives.

### Regions — constrain / remap the search onto a feasible set

| Region             | Effect                                      |
|--------------------|---------------------------------------------|
| `IdentityRegion`   | default, zero overhead                      |
| `BoxRegion`        | elementwise bounds `[lo, hi]`               |
| `OrthantRegion`    | OWL-QN-style L1 sparsity                    |
| `TrustRegion`      | adaptive `‖x_new − x‖₂ ≤ Δ`                 |
| `NoDecreaseRegion` | protect a secondary objective               |
| `Sequential`       | compose multiple regions (applied in order) |

```python
from qqn_jax.regions import BoxRegion, TrustRegion, Sequential

region = Sequential([
    BoxRegion(lo=0.0, hi=1.0),
    TrustRegion(radius=0.5),
])
solver = QQN(fun, region=region)
```

The line search navigates the *projected* path
`d_R(t) = project_R(x, x + d(t)) - x`, so descent/Wolfe guarantees stay meaningful on the feasible set. `region=None` is
identical to the unconstrained optimizer.

### Spline refinement

Reuses each line-search probe (value **and** slope) as a cubic-Hermite control point, then probes the spline's
stationary points to improve on the accepted step:

```python
QQN(fun, line_search="backtracking", spline=True)
QQN(fun, line_search="spline")  # equivalent shorthand
```

A spline candidate is accepted only if it strictly improves fitness, so it inherits the inner search's descent
guarantee.

---

## Theory in Brief

**The reframing.** Classical methods commit to *one* direction per iteration (gradient, momentum, or quasi-Newton) and
line-search along it. QQN refuses the binary choice: it builds a continuous curve connecting the two and turns *"which
direction?"* into *"where on the curve?"*. The 1-D search over `t`
replaces a discrete direction choice with a globally-anchored interpolation.

**The four axes.** QQN factors an optimizer into four orthogonal, swappable strategies — **gradient**, **oracle**
(`-H∇f` at `t=1`), **search** (line search over `t`), and **region** (feasible-set projection). Because
`d'(0) = -∇f` always holds, the oracle need not guarantee descent on its own — it can be aggressive.

**Guarantees** (given a sufficient-decrease line search):

- *Global convergence* — from the steepest-descent tangent at `t = 0`, **regardless of oracle quality**. A decreasing
  step always exists.
- *Superlinear convergence* — near the optimum, when the selected `t → 1`
  and the L-BFGS direction dominates.
- *Descent* — every accepted step decreases `f`, enforced by the search.

**`C⁰` is enough.** Monotone progress needs only continuity along the path, since the sufficient-decrease test compares
function *values*. Smoothness sharpens rate proofs but is not required for descent — making QQN suited to
piecewise-smooth objectives (ReLU, max-pool, hinge/L1).

**Classical methods are special cases.** Gradient descent (`t → 0`), L-BFGS (`t = 1`, default oracle), Newton
(exact-Hessian oracle, `t = 1`), momentum / Barzilai-Borwein / Anderson (oracle choices), trust-region / OWL-QN /
projected gradient (region choices), and conjugate gradient (CG-as-oracle)
all emerge as configurations of the four axes.

---

## When to Use QQN

QQN is **not** a drop-in replacement for Adam everywhere. It earns its keep on **ill-curved, anisotropic, full-batch**
landscapes where a robust line search is affordable.

| Situation                                                    | Prefer           |
|--------------------------------------------------------------|------------------|
| Large-scale, noisy, stochastic minibatch training            | **Adam**         |
| Tight memory budget, very high dimension                     | **Adam / SGD**   |
| Smooth, full-batch, ill-conditioned objective                | **QQN**          |
| Complex / anisotropic curvature where step tuning is brittle | **QQN**          |
| Parameter-free, self-tuning blend of GD and L-BFGS           | **QQN**          |
| Bound / orthant / trust constraints alongside curvature      | **QQN + region** |

On a 4-layer MLP (335k params) trained full-batch on Fashion-MNIST, QQN with a deep L-BFGS oracle reaches the `2e-2`
loss target in **2.64× fewer iterations** than standalone L-BFGS *and* cheaper per iteration (Armijo needs
~1.0–1.1 evals/it vs ~2.1 for the Optax zoom search inside L-BFGS). The speedup widens as the target tightens.

---

## License

[Apache 2.0](LICENSE)
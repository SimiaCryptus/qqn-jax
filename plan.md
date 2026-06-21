# QQN JAX Implementation Plan

## Overview

QQN will be implemented as a standalone PyPI package (`qqn-jax`) with JAX and JAXopt as dependencies. This approach gives full control over API design, release cadence, and documentation while remaining fully compatible with the JAX ecosystem.

## Package Structure

```
qqn_jax/
├── __init__.py
├── solver.py        # QQN solver (QQN class + QQNState)
├── line_search.py   # Wrappers around JAXopt line searches
├── lbfgs.py         # Wrappers around JAXopt LBFGS oracle
├── types.py         # chex/jaxtyping type definitions
├── utils.py         # Shared utilities
tests/
examples/
pyproject.toml
README.md
```

## Core Components

### 1. QQNState (NamedTuple)

Immutable state container, JIT-compatible:

```python
class QQNState(NamedTuple):
    iter: int
    value: float
    grad: jnp.ndarray
    lbfgs_state: LBFGSState
    step_size: float
    aux: Any
```

### 2. QQN Solver Class

Implements the JAXopt solver interface (`init_state`, `update`, `run`):

```python
class QQN(jaxopt.OptStep):
    def init_state(self, params, *args): ...
    def update(self, params, state, *args): ...
```

### 3. Quadratic Path Construction

Inside `update`, the three core components are assembled:

```python
# 1. Oracle: L-BFGS direction
_, lbfgs_state = self.lbfgs.update(params, state.lbfgs_state, *args)
qn_dir = lbfgs_state.direction

# 2. Gradient: steepest descent direction
grad_dir = -state.grad

# 3. Quadratic path: d(t) = t(1-t)(-∇f) + t²(-H∇f)
d = t * (1 - t) * grad_dir + t**2 * qn_dir

# 4. Search: line search over d(t)
alpha = self.ls.search(self.fun, params, d, *args)

# 5. Step
new_params = params + alpha * d
```

## Dependencies

```toml
[project]
dependencies = [
    "jax>=0.4",
    "jaxopt>=0.8",
    "chex>=0.1",
    "jaxtyping>=0.2",
]
```

## Design Constraints

- **All state in NamedTuples**: JIT-compatible, immutable, functional
- **No Python-level loops**: All iteration via JAX-compatible control flow (`lax.while_loop`, `lax.cond`)
- **Strongly typed**: All arrays typed via `chex.Array` / `jaxtyping`
- **Pure functions**: `init_state` and `update` are stateless and side-effect-free
- **Reuse JAXopt internals**: LBFGS oracle and line search are delegated to JAXopt, not reimplemented

## Line Search Strategy

The line search is a first-class component, not an implementation detail. It is responsible for:

1. Selecting `t` — the interpolation parameter (how much to trust gradient vs. oracle)
2. Selecting `α` — the step size along `d(t)`
3. Enforcing sufficient decrease (Armijo / Wolfe conditions)
4. Ensuring curvature updates fed back to L-BFGS remain valid (Strong Wolfe)

Default: Strong Wolfe line search (from JAXopt).
Fallback: Backtracking with Armijo condition.

## JAX Acceleration

Because QQN is implemented inside JAX's functional model, it automatically supports:

- `jax.jit` — XLA compilation and GPU/TPU dispatch
- `jax.vmap` — batching over multiple starting points
- `jax.pmap` — multi-device parallelism
- Differentiation through the optimizer (where applicable)

## Usage (Target API)

```python
from qqn_jax import QQN

# Basic usage
solver = QQN(fun, maxiter=100)
params, state = solver.run(init_params)

# JIT-compiled
solver_jit = jax.jit(QQN(fun).run)
params, state = solver_jit(init_params)

# Batched over initial points
batched = jax.vmap(QQN(fun).run, in_axes=(0,))
params, states = batched(init_params_batch)
```

## Implementation Phases

### Phase 1 — Core Solver
- [ ] `types.py`: Define `QQNState` and typed interfaces
- [ ] `solver.py`: Implement `QQN.init_state` and `QQN.update`
- [ ] `lbfgs.py`: Wrap JAXopt LBFGS as a single-step oracle
- [ ] `line_search.py`: Wrap JAXopt Strong Wolfe and Armijo line searches

### Phase 2 — Testing
- [ ] Unit tests for `init_state` and `update`
- [ ] JIT compilation smoke tests
- [ ] Convergence tests on standard benchmark functions (Rosenbrock, quadratics)
- [ ] Comparison benchmarks: QQN vs LBFGS vs GradientDescent

### Phase 3 — Packaging
- [ ] `pyproject.toml` with correct metadata and dependencies
- [ ] `README.md` with algorithm description and usage examples
- [ ] PyPI publish via `hatch` or `flit`

### Phase 4 — Documentation
- [ ] API reference (auto-generated from docstrings)
- [ ] Algorithm explainer (based on `algorithm.md`)
- [ ] Benchmark results and comparison plots


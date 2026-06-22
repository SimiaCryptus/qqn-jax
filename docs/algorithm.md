# QQN (Quasi-Quadratic-Newton) Algorithm Technical Documentation

## Overview

The QQN (Quasi-Quadratic-Newton) algorithm is a novel optimization method that
combines the robustness of steepest descent with the efficiency of L-BFGS through
a unique quadratic interpolation scheme. This implementation provides a
sophisticated approach to unconstrained (and, via projective regions, lightly
constrained) optimization that adaptively blends gradient descent and
quasi-Newton directions.

This document is the **comprehensive reference** for the complete algorithm as
implemented in `qqn-jax`. It covers the conceptual model, the mathematical
construction of the quadratic path, the four extension axes (gradient, oracle,
region, search), the concrete numerical procedures, the solver loop, and the
theoretical and practical guarantees. Companion documents drill into individual
components:

- [`oracles.md`](oracles.md) — the oracle abstraction and concrete oracles.
- [`regions.md`](regions.md) — projective regions for feasibility/preference.
- [`spline_search.md`](spline_search.md) — the cubic Hermite spline line search.

## Conceptual Role: A Combiner for Gradient + Oracle + Search

At its core, QQN is best understood as a **combiner** that unifies three
fundamental components of numerical optimization, with a fourth (regions) layered
on top as an optional projection:

1. **Gradient** (steepest descent): The raw signal from `-∇f(x)`, providing a
   reliable, locally valid descent direction.
2. **Oracle** (L-BFGS quasi-Newton by default): A learned approximation of
   curvature (`-H∇f(x)`), acting as a black-box oracle that encodes second-order
   information from historical gradient differences.
3. **Search** (line search strategy): The mechanism that traverses the quadratic
   path `d(t)` and selects a step that guarantees sufficient descent.
4. **Region** (projective constraint, optional): A pure projection
   `project_R(x, x + d(t))` that remaps each candidate point onto a feasible or
   preferred set, so the search navigates the *projected* path.

These components are not merely combined additively — the quadratic path
construction means the **search strategy is the glue** that makes the gradient and
oracle work together coherently. Without a robust line search, the interpolation
between directions has no principled stopping criterion and the algorithm loses
its convergence guarantees entirely.

The four components are **conceptually orthogonal and independently swappable**.
The solver threads their state through a single immutable `QQNState` and exposes
each as a configuration point, so alternative oracles, regions, or search
strategies can be substituted without touching the rest of the algorithm.

## Algorithm Description

### Core Concept

QQN operates by constructing a quadratic path between two search directions:

1. **Steepest descent direction**: `-∇f(x)` (negative gradient)
2. **Oracle direction**: `-H∇f(x)` (the quasi-Newton direction with approximate
   inverse Hessian `H`, supplied by the oracle's `t = 1` endpoint)

The algorithm searches along a parametric curve defined by:

```
d(t) = t(1-t)(-∇f) + t²(-H∇f)
```

where `t ∈ [0, 1]` is the interpolation parameter.

### The Quadratic Path: Geometry and Endpoints

The path `d(t)` is a vector-valued quadratic in `t`. Expanding:

```
d(t) = -t(1-t)·∇f - t²·H∇f
     = -t·∇f + t²·∇f - t²·H∇f
     = -t·∇f + t²·(∇f - H∇f)
```

Its endpoints and tangent at the origin are the key to its behavior:

- **`d(0) = 0`**: the path starts at the current iterate `x`.
- **`d'(0) = -∇f`**: the *initial tangent* of the path is exactly the steepest
  descent direction. This is the crucial property — for small `t`, moving along
  the path is moving along `-∇f`, so the path is guaranteed to begin as a descent
  direction whenever `∇f ≠ 0`.
- **`d(1) = -H∇f`**: at `t = 1` the path arrives exactly at the oracle (L-BFGS)
  direction.

Because `d'(0) = -∇f`, the directional derivative of `f` along the path at the
origin is `⟨∇f, d'(0)⟩ = -‖∇f‖² ≤ 0`. This is what anchors QQN's global
convergence: regardless of how poor the oracle direction is, the *beginning* of
the path always decreases `f`.

### Key Properties

- **t = 0**: Pure steepest descent direction (the path's tangent).
- **t = 1**: Pure oracle / L-BFGS direction.
- **0 < t < 1**: A smooth quadratic blend, weighting the gradient by `t(1-t)` and
  the oracle by `t²`.

This formulation ensures:

- The direction is always a descent direction for small enough steps (since
  `d'(0) = -∇f`).
- A smooth transition between conservative (gradient) and aggressive
  (quasi-Newton) steps.
- Adaptive behavior based on problem characteristics, discovered by the search
  rather than hand-tuned.




### Searching the Path Directly

The points along `d(t)` are **states**, not directions to be second-guessed by
a separate inner line search. The line search therefore traverses the path
parameter `t ∈ [0, 1]` *directly*: at each iteration a single search walks the
curve, evaluating candidate states `x + d(t)` and selecting the step `t` (and
hence the state) that satisfies sufficient decrease. There is no discretized
"blend grid" and no per-grid-point inner search — the curve itself is the
one-dimensional search space.

#### Invariance to gradient rescaling

Rescaling the gradient (or the oracle direction) does **not** change the
geometric path that `d(t)` traces through parameter space. Scaling a direction
only **distorts the parameterization** along the path — i.e. it changes how the
scalar `t` maps onto arc length — but the set of points (states) on the curve,
and therefore the candidates the line search can reach, is invariant. This is
why it is meaningless to "re-search" a chosen `t` with an inner line search that
rescales `d(t)`: doing so would only re-walk the very same curve under a
different clock.

### The Line Search Strategy: The Critical Component

**The line search is not an implementation detail — it is a first-class
algorithmic component** and the mechanism by which QQN's theoretical properties
are realized in practice.

The line search traverses the quadratic path `d(t)` over `t ∈ [0, 1]` and must:

- **Select the path parameter `t`**: Walk the curve to satisfy sufficient
   decrease conditions (e.g., Armijo/Wolfe conditions). Each probe `x + d(t)` is
   a *state* on the path, not a direction to be re-scaled.
- **Enforce descent**: Guarantee that `f(x + d(t)) < f(x)` (or report
  failure), which is the foundation of global convergence.
- **Exploit curvature**: A strong Wolfe condition on the line search ensures the
  curvature information `(s, y)` fed back into the L-BFGS oracle remains accurate
  and well-conditioned.
- **Navigate the feasible path**: When a region is configured, evaluate the
   *projected* candidate `project_R(x, x + d(t))` so the search respects
  constraints.

Walking `t` directly is what lets QQN **automatically discover the right blend**
of gradient and oracle directions without manual tuning. A poor line search can
cause the
algorithm to degenerate into neither effective gradient descent nor effective
quasi-Newton steps, losing the benefits of both.

> **Key insight**: The quadratic path `d(t)` defines a one-dimensional search
> space over states; the line search walks it directly. The quality of the
> overall optimization is therefore directly bounded by the quality of the line
> search.

#### Available Line Search Strategies

The solver registers several interchangeable strategies (all sharing a common
`LineSearchResult` return type and region-aware interface):

| Name | Method | Conditions | Notes |
| --- | --- | --- | --- |
| `strong_wolfe` | Optax zoom line search | Armijo + strong curvature | Keeps L-BFGS updates well-conditioned. |
| `backtracking` / `armijo` | Self-contained backtracking | Armijo sufficient decrease | `lax.while_loop`; robust fallback. |
| `hager_zhang` | Optax backtracking transform | Approximate Wolfe | Robust approximate-Wolfe scheme. |
| `fixed` | Constant step | None | Debugging / benchmarking baseline. |

The **spline** refinement is *not* a line-search strategy but an orthogonal,
boolean enhancement (`spline=True`). Because the path `d(t_i)` is consistent
across all measured points, every probe — regardless of the underlying line
search — can be reused as a control point. The spline is best understood as an
*expanded definition of the curve* rather than a competing search: it does not
replace the chosen line search but **wraps** it (`spline_wrap(inner_search)`),
first running the inner search and then probing the cubic Hermite spline's
stationary points to improve on the accepted step. When enabled, the spline
refinement therefore composes with — and genuinely augments — any chosen line
search.

##### Backtracking / Armijo

Starts at `init_step` and shrinks `α ← shrink·α` until
`f(x + α·d) ≤ f(x) + c1·α·⟨∇f, d⟩` holds or `max_iter` is reached. Implemented
with `lax.while_loop` for JIT/vmap compatibility.

##### Strong Wolfe

Delegates to Optax's `scale_by_zoom_linesearch`, enforcing both Armijo decrease
and the strong curvature condition. The transform rescales the supplied direction
by the discovered step size; the solver recovers `α` from the scaling and
recomputes value/grad at the (projected) accepted point.

##### Spline Search (Information-Reusing)

The spline refinement **wraps** any inner line search (`spline_wrap(inner)`),
treating every probe as a **reusable control point** carrying both a fitness
value `f(d(α))` and a directional derivative `m = ⟨∇f, d⟩`. After the inner
search accepts a step, it fits a piecewise **cubic Hermite spline** to the
active bracket and proposes additional probes at the spline's stationary points
(closed-form roots of a quadratic), keeping any that improve on the inner
result. A
crucial refinement is the **upstream/downstream symmetry rule**: tangents that
oppose a segment's secant slope are reflected to prevent spurious inflections,
phantom minima, and ill-conditioned segments. See
[`spline_search.md`](spline_search.md) for the full derivation.

### The Oracle: The `t = 1` Endpoint

The **oracle** is the component that supplies the `t = 1` endpoint `-H∇f`. The
default is **L-BFGS**, but the oracle is a swappable, pure-functional interface:

```python
class Oracle(NamedTuple):
    init:      Callable[[Params], OracleState]
    direction: Callable[[Params, Grad, OracleState], Tuple[Direction, OracleState]]
    update:    Callable[[OracleState, OracleInfo], OracleState]
```

Because the line search always retains the gradient direction's influence at the
path origin (`d'(0) = -∇f`), the oracle does **not** need to guarantee descent on
its own. Convergence is anchored by the steepest-descent contribution, leaving the
oracle free to be aggressive. This makes the oracle a natural extension point.

Concrete oracles (see [`oracles.md`](oracles.md)):

- **L-BFGS** (default): two-loop recursion over the most recent `m` curvature
  pairs `(s, y)`. Byte-for-byte equivalent to the original optimizer.
- **Momentum**: heavy-ball direction `-(β·v + (1-β)·∇f)`.
- **Shampoo**: structure-aware preconditioner via inverse matrix roots on a
  static refresh cadence.
- **Combinators**: `Fallback([O1, O2, ...])` uses the first valid direction (via
  `jnp.where`, no Python branching); `Blend` (stretch) takes a convex combination.

#### The L-BFGS Two-Loop Recursion

The default oracle computes `-H∇f` directly via the standard two-loop recursion
(Nocedal & Wright, Algorithm 7.4) over fixed-size circular buffers of curvature
pairs:

1. **History**: most-recent-first buffers of `s = Δx`, `y = Δ∇f`, and
   `ρ = 1/⟨y, s⟩`, plus a rolling scale `γ = ⟨y, s⟩ / ⟨y, y⟩`.
2. **Curvature safeguard**: a new pair is admitted only if `⟨y, s⟩ > ε`,
   protecting positive-definiteness on non-convex problems. Otherwise the history
   is left unchanged.
3. **First loop** (newest → oldest): `αᵢ = ρᵢ⟨sᵢ, q⟩`, `q ← q − αᵢ yᵢ`.
4. **Scaling**: `r = γ·q` applies the initial Hessian approximation `H₀ = γI`.
5. **Second loop** (oldest → newest): `βᵢ = ρᵢ⟨yᵢ, r⟩`,
   `r ← r + (αᵢ − βᵢ)sᵢ`.
6. **Direction**: return `-r = -H∇f`.

Unfilled history slots hold zeros and contribute nothing to either loop, so
masking is automatic and the result is exactly `-H∇f`. Both loops are expressed
with `lax.scan`, keeping the whole recursion JIT/vmap compatible.

### Projective Regions: Searching the Feasible Path

A **projective region** remaps a proposed update onto a feasible (or preferred)
set *inside* the line search loop. Rather than searching the raw path, the line
search navigates the **projected path**:

```
d_R(t) = project_R(x, x + d(t)) - x
```

This keeps the descent/Wolfe guarantees meaningful on the feasible path. Regions
are pure functions with the interface:

```python
class Region(NamedTuple):
    init:    Callable[[Params], RegionState]
    project: Callable[[Params, Candidate, RegionState], Candidate]
    update:  Callable[[RegionState, RegionInfo], RegionState]
```

Concrete regions (see [`regions.md`](regions.md)):

- **Box / Min-Max**: elementwise `clip(candidate, lo, hi)`.
- **Orthant** (OWL-QN style): zero coordinates that would flip sign, encouraging
  sparsity.
- **Trust-Region Sphere**: radially clip the step to `‖x_new − x‖ ≤ Δ`, with an
  adaptive radius driven by the ratio `ρ = ared/pred`.
- **Combinators**: `Sequential([R1, R2, ...])` composes projections in order;
  `Intersection` (stretch) approximates projection onto an intersection.

When `region=None`, the identity projection is used and behavior is byte-for-byte
equivalent to the un-regioned optimizer (zero overhead).

## The Solver Loop

QQN follows a JAXopt-style `init_state` / `update` / `run` interface with all
state held in a JIT-compatible `QQNState` NamedTuple:

```python
QQNState(
    iter,          # iteration counter
    value,         # current objective value f(x)
    grad,          # current gradient ∇f(x)
    oracle_state,  # e.g. L-BFGS history / momentum buffer
     step_size,     # last accepted path parameter t
    error,         # ‖∇f‖ (convergence metric)
    done,          # error ≤ tol
    aux,           # optional auxiliary output of the objective
    region_state,  # optional region state (e.g. trust radius)
)
```

### Initialization (`init_state`)

1. Evaluate `value, grad, aux` at the starting point.
2. Initialize the oracle state via `oracle.init(params)`.
3. Initialize the region state via `region.init(params)`.
4. Set `error = ‖∇f‖` and `done = error ≤ tol`.

### Single Iteration (`update`)

1. **Oracle**: query `qn_dir, _ = oracle.direction(params, grad, oracle_state)`
   for the `t = 1` endpoint `-H∇f`.
2. **Gradient**: form `grad_dir = -∇f`.
3. **Path + Search**: run a single configured line search that traverses the
    quadratic path `d(t) = t(1-t)·grad_dir + t²·qn_dir` over `t ∈ [0, 1]`,
    evaluating candidate states `x + d(t)` (each respecting the region via the
    projected path) and selecting the step `t`.
4. **Selection**: extract the accepted `new_params`, `new_value`, `new_grad`,
    and `step_size` (the chosen `t`).
5. **Oracle update**: assemble an `OracleInfo` (`params`, `new_params`, `grad`,
   `new_grad`, `t`, `α`) and call `oracle.update(...)` — e.g. push the new L-BFGS
   curvature pair `(s, y) = (x_new − x, ∇f_new − ∇f)`.
6. **Region update**: assemble a `RegionInfo` (with predicted/actual reduction,
   `t`, `α`) and call `region.update(...)` — e.g. grow/shrink the trust radius.
7. **Convergence**: recompute `error = ‖∇f_new‖` and `done = error ≤ tol`;
   increment `iter`.

### Driver (`run`)

`run` wraps the iteration in a `lax.while_loop` that continues while
`¬done ∧ iter < maxiter`, so the entire optimization is JIT/vmap compatible (e.g.
differentiable end-to-end and vectorizable over batched starting points).

## Public API

```python
QQN(
    fun,
    maxiter=100,
    tol=1e-5,
    history_size=10,
    line_search="strong_wolfe",   # or "backtracking"/"armijo"/"hager_zhang"/
                                   #    "fixed"
    line_search_options=None,     # dict forwarded to the line search (c1, c2, …)
     spline=False,                 # orthogonal cubic Hermite refinement (any LS)
    has_aux=False,
    oracle="lbfgs",               # "lbfgs"|"momentum"|"shampoo"|Oracle
    region=None,                  # Region | None
)
```

String shortcuts map to default-configured concrete components; explicit `Oracle`
or `Region` instances override them for full control. With the defaults
(`oracle="lbfgs"`, `region=None`), the optimizer reproduces the baseline behavior
exactly.

## Advantages

1. **Adaptive Behavior**: Automatically balances between conservative and
    aggressive steps via the line search walking the path, with no manual blend
    tuning.
2. **Robustness**: The path's `d'(0) = -∇f` property plus multiple line-search
   fallbacks ensure progress even when the oracle is poor.
3. **Efficiency**: L-BFGS (or other oracle) acceleration when appropriate;
   information-reusing spline search reduces evaluations.
4. **Smooth Transitions**: Quadratic interpolation avoids abrupt direction
   changes.
5. **Modular Design**: Gradient, oracle, search, and region are conceptually
   separable and independently swappable, making the algorithm extensible.
6. **Hardware-Friendly**: Pure, functional JAX throughout — composes with `jit`,
   `vmap`, `pmap`, and `grad`; the per-iteration line searches batch across the
   t-grid.

## Limitations

1. **Memory Requirements**: Stores L-BFGS history (`O(m×n)` where `m` is history
   size, `n` is parameter dimension); other oracles (e.g. Shampoo) may store
   larger preconditioner statistics.
2. **Computational Overhead**: Quadratic path evaluation across the t-grid adds
2. **Computational Overhead**: Evaluating the quadratic path and walking it with
    the line search adds modest per-iteration cost.
3. **Parameter Tuning**: Performance is sensitive to configuration (history size,
    line-search constants, region radii).
4. **Line Search Sensitivity**: The algorithm's effectiveness is highly sensitive
   to the line search implementation. An inexact or poorly tuned line search
   undermines both convergence speed and the quality of L-BFGS curvature updates.
5. **Region Non-Smoothness**: Projective regions (e.g. Orthant) can introduce
   discontinuities in `d_R(t)`; QQN relies on the line search's
   sufficient-decrease check to remain robust to these.

## Theoretical Guarantees

Under standard assumptions (smooth objective, bounded gradients):

- **Global Convergence**: Guaranteed by the steepest-descent contribution —
  because `d'(0) = -∇f`, a valid decreasing step always exists along any path
  `d(t_i)` for sufficiently small `α`.
- **Superlinear Convergence**: Near the optimum, when the L-BFGS direction
  dominates (the selected `t` approaches `1`), QQN inherits L-BFGS's superlinear
  behavior.
- **Descent Property**: Every accepted step decreases the function value,
  enforced by the line search's sufficient-decrease test.

> **Note on guarantees**: All three guarantees are contingent on the line search
> satisfying sufficient decrease conditions. The steepest-descent fallback
> provides global convergence only if the line search can always find a valid
> step along the path (which it can, given `d'(0) = -∇f`). The descent property is
> enforced *by* the line search, not independently of it. When a region is active,
> these guarantees hold on the *feasible* (projected) path `d_R(t)`.

## References

The QQN algorithm combines ideas from:

- L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) — the default oracle.
- Trust region methods (quadratic models, adaptive radius via `ρ = ared/pred`).
- Adaptive step size selection.
- Gradient descent with momentum (the Momentum oracle / heavy-ball flavor).
- Shampoo / Kronecker-factored preconditioning (the Shampoo oracle).
- Wolfe condition line searches (critical for curvature-update validity).
- Backtracking line search with Armijo conditions (fallback robustness).
- Cubic Hermite interpolation (the information-reusing spline search).
- OWL-QN (the Orthant region for sparsity).
# QQN: A Pluggable Optimizer Framework Built on Quadratic Path Interpolation

## Abstract

We present QQN (Quadratic Quasi-Newton), an optimization framework that reframes the classical "which direction?"
question of numerical optimization into "where on the curve?". Rather than committing to a single search direction per
iteration — gradient descent, momentum, or a quasi-Newton step — QQN constructs a continuous quadratic path that
smoothly interpolates between the steepest-descent direction and an *oracle* direction, then lets a line search discover
the optimal blend automatically. The central architectural contribution is not a single algorithm but a **factoring** of
optimization into four conceptually orthogonal, independently swappable axes: **gradient**, **oracle**, **search**, and
**region**. Many classical methods (L-BFGS, Newton, momentum, Barzilai-Borwein, trust-region, OWL-QN, projected
gradient) emerge as special cases of particular configurations of these axes. We describe the mathematical construction
of the quadratic and spline path methods, the pure-functional JAX implementation that composes with `jit`/`vmap`/`pmap`/
`grad`, and empirical results demonstrating that a deep-memory L-BFGS oracle on the quadratic path wins both the
iteration race and wall-clock time to target on smooth, ill-conditioned objectives.

---

## 1. Introduction

### 1.1 Orientation: The Anatomy of an Optimizer

We assume the reader is comfortable with the basic theory of numerical optimization — the language of gradients, descent
directions, and line searches. What we do *not* assume is a particular way of *carving up* an optimizer into parts,
because that carving is precisely our contribution. So before the innovations, a brief orientation.

A first-order iterative optimizer, viewed abstractly, repeatedly answers four questions at each iterate `x`:

1. **Gradient** — In which direction does the objective locally decrease? This is `-∇f(x)`, the one piece of information
   every method requires.
2. **Oracle** — Is there a *better-aimed* direction than the raw gradient? Curvature-aware methods answer with `-H∇f`
   (quasi-Newton, Newton), momentum methods with a smoothed history, adaptive methods with a per-coordinate rescaling.
3. **Search** — How far, and along what shape, do we move? Classically this is a line search that scales a *fixed*
   direction.
4. **Region** — Is the proposed point *allowed*? Constraints, trust regions, and sparsity structure all remap a proposed
   step onto a feasible set.

Classical methods answer all four questions at once, welded into a single monolithic procedure. L-BFGS *is* a particular
oracle plus a particular line search plus (implicitly) the whole real line as its region. Newton's method, OWL-QN,
projected gradient descent, trust-region methods — each fuses a specific choice on each axis into an indivisible
algorithm. The literature is a catalog of *points*, each proven, none of them *factored*.

### 1.2 The Classical Trade-off, and the Classical Reconciliation

Unconstrained smooth optimization has long been dominated by a trade-off between two families of methods. First-order
methods (gradient descent, momentum) use only the gradient `∇f(x)`. They are cheap, memory-light, and *robust*: the
negative gradient is always a descent direction. But they converge slowly on ill-conditioned problems whose loss
landscapes form long, narrow valleys. Second-order and quasi-Newton methods (Newton, BFGS, L-BFGS)
use curvature information `H ≈ ∇²f⁻¹` to take larger, better-aimed steps. They converge *fast* (superlinearly near a
minimum) but are *fragile*: the quasi-Newton direction `-H∇f` is only guaranteed to descend when `H` is
positive-definite — a condition that fails on non-convex objectives or when the curvature history is stale or
degenerate.

The classical reconciliation is to **pick one direction and then run a line search** along it. If the quasi-Newton
direction is good, the line search accepts a full step; if it is bad, the search backtracks. But this still commits to a
*single* direction per iteration. When the oracle direction is poor, a backtracking search along it can waste
evaluations without ever exploring the reliable gradient direction.

### 1.3 The QQN Idea: Blend, Don't Choose

QQN's core insight is to refuse the binary choice. Instead of selecting either the gradient or the quasi-Newton
direction, it constructs a **continuous quadratic path** that smoothly connects them:

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

This single curve has three decisive properties:

- `d(0) = 0` — the path starts at the current iterate `x`.
- `d'(0) = -∇f` — the path *begins* tangent to steepest descent, so it is guaranteed to decrease `f` for small `t`
  whenever `∇f ≠ 0`.
- `d(1) = -H∇f` — the path *ends* exactly at the quasi-Newton (oracle) step.

The line search then walks `t ∈ [0, 1]` directly. Near `t = 0` the path *is*
gradient descent (robustness); near `t = 1` it *is* the quasi-Newton step (speed). The search discovers the right blend
automatically, with no manual tuning, and inherits global convergence from the gradient tangent while retaining
superlinear behavior when the oracle direction dominates.

> **The reframing.** QQN turns "which direction?" into "where on the curve?".
> The one-dimensional search over `t` replaces the discrete choice between
> two competing directions with a continuous, globally-anchored
> interpolation.

### 1.4 Contributions

This paper makes contributions on two fronts, expository and constructive.

**The QQN innovations.** At the heart of the framework sit four ideas:

- **The quadratic path** — a geometrically principled interpolation between gradient and oracle, whose parabolic form is
  not an arbitrary choice but the simplest curve satisfying the three endpoint/tangent constraints of §1.3.
- **The strategy pattern** — the factoring of optimization into four orthogonal, independently swappable axes (gradient,
  oracle, search, region), each hidden behind a small pure-functional interface.
- **Provable convergence** — the tangent anchor `d'(0) = -∇f` guarantees a valid descent step exists on every path, so
  global convergence is inherited regardless of oracle quality, while superlinear behavior returns when the oracle
  dominates near the optimum.
- **Geometrically principled** — the path is a genuine geometric object, not a discretized blend grid: the curve *is*
  the search space, and the line search moves along it directly. The construction is intuitive and obeys the
  requirements of geometry, so a single scalar `t` names an actual state on the curve rather than a coefficient to be
  re-scaled. valid descent step exists on every path, so global convergence is inherited regardless of oracle quality,
  while superlinear behavior returns when the oracle dominates near the optimum.

**Cross-cutting enhancements.** Two orthogonal features layer on top of any configuration:

- **Temperature** — a Metropolis-style stochastic acceptance that permits *controlled exploration* (uphill moves) on
  non-convex landscapes, cooling to pure Armijo backtracking at `temperature = 0`.
- **Projective regions** — a means of *constraining the search* onto a feasible or preferred set, applied *inside* the
  line-search loop so that descent guarantees hold on the projected path.

**Core enhancements.** Three deeper extensions push the framework further:

- **Spline** — reuse *all* measured gradients, treating each line-search probe as a control point of a richer
  cubic-Hermite path model.
- **Partitioned oracles** — partition the weights (e.g. by layer) and treat each partition as its own oracle regime,
  making certain strategies faster.
- **Recursive descent** — parallelize the QQN path-building itself, remapping each step onto a low-dimensional regime
  solvable by an inner multidimensional continuous optimizer (e.g. a basic L-BFGS-driven QQN).

The deeper contribution is architectural rather than mathematical. Classical optimization is a literature of monolithic
methods — each proven, none of them *factored*. QQN treats four tangled concerns (direction, oracle, search, region) as
**independently swappable strategies** behind small, pure-functional interfaces. The parabola itself *falls out* of that
decomposition: once you stop thinking "L-BFGS is an algorithm" and start thinking "the direction is a component and the
search is a component," the question "what curve connects them?" becomes unavoidable, and the straight line stops
looking like a law of nature and starts looking like a hard-coded default nobody refactored.

---

## 2. The Objective Gradient Function: Required Input

The only mandatory input to QQN is an objective function `f(x)` from which a value and a gradient can be obtained. In
the JAX implementation this is supplied as a differentiable Python function; `jax.value_and_grad`
transparently produces `(f(x), ∇f(x))`. When the objective returns auxiliary data alongside its scalar value, a
`has_aux=True` flag threads that data through the solver.

Everything else in QQN — the oracle, the line search, the region — is an optional strategy layered on top of this single
required input. This is the architectural invariant that makes the framework pluggable: each strategy is a pure function
of quantities the solver already computes (parameters, gradients, along-path measurements), so strategies can be
substituted without touching the rest of the algorithm.

---

## 3. The Oracle Strategy

The **oracle** supplies the `t = 1` endpoint `-H∇f` of the quadratic path — the curvature-aware (or otherwise
accelerated) direction. The default oracle is L-BFGS, but the oracle is a swappable, pure-functional interface:

```python
class Oracle(NamedTuple):
    init: Callable[[Params], OracleState]
    direction: Callable[[Params, Grad, OracleState], Tuple[Direction, OracleState]]
    update: Callable[[OracleState, OracleInfo], OracleState]
```

Because the line search always retains the gradient direction's influence at the path origin (`d'(0) = -∇f`), the oracle
does **not** need to guarantee descent on its own. Convergence is anchored by the steepest-descent contribution, leaving
the oracle free to be aggressive. This makes the oracle a natural extension point.

### 3.1 Concrete Strategies

**Gradient.** The trivial oracle returns the negated gradient itself (`-∇f`). The path then degenerates to scaled
steepest descent. Though rarely used alone, this is the essential fallback for the earliest iterations — before any
curvature history has accumulated — and the terminal safety net for combinator oracles.

**Momentum.** A first-order accelerated (heavy-ball) direction. Instead of curvature, the oracle integrates a
decaying-weight history of the realized per-iteration parameter deltas `Δx = x_new − x`:

```
v_new     = β · v + (1 − β) · Δx
direction = -∇f + β · v
```

This smooths out the path and helps avoid oscillations, giving QQN a heavy-ball flavor at `t = 1` while retaining the
gradient at `t = 0`. The sole hyperparameter is the decay `β` (default `0.9`).

**Adam.** Adaptive moment estimation, combining momentum and RMSProp ideas. The oracle integrates a decaying-weight
first moment `m` (momentum) and a decaying-weight second moment `v` (energy) of the gradients, with the standard bias
correction:

```
m         = β₁·m + (1 − β₁)·∇f
v         = β₂·v + (1 − β₂)·∇f²
direction = − m̂ / (√v̂ + ε)
```

The hyperparameters `β₁`, `β₂`, `ε` follow the classical Adam defaults.

**L-BFGS.** The default oracle: a quasi-Newton estimate of `-H∇f` given a history of gradient/point measurements. It
maintains fixed-size circular buffers of the most recent `m` curvature pairs `(s, y) = (Δx, Δ∇f)` and computes the
direction directly via the standard two-loop recursion (Nocedal & Wright, Algorithm 7.4):

1. **History**: most-recent-first buffers of `s`, `y`, and `ρ = 1/⟨y, s⟩`, plus a rolling scale `γ = ⟨y, s⟩ / ⟨y, y⟩`.
2. **Curvature safeguard**: a new pair is admitted only if `⟨y, s⟩ > ε`
   (relative to the Cauchy-Schwarz scale), protecting positive-definiteness on non-convex problems.
3. **First loop** (newest → oldest): `αᵢ = ρᵢ⟨sᵢ, q⟩`, `q ← q − αᵢ yᵢ`.
4. **Scaling**: `r = γ·q` applies the initial Hessian approximation
   `H₀ = γI`.
5. **Second loop** (oldest → newest): `βᵢ = ρᵢ⟨yᵢ, r⟩`,
   `r ← r + (αᵢ − βᵢ)sᵢ`.
6. **Direction**: return `-r = -H∇f`.

Unfilled history slots hold zeros and contribute nothing to either loop, so masking is automatic and the whole recursion
is expressed with `lax.scan`, keeping it JIT/vmap compatible.

Beyond these four canonical strategies, the framework also implements a matrix-free Barzilai-Borwein **secant** oracle
(`O(n)` memory), a structure-aware **Shampoo** preconditioner, and an **Anderson**
acceleration oracle — the variational ideal that L-BFGS approximates.

### 3.2 The Foreach (Combinator) Strategy

Oracles compose. The **`Fallback([O₁, O₂, …])`** combinator uses the first oracle's direction when it is valid,
otherwise falls back to the next. Here validity is *descent*, not mere non-zeroness: a finite, non-zero quasi-Newton
direction that points uphill (`⟨∇f, d⟩ ≥ 0`) betrays a degenerate curvature estimate and triggers the fallback. All
selection uses
`jnp.where` / `lax.select` — no Python conditionals — so the combinator stays
`jit`-friendly, and a terminal safety net returns `-∇f` if *every* child produces an invalid direction.

A canonical example is `Fallback([LBFGSOracle(50), SecantOracle()])`: the secant oracle is dormant while the L-BFGS
history is valid, yet supplies a finite, curvature-aware direction the instant that history degenerates — carrying
curvature that a momentum fallback lacks. A **`Blend`** combinator (a fixed convex combination of oracle directions) is
also available.

**Pros and cons.** The combinator pattern is the primary mechanism for robustness: a deep-memory quasi-Newton oracle is
safeguarded by a lightweight backup that never divides or diverges. The cost is that all children are evaluated on every
step (both branches of a `where` are computed), so a fallback pays for the directions it does not use. In practice these
directions are cheap relative to the objective's own value/gradient evaluations.

**Hyperparameters.** The combinator itself is hyperparameter-free; its children carry their own (history size, momentum
decay, etc.). The ordering of the child list encodes priority and is the only structural choice.

### 3.3 Partitioned Oracles

A structural generalization of the oracle axis is to **partition the parameters** — for instance by layer, by tensor, or
by any user-supplied block structure — and run an independent oracle regime on each partition. Two motivations drive
this. First, several oracle strategies have superlinear per-step cost in the dimension of their working subspace
(Shampoo's Kronecker factors, dense secant updates); confining each oracle to a small partition makes those strategies
dramatically faster. Second, partitioning can be *statistically* beneficial: curvature that couples strongly within a
layer but weakly across layers is captured more faithfully by per-layer histories than by a single flattened buffer
whose fixed memory is spread thin across the whole parameter vector.

Because each partition still exposes the same `(init, direction, update)`
interface, the partitioned oracle is itself just an oracle — it composes with
`Fallback` and `Blend`, threads through the same `QQNState`, and remains
`vmap`/`jit`-compatible. The path construction is unchanged: the per-partition directions are concatenated back into a
single `-H∇f` endpoint before the quadratic path is formed, so the line search still traverses one scalar `t`.

### 3.4 Orthogonal Concern: Feeding Line-Search Probes to the Oracle

An optional, orthogonal enhancement forwards every gradient evaluated *during the line search* — not just the accepted
point — into the oracle's curvature memory. Because the line search probes lie on the single ray
`x + α·d`, they are collinear and only ever re-estimate curvature *along*
`d`. Replaying too many of them would flush genuine cross-iteration curvature from a fixed-size buffer, so the replay
count is capped and gated:
only probes that (a) strictly decrease the objective relative to the current iterate and (b) lie on the accepted side of
the path are admitted. This descent gate is the documented fix against history-polluting rejected probes. The feature is
off by default and composes with any oracle.

---

## 4. The Path Strategy

The path is the geometric object the line search traverses. Three path strategies span a spectrum from
information-discarding to information-reusing.

### 4.1 Linear

The linear path discards the gradient entirely and samples the straight chord from the origin (`α = 0`) to the oracle
endpoint (`α = 1`):

```
x + α·direction,   α ∈ [0, 1]
```

It keeps the lowest-value feasible sample found. This is the deliberate opposite of the spline: where the spline reuses
every probe's *gradient*, the linear refinement throws that information away and interpolates value-only. When the
direction degenerates to the negated gradient (no genuine oracle point), the samples still interpolate along the
gradient ray, so a sensible step is recovered. The linear path is primarily a control — a baseline against which the
value of the gradient information can be measured.

### 4.2 Quadratic

The quadratic path is the simplest path that *follows the gradient*. It is the vector-valued quadratic

```
d(t) = -t(1-t)·∇f - t²·H∇f
    = -t·∇f + t²·(∇f - H∇f)
```

whose endpoints and initial tangent were derived in Section 1.3. Because
`d'(0) = -∇f`, the directional derivative of `f` along the path at the origin is `⟨∇f, d'(0)⟩ = -‖∇f‖² ≤ 0`. This is
what anchors QQN's global convergence: regardless of how poor the oracle direction is, the *beginning*
of the path always decreases `f`.

The key properties bear restating:

- **t = 0**: pure steepest descent direction (the path's tangent).
- **t = 1**: pure oracle / L-BFGS direction.
- **0 < t < 1**: a smooth quadratic blend, weighting the gradient by
  `t(1-t)` and the oracle by `t²`.

The points along `d(t)` are **states**, not directions to be re-scaled by a separate inner line search. The line search
traverses the parameter
`t ∈ [0, 1]` *directly*: each probe `x + d(t)` is a state on the curve. There is no discretized "blend grid" and no
per-grid-point inner search — the curve itself is the one-dimensional search space.

> **Invariance to rescaling.** Rescaling the gradient (or the oracle
> direction) does *not* change the geometric path that `d(t)` traces through
> parameter space. Scaling only distorts the *parameterization* — how the
> scalar `t` maps onto arc length — but the set of reachable states is
> invariant. This is why it is meaningless to "re-search" a chosen `t` with
> an inner line search that rescales `d(t)`.

### 4.3 Spline

The quadratic path is fixed by the value and gradient measured at the *origin* (`t = 0`) together with the oracle
endpoint. But the very first thing a line search does is evaluate the objective — both its value and its gradient — at
the oracle point (`t = 1`). Once that measurement is in hand, we know the value and slope at *both* ends of the path,
and the quadratic, which only used slope information at the origin, is no longer the richest curve consistent with what
we have measured.

The first obvious enhancement, then, is to promote the path to a **cubic**:
a cubic Hermite segment is the simplest curve that can honor a value *and* a slope at each endpoint, so it incorporates
all the known information rather than discarding the oracle-point gradient. And once we are fitting a cubic from
endpoint measurements, nothing stops us from adding *more* endpoints. Every subsequent probe during the line search also
yields a value
`f(d(t))` and a directional derivative `m = ⟨∇f, d'(t)⟩`; each such measurement becomes another **control point**, and
the single cubic segment generalizes to a piecewise-cubic spline.

The spline path thus extends the quadratic by utilizing *all* gradient evaluations. The standard quadratic-path search
discards most of this information after each step; the spline treats each measurement as a reusable control point
carrying both value and slope.

The interpolation is a **piecewise cubic Hermite spline** over the parameter
`t`. Each control point `i` stores `(t_i, f_i, m_i)`. For two adjacent control points, with `h = t₁ − t₀` and normalized
`s = (t − t₀)/h`, the cubic Hermite basis gives:

```
f(s) = h00(s)·f₀ + h10(s)·h·m₀ + h01(s)·f₁ + h11(s)·h·m₁
```

To propose the next step, we differentiate and solve `f'(s) = 0`. Since
`f(s)` is cubic, `f'(s)` is quadratic, yielding at most two roots in closed form:

```
A =  6·f₀ + 3·h·m₀ − 6·f₁ + 3·h·m₁
B = −6·f₀ − 4·h·m₀ + 6·f₁ − 2·h·m₁
C =          h·m₀
```

with `s = (−B ± √(B² − 4AC)) / (2A)`, guarded for the degenerate near-linear case. Any real root in `[0, 1]` maps back
via `t = t₀ + s·h` and becomes a candidate minimizer.

**Upstream/downstream symmetry.** A naive Hermite construction inserts the measured tangent `m_i` with its raw sign. If
a control point's gradient "goes against" the local trend, the resulting cubic can develop a spurious inflection,
overshoot, or non-monotone loop — producing phantom minima, oscillating step proposals, and ill-conditioned segments.
The correction compares each tangent's orientation to the segment's secant slope
`Δ = (f₁ − f₀)/(t₁ − t₀)` and reflects tangents that oppose the established flow. The terrain analogy is instructive:
the spline models a watercourse, and a valley is the same valley whether traversed upstream or downstream — so we treat
orientation as a symmetric feature.

> **Soundness caveat.** Reflecting a *measured* directional derivative is a
> heuristic, not a proven-safe operation. Safety rests entirely on the outer
> line search: the spline only ever *proposes* candidates, and a candidate is
> accepted **only if it strictly improves fitness**. The descent guarantee is
> therefore inherited from the inner search's sufficient-decrease test, not
> from the reflection rule.

Crucially, the spline is **not** a competing line search but an *expanded definition of the curve* that **wraps** any
inner search (`spline_wrap(inner)`). Because the path `d(t_i)` is consistent across all measured points, every probe —
regardless of the underlying line search — can be reused as a control point. The wrapper first runs the inner search,
then probes the spline's stationary points to improve on the accepted step.

---

## 5. The Line Search Strategy

**The line search is not an implementation detail — it is a first-class algorithmic component** and the mechanism by
which QQN's theoretical properties are realized in practice. The line search traverses the path over
`t ∈ [0, 1]` and must:

- **Select the path parameter `t`** — walk the curve to satisfy sufficient-decrease conditions (Armijo/Wolfe). Each
  probe `x + d(t)` is a state, not a direction to be re-scaled.
- **Enforce descent** — guarantee `f(x + d(t)) < f(x)` (or report failure), the foundation of global convergence.
- **Exploit curvature** — a strong Wolfe condition keeps the curvature information `(s, y)` fed back into the L-BFGS
  oracle accurate and well-conditioned.
- **Navigate the feasible path** — when a region is configured, evaluate the *projected* candidate
  `project_R(x, x + d(t))`.

Walking `t` directly is what lets QQN **automatically discover the right blend** of gradient and oracle without manual
tuning. The quality of the overall optimization is directly bounded by the quality of the line search.

### 5.1 Temperature: Orthogonal Early Acceptance

Orthogonal to the inner search strategy is a **temperature** parameter that layers a Metropolis-style stochastic
acceptance on top of the Armijo test. A step that fails Armijo may still be accepted (an *uphill climb*) with
probability `exp(−ΔE / T)`, where `ΔE = f(x + α·d) − f(x)` and `T` is a geometrically-cooled temperature. This is used
for early acceptance and exploration on non-convex landscapes. With the default `temperature = 0.0`
the stochastic path is disabled entirely and the search reduces to plain Armijo backtracking. A deterministic PRNG seed
keeps the whole search JIT/vmap compatible and reproducible.

### 5.2 Concrete Strategies

**Backtracking / Armijo.** The robust default. Starting at `init_step`, it shrinks `α ← shrink·α` until the Armijo
sufficient-decrease condition
`f(x + α·d) ≤ f(x) + c1·α·⟨∇f, d⟩` holds or `max_iter` is reached. Implemented with `lax.while_loop` for JIT/vmap
compatibility.

**Armijo-Wolfe (Strong Wolfe).** Delegates to Optax's zoom line search, enforcing both Armijo decrease and the strong
curvature condition. This keeps L-BFGS updates well-conditioned but can *over-restrict* the quadratic-path step on some
benchmarks, so it is not the default.

The framework additionally registers a **Hager-Zhang** approximate-Wolfe search, a **fixed**-step baseline for
debugging, and a **null** search that unconditionally accepts the `t = 1` oracle point.

### 5.3 The Foreach (Wrapping) Strategy

The spline and linear refinements are *foreach*-style wrappers that compose with any inner line search: they run the
inner search first, then augment its result by probing additional candidates along the consistent path (Section 4).
Because they only ever accept a candidate that strictly improves on the inner result, they inherit the inner search's
descent guarantee.

**Pros and cons.** Wrapping adds a modest number of extra probes per iteration in exchange for a richer path model and
cheaper backtracking. The spline reuses gradient information the inner search would otherwise discard; the linear
wrapper deliberately discards it as a control. The cost is the extra evaluations, which are tracked honestly in the eval
accounting.

**Hyperparameters.** The wrappers expose a probe budget (`spline_max_iter`,
`num_samples`) that trades evaluation cost against model fidelity.

### 5.4 Available Strategies at a Glance

| Name                      | Method                      | Conditions                 | Notes                                            |
|---------------------------|-----------------------------|----------------------------|--------------------------------------------------|
| `armijo` / `backtracking` | self-contained backtracking | Armijo sufficient decrease | robust default; `lax.while_loop`                 |
| `strong_wolfe`            | Optax zoom                  | Armijo + strong curvature  | keeps L-BFGS well-conditioned; can over-restrict |
| `hager_zhang`             | Optax backtracking          | approximate Wolfe          | robust approximate-Wolfe scheme                  |
| `fixed`                   | constant step               | none                       | debugging / benchmarking baseline                |
| `null`                    | accept `t = 1`              | none                       | pure oracle-endpoint step                        |

### 5.5 Recursive Descent

The line search over `t` is itself a one-dimensional continuous optimization problem — and QQN is a continuous
optimizer. This suggests a recursive possibility. Rather than parameterizing the path with a single scalar `t`, we can
**parallelize the path-building** by allowing several path parameters at once (for instance, one per partition of §3.3,
or a small set of blend coefficients over multiple oracle directions). Each outer QQN step then remaps to a
*low-dimensional* continuous subproblem — a handful of blend coordinates rather than the full parameter dimension —
which is itself solved by an inner multidimensional continuous optimizer, e.g. a basic L-BFGS-driven QQN.

Because the inner problem lives in a tiny space, the inner optimizer is cheap and its own curvature estimate is dense
and accurate. The outer loop supplies the (gradient, oracle) endpoints; the inner loop discovers a *multi*-parameter
blend that no single scalar `t` could express. The construction is self-similar — QQN wrapping QQN — and inherits the
same descent guarantee at each level, because the innermost search still bottoms out in a sufficient-decrease test along
a gradient-tangent path.

---

## 6. Regions

A **projective region** remaps a proposed update onto a feasible (or preferred) set *inside* the line-search loop.
Rather than searching the raw path, the line search navigates the **projected path**:

```
d_R(t) = project_R(x, x + d(t)) - x
```

This keeps the descent/Wolfe guarantees meaningful on the feasible path. Regions are pure functions with an `init` /
`project` / `update` interface mirroring the oracle abstraction. When `region=None`, the identity projection is used and
behavior is byte-for-byte equivalent to the un-regioned optimizer (zero overhead).

### 6.1 Orthant

The **Orthant** region (OWL-QN style) constrains each step to remain within the orthant defined by the current point's
signs, clamping at zero any coordinate that would cross zero. This encourages sparsity: a coordinate that starts at
exactly zero stays at zero. The projection is a pure elementwise clamp, trivially `vmap`/`jit`-able. Paired with the
L-BFGS oracle it reproduces OWL-QN.

### 6.2 Trust Region

The **Trust-Region Sphere** enforces `‖x_new − x‖₂ ≤ Δ` by radially clipping the step. With `adaptive=True` the radius
grows/shrinks according to the ratio `ρ = ared/pred` of actual to predicted reduction. The predicted reduction is the
*exact* along-path model `pred(t) = −⟨∇f, d(t)⟩`, which requires no separate curvature term because the path's curvature
is already fully encoded in `d(t)`.

> **Chord/arc caveat.** On QQN's *curved* path, the radial clip measures
> chord length `‖x_new − x‖` while the predicted-reduction model integrates
> along arc length. This mismatch makes a naive `ρ < 0.25` shrink rule
> over-react. The implementation therefore shrinks only on a genuinely poor
> `ρ`, shrinks gently, holds the radius in a wide acceptable band, and never
> lets the radius fall below a step that just demonstrably succeeded.

### 6.3 No-Decrease

The **No-Decrease** region (a multi-objective guard) constrains the step so it does not increase the loss on a secondary
objective `g`. Given `∇g(x)`, it projects the step onto the half-space `{s : ⟨∇g, s⟩ ≤ 0}`, removing only the
`g`-increasing component:

```
s_proj = step − relu(⟨∇g, step⟩) / (‖∇g‖² + ε) · ∇g
```

This is the geometry of continual learning and constrained fine-tuning:
descent on `g` passes through untouched while the primary objective is optimized. It requires one extra gradient of `g`
per projection and is gated behind explicit opt-in.

### 6.4 Value Bounds (Box)

The **Box** region enforces elementwise bounds `lo ≤ x_new ≤ hi` via a simple
`clip(candidate, lo, hi)`. Bounds may be `None` on either side (mapped to ±inf). A related **Quantization** region
confines each weight to the rounding cell of its starting value, drawing the optimizer toward representable grid values
without hard-snapping — the geometric counterpart to a quantization-delta regularizer.

Regions compose via a **`Sequential`** combinator, applying projections in order (e.g. box ∩ trust-region), with an
optional **`Intersection`**
combinator for Dykstra-style alternating projection.

---

## 7. The Solver Loop

QQN threads the state of all four axes through a single immutable
`QQNState` NamedTuple and follows a JAXopt-style `init_state` / `update` /
`run` interface.

**Initialization** evaluates value, grad, and aux at the starting point; initializes the oracle and region states; and
sets the convergence metric
`error = ‖∇f‖`.

**A single iteration** proceeds:

1. **Oracle**: query the `t = 1` endpoint `-H∇f`.
2. **Gradient**: form the tangent `-∇f`.
3. **Path + Search**: run a single configured line search that traverses
   `d(t) = t(1-t)·(-∇f) + t²·(-H∇f)` over `t ∈ [0, 1]`, evaluating projected candidate states and selecting the step
   `t`.
4. **Selection**: extract `new_params`, `new_value`, `new_grad`, `step_size`.
5. **Oracle update**: push the new curvature pair (admitted only if
   `⟨y, s⟩ > ε`).
6. **Region update**: update adaptive state (e.g. trust radius) from the predicted/actual reduction.
7. **Convergence**: recompute `error` and `done`; increment `iter`. A run also terminates if an iterate becomes
   non-finite, so a single bad start in a `vmap` batch does not waste the batch's remaining iterations.

The **driver** wraps the iteration in a `lax.while_loop` that continues while
`¬done ∧ iter < maxiter`, so the entire optimization is JIT/vmap compatible — differentiable end-to-end and vectorizable
over batched starting points.

The four components are **conceptually orthogonal and independently swappable**. The solver exposes each as a
configuration point, so alternative oracles, regions, or search strategies can be substituted without touching the rest
of the algorithm.

---

## 8. Test Method

### 8.1 Problems

We evaluate on two families of objectives.

**Analytical.** Classical hard-conditioned test functions — Rosenbrock, Rastrigin, Ackley, and related landscapes —
probe convergence on ill-conditioned and multi-modal surfaces where naive direction choices stall.

**Neural.** Simple multi-layer perceptrons trained full-batch on MNIST and Fashion-MNIST, exercised across a range of
activation functions (ReLU, sigmoid, tanh, gelu, swish, sine, gaussian, and unconventional rolling-window activations).
These are merely-piecewise-smooth in the ReLU case, testing whether QQN's `C⁰`-along-the-path descent argument holds up
where the second-order rate proofs do not strictly apply.

### 8.2 Measures

Three measures capture optimizer quality, with differing honesty:

- **Iterations** (bad — not representative): a raw iteration count ignores the varying per-iteration cost of different
  line searches (the Optax zoom search inside standalone L-BFGS costs ~2.1 evaluations/iteration versus
  ~1.0–1.1 for bare Armijo).
- **Time** (honest): wall-clock to a loss target reflects the true cost the user pays, but is sensitive to hardware and
  JIT warmup.
- **Evaluations** (stable): the cumulative count of value-and-gradient evaluations is the most reproducible,
  hardware-independent measure. The implementation tracks this explicitly through every line-search probe, spline probe,
  aux recompute, and probe-value recovery.

### 8.3 Analysis

The benchmark harness generates, for each optimizer configuration, a Pareto frontier (loss vs. time), an
iteration-efficiency leaderboard, a cost-aware (evaluations-to-target) leaderboard, a target-sensitivity profile
(iterations to reach each of a sequence of loss milestones), and inter-milestone cost breakdowns. Optimizer
configurations are generated as the *cross product* of the four orthogonal axes — oracle × line-search × spline/linear ×
region × probe-feeding — so the effect of each axis can be isolated.

Empirically, on a 4-layer `tanh,gelu,tanh` MLP (335k params) trained full-batch on Fashion-MNIST, QQN with a **deep
L-BFGS oracle** decisively wins both the iteration race *and* wall-clock to the `2e-2` loss target, achieving a 2.64×
iteration speedup over standalone L-BFGS while also being cheaper per iteration (16.08 ms/it vs 20.71 ms/it). The Pareto
frontier is entirely QQN; standalone L-BFGS is dominated. The speedup *widens as the target tightens*, reflecting the
second-order advantage in the fine-tuning regime where first-order baselines exhaust their budget. Among oracle choices,
the L-BFGS oracle wins — momentum, Anderson, and the matrix-free secant cannot match the dominant-subspace capture of a
deep L-BFGS history on an anisotropic Hessian — and the curvature/memory lever is monotone in iterations, with the
wall-clock knee at a history size of 80–120.

---

## 9. Theoretical Guarantees

Under standard assumptions (smooth objective, bounded gradients), and contingent on a line search that satisfies
sufficient-decrease conditions:

- **Global convergence** — guaranteed by the steepest-descent contribution. Because `d'(0) = -∇f`, a valid decreasing
  step always exists along any path
  `d(t)` for sufficiently small `α`, *regardless of oracle direction quality*.
- **Superlinear convergence** — near the optimum, when the L-BFGS direction dominates (the selected `t` approaches `1`),
  QQN inherits L-BFGS's superlinear behavior.
- **Descent property** — every accepted step decreases the function value, enforced by the line search's
  sufficient-decrease test.

All three guarantees are contingent on the line search. The steepest-descent fallback provides global convergence only
because the line search can always find a valid step along the path (which it can, given `d'(0) = -∇f`). The descent
property is enforced *by* the line search. When a region is active, these guarantees hold on the *feasible* (projected)
path `d_R(t)`.

Notably, the *hybrid algorithm itself* requires only **`C⁰` continuity along the path** to make monotone progress — the
sufficient-decrease test compares function *values*. Smoothness sharpens the rate proofs and strengthens the oracle but
is not a precondition for descent, making QQN well-suited to piecewise-smooth objectives (ReLU networks, max-pooling,
hinge/L1 terms).

---

## 10. Related Work and Equivalences

QQN's factoring is what makes a broad catalog of equivalences possible: many classical methods are QQN with one or two
axes fixed to a canonical choice.

- **Gradient descent** is the `t → 0` regime (fixed search, suppressed oracle).
- **L-BFGS** is the `t = 1` corner with the default oracle.
- **Newton's method** arises from an exact-Hessian oracle accepting `t = 1`.
- **Momentum, Barzilai-Borwein, Anderson acceleration** arise from oracle choices.
- **Trust regions, OWL-QN, projected gradient descent** arise from region choices.
- **Conjugate gradient** arises from a CG-as-oracle configuration (the conjugacy `β` correction encoded in the oracle,
  not the search).

These equivalences follow from three structural facts: the tangent anchor (`d'(0) = -∇f`) means every configuration
contains gradient descent as its
`t → 0` limit; the endpoint (`d(1) = -H∇f`) means whatever the oracle proposes is reachable at `t = 1`; and projection
inside the search means constrained methods arise by remapping the path without altering the gradient/oracle/search
machinery.

QQN itself draws on L-BFGS, trust-region methods, adaptive step selection, momentum, Shampoo / Kronecker-factored
preconditioning, Wolfe and Armijo line searches, cubic Hermite interpolation, and OWL-QN.

---

## 11. Conclusion

QQN is a framework, not merely an algorithm. Its central quadratic path
`d(t) = t(1-t)(-∇f) + t²(-H∇f)` is the geometric consequence of a design decision: to treat direction, oracle, search,
and region as independently swappable strategies rather than welded-together parts of a monolithic method. From that
decomposition the parabola falls out — it is the simplest curve that begins tangent to steepest descent (for
globalization) and ends at the oracle direction (for speed) — and with it a configuration space in which classical
optimizers are special cases.

The framework is implemented as pure, functional JAX so that every component composes with `jit`, `vmap`, `pmap`, and
`grad`; state threads through a single immutable `QQNState`; and the whole optimization is a differentiable,
vectorizable operation. Empirically, the quadratic path with a deep-memory L-BFGS oracle wins both the iteration race
and wall-clock time to target on smooth, ill-conditioned objectives, with the value proposition compounding as the
target tightens.

The contribution is architectural. The math was downstream of the architecture: once you stop asking "which direction?"
and start asking "where on the curve?", the straight line stops looking like a law of nature and starts looking like a
hard-coded default nobody refactored.

---

## References

- Nocedal, J. & Wright, S. *Numerical Optimization* — L-BFGS two-loop recursion (Algorithm 7.4).
- Broyden–Fletcher–Goldfarb–Shanno (BFGS) and limited-memory L-BFGS.
- Trust-region methods (quadratic models, adaptive radius via `ρ = ared/pred`).
- Wolfe-condition line searches; Armijo backtracking.
- Cubic Hermite interpolation (the information-reusing spline search).
- OWL-QN (the Orthant region for sparsity).
- Shampoo / Kronecker-factored preconditioning.
- Anderson / Pulay acceleration.
- Gradient descent with momentum (heavy-ball).
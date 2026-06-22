---
documents:
  - ../mnist_comparison.log
related:
  - algorithm.md
  - ../README.md
  - ../examples/mnist_comparison.py
  - ../qqn_jax/solver.py
  - ../qqn_jax/line_search.py
  - ../qqn_jax/oracles.py
  - ../qqn_jax/spline_search.py
  - ../qqn_jax/regions.py
---

# Experimental Results: MNIST Optimizer Comparison

This document summarizes the empirical validation of QQN against standard
baselines and across its own swappable components. The driving experiment is
[`examples/mnist_comparison.py`](../examples/mnist_comparison.py); the raw
output it produced is captured in
[`mnist_comparison.log`](../mnist_comparison.log).

## Experimental Setup

The benchmark frames MNIST classification as a **full-batch, deterministic**
optimization problem — a softmax (multinomial logistic regression) classifier
with L2 regularization (`l2 = 1e-4`). The full-batch framing is deliberate: it
keeps the comparison apples-to-apples for the second-order methods (QQN and
L-BFGS), which assume a smooth, deterministic objective.

| Setting        | Value                                                      |
|----------------|------------------------------------------------------------|
| Classes        | 10                                                         |
| Train samples  | 5000                                                       |
| Test samples   | 1000                                                       |
| Max iterations | 500                                                        |
| Model          | Softmax / multinomial logistic regression                  |
| Loss           | Cross-entropy + `0.5·l2·‖params‖²` (`l2 = 1e-4`)           |
| Init           | Shared `PRNGKey(42)` so every optimizer starts identically |

Data is loaded from real MNIST via `tensorflow.keras` or `torchvision` when
available, and falls back to a synthetic Gaussian-blob dataset otherwise, so
the experiment always runs.

All QQN variants are run **one update at a time** (via `solver.init_state` +
a JIT-compiled `solver.update`) to record the full loss trajectory; the Optax
baselines (`SGD`, `Adam`, `L-BFGS`) use their own JIT-compiled step loops.

The default QQN configuration uses the **L-BFGS oracle** (`history_size=10`),
the **Armijo backtracking line search**, and **no region**.

### Shared, Fair Termination Bounds

A key feature of the experiment is that **every optimizer races against the
same termination criteria**, making the comparison strictly apples-to-apples.
Rather than each method using its own private stopping rule, all share:

| Bound         |    Value | Meaning                                       |
|---------------|---------:|-----------------------------------------------|
| `f_target`    | `1.1e-1` | Stop once full-batch loss `≤` this value.     |
| `gtol`        | `1.0e-4` | Stop once `‖∇f‖ ≤` this value (stationarity). |
| `time_budget` | `15.0` s | Hard wall-clock cap per optimizer.            |

The summary table records, for every method, the iteration (`->target`) and
wall-clock time (`t->tgt`) at which the shared loss/gradient target was first
reached — or `—` when the method did not reach it within the iteration limit.
The `f_target` was deliberately tuned to `1.1e-1` so the `->target` / `t->tgt`
columns become *informative*. The `time_budget` is set to `15.0`s (and Shampoo
switched to a *blocked* preconditioner) so the comparison stays meaningful
while still capping runaways.

### Convergence-Rate Milestones

Beyond a single time-to-target, the experiment also records a full
**convergence-rate profile**: a tuple of descending loss thresholds
(`5.0e-1`, `2.0e-1`, `1.5e-1`, `1.2e-1`) and, per optimizer, the first
iteration at which each threshold is crossed. This separates *early-phase*
descent speed (large-loss milestones) from *late-phase* refinement
(small-loss milestones) far more sharply than the final target alone,
surfacing methods that descend fast early but stall late (e.g. momentum) vs.
those that accelerate near the optimum (e.g. deep-memory QQN).

### Reported Metrics

The summary table reports several derived efficiency metrics:

- **`ms/it`** — mean wall-clock cost per accepted iteration (total wall-time
  divided by the number of accepted iterations), a clean per-step cost metric.
- **`vs LBFGS`** — speedup factor in iterations-to-target relative to the
  classical L-BFGS baseline (`lbfgs_iters / variant_iters`); values above
  `1.00x` indicate a variant reaches the shared target in fewer iterations
  than L-BFGS.
- **`AUC`** — the **trajectory AUC**: `log10(loss)` integrated over the
  normalized iteration axis (trapezoid rule). A *lower* (more negative) AUC
  means the optimizer spent its whole trajectory at lower loss, rewarding
  fast early descent **and** deep late refinement simultaneously — a far more
  discriminating single-scalar summary than a single time-to-target.

## Baseline Comparison

With all defaults (L-BFGS oracle, Armijo line search, no region), QQN reaches
a substantially lower full-batch loss than the first-order baselines and is
competitive with — and faster than — Optax's L-BFGS, all converging to the
shared `f_target = 1.1e-1` within the iteration/time budget.

| Optimizer | Final loss | Iters | Train acc | Test acc | Time (s) | ->target | vs LBFGS |
|-----------|-----------:|------:|----------:|---------:|---------:|---------:|---------:|
| QQN       |  1.096e-01 |    65 |    0.9902 |   0.8700 |    1.534 |       65 |    1.08x |
| L-BFGS    |  1.098e-01 |    70 |    0.9910 |   0.8750 |    2.268 |       70 |    1.00x |
| Adam      |  1.100e-01 |   263 |    0.9898 |   0.8810 |    0.546 |      263 |    0.27x |
| SGD       |  2.266e-01 |   500 |    0.9422 |   0.8900 |    0.611 |        — |        — |

**Observations:**

- QQN reaches the shared loss target in **65 iterations**, fewer than Optax's
  L-BFGS (70, a **1.08×** iteration speedup) and running ~**1.5× faster** in
  wall-clock time, owing to its cheap Armijo backtracking search.
- **Adam** also reaches the target but needs **263 iterations** — far more
  than the quasi-Newton methods (a `0.27x` iteration speedup vs L-BFGS) —
  though its per-iteration cost is so low (~2.08 ms/it) that it is the fastest
  in wall-clock time (0.546s) on this problem.
- **SGD never reaches** the `f_target = 1.1e-1` target within 500 iterations,
  plateauing at `2.266e-01`.
- Test accuracy is similar across the strong optimizers; the differentiator
  here is optimization speed and iterations-to-target, not generalization
  (Adam actually has the highest test accuracy at 0.8810).

## QQN Component Sweeps (A/B Comparisons)

Because gradient, oracle, line search, and region are conceptually orthogonal
and independently swappable, the experiment runs controlled A/B sweeps where
each pair isolates a single variable against a named baseline. Each pair's
first entry is the baseline; later entries report deltas against it.

### Oracle: L-BFGS History Depth

Deeper L-BFGS history monotonically reduces the **iterations to target**, with
clear diminishing returns past size 50 and a hard plateau at size 100. The
converged final loss is essentially flat across depths (every variant hits the
shared target), so the lever here is *speed of convergence*, not final loss.

| Variant  | History | Final loss | Iters | ->target | Time (s) | AUC   |
|----------|--------:|-----------:|------:|---------:|---------:|------:|
| QQN-L5   |       5 |  1.093e-01 |    80 |       80 |    1.386 | -0.73 |
| QQN      |      10 |  1.096e-01 |    65 |       65 |    1.534 | -0.70 |
| QQN-L20  |      20 |  1.096e-01 |    59 |       59 |    1.181 | -0.68 |
| QQN-L50  |      50 |  1.097e-01 |    43 |       43 |    1.042 | -0.61 |
| QQN-L100 |     100 |  1.097e-01 |    43 |       43 |    1.115 | -0.61 |

The sweep `L5 > L10 > L20 > L50` in iterations-to-target (80 → 65 → 59 → 43)
confirms richer curvature memory accelerates convergence, but the count
plateaus exactly at 43 iterations from size 50 onward (`L50 == L100`) while
wall-time keeps growing — so very deep histories (L100) buy *no* extra speed
for extra cost on this problem.

### Oracle: Secant (Barzilai-Borwein)

The **secant** oracle is a matrix-free, `O(n)`-memory curvature estimate that
reuses the *realized* step's secant `(s, y)` to form a Barzilai-Borwein step
`α = ⟨s,s⟩/⟨s,y⟩`, then proposes `-α·∇f`. It carries no Hessian and no history
buffers — it probes how much curvature lives in a *single* realized step.

| Variant | Oracle             | Final loss | Iters | ->target | AUC   |
|---------|--------------------|-----------:|------:|---------:|------:|
| QQN-Sec | Secant (BB1, O(n)) |  1.097e-01 |   311 |      311 | -0.74 |

`QQN-Sec` does eventually reach the shared target, but needs **311 iterations**
(a `0.23x` iteration speedup vs L-BFGS) — far more than any L-BFGS depth, yet
far fewer than the momentum oracles (which never reach it). Notably its
**trajectory AUC of `-0.74` is the best among all QQN variants** (and second
only to Adam overall): the BB step descends fast and deep on average even
though it takes many iterations to formally cross the tight target. This makes
the single-step secant a strong, zero-storage curvature signal.

### Oracle: Momentum (heavy-ball) `beta`

The momentum oracle is a first-order accelerator and, as expected, **never
reaches the target** within 500 iterations. Notably, *lighter* damping
converges to a lower loss on this problem (the sweep is monotone in `beta`).

| Variant   | beta | Final loss | Iters | ->target | Time (s) |
|-----------|-----:|-----------:|------:|---------:|---------:|
| QQN-Mom01 | 0.01 |  1.892e-01 |   500 |        — |    5.496 |
| QQN-Mom10 | 0.10 |  1.940e-01 |   500 |        — |    5.441 |
| QQN-Mom50 | 0.50 |  2.265e-01 |   500 |        — |    5.550 |
| QQN-Mom   | 0.90 |  3.419e-01 |   500 |        — |    6.257 |

Near-zero momentum (`beta = 0.01`) effectively collapses toward steepest
descent, which on this smooth full-batch problem outperforms heavier momentum
(`Mom01 < Mom10 < Mom50 < Mom` in loss). All momentum variants exhaust the
full 500-iteration budget without reaching the target. The convergence-rate
profile is revealing: the lighter-damping variants (`Mom01`, `Mom10`) do
eventually cross the `2.0e-1` milestone (at iterations 410 and 449
respectively), while the heavier `Mom50`/`Mom` never do.

### Oracle: Accelerator Class (Momentum vs Shampoo)

The structure-aware **Shampoo** oracle recomputes inverse matrix roots on a
static cadence (`update_freq=25`) over a *blocked* preconditioner
(`block_size=64`), so the per-refresh eigendecomposition is `O(block³)`
instead of `O(n³)`. Even so, on this high-dimensional softmax problem the
refresh is expensive: `QQN-Sh` exhausts the shared **15-second wall-clock
budget** after only **9 iterations** and lands at a much higher loss than even
the momentum oracle.

| Variant   | Oracle              | Final loss | Iters | ->target | Time (s) |
|-----------|---------------------|-----------:|------:|---------:|---------:|
| QQN-Mom10 | Momentum (beta=0.1) |  1.940e-01 |   500 |        — |    5.441 |
| QQN-Sh    | Shampoo (block=64)  |  8.883e-01 |     9 |        — |   16.141 |

Shampoo's blocked inverse-root refresh still does not amortize well at this
scale (~1793 ms/it); it is the only oracle to exhaust the time budget before
reaching `maxiter`, and the only variant whose convergence-rate profile never
crosses even the loosest `5.0e-1` milestone.

### Region: Trust-Region Radius and Adaptivity — a Cautionary Result

> **Important:** with the current honest predicted-reduction model and the
> default Armijo (init_step=1.0) line search, the *adaptive* trust-region
> **destabilizes** convergence on this problem. Several trust-region variants
> that previously won the race now **stall and never reach the target**.

| Variant   | Radius | Adaptive | Final loss | ->target | Time (s) |
|-----------|-------:|:--------:|-----------:|---------:|---------:|
| QQN-TRfix |   1.00 |    no    |  1.099e-01 |       68 |    1.357 |
| QQN-TR025 |   0.25 |   yes    |  1.983e+00 |        — |    6.897 |
| QQN-TR    |   1.00 |   yes    |  7.772e-01 |        — |    6.820 |
| QQN-TR2   |   2.00 |   yes    |  6.256e-01 |        — |    6.551 |

The only trust-region variant to reach the target is the **fixed** radius
`QQN-TRfix` (68 iterations). Every *adaptive* radius (`TR025`, `TR`, `TR2`)
plateaus early: `QQN-TR` flatlines at `7.772e-01` and `QQN-TR2` at
`6.256e-01`. The tighter radius is the worst (`TR025` ends at `1.983e+00`,
i.e. *above* its start). This is a sharp reversal from earlier runs where the
adaptive trust-region marginally accelerated convergence: the current honest
`pred = -⟨∇f, d(t)⟩` model, combined with the radial step-clipping, drives the
adaptive radius to over-shrink and stall the search.

The **adaptivity** A/B makes this stark:

| Variant   | Adaptive | Final loss | ->target | Time (s) |
|-----------|:--------:|-----------:|---------:|---------:|
| QQN-TRfix |    no    |  1.099e-01 |       68 |    1.357 |
| QQN-TR    |   yes    |  7.772e-01 |        — |    6.820 |

Switching the radius from fixed to adaptive moves the result from *converged
at 68 iterations* to *never reaches the target*. **Use a fixed trust-region
radius (or no region) on this class of well-conditioned smooth problem.**

### Region: Combinator and Orthant Sparsity

The combinator `Sequential([Box, TrustRegion])` composes two projections in
order at negligible extra cost, and the orthant region is the only QQN region
to induce measurable weight sparsity.

| Variant  | Configuration                    | Final loss | Sparsity | ->target |
|----------|----------------------------------|-----------:|---------:|---------:|
| QQN-Box  | BoxRegion(-2, 2)                 |  1.098e-01 |   0.0000 |       65 |
| QQN-Orth | OrthantRegion (OWL-QN-style)     |  1.100e-01 |   0.0027 |       70 |
| QQN-Seq  | Sequential([Box(-2,2), TR(1.0)]) |  7.772e-01 |   0.0001 |        — |

The **box** region adds negligible cost while bounding weights (reaches target
at 65). The **orthant** region induces measurable sparsity (0.0027). The
`Sequential` combinator *inherits the adaptive trust-region's stall*
(`7.772e-01`, never reaching target): composition itself is correct, but its
nested adaptive `TrustRegion(1.0)` carries the same destabilizing behavior
documented above.

### Line Search (at fixed oracle depth, L-BFGS-10)

The line search choice has negligible effect on the *iterations-to-target* for
the **backtracking/Armijo** family but a large effect on **wall-time** — and a
*dramatic* effect for **strong-Wolfe**, which fails to converge here.

| Variant  | Line search   | Final loss | ->target | Time (s) | AUC   |
|----------|---------------|-----------:|---------:|---------:|------:|
| QQN      | armijo        |  1.096e-01 |       65 |    1.534 | -0.70 |
| QQN-BT   | backtracking  |  1.096e-01 |       65 |    1.277 | -0.70 |
| QQN-Spln | armijo+spline |  1.094e-01 |       62 |    2.590 | -0.71 |
| QQN-SW   | strong_wolfe  |  4.077e-01 |        — |    7.492 | -0.38 |

The default search is Armijo backtracking (`QQN`, 1.534s); the dedicated
`QQN-BT` backtracking variant is the cheapest robust search (1.277s) and
reaches the target in the same 65 iterations. The cubic Hermite spline
refinement (`QQN-Spln`) reaches the target slightly earlier (62) but at ~1.7×
the wall-time. **Strong-Wolfe (`QQN-SW`) fails to converge** on this problem,
plateauing at `4.077e-01` after exhausting all 500 iterations — its tight
curvature condition over-restricts the step along the quadratic path here.

### Spline Refinement (orthogonal enhancement)

The spline is **not** a line-search strategy but a boolean enhancement
(`spline=True`) that *wraps* any chosen line search (`spline_wrap(inner_search)`).
It reuses every probe along the consistent path as a cubic Hermite control
point, performs a **superlinear extension probe** when the downstream tangent
still descends (`m1 < 0`), and probes the spline's stationary points to
improve on the inner search's accepted step.

| Variant       | Configuration                           | Final loss | ->target |
|---------------|-----------------------------------------|-----------:|---------:|
| QQN-Spln      | armijo + spline                         |  1.094e-01 |       62 |
| QQN-BTSpln    | backtracking + spline                   |  1.094e-01 |       62 |
| QQN-L50Spln   | L50 oracle + spline                     |  1.092e-01 |       44 |
| QQN-L100Spln  | L100 oracle + spline                    |  1.092e-01 |       44 |
| QQN-SplnTR    | armijo + spline + adaptive trust-region |  1.096e-01 |       66 |
| QQN-L50SplnTR | L50 + spline + adaptive trust-region    |  1.095e-01 |       45 |

The spline refinement notably **sharpens the deep-memory trajectory**:
`QQN-L50Spln` reaches the target in **44 iterations** (vs the spline-less L50
baseline at 43 — essentially tied) with the **lowest loss observed across the
whole study (`1.092e-01`)** and the **fewest iterations of any converging
variant**. Crucially, the spline-wrapped variants are *immune* to the adaptive
trust-region stall that afflicts the bare `QQN-TR` family: `QQN-SplnTR` (66)
and `QQN-L50SplnTR` (45) both converge cleanly, because the spline's
region-projected probes and strict-improvement gating keep the search
monotone even when the radius adapts. On the smooth convex objective the extra
per-probe spline fitting costs roughly ~2× wall-time for the shallow-memory
variants.

## Best-of-Breed Combinations

The strongest **converging** stacks here are the **deep-memory + spline**
combinations, not the deep-memory + bare-adaptive-trust-region stacks (which
stall, see below). The experiment also probes several *performance-tuned*
stacks that warm-start the line search at a larger initial step and
concentrate the t-grid near the pure-oracle endpoint.

| Variant       | Configuration                           | Final loss | ->target | Time (s) |
|---------------|-----------------------------------------|-----------:|---------:|---------:|
| QQN-L50       | L50 oracle (no region)                  |  1.097e-01 |       43 |    1.042 |
| QQN-L100      | L100 oracle (no region)                 |  1.097e-01 |       43 |    1.115 |
| QQN-L50Spln   | L50 + spline                            |  1.092e-01 |       44 |    2.334 |
| QQN-L100Spln  | L100 + spline                           |  1.092e-01 |       44 |    2.385 |
| QQN-L50SplnTR | L50 + spline + adaptive trust-region    |  1.095e-01 |       45 |    2.456 |
| QQN-Best      | L50 + BT + spline + adaptive TR         |  1.095e-01 |       45 |    2.467 |
| QQN-L20HZ     | L20 + Hager-Zhang                       |  1.093e-01 |       62 |    1.213 |

The fewest iterations to target (**43**) are reached by the **bare deep-memory
oracles** `QQN-L50` and `QQN-L100`, at the lowest wall-time (~1.04–1.12s) — a
strong pareto point on iterations vs. time. The **lowest loss** (`1.092e-01`)
is reached by the deep-memory + spline combos `QQN-L50Spln` / `QQN-L100Spln`
at 44 iterations.

> **Critical caveat — the bare deep-memory + adaptive-trust-region stacks
> stall.** Combos such as `QQN-L50TR`, `QQN-L100TR`, `QQN-L50BTTR`, and
> `QQN-L50Tnear1` all **fail to converge**, plateauing at `7.772e-01` after
> the full 500 iterations. The warm-started backtracking stacks
> (`QQN-L50BTTR+`, `QQN-L50BTTR++`, `QQN-L50Endpt`, `QQN-Fast`,
> `QQN-Champion`) fare *even worse* (final losses `0.87`–`1.08`), because
> probing beyond `α = 1` interacts badly with the radial trust-region clip and
> the over-shrinking adaptive radius. **On this problem, the adaptive
> trust-region is the single most fragile component**, and any stack that
> relies on it (without the spline's monotone gating) stalls.

The robust path to the best loss/iteration trade-off here is therefore:
**deep L-BFGS memory (L50/L100) + backtracking + the spline refinement**, and
*without* a bare adaptive trust-region.

### Pareto Frontier (loss vs. wall-time)

The experiment reports the **Pareto frontier** of non-dominated variants:
those for which no other variant is both faster *and* lower-loss.

| Variant     | Final loss | Time (s) |
|-------------|-----------:|---------:|
| Adam        | 1.0999e-01 |    0.546 |
| QQN-L50     | 1.0968e-01 |    1.042 |
| QQN-L20     | 1.0959e-01 |    1.181 |
| QQN-L20HZ   | 1.0932e-01 |    1.213 |
| QQN-L5      | 1.0931e-01 |    1.386 |
| QQN-L50Spln | 1.0920e-01 |    2.334 |

Adam anchors the cheap-but-higher-loss end of the frontier; the QQN variants
trade increasing wall-time for progressively lower loss, with `QQN-L50` the
standout efficiency/quality balance among the quasi-Newton methods (reaching
`1.097e-01` in close to a single second), and `QQN-L50Spln` the lowest-loss
frontier point overall (`1.092e-01`).

### Trajectory-AUC Leaderboard

Ranking optimizers by the single-scalar **trajectory AUC** (lower = faster
*overall* descent — both early and late phase) gives a complementary view to
iterations-to-target:

| Variant   | AUC    | Final loss | Time (s) |
|-----------|-------:|-----------:|---------:|
| Adam      | -0.821 | 1.0999e-01 |    0.546 |
| QQN-Sec   | -0.739 | 1.0974e-01 |    4.018 |
| QQN-L5    | -0.726 | 1.0931e-01 |    1.386 |
| QQN-Orth  | -0.715 | 1.1000e-01 |    1.375 |
| QQN-Spln  | -0.706 | 1.0941e-01 |    2.590 |
| QQN-BTSpln| -0.706 | 1.0941e-01 |    2.481 |
| L-BFGS    | -0.705 | 1.0977e-01 |    2.268 |

**Adam leads the AUC leaderboard** (its many cheap iterations keep it at low
loss throughout), with the matrix-free **secant oracle (`QQN-Sec`) a close
second** — a striking result for a zero-storage curvature estimate. The
spline-augmented variants and L-BFGS cluster just behind.

### Convergence-Rate Profile

The milestone profile crisply separates early- from late-phase descent. The
fastest variants to reach the tightest `1.2e-1` milestone are the
**deep-memory + spline** combos:

| Variant      | `≤5.0e-1` | `≤2.0e-1` | `≤1.5e-1` | `≤1.2e-1` |
|--------------|----------:|----------:|----------:|----------:|
| QQN-L50Spln  |         7 |        21 |        29 |        38 |
| QQN-L100Spln |         7 |        21 |        29 |        38 |
| QQN-L50      |         7 |        22 |        30 |        38 |
| QQN-L100     |         7 |        22 |        30 |        38 |
| QQN-L50SplnTR|         8 |        24 |        31 |        39 |
| QQN-Best     |         8 |        24 |        31 |        39 |
| QQN-L20      |         7 |        22 |        33 |        46 |
| QQN (L10)    |         7 |        24 |        35 |        52 |
| L-BFGS       |         7 |        25 |        37 |        54 |
| Adam         |         8 |        42 |        77 |       167 |

The profile makes the deep-memory advantage stark: `QQN-L50Spln` crosses the
`1.2e-1` milestone at iteration **38**, while the L10 baseline takes 52,
L-BFGS takes 54, and Adam takes 167. The momentum oracles, SGD, and **all
bare adaptive-trust-region stacks** never reach even the `1.5e-1` milestone
(and most never cross `2.0e-1`).

## Combinator and Constraint Variants

The experiment also exercises the combinator oracles and regions to confirm
they run correctly and produce sensible behavior:

| Variant    | Configuration                    | Final loss | Sparsity | ->target |
|------------|----------------------------------|-----------:|---------:|---------:|
| QQN-Fall   | Fallback([L-BFGS(10), Momentum]) |  1.096e-01 |   0.0000 |       65 |
| QQN-Box    | BoxRegion(-2, 2)                 |  1.098e-01 |   0.0000 |       65 |
| QQN-Orth   | OrthantRegion (OWL-QN-style)     |  1.100e-01 |   0.0027 |       70 |
| QQN-L20Box | L-BFGS(20) + BoxRegion(-2, 2)    |  1.100e-01 |   0.0000 |       60 |
| QQN-L50Sec | Fallback([L-BFGS(50), Secant])+TR|  7.772e-01 |   0.0001 |        — |

- **Fallback** reproduces the L-BFGS baseline exactly here (1.096e-01, target
  at iteration 65), because the L-BFGS direction is always valid (finite,
  non-zero), so the momentum fallback never triggers.
- **OrthantRegion** is the only configuration to induce measurable weight
  sparsity (0.0027), as expected from its sign-preserving projection.
- The **box** and **L20+box** constraints add negligible cost while keeping
  weights bounded; the deeper L20 oracle lets `QQN-L20Box` reach the target
  in 60 iterations, ahead of the shallow box variant (65).
- **`QQN-L50Sec`** (a `Fallback([L-BFGS(50), Secant])` paired with an adaptive
  `TrustRegion`) **stalls** at `7.772e-01` — the oracle fallback is sound, but
  it again inherits the adaptive-trust-region instability documented above.

## Loss Trajectories

The log records a compact log10-scale, sampled view of every trajectory. The
qualitative picture (over 10 sampled points across the run):

- **QQN (and most L-BFGS-10 variants)** drop from `0.36` to roughly `-0.96`
  in log10 loss by the end of the run.
- The **deep-memory + spline combos** (`QQN-L50Spln`, `QQN-L100Spln`,
  `QQN-Best`, `QQN-L50SplnTR`) match the deepest trajectory, reaching `-0.96`
  by the end.
- **Adam** descends fastest in log10 terms, reaching `-0.93` by the seventh
  sample and `-0.96` by the end (it just needs many cheap iterations).
- The **secant oracle** (`QQN-Sec`) descends fast and deep, reaching `-0.96`
  by the end despite needing many iterations to formally cross the target.
- **SGD and the heavier momentum oracles** plateau between `-0.47` and
  `-0.64` (lighter momentum `QQN-Mom01`/`QQN-Mom10` reaching `-0.72`).
- **The bare adaptive-trust-region stacks** (`QQN-TR`, `QQN-L50TR`,
  `QQN-L50BTTR`, …) flatline early at `-0.11` (loss `7.772e-01`); the
  warm-started + TR stacks plateau even higher (`-0.00` to `+0.03`).
- **Shampoo** (`QQN-Sh`) barely moves before exhausting the time budget,
  reaching only `-0.05` after 9 iterations.

## Key Takeaways

1. **QQN converges to the shared target in fewer iterations than L-BFGS** on a
   smooth, deterministic full-batch problem, at a fraction of the wall-time,
   and clearly outperforms SGD (which never reaches the target). Adam reaches
   the target but needs ~4× the iterations of the quasi-Newton methods.
2. **L-BFGS history depth is the dominant convergence-speed lever**, cutting
   iterations-to-target from 80 (L5) to 43 (L50), with diminishing returns
   past size 50 and a hard plateau at 100 (`L50 == L100` at 43 iterations).
3. **The line search choice trades wall-time, not convergence speed**, within
   the backtracking/Armijo family — backtracking/Armijo is the clear
   efficiency winner. The cubic Hermite spline matches their
   iterations-to-target while reaching a slightly *lower* loss, at ~2× the
   time. **Strong-Wolfe fails to converge** on this problem (its tight
   curvature condition over-restricts the path step).
4. **The spline refinement composes with any line search** (it wraps the inner
   search rather than replacing it), achieves the **lowest loss observed**
   (`1.092e-01` for `QQN-L50Spln`), and — thanks to its region-projected,
   strict-improvement-gated probes — is **immune to the adaptive
   trust-region instability** that stalls the bare TR stacks.
5. **The adaptive trust-region is the single most fragile component here.**
   Under the current honest predicted-reduction model, *adaptive*
   trust-region stacks (`QQN-TR`, `QQN-L50TR`, `QQN-L50BTTR`, and the
   warm-started variants) **stall and never reach the target**. A *fixed*
   radius (`QQN-TRfix`) converges cleanly (68 iterations). On this class of
   well-conditioned smooth problem, prefer no region (or a fixed radius / box
   / orthant) over an adaptive trust-region.
6. **The matrix-free secant oracle is a strong, zero-storage curvature
   signal**: `QQN-Sec` reaches the target (in 311 iterations) and posts the
   **best trajectory AUC of any QQN variant** (`-0.74`), crushing plain
   momentum at `O(n)` memory and no Hessian.
7. **Oracle choice matters more than search or region**: deep L-BFGS dominates,
   the secant trails but reaches target, momentum never converges, and the
   blocked Shampoo oracle does not scale to this high-dimensional problem
   within the time budget.
8. **The convergence-rate milestone profile confirms the deep-memory edge**:
   the L50/L100 (+spline) combos cross the tight `1.2e-1` milestone at
   iteration 38 — far ahead of the L10 baseline (52), L-BFGS (54), and Adam
   (167).

## Reproducing

```bash
pip install -e ".[dev]"
python examples/mnist_comparison.py
```

The script prints the summary table (including `ms/it` per-iteration cost,
the `->target` iteration and `t->tgt` wall-clock time at which the shared
loss/gradient target was first hit, the `vs LBFGS` iteration speedup, and the
`AUC` trajectory integral), the Pareto frontier of non-dominated variants, the
trajectory-AUC leaderboard, the convergence-rate profile (first iteration
crossing each loss milestone), sampled log10 trajectories, and the controlled
A/B comparison report. It also saves both a `mnist_comparison.png` (loss vs.
iteration) and a `mnist_comparison_time.png` (loss vs. wall-clock time)
convergence plot when `matplotlib` is available.
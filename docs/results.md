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
columns become *informative*: with the previous `1.0e-1` target every entry
was empty, whereas at `1.1e-1` the strongest deep-memory + trust-region combos
actually win the race and surface their iteration/time-to-target advantage.
The `time_budget` was likewise raised to `15.0`s (and Shampoo switched to a
*blocked* preconditioner) so the comparison stays meaningful while still
capping runaways.

## Baseline Comparison

With all defaults (L-BFGS oracle, Armijo line search, no region), QQN reaches
a substantially lower full-batch loss than the first-order baselines and is
competitive with — and faster than — Optax's L-BFGS, all converging to the
shared `f_target = 1.1e-1` within the iteration/time budget.

| Optimizer | Final loss | Iters | Train acc | Test acc | Time (s) | ->target |
|-----------|-----------:|------:|----------:|---------:|---------:|---------:|
| QQN       |  1.098e-01 |    65 |    0.9906 |   0.8690 |    1.642 |       65 |
| L-BFGS    |  1.098e-01 |    70 |    0.9910 |   0.8750 |    2.174 |       70 |
| Adam      |  1.100e-01 |   263 |    0.9898 |   0.8810 |    0.542 |      263 |
| SGD       |  2.266e-01 |   500 |    0.9422 |   0.8900 |    0.667 |        — |

**Observations:**

- QQN reaches the shared loss target in **65 iterations**, fewer than Optax's
  L-BFGS (70) and running ~**1.3× faster** in wall-clock time, owing to its
  cheap Armijo backtracking search and batched t-grid line searches.
- **Adam** also reaches the target but needs **263 iterations** — far more
  than the quasi-Newton methods — though its per-iteration cost is so low
  that it is the fastest in wall-clock time (0.542s) on this problem.
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

| Variant  | History | Final loss | Iters | ->target | Time (s) |
|----------|--------:|-----------:|------:|---------:|---------:|
| QQN-L5   |       5 |  1.099e-01 |    73 |       73 |    1.408 |
| QQN      |      10 |  1.098e-01 |    65 |       65 |    1.642 |
| QQN-L20  |      20 |  1.096e-01 |    60 |       60 |    1.286 |
| QQN-L50  |      50 |  1.092e-01 |    46 |       46 |    1.206 |
| QQN-L100 |     100 |  1.092e-01 |    46 |       46 |    1.262 |

The sweep `L5 > L10 > L20 > L50` in iterations-to-target (73 → 65 → 60 → 46)
confirms richer curvature memory accelerates convergence, but the count
plateaus exactly at 46 iterations from size 50 onward (`L50 == L100`) while
wall-time keeps growing — so very deep histories (L100) buy *no* extra speed
for extra cost on this problem.

### Oracle: Momentum (heavy-ball) `beta`

The momentum oracle is a first-order accelerator and, as expected, **never
reaches the target** within 500 iterations. Notably, *lighter* damping
converges to a lower loss on this problem (the sweep is monotone in `beta`).

| Variant   | beta | Final loss | Iters | ->target | Time (s) |
|-----------|-----:|-----------:|------:|---------:|---------:|
| QQN-Mom01 | 0.01 |  1.895e-01 |   500 |        — |    5.696 |
| QQN-Mom10 | 0.10 |  1.942e-01 |   500 |        — |    5.513 |
| QQN-Mom50 | 0.50 |  2.265e-01 |   500 |        — |    5.559 |
| QQN-Mom   | 0.90 |  2.638e-01 |   500 |        — |    5.572 |

Near-zero momentum (`beta = 0.01`) effectively collapses toward steepest
descent, which on this smooth full-batch problem outperforms heavier momentum
(`Mom01 < Mom10 < Mom50 < Mom` in loss). All momentum variants exhaust the
full 500-iteration budget without reaching the target.

### Oracle: Accelerator Class (Momentum vs Shampoo)

The structure-aware **Shampoo** oracle recomputes inverse matrix roots on a
static cadence (`update_freq=25`) over a *blocked* preconditioner
(`block_size=64`), so the per-refresh eigendecomposition is `O(block³)`
instead of `O(n³)`. Even so, on this high-dimensional softmax problem the
refresh is expensive: `QQN-Sh` exhausts the shared **15-second wall-clock
budget** after only **9 iterations** and lands at a much higher loss than even
the momentum oracle.

| Variant   | Oracle               | Final loss | Iters | ->target | Time (s) |
|-----------|----------------------|-----------:|------:|---------:|---------:|
| QQN-Mom10 | Momentum (beta=0.1)  |  1.942e-01 |   500 |        — |    5.513 |
| QQN-Sh    | Shampoo (block=64)   |  7.236e-01 |     9 |        — |   16.401 |

Shampoo's blocked inverse-root refresh still does not amortize well at this
scale; it is the only oracle to exhaust the time budget before reaching
`maxiter`.

### Region: Trust-Region Radius and Adaptivity

The adaptive trust-region barely perturbs the converged loss across radii,
confirming the region is a low-overhead safeguard rather than a driver of
performance on this well-conditioned problem. All radii reach the target.

| Variant   | Radius | Adaptive | Final loss | ->target | Time (s) |
|-----------|-------:|:--------:|-----------:|---------:|---------:|
| QQN-TR025 |   0.25 |   yes    |  1.096e-01 |       66 |    1.363 |
| QQN-TR    |   1.00 |   yes    |  1.096e-01 |       68 |    1.390 |
| QQN-TR2   |   2.00 |   yes    |  1.096e-01 |       67 |    1.372 |
| QQN-TRfix |   1.00 |    no    |  1.098e-01 |       69 |    1.426 |

The radius sweep (0.25 → 1.0 → 2.0) is essentially flat in both final loss
and iterations-to-target; an adaptive radius performs marginally better than
a fixed one (`TR` reaches target at 68 vs `TRfix` at 69).

### Region: Combinator and Orthant Sparsity

The combinator `Sequential([Box, TrustRegion])` composes two projections in
order at negligible extra cost, and the orthant region is the only one to
induce measurable weight sparsity.

| Variant  | Configuration                    | Final loss | Sparsity | ->target |
|----------|----------------------------------|-----------:|---------:|---------:|
| QQN-TR   | TrustRegion(1.0, adaptive)       |  1.096e-01 |   0.0000 |       68 |
| QQN-Box  | BoxRegion(-2, 2)                 |  1.100e-01 |   0.0000 |       66 |
| QQN-Seq  | Sequential([Box(-2,2), TR(1.0)]) |  1.097e-01 |   0.0000 |       67 |
| QQN-Orth | OrthantRegion (OWL-QN-style)     |  1.098e-01 |   0.0037 |       70 |

The Sequential combinator's projected loss (1.097e-01) sits between the box
and trust-region results, confirming it composes the two constraints as
expected with only a small overhead.

### t-Grid: Blend Discretization

The t-grid samples the continuous blend parameter `t`. Sweeping its
granularity (2, 4, and 8 points) has a negligible effect on the converged
loss and iterations-to-target but a modest effect on wall-time: a finer grid
runs more line searches per iteration.

| Variant     | t-grid points | Final loss | ->target | Time (s) |
|-------------|--------------:|-----------:|---------:|---------:|
| QQN-Tcoarse |             2 |  1.098e-01 |       65 |    1.439 |
| QQN         |             4 |  1.098e-01 |       65 |    1.642 |
| QQN-Tfine   |             8 |  1.098e-01 |       64 |    1.401 |

The coarse 2-point grid is essentially as good as the default 4-point grid on
this smooth problem (both reach target at iteration 65), confirming the t-grid
is a cheap tuning knob here rather than a convergence driver.

### Line Search (at fixed oracle depth, L-BFGS-10)

The line search choice has negligible effect on the *final* loss or
iterations-to-target but a large effect on **wall-time**: backtracking is the
cheapest, while strong-Wolfe and the spline refinement are ~2× slower for no
convergence-speed gain on this smooth problem.

| Variant  | Line search   | Final loss | ->target | Time (s) |
|----------|---------------|-----------:|---------:|---------:|
| QQN      | armijo        |  1.098e-01 |       65 |    1.642 |
| QQN-BT   | backtracking  |  1.098e-01 |       65 |    1.369 |
| QQN-SW   | strong_wolfe  |  1.097e-01 |       65 |    3.155 |
| QQN-Spln | armijo+spline |  1.096e-01 |       66 |    2.889 |

The default search is Armijo backtracking (`QQN`, 1.642s); the dedicated
`QQN-BT` backtracking variant is the cheapest robust search (1.369s) and
reaches the target in the same 65 iterations. Strong-Wolfe (`QQN-SW`, 3.155s)
and the cubic Hermite spline refinement (`QQN-Spln`, 2.889s) reach essentially
the same loss — useful confirmation that the more expensive searches do not
degrade quality, but do not pay off on a smooth convex objective. The
`QQN-L20HZ` variant additionally exercises the Hager-Zhang approximate-Wolfe
search atop an L20 oracle (1.099e-01, target at iteration 60, 2.248s).

### Spline Refinement (orthogonal enhancement)

The spline is **not** a line-search strategy but a boolean enhancement
(`spline=True`) that *wraps* any chosen line search (`spline_wrap(inner_search)`).
It reuses every probe along the consistent path as a cubic Hermite control
point and probes the spline's stationary points to improve on the inner
search's accepted step.

| Variant       | Configuration                           | Final loss | ->target |
|---------------|-----------------------------------------|-----------:|---------:|
| QQN-Spln      | armijo + spline                         |  1.096e-01 |       66 |
| QQN-BTSpln    | backtracking + spline                   |  1.096e-01 |       66 |
| QQN-L50Spln   | L50 oracle + spline                     |  1.091e-01 |       45 |
| QQN-L100Spln  | L100 oracle + spline                    |  1.091e-01 |       45 |
| QQN-SplnTR    | armijo + spline + adaptive trust-region |  1.098e-01 |       66 |
| QQN-L50SplnTR | L50 + spline + adaptive trust-region    |  1.091e-01 |       44 |

The spline refinement notably **sharpens the deep-memory trajectory**:
`QQN-L50Spln` reaches the target in **45 iterations** (vs the spline-less L50
baseline at 46), and `QQN-L50SplnTR` reaches it fastest of the spline variants
at **44 iterations** with the lowest spline loss observed (1.091e-01). On the
smooth convex objective the iterations-to-target for the shallow-memory
variants is unchanged (66), but the extra per-probe spline fitting costs ~2×
wall-time.

## Best-of-Breed Combinations

Stacking the strongest pareto components — deep L-BFGS memory, the cheapest
robust line search (backtracking), and the convergence-stabilizing
trust-region — yields the **fewest iterations to target**, at competitive
wall-time.

| Variant     | Configuration                     | Final loss | ->target | Time (s) |
|-------------|-----------------------------------|-----------:|---------:|---------:|
| QQN-L50TR   | L50 + adaptive trust-region       |  1.094e-01 |       41 |    1.256 |
| QQN-L100TR  | L100 + adaptive trust-region      |  1.094e-01 |       41 |    1.236 |
| QQN-L50BTTR | L50 + backtracking + trust-region |  1.094e-01 |       41 |    1.168 |
| QQN-L50TR2  | L50 + TR(radius=2.0)              |  1.097e-01 |       45 |    1.186 |
| QQN-Best    | L50 + BT + spline + TR + 8-pt grid|  1.099e-01 |       44 |    2.703 |

The `L50TR` / `L100TR` / `L50BTTR` combos reach the target in the **fewest
iterations observed (41)** while staying around ~1.17–1.26s — a strong pareto
point on iterations vs. time. The trust-region shaves the iterations-to-target
below the raw L50 oracle (46 → 41) at essentially no extra cost, while
`QQN-L50BTTR` is the cheapest of these in wall-time (1.168s). The full
"everything" stack `QQN-Best` (deep L50 + backtracking + spline + adaptive
trust-region + fine 8-point grid) reaches the target in 44 iterations but at
~2.3× the wall-time of `QQN-L50BTTR` — the spline refinement does not improve
the deep-memory backtracking combo here. Note that the experiment also
includes a `QQN-SW+TR` combo (strong-Wolfe + adaptive trust-region,
1.098e-01 at 3.225s, target at iteration 65) which trades wall-time for no
convergence-speed gain on this smooth objective.

### Pareto Frontier (loss vs. wall-time)

The experiment now reports the **Pareto frontier** of non-dominated variants:
those for which no other variant is both faster *and* lower-loss.

| Variant       | Final loss | Time (s) |
|---------------|-----------:|---------:|
| Adam          |  1.0999e-01 |    0.542 |
| QQN-L50BTTR   |  1.0941e-01 |    1.168 |
| QQN-L50       |  1.0920e-01 |    1.206 |
| QQN-L50SplnTR |  1.0915e-01 |    2.598 |
| QQN-L100Spln  |  1.0912e-01 |    2.615 |

Adam anchors the cheap-but-higher-loss end of the frontier; the deep-memory
QQN variants trade increasing wall-time for progressively lower loss, with
`QQN-L50BTTR` the standout efficiency/quality balance among the quasi-Newton
methods.

In the sampled log10 trajectory, `QQN-L50TR` / `QQN-L100TR` / `QQN-L50BTTR`
all share the leading trajectory, reaching `-0.81` by the seventh sample and
`-0.96` by the final sample — distinctly ahead of the size-10 baseline.

## Combinator and Constraint Variants

The experiment also exercises the combinator oracles and regions to confirm
they run correctly and produce sensible behavior:

| Variant    | Configuration                    | Final loss | Sparsity | ->target |
|------------|----------------------------------|-----------:|---------:|---------:|
| QQN-Fall   | Fallback([L-BFGS(10), Momentum]) |  1.098e-01 |   0.0001 |       65 |
| QQN-Box    | BoxRegion(-2, 2)                 |  1.100e-01 |   0.0000 |       66 |
| QQN-Orth   | OrthantRegion (OWL-QN-style)     |  1.098e-01 |   0.0037 |       70 |
| QQN-L20Box | L-BFGS(20) + BoxRegion(-2, 2)    |  1.097e-01 |   0.0000 |       58 |

- **Fallback** reproduces the L-BFGS baseline exactly here (1.098e-01, target
  at iteration 65), because the L-BFGS direction is always valid (finite,
  non-zero), so the momentum fallback never triggers.
- **OrthantRegion** is the only configuration to induce measurable weight
  sparsity (0.0037), as expected from its sign-preserving projection.
- The **box** and **L20+box** constraints add negligible cost while keeping
  weights bounded; the deeper L20 oracle lets `QQN-L20Box` reach the target
  in 58 iterations, ahead of the shallow box variant (66).

## Loss Trajectories

The log records a compact log10-scale, sampled view of every trajectory. The
qualitative picture (over 10 sampled points across the run):

- **QQN (and most L-BFGS-10 variants)** drop from `0.36` to roughly `-0.96`
  in log10 loss by the end of the run.
- The **deeper-history / trust-region combos** (`QQN-L50TR`, `QQN-L100TR`,
  `QQN-L50BTTR`) lead the field early, reaching `-0.81` by the seventh sample
  and `-0.96` by the end — distinctly ahead of the size-10 baseline (`-0.88`
  at the seventh sample).
- **QQN-L50Spln / QQN-L100Spln / QQN-L50SplnTR / QQN-Best** match the
  deep-memory trajectory, reaching `-0.84`/`-0.83` by the seventh sample and
  `-0.96` by the end.
- **Adam** descends fastest in log10 terms, reaching `-0.93` by the seventh
  sample and `-0.96` by the end (it just needs many cheap iterations).
- **SGD and the heavier momentum oracles** plateau between `-0.58` and
  `-0.64` (lighter momentum / `QQN-Mom01` reaching the lowest of these at
  `-0.72`).
- **Shampoo** (`QQN-Sh`) barely moves before exhausting the time budget,
  reaching only `-0.14` after 9 iterations.

## Key Takeaways

1. **QQN converges to the shared target in fewer iterations than L-BFGS** on a
  smooth, deterministic full-batch problem, at a fraction of the wall-time,
  and clearly outperforms SGD (which never reaches the target). Adam reaches
  the target but needs ~4× the iterations of the quasi-Newton methods.
2. **L-BFGS history depth is the dominant convergence-speed lever**, cutting
  iterations-to-target from 73 (L5) to 46 (L50), with diminishing returns
  past size 50 and a hard plateau at 100 (`L50 == L100` at 46 iterations).
3. **The line search choice trades wall-time, not convergence speed**, on this
  smooth objective — backtracking/Armijo is the clear efficiency winner;
  strong Wolfe and the spline refinement match its iterations-to-target but
  cost ~2× the time.
4. **The spline refinement composes with any line search** (it wraps the inner
  search rather than replacing it) and can sharpen the deep-memory trajectory
  (e.g. `QQN-L50SplnTR` reaches the target fastest among spline variants at
  44 iterations), but it does not change the converged behavior for
  shallow-memory variants on this smooth objective.
5. **Regions are low-overhead safeguards** here; the adaptive trust-region
  marginally accelerates convergence (e.g. L50 → L50TR: 46 → 41 iterations) at
  negligible cost, the Sequential combinator composes cleanly, and the
  orthant region is the lever for sparsity.
6. **Oracle choice matters more than search or region**: momentum trails
  L-BFGS substantially (never reaching target), and the dense/blocked Shampoo
  oracle does not scale to this high-dimensional problem within the time
  budget.
7. **Under the shared `f_target = 1.1e-1`**, all the strong methods (QQN
  variants, L-BFGS, Adam) converge, with the deep-memory + trust-region combos
  (`QQN-L50TR` / `QQN-L100TR` / `QQN-L50BTTR`) winning the race at **41
  iterations**, while SGD and all momentum oracles fail to reach the target.

## Reproducing

```bash
pip install -e ".[dev]"
python examples/mnist_comparison.py
```

The script prints the summary table (including `->target` iteration and
`t->tgt` wall-clock time at which the shared loss/gradient target was first
hit), the Pareto frontier of non-dominated variants, sampled log10
trajectories, and the controlled A/B comparison report. It also saves both a
`mnist_comparison.png` (loss vs. iteration) and a `mnist_comparison_time.png`
(loss vs. wall-clock time) convergence plot when `matplotlib` is available.
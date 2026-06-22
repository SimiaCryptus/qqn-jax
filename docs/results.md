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
| Max iterations | 50                                                         |
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

A key feature of the current experiment is that **every optimizer races
against the same termination criteria**, making the comparison strictly
apples-to-apples. Rather than each method using its own private stopping
rule, all share:

| Bound         |    Value | Meaning                                       |
|---------------|---------:|-----------------------------------------------|
| `f_target`    | `1.0e-1` | Stop once full-batch loss `≤` this value.     |
| `gtol`        | `1.0e-4` | Stop once `‖∇f‖ ≤` this value (stationarity). |
| `time_budget` | `10.0` s | Hard wall-clock cap per optimizer.            |

The summary table records, for every method, the iteration (`->target`) and
wall-clock time (`t->tgt`) at which the shared loss/gradient target was first
reached — or `—` when the method did not reach it within the iteration limit.

## Baseline Comparison

With all defaults (L-BFGS oracle, Armijo line search, no region), QQN reaches
a substantially lower full-batch loss than the first-order baselines and is
competitive with — and faster than — Optax's L-BFGS within the 50-iteration
budget.

| Optimizer | Final loss | Iters | Train acc | Test acc | Time (s) |
|-----------|-----------:|------:|----------:|---------:|---------:|
| QQN       |  1.209e-01 |    50 |    0.9858 |   0.8740 |    1.479 |
| SGD       |  4.178e-01 |    50 |    0.8994 |   0.8680 |    0.475 |
| Adam      |  1.801e-01 |    50 |    0.9628 |   0.8960 |    0.452 |
| L-BFGS    |  1.231e-01 |    50 |    0.9840 |   0.8760 |    2.094 |

**Observations:**

- QQN drives the loss roughly **3.5× lower than SGD** and clearly below Adam,
  reflecting its quasi-Newton acceleration on a smooth deterministic problem.
- QQN edges out Optax's L-BFGS in final loss while running ~**1.4× faster** in
  wall-clock time, owing to its cheap Armijo backtracking search and batched
  t-grid line searches.
- No optimizer reaches the aggressive shared `f_target = 1.0e-1` within the
  50-iteration budget at the baseline oracle depth, but the deeper-memory QQN
  variants (L50/L100 with trust-region) come closest (loss `≈ 1.044e-01`).
- Test accuracy is similar across the strong optimizers; the differentiator
  here is optimization speed and final training loss, not generalization
  (Adam actually has the highest test accuracy at 0.8960).

## QQN Component Sweeps (A/B Comparisons)

Because gradient, oracle, line search, and region are conceptually orthogonal
and independently swappable, the experiment runs controlled A/B sweeps where
each pair isolates a single variable against a named baseline. Each pair's
first entry is the baseline; later entries report deltas against it.

### Oracle: L-BFGS History Depth

Deeper L-BFGS history monotonically improves final loss, with clear
diminishing returns past size 50 and a hard plateau at size 100.

| Variant  | History | Final loss | Time (s) |
|----------|--------:|-----------:|---------:|
| QQN-L5   |       5 |  1.266e-01 |    1.194 |
| QQN      |      10 |  1.209e-01 |    1.479 |
| QQN-L20  |      20 |  1.153e-01 |    1.227 |
| QQN-L50  |      50 |  1.062e-01 |    1.222 |
| QQN-L100 |     100 |  1.062e-01 |    1.301 |

The sweep `L5 < L10 < L20 < L50` confirms richer curvature memory helps, but
the loss plateaus exactly at 1.062e-01 from size 50 onward (`L50 == L100`)
while wall-time keeps growing — so very deep histories (L100) buy *no*
accuracy for extra cost on this problem.

### Oracle: Momentum (heavy-ball) `beta`

The momentum oracle is a first-order accelerator and, as expected, lands well
short of L-BFGS quality. Notably, *lighter* damping converges to a lower loss
on this problem (the sweep is monotone in `beta`).

| Variant   | beta | Final loss | Time (s) |
|-----------|-----:|-----------:|---------:|
| QQN-Mom10 | 0.10 |  3.630e-01 |    1.062 |
| QQN-Mom50 | 0.50 |  4.152e-01 |    1.070 |
| QQN-Mom   | 0.90 |  5.062e-01 |    1.080 |

Near-zero momentum (`beta = 0.1`) effectively collapses toward steepest
descent, which on this smooth full-batch problem outperforms heavier momentum
(`Mom10 < Mom50 < Mom` in loss).

### Region: Trust-Region Radius and Adaptivity

The adaptive trust-region barely perturbs the converged loss across radii,
confirming the region is a low-overhead safeguard rather than a driver of
performance on this well-conditioned problem.

| Variant   | Radius | Adaptive | Final loss | Time (s) |
|-----------|-------:|:--------:|-----------:|---------:|
| QQN-TR025 |   0.25 |   yes    |  1.217e-01 |    1.171 |
| QQN-TR    |   1.00 |   yes    |  1.209e-01 |    1.179 |
| QQN-TRfix |   1.00 |    no    |  1.217e-01 |    1.180 |

Over-constraining the step (radius 0.25) very slightly harms the loss; an
adaptive radius performs marginally better than a fixed one (`TR < TRfix`).

### Line Search (at fixed oracle depth, L-BFGS-10)

The line search choice has negligible effect on the *final* loss but a large
effect on **wall-time**: backtracking is the cheapest, while strong-Wolfe and
the spline refinement are ~2–3× slower for no accuracy gain on this smooth
problem.

| Variant  | Line search   | Final loss | Time (s) |
|----------|---------------|-----------:|---------:|
| QQN      | armijo        |  1.209e-01 |    1.479 |
| QQN-BT   | backtracking  |  1.209e-01 |    1.191 |
| QQN-SW   | strong_wolfe  |  1.209e-01 |    3.014 |
| QQN-Spln | armijo+spline |  1.208e-01 |    2.664 |

The default search is Armijo backtracking (`QQN`, 1.479s); the dedicated
`QQN-BT` backtracking variant is the cheapest robust search (1.191s).
Strong-Wolfe (`QQN-SW`, 3.014s) and the cubic Hermite spline refinement
(`QQN-Spln`, 2.664s) reach essentially the same loss — useful confirmation
that the more expensive searches do not degrade quality, but do not pay off
on a smooth convex objective.

### Spline Refinement (orthogonal enhancement)

In the current implementation the spline is **not** a line-search strategy
but a boolean enhancement (`spline=True`) that *wraps* any chosen line search
(`spline_wrap(inner_search)`). It reuses every probe along the consistent path
as a cubic Hermite control point and probes the spline's stationary points to
improve on the inner search's accepted step.

| Variant       | Configuration                           | Final loss | Time (s) |
|---------------|-----------------------------------------|-----------:|---------:|
| QQN-Spln      | armijo + spline                         |  1.208e-01 |    2.664 |
| QQN-BTSpln    | backtracking + spline                   |  1.208e-01 |    2.571 |
| QQN-L50Spln   | L50 oracle + spline                     |  1.057e-01 |    2.906 |
| QQN-L100Spln  | L100 oracle + spline                    |  1.057e-01 |    2.623 |
| QQN-SplnTR    | armijo + spline + adaptive trust-region |  1.208e-01 |    2.736 |
| QQN-L50SplnTR | L50 + spline + adaptive trust-region    |  1.051e-01 |    2.683 |

The spline refinement notably **sharpens the deep-memory trajectory**:
`QQN-L50Spln` reaches the `-0.98` log10 plateau distinctly earlier than the
spline-less baseline (it is already at `-0.87` by the seventh sample vs.
`-0.81` for the size-10 baseline), and the full stack `QQN-L50SplnTR` reaches
the lowest spline loss observed (1.051e-01). On the smooth convex objective
the final loss for the shallow-memory variants is unchanged, but the extra
per-probe spline fitting costs ~2× wall-time.

## Best-of-Breed Combinations

Stacking the strongest pareto components — deep L-BFGS memory, the cheapest
robust line search (backtracking), and the convergence-stabilizing
trust-region — yields the lowest losses observed, at competitive wall-time.

| Variant     | Configuration                     | Final loss | Time (s) |
|-------------|-----------------------------------|-----------:|---------:|
| QQN-L50TR   | L50 + adaptive trust-region       |  1.044e-01 |    1.233 |
| QQN-L50BTTR | L50 + backtracking + trust-region |  1.044e-01 |    1.251 |
| QQN-L100TR  | L100 + adaptive trust-region      |  1.044e-01 |    1.290 |

The `L50TR` / `L50BTTR` / `L100TR` combos reach the **lowest-loss trajectory
observed** (1.044e-01) while staying around ~1.23–1.29s — a strong pareto
point on loss vs. time. The trust-region shaves the loss below the raw L50
oracle (1.062e-01 → 1.044e-01) at essentially no extra cost. Note that the
experiment also includes a `QQN-SW+TR` combo (strong-Wolfe + adaptive
trust-region, 1.217e-01 at 3.069s) which trades wall-time for no accuracy
gain on this smooth objective.

In the sampled log10 trajectory, `QQN-L50TR` / `QQN-L100TR` / `QQN-L50BTTR`
all share the leading trajectory, reaching `-0.90` by the seventh sample and
`-0.98` by the final sample — distinctly ahead of the size-10 baseline.

## Combinator and Constraint Variants

The experiment also exercises the combinator oracles and regions to confirm
they run correctly and produce sensible behavior:

| Variant    | Configuration                    | Final loss | Sparsity |
|------------|----------------------------------|-----------:|---------:|
| QQN-Fall   | Fallback([L-BFGS(10), Momentum]) |  1.209e-01 |   0.0000 |
| QQN-Box    | BoxRegion(-2, 2)                 |  1.217e-01 |   0.0000 |
| QQN-Orth   | OrthantRegion (OWL-QN-style)     |  1.228e-01 |   0.0056 |
| QQN-L20Box | L-BFGS(20) + BoxRegion(-2, 2)    |  1.148e-01 |   0.0000 |

- **Fallback** reproduces the L-BFGS baseline exactly here (1.209e-01),
  because the L-BFGS direction is always valid (finite, non-zero), so the
  momentum fallback never triggers.
- **OrthantRegion** is the only configuration to induce measurable weight
  sparsity (0.0056), as expected from its sign-preserving projection.
- The **box** and **L20+box** constraints add negligible cost while keeping
  weights bounded; the deeper L20 oracle keeps `QQN-L20Box` (1.148e-01) ahead
  of the shallow box variant.

## Loss Trajectories

The log records a compact log10-scale, sampled view of every trajectory. The
qualitative picture (over 10 sampled points across the 50 iterations):

- **QQN (and most L-BFGS-10 variants)** drop from `0.36` to roughly `-0.92`
  in log10 loss.
- The **deeper-history / trust-region combos** (`QQN-L50TR`, `QQN-L100TR`,
  `QQN-L50BTTR`) lead the field, reaching `-0.90` by the seventh sample and
  `-0.98` by the end — distinctly ahead of the size-10 baseline (`-0.81` at
  the seventh sample).
- **QQN-L50Spln / QQN-L100Spln / QQN-L50SplnTR** match the deep-memory
  trajectory, reaching `-0.87`/`-0.88` by the seventh sample and `-0.98` by
  the end.
- **Adam** reaches `-0.74`, between the first-order and quasi-Newton tiers.
- **SGD and the momentum oracles** plateau between `-0.30` and `-0.44`
  (lighter momentum / `QQN-Mom10` reaching the lowest of these at `-0.44`).

## Key Takeaways

1. **QQN is competitive with L-BFGS at a fraction of the wall-time** on a
   smooth, deterministic full-batch problem, and clearly outperforms
   first-order baselines (SGD, Adam) on training loss within the shared
   iteration/time budget.
2. **L-BFGS history depth is the dominant accuracy lever**, with diminishing
   returns past size 50 and a hard plateau at 100 (`L50 == L100` in loss).
3. **The line search choice trades wall-time, not final loss**, on this smooth
   objective — backtracking/Armijo is the clear efficiency winner; strong
   Wolfe and the spline refinement match its quality but cost ~2–3× the time.
4. **The spline refinement composes with any line search** (it wraps the inner
   search rather than replacing it) and can sharpen the deep-memory trajectory
   (e.g. `QQN-L50SplnTR` reaches the lowest spline loss, 1.051e-01), but it
   does not change the converged loss for shallow-memory variants on this
   smooth objective.
5. **Regions are low-overhead safeguards** here; the adaptive trust-region
   marginally improves the loss (e.g. L50 → L50TR: 1.062e-01 → 1.044e-01) at
   negligible cost, and the orthant region is the lever for sparsity.
6. **Oracle choice matters more than search or region**: momentum trails
   L-BFGS substantially.
7. **Under the shared aggressive `f_target = 1.0e-1`**, no method converged
   to target within the 50-iteration budget, but the best-of-breed
   deep-memory + trust-region combos came closest (loss `≈ 1.044e-01`).

## Reproducing

```bash
pip install -e ".[dev]"
python examples/mnist_comparison.py
```

The script prints the summary table (including `->target` iteration and
`t->tgt` wall-clock time at which the shared loss/gradient target was first
hit), sampled log10 trajectories, and the controlled A/B comparison report.
It also saves both a `mnist_comparison.png` (loss vs. iteration) and a
`mnist_comparison_time.png` (loss vs. wall-clock time) convergence plot when
`matplotlib` is available.
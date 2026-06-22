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

| Setting       | Value                                            |
|---------------|--------------------------------------------------|
| Classes       | 10                                               |
| Train samples | 5000                                             |
| Test samples  | 1000                                             |
| Max iterations| 100                                              |
| Model         | Softmax / multinomial logistic regression        |
| Loss          | Cross-entropy + `0.5·l2·‖params‖²` (`l2 = 1e-4`)  |
| Init          | Shared `PRNGKey(42)` so every optimizer starts identically |

Data is loaded from real MNIST via `tensorflow.keras` or `torchvision` when
available, and falls back to a synthetic Gaussian-blob dataset otherwise, so
the experiment always runs.

All QQN variants are run **one update at a time** (via `solver.init_state` +
a JIT-compiled `solver.update`) to record the full loss trajectory; the Optax
baselines (`SGD`, `Adam`, `L-BFGS`) use their own JIT-compiled step loops.

## Baseline Comparison

With all defaults (L-BFGS oracle, Armijo line search, no region), QQN reaches
a substantially lower full-batch loss than the first-order baselines and is
competitive with — and faster than — Optax's L-BFGS.

| Optimizer | Final loss   | Iters | Train acc | Test acc | Time (s) |
|-----------|-------------:|------:|----------:|---------:|---------:|
| QQN       | 1.034e-01    | 100   | 0.9930    | 0.8770   | 0.936    |
| SGD       | 3.422e-01    | 100   | 0.9148    | 0.8740   | 0.457    |
| Adam      | 1.373e-01    | 100   | 0.9788    | 0.8880   | 0.446    |
| L-BFGS    | 1.039e-01    | 100   | 0.9934    | 0.8780   | 2.156    |

**Observations:**

- QQN drives the loss roughly **3× lower than SGD** and clearly below Adam,
  reflecting its quasi-Newton acceleration on a smooth deterministic problem.
- QQN matches Optax's L-BFGS in final loss while running ~**3× faster** in
  wall-clock time, owing to its cheap Armijo backtracking search and batched
  t-grid line searches.
- Test accuracy is similar across the strong optimizers; the differentiator
  here is optimization speed and final training loss, not generalization.

## QQN Component Sweeps (A/B Comparisons)

Because gradient, oracle, line search, and region are conceptually orthogonal
and independently swappable, the experiment runs controlled A/B sweeps where
each pair isolates a single variable against a named baseline.

### Oracle: L-BFGS History Depth

Deeper L-BFGS history monotonically improves final loss, with clear
diminishing returns past size 50 and a plateau at size 100–200.

| Variant   | History | Final loss | Time (s) |
|-----------|--------:|-----------:|---------:|
| QQN-L5    | 5       | 1.047e-01  | 0.747    |
| QQN       | 10      | 1.034e-01  | 0.936    |
| QQN-L20   | 20      | 1.028e-01  | 0.777    |
| QQN-L50   | 50      | 1.025e-01  | 0.845    |
| QQN-L100  | 100     | 1.024e-01  | 0.924    |
| QQN-L200  | 200     | 1.024e-01  | 1.154    |

The sweep `L5 < L10 < L20 < L50 < L100 ≈ L200` confirms richer curvature
memory helps, but the loss plateaus near 1.024e-01 while wall-time keeps
growing — so very deep histories (L200) buy no accuracy for extra cost.

### Oracle: Momentum (heavy-ball) `beta`

The momentum oracle is a first-order accelerator and, as expected, lands well
short of L-BFGS quality. Notably, *lighter* damping converges to a lower loss
on this problem (the sweep is monotone in `beta`).

| Variant    | beta  | Final loss |
|------------|------:|-----------:|
| QQN-Mom10  | 0.10  | 2.977e-01  |
| QQN-Mom50  | 0.50  | 3.412e-01  |
| QQN-Mom    | 0.90  | 4.044e-01  |
| QQN-Mom99  | 0.99  | 4.150e-01  |

Near-zero momentum (`beta = 0.1`) effectively collapses toward steepest
descent, which on this smooth full-batch problem outperforms heavier momentum.

### Oracle: Shampoo `update_freq`

The Shampoo (structure-aware) oracle reaches a moderate loss but is **orders
of magnitude slower** here, dominated by the dense inverse-root computations
on the flat parameter block.

| Variant     | update_freq | Final loss | Time (s) |
|-------------|------------:|-----------:|---------:|
| QQN-Shmp    | 1           | 2.728e-01  | 180.0    |
| QQN-Shmp5   | 5           | 2.728e-01  | 173.8    |
| QQN-Shmp20  | 20          | 2.728e-01  | 175.8    |

Amortizing the preconditioner refresh (`update_freq = 20`) barely moves the
wall-time, indicating the cost is dominated by the per-step `g gᵀ`
accumulations of a large dense matrix rather than the root refresh itself.
Shampoo is best suited to genuinely structured (matrix-shaped) parameters,
not the flat softmax vector used in this benchmark.

### Region: Trust-Region Radius

The adaptive trust-region barely perturbs the converged loss across radii,
confirming the region is a low-overhead safeguard rather than a driver of
performance on this well-conditioned problem.

| Variant   | Radius | Adaptive | Final loss |
|-----------|-------:|:--------:|-----------:|
| QQN-TR025 | 0.25   | yes      | 1.036e-01  |
| QQN-TR05  | 0.50   | yes      | 1.035e-01  |
| QQN-TR    | 1.00   | yes      | 1.035e-01  |
| QQN-TR20  | 2.00   | yes      | 1.035e-01  |
| QQN-TRfix | 1.00   | no       | 1.036e-01  |

Over-constraining the step (radius 0.25) very slightly harms the loss; an
adaptive radius performs marginally better than a fixed one.

### Line Search (at fixed oracle depth)

The line search choice has negligible effect on the *final* loss but a large
effect on **wall-time**: backtracking is the cheapest, strong-Wolfe and
Hager-Zhang are ~2× slower for no accuracy gain on this smooth problem.

| Variant   | Line search   | Final loss | Time (s) |
|-----------|---------------|-----------:|---------:|
| QQN-L20   | armijo        | 1.028e-01  | 0.777    |
| QQN-L20BT | backtracking  | 1.028e-01  | 0.772    |
| QQN-L20HZ | hager_zhang   | 1.029e-01  | 1.625    |
| QQN-L50   | armijo        | 1.025e-01  | 0.845    |
| QQN-L50BT | backtracking  | 1.025e-01  | 0.853    |
| QQN-L50HZ | hager_zhang   | 1.025e-01  | 1.674    |

The full table also shows `QQN-SW` (strong-Wolfe, 2.632s) and `QQN-Spln`
(cubic Hermite spline refinement, 2.375s) reaching essentially the same loss
as the cheap default — useful confirmation that the more expensive searches do
not degrade quality, but do not pay off on a smooth convex objective. The
baseline default search is Armijo backtracking (`QQN`, 0.936s); the dedicated
`QQN-BT` backtracking variant is the cheapest robust search (0.756s).

## Best-of-Breed Combinations

Stacking the strongest pareto components — deep L-BFGS memory, the cheapest
robust line search (backtracking), and the convergence-stabilizing
trust-region — yields the lowest losses observed, at competitive wall-time.

| Variant      | Configuration                              | Final loss | Time (s) |
|--------------|--------------------------------------------|-----------:|---------:|
| QQN-L50TR    | L50 + adaptive trust-region                | 1.024e-01  | 0.917    |
| QQN-L50BTTR  | L50 + backtracking + trust-region          | 1.024e-01  | 1.008    |
| QQN-L100TR   | L100 + adaptive trust-region               | 1.024e-01  | 0.989    |
| QQN-L100BT   | L100 + backtracking                        | 1.024e-01  | 0.970    |

The `L50TR` / `L50BTTR` combos reach the lowest-loss trajectory (see the
sampled log10 trajectory: `-0.99` plateau reached earlier than the baseline)
while staying around ~0.92–1.01s — a strong pareto point on loss vs. time.
Note that the experiment also includes a `QQN-SW+TR` combo (strong-Wolfe +
adaptive trust-region, 1.035e-01 at 2.539s) which trades wall-time for no
accuracy gain on this smooth objective.

## Combinator and Constraint Variants

The experiment also exercises the combinator oracles and regions to confirm
they run correctly and produce sensible behavior:

| Variant    | Configuration                                  | Final loss | Sparsity |
|------------|------------------------------------------------|-----------:|---------:|
| QQN-Fall   | Fallback([L-BFGS(10), Momentum])               | 1.034e-01  | 0.0001   |
| QQN-Box    | BoxRegion(-2, 2)                               | 1.037e-01  | 0.0000   |
| QQN-Orth   | OrthantRegion (OWL-QN-style)                   | 1.040e-01  | 0.0008   |
| QQN-Stack  | Fallback oracle + Sequential(Box, Trust)       | 1.036e-01  | 0.0001   |
| QQN-L20Box | L-BFGS(20) + BoxRegion(-2, 2)                  | 1.031e-01  | 0.0000   |

- **Fallback** reproduces the L-BFGS baseline exactly here, because the
  L-BFGS direction is always valid (finite, non-zero), so the momentum
  fallback never triggers.
- **OrthantRegion** is the only configuration to induce measurable weight
  sparsity (0.0008), as expected from its sign-preserving projection.
- The **box** and **stacked** constraints add negligible cost while keeping
  weights bounded.

## Loss Trajectories

The log records a compact log10-scale, sampled view of every trajectory. The
qualitative picture:

- **QQN (and most L-BFGS variants)** drop from `0.36` to roughly `-0.99` in
  log10 loss, with the deeper-history / trust-region combos reaching the
  `-0.99` plateau **earlier** (e.g. `QQN-L50TR` is at `-0.97` by the third
- **Adam** reaches `-0.86`, between the first-order and quasi-Newton tiers.
- **SGD and the momentum oracles** plateau between `-0.39` and `-0.53`.
- **Shampoo** plateaus around `-0.56`.

## Key Takeaways

1. **QQN is competitive with L-BFGS at a fraction of the wall-time** on a
   smooth, deterministic full-batch problem, and clearly outperforms
   first-order baselines (SGD, Adam) on training loss.
2. **L-BFGS history depth is the dominant accuracy lever**, with diminishing
   returns past size 50 and a hard plateau at 100–200.
3. **The line search choice trades wall-time, not final loss**, on this smooth
   objective — backtracking/Armijo is the clear efficiency winner; strong
   Wolfe, Hager-Zhang, and the spline refinement match its quality but cost
   ~2–3× the time.
4. **Regions are low-overhead safeguards** here; the trust-region barely moves
   the loss across radii, and the orthant region is the lever for sparsity.
5. **Oracle choice matters more than search or region**: momentum and Shampoo
   (the latter at extreme cost on flat parameters) trail L-BFGS substantially.

## Reproducing

```bash
pip install -e ".[dev]"
python examples/mnist_comparison.py
```

The script prints the summary table, sampled log10 trajectories, and the
controlled A/B comparison report, and saves a `mnist_comparison.png`
convergence plot when `matplotlib` is available.
   sample vs. `-0.81` for the size-10 baseline; `QQN-L50TR`/`QQN-L100TR` reach
   `-0.72` by the third sample and plateau at `-0.99`).
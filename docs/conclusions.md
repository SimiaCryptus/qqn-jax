---
documents:
  - results.md
related:
  - algorithm.md
  - regions.md
  - oracles.md
  - spline_search.md
---

# Conclusions

This document synthesizes the findings from the QQN experimental evaluation
(see [`results.md`](results.md)) and assesses how the empirical evidence
supports the algorithmic claims made in [`algorithm.md`](algorithm.md),
[`oracles.md`](oracles.md), [`regions.md`](regions.md), and
[`spline_search.md`](spline_search.md).

## Summary of Findings

The MNIST optimizer comparison validated QQN as a practical, competitive
optimizer on a smooth, deterministic, full-batch problem. Because every
optimizer raced against the **same shared termination bounds** (`f_target =
1.1e-1`, `gtol = 1.0e-4`, `time_budget = 15.0s`), the headline metric is
*iterations-to-target* — the iteration at which the shared loss/gradient
target was first reached. The headline results are:

- **QQN reaches the shared target in fewer iterations than L-BFGS at a
  fraction of the cost.** On the softmax MNIST benchmark, QQN reached the
  shared `f_target = 1.1e-1` in **65 iterations** (final loss `1.098e-01`) —
  fewer than Optax's L-BFGS (70 iterations) — while running roughly **1.3×
  faster** in wall-clock time (1.642s vs. 2.174s).
- **QQN clearly beats first-order baselines on convergence speed.** It reached
  the target in ~4× fewer iterations than Adam (which needed 263), and SGD
  **never reached** the target within 500 iterations (plateauing at
  `2.266e-01`), confirming the benefit of quasi-Newton acceleration on smooth
  deterministic objectives.
- **The four-axis modular design behaves as specified.** Each swappable
  component (gradient, oracle, search, region) could be substituted
  independently, and the defaults (`oracle="lbfgs"`, `region=None`) reproduced
  the baseline behavior exactly.
- **All strong methods reached the shared `f_target = 1.1e-1`.** The target was
  deliberately tuned to `1.1e-1` (the previous `1.0e-1` was unreachable by
  every method within budget) so the iterations-to-target column became
  informative. Under it, the best-of-breed deep-memory + trust-region combos
  won the race at **41 iterations**.

## Validation of Core Algorithmic Claims

### The Combiner Model Holds

The central thesis of [`algorithm.md`](algorithm.md) — that QQN is a
**combiner** of orthogonal, independently swappable components — is borne out
by the controlled A/B sweeps. Swapping the oracle, line search, or region in
isolation produced predictable, decomposable effects on iterations-to-target
and wall-time, with no cross-component coupling that would undermine the
modularity claim.

### Global Convergence via the Steepest-Descent Anchor

Across every oracle (including the deliberately aggressive Shampoo and the
weak high-`beta` momentum oracle), the line search always returned a
decreasing step or rejected it. This is consistent with the theoretical
guarantee that the path property `d'(0) = -∇f` anchors global convergence
regardless of oracle quality, leaving the oracle free to be aggressive.

## Component-Level Conclusions

### Oracle Choice Is the Dominant Lever

- **L-BFGS history depth** is the single most important convergence-speed
  lever, with a monotone reduction in iterations-to-target `L5 > L10 > L20 >
 L50` (73 → 65 → 60 → 46), clear diminishing returns past size 50, and a hard
  plateau at 100 (`L50 == L100` at **46 iterations**). The converged final
  loss is essentially flat across depths (every variant hits the shared
  target), so the lever here is *speed of convergence*, not final loss. Very
  deep histories (L100) buy *no* extra speed for their additional cost on this
  problem.
- **Momentum** behaved as a first-order accelerator and **never reached the
  target** within 500 iterations; notably, lighter damping (`beta = 0.01`) —
  which collapses toward steepest descent — converged to a lower loss than
  heavier momentum on this smooth problem (the sweep is monotone in `beta`).
- **Shampoo** did not scale to this high-dimensional softmax problem: even with
  a *blocked* preconditioner (`block_size=64`, `update_freq=25`), its dense
  inverse-root refresh exhausted the 15-second wall-clock budget after only
  9 iterations, landing at a much higher loss than even the momentum oracle.

### Line Search Trades Time, Not Convergence Speed

On this smooth convex objective, the line search choice had negligible effect
on the iterations-to-target (or converged loss) but a large effect on
wall-time. Armijo backtracking was the clear efficiency winner (`QQN-BT`,
1.369s, target at iteration 65); strong-Wolfe (3.155s), Hager-Zhang, and the
spline refinement (2.889s) matched its iterations-to-target at ~2× the cost.
This confirms that the more expensive searches do **not** degrade quality —
they simply do not pay off on a well-conditioned objective where curvature
information is easy to exploit.

### The Spline Refinement Composes, As Designed

Consistent with [`spline_search.md`](spline_search.md), the spline behaves as
an orthogonal enhancement that *wraps* (rather than replaces) the inner search.
It did not change the iterations-to-target for shallow-memory variants on the
smooth objective (unchanged at 66), but it measurably **sharpened the
deep-memory trajectory** — `QQN-L50Spln` reached the target in **45
iterations** (vs the spline-less L50 baseline at 46), and the full stack
`QQN-L50SplnTR` reached it fastest among the spline variants at **44
iterations** with the lowest spline loss observed (`1.091e-01`). The extra
per-probe spline fitting costs roughly **2×** wall-time, which did not pay off
for shallow-memory variants on this smooth objective.

### Regions Are Low-Overhead Safeguards

The adaptive trust-region barely perturbed the converged loss across radii
(0.25 → 1.0 → 2.0 is essentially flat), confirming regions function as cheap
safeguards rather than performance drivers on a well-conditioned problem. When
stacked atop a deep oracle, the adaptive trust-region did *accelerate*
convergence (e.g. L50 → L50TR: 46 → 41 iterations-to-target) at negligible
cost. An adaptive radius performed marginally better than a fixed one (`TR`
reached target at 68 vs `TRfix` at 69). The orthant region was the only
configuration to induce measurable weight sparsity (`0.0037`), exactly as its
sign-preserving projection predicts. Box and stacked (`Sequential`) constraints
added negligible cost.

### The t-Grid Is a Cheap Tuning Knob

Sweeping the t-grid granularity (2, 4, and 8 points) had a negligible effect on
iterations-to-target (65, 65, 64) and converged loss, with only a modest effect
on wall-time (a finer grid runs more line searches per iteration). The coarse
2-point grid was essentially as good as the default 4-point grid on this smooth
problem, confirming the t-grid is a tuning knob here rather than a convergence
driver.

### Combinators Work Correctly

`Fallback([L-BFGS, Momentum])` reproduced the L-BFGS baseline exactly (target
at iteration 65), because the L-BFGS direction was always valid and the
momentum fallback never triggered — the intended behavior. Stacked
oracle/region combinators (e.g. `Sequential([Box, TrustRegion])`,
`L-BFGS(20) + BoxRegion`) ran correctly and produced sensible results; the
deeper L20 oracle let `QQN-L20Box` reach the target in 58 iterations, ahead of
the shallow box variant (66).

## Best-of-Breed Recommendation

Stacking the strongest pareto components — deep L-BFGS memory (size 50),
backtracking line search, and an adaptive trust-region — yielded the **fewest
iterations to target** (`QQN-L50BTTR` at **41 iterations**) at competitive
wall-time (~1.17s), making it the standout efficiency/quality balance among the
quasi-Newton methods on the Pareto frontier. For smooth, deterministic,
full-batch problems, this configuration represents a strong default: most of
the convergence speed of the deepest histories with the cheapest robust search
and a low-overhead convergence safeguard. The full "everything" stack
(`QQN-Best`: L50 + backtracking + spline + adaptive trust-region + 8-point
grid) reached the target in 44 iterations but at ~2.3× the wall-time, showing
the spline refinement does not improve the deep-memory backtracking combo here.

## Limitations and Caveats

The conclusions above are drawn from a **single, smooth, deterministic,
full-batch convex benchmark** (softmax MNIST). They should be read with the
following caveats:

- **Smoothness flatters cheap searches.** On non-smooth or ill-conditioned
  objectives, the stronger Wolfe/Hager-Zhang searches and the spline
  refinement may pay off where they did not here.
- **Generalization was not the differentiator.** Test accuracy was similar
  across the strong optimizers; these results concern optimization speed and
  final training loss, not generalization (Adam in fact had the highest test
  accuracy at 0.8810).
- **Structured parameters change the oracle ranking.** The flat softmax
  parameter block here favors L-BFGS; on genuinely matrix-shaped models a
  structure-aware preconditioner (e.g. the Shampoo oracle) may compete
  differently, and its blocked inverse-root cost may amortize better.

## Overall Assessment

The empirical evidence supports QQN's central design claims: it is a competitive
quasi-Newton optimizer whose modular four-axis architecture (gradient, oracle,
search, region) behaves as specified, with the steepest-descent path anchor
delivering robust convergence and the L-BFGS oracle delivering the bulk of the
convergence speed. The line search and region axes are best understood as
tunable trade-offs — efficiency and safety/acceleration levers, respectively —
rather than primary final-loss drivers on smooth problems.
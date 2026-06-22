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
optimizer on a smooth, deterministic, full-batch problem. The headline
results are:

- **QQN matches L-BFGS quality at a fraction of the cost.** On the softmax
   MNIST benchmark, QQN reached a final training loss of `1.209e-01` —
   edging out Optax's L-BFGS (`1.231e-01`) — while running roughly **1.4×
   faster** in wall-clock time (1.467s vs. 2.098s).
- **QQN clearly beats first-order baselines on training loss.** It drove the
   loss roughly **3.5× lower than SGD** and clearly below Adam, confirming the
   benefit of quasi-Newton acceleration on smooth deterministic objectives.
- **The four-axis modular design behaves as specified.** Each swappable
   component (gradient, oracle, search, region) could be substituted
   independently, and the defaults (`oracle="lbfgs"`, `region=None`) reproduced
   the baseline behavior exactly.
- **No method reached the aggressive shared `f_target = 1.0e-1`** within the
   50-iteration budget; the best-of-breed deep-memory + trust-region combos
   came closest (loss `≈ 1.044e-01`).

## Validation of Core Algorithmic Claims

### The Combiner Model Holds

The central thesis of [`algorithm.md`](algorithm.md) — that QQN is a
**combiner** of orthogonal, independently swappable components — is borne out
by the controlled A/B sweeps. Swapping the oracle, line search, or region in
isolation produced predictable, decomposable effects on loss and wall-time,
with no cross-component coupling that would undermine the modularity claim.

### Global Convergence via the Steepest-Descent Anchor

Across every oracle (including the deliberately aggressive Shampoo and the
weak high-`beta` momentum oracle), the line search always returned a
decreasing step or rejected it. This is consistent with the theoretical
guarantee that the path property `d'(0) = -∇f` anchors global convergence
regardless of oracle quality, leaving the oracle free to be aggressive.

## Component-Level Conclusions

### Oracle Choice Is the Dominant Lever

- **L-BFGS history depth** is the single most important accuracy lever, with a
   monotone improvement `L5 < L10 < L20 < L50`, clear diminishing
   returns past size 50, and a hard plateau at 100 (`L50 == L100` at
   `1.061e-01`). Very deep histories (L100) buy *no* accuracy for their extra
   cost on this problem.
- **Momentum** behaved as a first-order accelerator and trailed L-BFGS
   substantially; lighter damping (`beta = 0.1`) — which collapses toward
   steepest descent — outperformed heavier momentum on this smooth problem.
- **Shampoo** did not scale to this high-dimensional softmax problem: its
   dense inverse-root refresh exhausted the 10-second wall-clock budget after
   only 6 iterations, landing at a much higher loss than even momentum.

### Line Search Trades Time, Not Final Loss

On this smooth convex objective, the line search choice had negligible effect
on the converged loss but a large effect on wall-time. Armijo backtracking was
the clear efficiency winner; strong-Wolfe, Hager-Zhang, and the spline
refinement matched its quality at ~2–3× the cost. This confirms that the more
expensive searches do **not** degrade quality — they simply do not pay off on
a well-conditioned objective where curvature information is easy to exploit.

### The Spline Refinement Composes, As Designed

Consistent with [`spline_search.md`](spline_search.md), the spline behaves as
an orthogonal enhancement that *wraps* (rather than replaces) the inner search.
It did not change the converged loss on the smooth objective, but it
measurably **sharpened the deep-memory trajectory** — `QQN-L50Spln` reached the
`-0.98` log10 plateau distinctly earlier than the spline-less baseline (it is
already at `-0.87` by the seventh sample vs. `-0.81` for the size-10 baseline),
and the full stack `QQN-L50SplnTR` reached the lowest spline loss observed
(`1.051e-01`). The extra per-probe spline fitting costs roughly **2×**
wall-time, which did not pay off for shallow-memory variants on this smooth
objective.

### Regions Are Low-Overhead Safeguards

The adaptive trust-region barely perturbed the converged loss across radii,
confirming regions function as cheap safeguards rather than performance drivers
on a well-conditioned problem. When stacked atop a deep oracle, the adaptive
trust-region did shave the loss slightly (e.g. L50 → L50TR: `1.062e-01 →
1.044e-01`) at negligible cost. The orthant region was the only configuration
to induce measurable weight sparsity (`0.0056`), exactly as its sign-preserving
projection predicts. Box and stacked constraints added negligible cost.

### Combinators Work Correctly

`Fallback([L-BFGS, Momentum])` reproduced the L-BFGS baseline exactly, because
the L-BFGS direction was always valid and the momentum fallback never
triggered — the intended behavior. Stacked oracle/region combinators ran
correctly and produced sensible results.

## Best-of-Breed Recommendation

Stacking the strongest pareto components — deep L-BFGS memory (size 50),
backtracking line search, and an adaptive trust-region — yielded the lowest
observed losses (`1.044e-01`) at competitive wall-time (~1.23–1.29s). For smooth,
deterministic, full-batch problems, this configuration represents a strong
default: most of the accuracy of the deepest histories with the cheapest robust
search and a low-overhead convergence safeguard.

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
   accuracy at 0.8960).
- **Structured parameters change the oracle ranking.** The flat softmax
   parameter block here favors L-BFGS; on genuinely matrix-shaped models a
   structure-aware preconditioner (e.g. the Shampoo oracle) may compete
   differently, and its dense inverse-root cost may amortize better.

## Overall Assessment

The empirical evidence supports QQN's central design claims: it is a competitive
quasi-Newton optimizer whose modular four-axis architecture (gradient, oracle,
search, region) behaves as specified, with the steepest-descent path anchor
delivering robust convergence and the L-BFGS oracle delivering the bulk of the
accuracy. The line search and region axes are best understood as tunable
trade-offs — efficiency and safety levers, respectively — rather than primary
accuracy drivers on smooth problems.
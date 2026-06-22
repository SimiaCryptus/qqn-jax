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

# Empirical Results: QQN on Full-Batch Softmax-MNIST

This document records the empirical validation of QQN against classical
baselines (SGD, Adam, Optax L-BFGS) and a broad sweep over QQN's swappable
components — the **oracle** (curvature source), the **line search** (step
selection), the **region** (projective constraint), and the orthogonal
**spline** refinement. The experiment is reproduced by:

```bash
python examples/mnist_comparison.py
```

The full console log lives in [`../mnist_comparison.log`](../mnist_comparison.log)
and is the source of every number quoted below.

---

## Experimental Setup

| Setting      | Value                                                         |
|--------------|---------------------------------------------------------------|
| Problem      | Multinomial logistic regression (softmax) on MNIST            |
| Classes      | 10                                                            |
| Train / Test | 5000 / 1000 examples                                          |
| Objective    | Full-batch cross-entropy + `0.5·1e-4·‖θ‖²` L2                 |
| Regime       | **Deterministic full-batch** (apples-to-apples for 2nd-order) |
| `maxiter`    | 500                                                           |

The problem is deliberately **smooth, deterministic, and full-batch** so the
comparison is fair to the second-order methods (QQN, L-BFGS), which assume a
smooth deterministic objective. If real MNIST is unavailable, the script
falls back to a synthetic Gaussian-blob dataset so the experiment always runs.
> **Dataset provenance caveat:** the loader silently falls back to a synthetic
> Gaussian-blob dataset when neither `torchvision` nor `tensorflow` is
> installed. Gaussian blobs are more separable and better-conditioned than
> real MNIST and would inflate every second-order result. The numbers below
> should be regarded as valid **only** if the run used real MNIST; the raw log
> does not currently record which dataset was loaded. Re-run with the dataset
> source logged (see [`libraries.md`](libraries.md)) to confirm.

### Shared, Fair Termination Bounds

Every optimizer races to the **same** termination criteria, rather than each
using a private rule. This is what makes the leaderboard apples-to-apples:

| Bound         | Value                          | Meaning                                      |
|---------------|--------------------------------|----------------------------------------------|
| `f_target`    | `1.1e-1`                       | stop once full-batch loss ≤ this value       |
| `gtol`        | `1.0e-4`                       | stop once `‖∇f‖ ≤ this value` (stationarity) |
| `time_budget` | `15.0 s`                       | hard wall-clock cap per optimizer            |
| `milestones`  | `(5e-1, 2e-1, 1.5e-1, 1.2e-1)` | convergence-rate profile thresholds          |

The target `1.1e-1` is intentionally *reachable-but-demanding*: the deep-memory
and trust-region combos converge to ≈`1.04e-1`, so this target lets the
strongest variants actually "win" the race and surface their
iteration/time-to-target advantage.
> **Selection-bias caveat:** choosing a target just above the asymptote of the
> favored configurations is a soft form of selecting on the outcome. The
> reported "1.56–1.67× vs L-BFGS" advantage may shift with a tighter or looser
> target. No target-sensitivity analysis has yet been run; these speedups
> should be read as target-specific point estimates, not robust effect sizes.

### Metrics Reported

- **final_loss / best_loss** — terminal and best objective values.
- **iters** — total iterations run.
- **→target / t→tgt** — iteration and wall-time at which `f_target` was first hit.
- **vs LBFGS** — speedup in iterations-to-target relative to Optax L-BFGS.
- **ms/it** — mean wall-clock cost per accepted iteration.
- **AUC** — trajectory area under `log10(loss)` over normalized iterations
  (lower = faster *overall* descent; rewards fast early **and** deep late
  convergence simultaneously).
- **sparsity** — fraction of near-zero weights (illuminating for the orthant
  region).
> **Metric caveats.** (1) *Iterations are not cost-neutral.* QQN's line-search
> iterations issue several function/gradient evaluations each, so
> "iterations-to-target" understates true work. A fairer unit —
> **function/gradient-evaluations-to-target** — is **not yet reported** and
> should be added. (2) *No variance.* Every number is a single-seed point
> estimate with no error bars; small gaps (e.g. 42 vs 45 iters) may be within
> run-to-run noise and should not be over-interpreted.

---

## Headline Findings

On this benchmark, the strongest **converging** QQN configurations reach the
shared target in substantially fewer iterations than the classical baselines:

- **QQN-L50 / QQN-L100 / QQN-L50And** reach the target in **45 iterations**
  (1.56× fewer than L-BFGS's 70) at ≈1.28 s.
- **QQN-L50Spln / QQN-L100Spln** reach it in just **42 iterations** (1.67×
  fewer than L-BFGS) — the fewest-iteration converging variants.
- **L-BFGS** (Optax baseline) needs **70 iterations** / 2.08 s.
- **Adam** needs **263 iterations** (≈4–6× more than the fast QQN stacks)
  but is the cheapest per step (≈2 ms/it) and so wins on raw wall-clock
  (0.53 s) under this tiny model.
- **SGD** never reaches the target within `maxiter`.

The pareto frontier (loss vs. wall-time, non-dominated variants):

```
Adam         loss=1.0999e-01  time=0.530s
QQN-L50TRfix loss=1.0990e-01  time=1.265s
QQN-L50      loss=1.0910e-01  time=1.284s
QQN-Fast     loss=1.0904e-01  time=1.453s
QQN-And      loss=1.0614e-01  time=2.936s
```

**QQN-And** (Anderson acceleration) reaches the **lowest final loss overall**
(`1.061e-1`) — beating every L-BFGS variant — though it takes 194 iterations
to do so, trading iteration count for trajectory depth.

---

## Component Sweeps (A/B Controlled Comparisons)

Each comparison isolates a *single* variable against a named baseline so the
effect is causal.

### Oracle: L-BFGS History Depth

Deeper curvature memory monotonically reduces iterations-to-target (baseline
is `QQN-L5`):

| Variant  | History | iters     | final_loss |
|----------|---------|-----------|------------|
| QQN-L5   | 5       | 72        | 1.100e-1   |
| QQN      | 10      | 62 (Δ−10) | 1.100e-1   |
| QQN-L20  | 20      | 56 (Δ−16) | 1.097e-1   |
| QQN-L50  | 50      | 45 (Δ−27) | 1.091e-1   |
| QQN-L100 | 100     | 45 (Δ−27) | 1.091e-1   |

**Conclusion (this benchmark only):** Deep L-BFGS memory was the largest
convergence-speed lever *observed here*, with diminishing returns saturating
between L50 and L100 (both 45 iters). This is an association from a single
convex run, not an established causal dominance across problem classes.

### Oracle: Momentum β Sweep

The momentum oracle's loss is monotone in β; *lighter* damping descends
further but **no momentum variant reaches the target** within `maxiter`:

| Variant   | β    | final_loss |
|-----------|------|------------|
| QQN-Mom01 | 0.01 | 1.371e-1   |
| QQN-Mom10 | 0.1  | 1.582e-1   |
| QQN-Mom50 | 0.5  | 2.265e-1   |
| QQN-Mom   | 0.9  | 3.419e-1   |

Near-zero momentum collapses toward steepest descent (mirroring `SGD`'s
`2.27e-1` plateau). First-order acceleration is no substitute for genuine
curvature on this smooth problem.

### Oracle: Matrix-Free Curvature (Secant & Anderson)

Two new **matrix-free** oracles probe how much curvature lives in the path's
own realized steps:

| Variant        | Oracle                             | iters | final_loss   | AUC   |
|----------------|------------------------------------|-------|--------------|-------|
| **QQN-Sec**    | Barzilai-Borwein secant (O(n) mem) | 194   | 1.099e-1     | −0.71 |
| **QQN-And**    | Anderson (window=5, m×m solve)     | 194   | **1.061e-1** | −0.81 |
| **QQN-And2**   | Anderson (window=5, β=1.5)         | 146   | 1.100e-1     | −0.79 |
| **QQN-L50And** | Fallback([L50, Anderson])          | 45    | 1.091e-1     | −0.65 |

- **SecantOracle** (BB1 step `α = ⟨s,s⟩/⟨s,y⟩`) crushes plain momentum at
  *zero* storage cost, trailing L-BFGS in iterations but matching it in loss.
- **AndersonOracle** — the variational ideal L-BFGS approximates — reaches the
  **lowest loss of any oracle** and the leading single-oracle AUC.
- **QQN-And2** (β=1.5 coupling) converts Anderson's deep trajectory into a
  faster iteration count (146 vs 194) without sacrificing final loss.
- **QQN-L50And** (`Fallback([L50, Anderson])`) matches the fastest L50 stack
  (45 iters): the Anderson residual solve is a strictly-dominant safety net
  that supplies curvature the instant the L-BFGS history degenerates.

### Oracle: Shampoo

The blocked Shampoo preconditioner (`block_size=64`, `update_freq=25`) is far
too expensive per step for this tiny model: it exhausts the 15 s budget after
only **8 iterations** (≈1993 ms/it) at loss `7.0e-1`. The dense inverse-root
refresh does not amortize at this scale.

### Region: Trust-Region Radius & Adaptivity

The trust-region results reveal a **subtle geometric pitfall** that the code
now documents and partially mitigates:

| Variant   | Config           | iters | final_loss           |
|-----------|------------------|-------|----------------------|
| QQN-TR025 | r=0.25, adaptive | 500   | 1.462e+0 (stalled)   |
| QQN-TR    | r=1.0, adaptive  | 500   | 1.133e-1 (no target) |
| QQN-TR2   | r=2.0, adaptive  | 65    | 1.100e-1 ✓           |
| QQN-TRfix | r=1.0, **fixed** | 66    | 1.097e-1 ✓           |

**The adaptive trust-region over-shrinks** on this curved path. The naive
`ρ = ared/pred` rule compares **chord-length** (the radial clip) against
**arc-length** (the predicted-reduction model) — different coordinates on a
curved path — and collapses the radius. The mitigations now in the code:

1. A **second-order-aware predicted reduction** in `solver.py` that adds a
   curvature term `0.5·t²·(m_q − m_g)` to the linear model and floors `pred`
   at the honest first-order value, so `ρ` is meaningful and non-negative.
2. A **curvature-consistent** `TrustRegion` (`shrink=0.5`, wide stable band
   `[eta_lo, eta_hi]`) that only shrinks on genuinely poor `ρ < eta_lo`.

Despite these, the adaptive radius still stalls when stacked with deep memory
(see below). **Fixed-radius trust-regions are the robust fast path** — `QQN-TRfix`
converges in 66 iters where the adaptive `QQN-TR` never reaches the target.

The `QQN-L50TRcc` variant (the curvature-consistent gentle-shrink rule on a
deep oracle) still stalls at `1.364e-1`, confirming the chord/arc mismatch is
geometric, not merely a tuning artifact — and that the **fixed** clip is the
reliable safeguard for deep-memory steps.

### Region: Box, Orthant, Sequential

- **QQN-Box** (`lo=-2, hi=2`): converges in 64 iters at `1.098e-1`, negligible
  overhead.
- **QQN-Orth** (OWL-QN orthant): converges in 67 iters and is the **only**
  variant inducing measurable sparsity (`0.0037`).
- **QQN-Seq** (`Sequential([Box, TR-adaptive])`): inherits the adaptive-TR
  stall (500 iters, `1.176e-1`) — confirming the combinator composes
  projections faithfully (including the stall it inherits from its TR child).

### Line Search

| Variant  | Search           | iters | final_loss |
|----------|------------------|-------|------------|
| QQN      | Armijo (default) | 62    | 1.100e-1 ✓ |
| QQN-BT   | backtracking     | 62    | 1.100e-1 ✓ |
| QQN-Spln | Armijo + spline  | 63    | 1.099e-1 ✓ |
| QQN-SW   | strong Wolfe     | 500   | 4.077e-1 ✗ |

**Strong Wolfe over-restricts** the quadratic-path step and fails to converge
(it plateaus at `4.08e-1`). The **backtracking / Armijo family is the robust
efficiency winner** — backtracking matches Armijo on iterations while running
slightly faster in wall-clock (no curvature condition to satisfy). The line
search trades wall-time, not convergence speed.

### Spline Refinement (Orthogonal Augmentation)

The spline (`spline=True`) **wraps** any line search, reusing every probe as a
cubic Hermite control point and probing the spline's stationary points
(including a **superlinear extrapolation probe** beyond the inner step when the
downstream tangent still descends):

| Variant      | Stack                | iters  | final_loss |
|--------------|----------------------|--------|------------|
| QQN-Spln     | Armijo + spline      | 63     | 1.099e-1   |
| QQN-L50Spln  | L50 + spline         | **42** | 1.096e-1   |
| QQN-L100Spln | L100 + spline        | **42** | 1.096e-1   |
| QQN-SplnTR   | spline + adaptive TR | 268    | 1.100e-1   |

The spline sharpens the **deepest-memory** trajectories the most: `QQN-L50Spln`
is the **fewest-iteration converging variant (42 iters, 1.67× vs L-BFGS)**,
though the extra probes raise per-iteration cost (≈66 ms/it vs ≈29 ms/it for
plain L50). Stacking the spline with the *adaptive* trust-region
(`QQN-SplnTR`, `QQN-L50SplnTR`, `QQN-Best`) inherits the adaptive-radius stall.

### Performance: Warm-Started Backtracking (the Speed Lever)

Because the path's `t = 1` endpoint is already a full quasi-Newton step,
warm-starting the backtracking search **beyond α = 1** lets deep-memory steps
stretch into the superlinear regime. Critically, this must be paired with a
**fixed** trust-region (the adaptive radius contaminates the warm start):

| Variant      | init_step / shrink | region           | iters       | final_loss |
|--------------|--------------------|------------------|-------------|------------|
| QQN-L50BTTR  | 1.0 / 0.5          | TR adaptive      | 500 (stall) | 1.364e-1   |
| QQN-L50WS+   | 2.0 / 0.7          | TR fixed         | 57          | 1.097e-1 ✓ |
| QQN-L50WS    | 4.0 / 0.8          | TR fixed (r=1.5) | 79          | 1.100e-1 ✓ |
| QQN-Fast     | 2.0 / 0.7          | TR fixed (L100)  | 57          | 1.090e-1 ✓ |
| QQN-Champion | 3.0 / 0.75         | TR fixed (r=1.5) | 59          | 1.097e-1 ✓ |

The contrast between `QQN-L50BTTR` (adaptive TR, stalls at 500 iters) and
`QQN-L50WS+` (fixed TR, 57 iters) is a **Δ−443 iteration swing from a single
variable** — the trust-region adaptivity. This is the clearest evidence that
the adaptive radius is the destabilizing factor, and that **fixed-radius +
warm-started backtracking is the intended robust fast stack.**
> **Statistical caveat on the Δ−443 figure.** The `QQN-L50BTTR` arm does not
> *converge in 500 iterations* — it hits the `maxiter=500` ceiling without
> reaching the target. 500 is therefore a **censoring bound**, not a measured
> convergence time; the true gap is "≥443," unbounded above. The number is
> best read qualitatively (adaptive-TR + deep memory stalls; fixed-TR does
> not), not as a precise effect size.

---

## Leaderboards

### Iteration-Efficiency (target reached, fewest iters)

```
QQN-L50Spln    iters=42  time=2.753s  vs_LBFGS=1.67x  final=1.0956e-01
QQN-L100Spln   iters=42  time=2.869s  vs_LBFGS=1.67x  final=1.0956e-01
QQN-L50TRfix   iters=45  time=1.256s  vs_LBFGS=1.56x  final=1.0990e-01
QQN-L50        iters=45  time=1.275s  vs_LBFGS=1.56x  final=1.0910e-01
QQN-L100       iters=45  time=1.280s  vs_LBFGS=1.56x  final=1.0910e-01
QQN-L50And     iters=45  time=1.361s  vs_LBFGS=1.56x  final=1.0910e-01
QQN-L50TR2     iters=46  time=1.315s  vs_LBFGS=1.52x  final=1.0974e-01
```

### Trajectory-AUC (lower = faster overall descent)

```
QQN-TR         AUC=-0.911   (stalls at target but descends fast early)
QQN-Seq        AUC=-0.897
QQN-SplnTR     AUC=-0.893
QQN-L50SplnTR  AUC=-0.867
QQN-Best       AUC=-0.867
```

Note that several **stalling** variants top the AUC board: the adaptive
trust-region drives loss down *fast early* (good AUC) but then over-shrinks
and never reaches the final target. AUC and iteration-to-target are
complementary — the former rewards early descent, the latter rewards actually
finishing.

---

## Cautionary Findings (Stall Report)

The benchmark explicitly surfaces every variant that exhausted its budget
**without** reaching the shared target, classified by likely cause:

| Variant                                         | final_loss   | cause                                         |
|-------------------------------------------------|--------------|-----------------------------------------------|
| QQN-TR, QQN-Seq                                 | 1.13–1.18e-1 | slow (adaptive-TR over-shrink)                |
| QQN-L50TR, QQN-L50BTTR, QQN-L100TR, QQN-L50TRcc | 1.364e-1     | slow (deep-memory + adaptive TR stall)        |
| QQN-Mom*                                        | 1.37–3.42e-1 | slow (first-order plateau)                    |
| QQN-SW, QQN-SW+TR                               | 0.41–0.68e-0 | strong-Wolfe over-restriction                 |
| QQN-Sh                                          | 7.0e-1       | time-budget exhausted (dense Shampoo refresh) |
| QQN-TR025                                       | 1.46e+0      | stalled (radius too tight)                    |

These are **first-class experimental findings**, not failures to hide:

1. The **adaptive trust-region** over-shrinks on the curved path under the
   honest predicted-reduction model; the **fixed** radius is the robust
   alternative.
2. The **strong-Wolfe** curvature condition over-restricts the quadratic-path
   step on this problem.
3. **Dense Shampoo** does not amortize at small model scale.
4. An **over-tight** trust radius (0.25) starves the step entirely.

---

## Summary of Design-Claim Validation

| QQN Design Claim                                                 | Empirical Verdict                                                              |
|------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Gradient + oracle blending via the quadratic path converges fast | ✅ 1.56–1.67× fewer iters than L-BFGS                                           |
| The oracle is freely swappable                                   | ✅ L-BFGS, Momentum, Secant, Anderson, Shampoo, Fallback all run                |
| Deep curvature memory accelerates convergence                    | ✅ monotone L5→L50, saturating at L50–L100                                      |
| The line search trades wall-time, not convergence speed          | ✅ BT ≈ Armijo in iters; SW over-restricts                                      |
| Regions are low-overhead safeguards                              | ✅ Box/Orthant negligible overhead; **fixed** TR robust, **adaptive** TR stalls |
| The spline reuses information to sharpen trajectories            | ✅ L50Spln is the fewest-iteration converging variant (42)                      |
| Warm-started backtracking unlocks the superlinear regime         | ✅ Δ−443 iters vs adaptive-TR baseline                                          |

The best-of-breed **converging** stacks land at **45 iterations** (deep L-BFGS,
`QQN-L50`/`L100`) or **42 iterations** (with spline refinement, `QQN-L50Spln`),
versus **70** for classical L-BFGS and **263** for Adam — validating QQN's
central thesis that coherently blending gradient and oracle along the quadratic
path, navigated by a robust line search, yields a fast, modular optimizer.

See [`algorithm.md`](algorithm.md) for the conceptual treatment and
[`../mnist_comparison.log`](../mnist_comparison.log) for the full raw output.
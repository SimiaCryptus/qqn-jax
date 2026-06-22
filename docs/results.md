---
documents:
   - ../results/fashion_mnist_mlp_comparison*.log
related:
  - algorithm.md
  - ../README.md
   - ../examples/fashion_mnist_mlp_comparison.py
  - ../qqn_jax/solver.py
  - ../qqn_jax/line_search.py
  - ../qqn_jax/oracles.py
  - ../qqn_jax/spline_search.py
  - ../qqn_jax/regions.py
---

# Empirical Results: QQN on Full-Batch Fashion-MNIST MLP

This document records the empirical validation of QQN against classical
baselines (SGD, Adam, Optax L-BFGS) and a broad sweep over QQN's swappable
components — the **oracle** (curvature source), the **line search** (step
selection), the **region** (projective constraint), and the orthogonal **spline**
refinement. The benchmark additionally exercises the **probe-feeding** lever
(`feed_probes_to_oracle=True`), which forwards every gradient evaluated *during*
the line search into the oracle's curvature memory. The experiment is reproduced
by:

> **Note on the swappable component catalogue.** Beyond the variants exercised
> in this run, the implementation also exposes additional oracles
> (`SecantOracle`, `AndersonOracle`, and the string shortcuts `"secant"`,
> `"anderson"`, `"anderson+secant"`, `"lbfgs+secant"`) and regions
> (`OrthantRegion` for OWL-QN-style sparsity, `NoDecreaseRegion` for
> multi-objective / continual-learning guards). The solver additionally
> supports `feed_probes_to_oracle=True`, which forwards every gradient
> evaluated *during* the line search into the oracle's curvature memory (not
> just the accepted point) via fixed-size, JIT/vmap-compatible probe buffers.
> The `QQN-L50P`, `QQN-MaxP`, `QQN-L80P`, `QQN-UltraP`, `QQN-Max`, and
> `QQN-Fast` variants are now first-class members of the benchmark sweep (see
> `examples/fashion_mnist_mlp_comparison.py`).

```bash
python examples/fashion_mnist_mlp_comparison.py
```

The full console log lives in
[`../results/fashion_mnist_mlp_comparison_20260622_142624.log`](../results/fashion_mnist_mlp_comparison_20260622_142624.log)
and is the source of every number quoted below.

> **Headline caveat (this run).** Under the configuration recorded in the
> current log (`activation=sigmoid,relu,gaussian` cycled across a deeper
> 4-hidden-layer network, `f_target=5.0e-2`, `time_budget=15.0 s`),
> **no optimizer reached the shared loss target** within the wall-clock
> budget — including L-BFGS. This is a markedly harder problem instance than
> the historical run, and the numbers below should be read as a
> *budget-limited* comparison (who descended furthest in 15 s), **not** an
> iterations-to-target race. The earlier "1.45× fewer iterations than L-BFGS"
> result does **not** reproduce under this configuration.

---

## Experimental Setup

| Setting      | Value                                                         |
|--------------|---------------------------------------------------------------|
| Problem      | Multi-layer MLP (configurable activation) on Fashion-MNIST    |
| Architecture | `x -> 64 -> 64 -> 64 -> 64 -> 10` (4 hidden layers, `HIDDEN=64`) |
| Activation   | `sigmoid,relu,gaussian` (cycled per hidden layer → `sigmoid,relu,gaussian,sigmoid`) |
| Classes      | 10                                                            |
| Train / Test | 15000 / 2000 examples (`N_TRAIN` / `N_TEST`)                  |
| Parameters   | 63 370                                                        |
| Objective    | Full-batch cross-entropy + `0.5·1e-4·‖θ‖²` L2 (non-convex)   |
| Regime       | **Deterministic full-batch** (apples-to-apples for 2nd-order) |
| `maxiter`    | 100000 (effectively unbounded; runs stop on target/budget)   |

The problem is deliberately **deterministic and full-batch** so the comparison is
fair to the second-order methods (QQN, L-BFGS). The hidden nonlinear layers make
the objective **non-convex**, and the *mixed* activation stack
(`sigmoid,relu,gaussian` cycled across the four hidden layers) makes this a
sterner, less well-conditioned test than a uniform-activation network. If real
Fashion-MNIST is unavailable, the script falls back to a synthetic
Gaussian-blob dataset so the experiment always runs.

> **Dataset provenance caveat:** the loader silently falls back to a synthetic
> Gaussian-blob dataset when neither `torchvision` nor `tensorflow` is
> installed. Gaussian blobs are more separable and better-conditioned than
> real Fashion-MNIST and would inflate every second-order result. The numbers
> below are valid: the log confirms `[data] Loaded fashion_mnist via
> tensorflow.keras.`

> **Note on script defaults vs. this log.** The committed
> `examples/fashion_mnist_mlp_comparison.py` currently defaults to a *larger,
> deeper* configuration (`N_TRAIN=20000`, `N_TEST=3000`, `DEPTH=5`,
> `f_target=4.0e-2`, `time_budget=20.0 s`, and additional `QQN-L80`,
> `QQN-L80P`, `QQN-UltraP` variants). The current log was produced under the
> *earlier* configuration listed above (`N_TRAIN=15000`, `N_TEST=2000`,
> `DEPTH=4`, `f_target=5.0e-2`, `time_budget=15.0 s`). Re-running the script
> as committed will therefore produce different absolute numbers and a
> different variant roster. See the **Re-run caveat** at the end.

### Configurable Architecture & Activations

The benchmark is highly configurable via environment variables (see the script
docstring):

| Env var        | Meaning                                            | Default (committed script) |
|----------------|----------------------------------------------------|----------------------------|
| `DATASET`      | `mnist` or `fashion_mnist`                          | `fashion_mnist`            |
| `HIDDEN_SIZES` | Comma-separated hidden widths (e.g. `128,64`)       | —                          |
| `HIDDEN`       | Uniform hidden-layer width                          | `64`                       |
| `DEPTH`        | Number of hidden layers                             | `5`                        |
| `ACTIVATION`   | Activation name(s); comma-list mixes per layer      | `sigmoid,relu,gaussian`    |
| `N_TRAIN`      | Full-batch training-set size                        | `20000`                    |
| `N_TEST`       | Test-set size                                       | `3000`                     |

Supported activations: `relu`, `sigmoid`, `sine`, `gaussian`, `triangle`,
`sawtooth`, `logabs`, `tanh`, `gelu`, `swish`, `softplus`, `abs`, `identity`.
A comma-separated `ACTIVATION` list assigns different activations to different
hidden layers (cycled if shorter than the layer count); the output layer is
always linear. Initialization is He-style for ReLU layers and Glorot/Xavier
otherwise.

### Shared, Fair Termination Bounds

Every optimizer races to the **same** termination criteria, rather than each
using a private rule. This is what makes the leaderboard apples-to-apples
(the bounds shown are those used for the current log):

| Bound         | Value                          | Meaning                                      |
|---------------|--------------------------------|----------------------------------------------|
| `f_target`    | `5.0e-2`                       | stop once full-batch loss ≤ this value       |
| `gtol`        | `1.0e-8`                       | stop once `‖∇f‖ ≤ this value` (stationarity) |
| `time_budget` | `15.0 s`                       | hard wall-clock cap per optimizer            |
| `milestones`  | `(1e0, 5e-1, 2e-1, 1e-1)`      | convergence-rate profile thresholds          |

On this harder problem instance the target `5.0e-2` proved **unreachable for
every optimizer** within the 15 s budget (the best, L-BFGS, plateaued at
`5.108e-2`). The looser milestones `(1e0, 5e-1, 2e-1, 1e-1)` therefore carry
the bulk of the analysis, profiling the coarse-descent phase that each method
actually reaches.

> **Selection-bias caveat:** choosing a target near the asymptote of the
> favored configurations is a soft form of selecting on the outcome. The
> target-sensitivity profile below mitigates this by reporting iterations
> across a *range* of targets, but on this run only the loosest target
> (`2e-1`) was reached by any QQN variant.

### Target-Sensitivity Profile

To address the selection-bias caveat, the script additionally reports
iterations-to-target across a **range** of targets (the `target_profile`
`(2.0e-1, 1.0e-1, 7.0e-2, 5.0e-2)` for this log; the committed script extends
it to `4.0e-2`), plus a dedicated **vs-LBFGS speedup stability** check for
`QQN-L50`, `QQN-L50P`, and `QQN-MaxP` across those targets. On this run only
the `2e-1` target was reachable by QQN variants (see below).

### Metrics Reported

- **final_loss / best_loss** — terminal and best objective values.
- **iters** — total iterations run.
- **→target / t→tgt** — iteration and wall-time at which `f_target` was first hit.
- **vs LBFGS** — speedup in iterations-to-target relative to Optax L-BFGS.
- **ms/it** — mean wall-clock cost per accepted iteration.
- **AUC** — trajectory area under `log10(loss)` over normalized iterations
- **evals** — *cost-aware* estimated function/gradient-evaluations-to-target
   (iterations × per-method evaluation multiplicity). QQN line-search iterations
   issue several value/grad probes each, so this is a fairer unit than raw
   iterations. See `_estimate_evals_per_iter` for the per-method heuristics.
  (lower = faster *overall* descent; rewards fast early **and** deep late
  convergence simultaneously).
- **train_acc / test_acc** — training and test accuracy at termination.

> **Metric caveats.** (1) *Iterations are not cost-neutral.* QQN's line-search
> iterations issue several function/gradient evaluations each, so
> "iterations-to-target" understates true work. The script reports a
> cost-aware **evals-to-target** unit (and a dedicated cost-aware leaderboard),
> though the per-iteration multiplicities are conservative *analytic estimates*,
> not measured counts. (2) *No variance.* Every number is a single-seed point
> estimate with no error bars; small gaps may be within run-to-run noise and
> should not be over-interpreted. (3) *Budget-limited run.* Because no
> optimizer reached `f_target` here, the `→target`, `evals`, and `vs LBFGS`
> columns are empty (`—`) for almost every variant.

---

## Headline Findings

On this harder non-convex instance (`sigmoid,relu,gaussian` mixed activations,
4 hidden layers), **no optimizer reaches the `5.0e-2` target within the 15 s
budget.** The comparison therefore measures *how far each method descends* in a
fixed wall-clock budget:

- **L-BFGS** is the clear best on final loss, reaching **`5.108e-2`** (just
   above the target) in 2 671 iterations — its cheap per-iteration cost
   (≈5.6 ms/it) lets it run far more iterations than any QQN variant.
- **QQN-L50** is the best QQN variant on final loss (**`1.043e-1`**, 426
   iters), followed by **QQN-L50And** (`1.154e-1`, 395 iters).
- **Adam** (`1.189e-1`, 6 838 iters) and **SGD** (`1.308e-1`, 6 986 iters)
   descend respectably thanks to their very cheap per-step cost (≈2.2 ms/it),
   running an order of magnitude more iterations than the second-order methods.
- The **probe-fed variants catastrophically diverge** on this instance:
   `QQN-L50P` and `QQN-MaxP` both blow up to a final loss of `2.302e+0`
   (≈`log(10)`, i.e. uniform-class predictions) with ≈10% train/test accuracy.
   Probe-feeding is *actively harmful* under this configuration (see below).

The Pareto frontier (loss vs. wall-time, non-dominated variants) collapses to a
single point, because every method exhausts the same 15 s budget and L-BFGS
dominates on loss:

```
L-BFGS       loss=5.1075e-02  time=15.000s
```

**L-BFGS dominates outright** on this run — lowest loss at the shared budget.
Among QQN variants, deep L-BFGS memory (`QQN-L50`) remains the strongest
configuration, but it does **not** overtake classical L-BFGS here: its higher
per-iteration cost (≈35 ms/it vs ≈5.6 ms/it) means it completes only 426
iterations to L-BFGS's 2 671 in the same wall-clock window.

---

## Component Sweeps (A/B Controlled Comparisons)

Each comparison isolates a *single* variable against a named baseline so the
effect is causal. Because no variant reached the target, the comparisons use
**final loss at the 15 s budget** and the **`2e-1` milestone** as the primary
yardsticks.

### Oracle: L-BFGS History Depth

Deeper curvature memory yields a markedly lower final loss (baseline is `QQN`
with `history_size=10`):

| Variant  | History | iters | final_loss | `→2e-1` iter |
|----------|---------|-------|------------|--------------|
| QQN      | 10      | 446   | 3.138e-1   | —            |
| QQN-L20  | 20      | 408   | 1.638e-1   | 354          |
| QQN-L50  | 50      | 426   | 1.043e-1   | 282          |

**Conclusion (this benchmark only):** Deep L-BFGS memory remains the largest
single QQN convergence-quality lever on this non-convex problem — `QQN-L50`
reaches a final loss roughly **3× lower** than the `history=10` baseline, and
is the only QQN family member to cross the `2e-1` milestone fastest (282
iters). This is an association from a single non-convex run, not an established
causal dominance across problem classes.

### Oracle: Momentum

**No momentum variant descends meaningfully** within the budget on this harder
non-convex problem:

| Variant      | Config              | final_loss   | iters |
|--------------|---------------------|--------------|-------|
| QQN-Mom      | β=0.9               | 1.084e+0     | 435   |
| QQN-Mom-S    | β=0.9 + spline      | 1.448e+0     | 260   |
| QQN-Mom-S-BT | β=0.9 + spline + BT | 1.444e+0     | 261   |

All momentum variants plateau far above the milestones (none crosses even the
`5e-1` line) and exhaust the 15 s budget. First-order acceleration is no
substitute for genuine curvature on this non-convex problem; the spline
augmentation does not rescue the momentum oracle (and the spline's per-iteration
overhead actually *reduces* the iteration count it can afford in the budget).

### Oracle: Matrix-Free Curvature (Secant & Anderson)

Two **matrix-free** oracles probe how much curvature lives in the path's own
realized steps:

| Variant        | Oracle                             | iters | final_loss   | AUC   |
|----------------|------------------------------------|-------|--------------|-------|
| **QQN-Sec**    | Barzilai-Borwein secant (O(n) mem) | 409   | 4.154e-1     | −0.17 |
| **QQN-And**    | Anderson (window=5, m×m solve, β)  | 332   | 5.241e-1     | −0.03 |
| **QQN-L50And** | Fallback([L50, Anderson])          | 395   | 1.154e-1     | −0.43 |

- **SecantOracle** (BB1 step `α = ⟨s,s⟩/⟨s,y⟩`) descends to `4.154e-1` (test
   accuracy 85.1%) but stalls well above the milestones.
- **AndersonOracle** alone fares worst of the curvature oracles on this run
   (`5.241e-1`, test accuracy 81.8%): the non-convex loss surface exposes its
   sensitivity to the quality of the residual window.
- **QQN-L50And** (`Fallback([L50, Anderson])`) tracks `QQN-L50` closely
   (`1.154e-1` vs `1.043e-1`, identical `2e-1` milestone at 282 iters): the
   Anderson residual solve acts as a safety net that supplies curvature the
   instant the L-BFGS history degenerates, without materially slowing
   convergence when L-BFGS is healthy. It does not *quite* match bare
   `QQN-L50` on this instance.

The Anderson oracle exposes a **coupling constant `β`** (the classical mixing
parameter): `β = 1` recovers the pure Type-II update, while `β > 1` lets the
deep-residual descent stretch. Its `(m × m)` solve uses a scale-aware Tikhonov
ridge anchored to the Gram trace plus an absolute diagonal floor to guarantee
SPD-ness even on a degenerate window.

> **Fallback validity is descent, not non-zeroness.** The `Fallback`
> combinator selects the first oracle whose direction is finite, non-zero
> **and a genuine descent direction** (`⟨∇f, d⟩ < 0`). A finite, non-zero
> quasi-Newton direction that points uphill triggers the fallback, and a
> terminal steepest-descent safety net guarantees the `t = 1` endpoint can
> never be a non-descent or NaN direction.

> **Note:** On the convex softmax-MNIST benchmark, Anderson achieved leading
> AUC and the lowest final loss. On this non-convex MLP, the Anderson oracle
> alone is the *weakest* curvature oracle. The `Fallback([L50, Anderson])`
> pairing remains robust by delegating to L-BFGS when Anderson degenerates.

### Oracle: Probe-Feeding (⚠️ Harmful on This Instance)

The solver's `feed_probes_to_oracle=True` lever forwards **every gradient
evaluated during the line search** — not just the accepted point — into the
L-BFGS curvature memory. The line search already computes these gradients while
walking the path, so the extra `(s, y)` curvature pairs are obtained
essentially for free (no additional function/gradient evaluations are charged).
Internally, the line-search `LineSearchResult` carries fixed-size, JIT/vmap-safe
`probe_params` / `probe_grads` / `probe_valid` buffers (`max_probes=32` by
default), and the L-BFGS oracle replays them oldest-first via
`update_lbfgs_history_batch` before appending the accepted point as the newest
pair.

Two probe-fed variants are benchmarked:

| Variant    | Stack                                                          | Probe-fed | final_loss | test_acc |
|------------|----------------------------------------------------------------|-----------|------------|----------|
| QQN-L50P   | L-BFGS (history=50) + Armijo                                    | ✅         | 2.302e+0   | 9.85%    |
| QQN-MaxP   | Fallback([L50, Anderson]) + warm BT + fixed TR(r=2) + spline   | ✅         | 2.302e+0   | 9.85%    |

**On this configuration, probe-feeding catastrophically degrades convergence.**
Both probe-fed variants diverge to `2.302e+0` — exactly `ln(10)`, the loss of a
classifier emitting uniform class probabilities — with ≈10% (chance-level)
train and test accuracy. The loss trajectory shows them flat-lining at the
initial loss (`log10 ≈ 0.36`) from the first iteration onward.

The likely mechanism: replaying *intermediate* line-search probes — which are
by construction points the search **rejected** for failing sufficient decrease
— pollutes the L-BFGS curvature memory with poorly-conditioned `(s, y)` pairs.
On the well-conditioned convex benchmark these probes were benign-to-helpful;
on this ill-conditioned mixed-activation surface they corrupt the Hessian
approximation badly enough to destroy the `t = 1` endpoint. The bare-vs-fed
comparison is stark: `QQN-L50` reaches `1.043e-1` while `QQN-L50P` never
descends at all.

> **Verdict revised.** Probe-feeding was previously documented as a "free
> curvature boost." On this harder instance it is **actively harmful** and
> should be treated as an *experimental, problem-dependent* lever — not a
> default-on optimization. The curvature it harvests is only useful when the
> rejected-probe `(s, y)` pairs are well-conditioned, which is not guaranteed
> on non-convex surfaces.

### Oracle: Shampoo

The Shampoo oracle is not included in this benchmark run. On the prior convex
softmax-MNIST benchmark, the blocked Shampoo preconditioner (`block_size=64`,
`update_freq=25`) exhausted the time budget after only ~9 iterations (≈1796
ms/it) at loss `6.8e-1`. The dense inverse-root refresh does not amortize at
this model scale.

### Region: Trust-Region

| Variant   | Config           | iters | final_loss | test_acc |
|-----------|------------------|-------|------------|----------|
| QQN-TR    | r=1.0, adaptive  | 399   | 4.796e-1   | 83.8%    |

The adaptive trust-region (`QQN-TR`, `r=1.0`) descends only to `4.796e-1` on
this run — substantially *worse* than the unconstrained `QQN` baseline
(`3.138e-1`). On this harder mixed-activation surface the adaptive radius is
over-conservative, repeatedly shrinking the step and slowing descent. The
mitigations in the code (exact along-path predicted reduction, progress floor,
gentle shrink) keep it from collapsing entirely, but it does not act as a
net-positive safeguard here. See the algorithm documentation for details.

### Region: Box

- **QQN-Box** (`lo=-2, hi=2`): descends to `3.232e-1` in 395 iterations (test
   accuracy 86.8% — the **highest test accuracy of any QQN variant**). The box
   constraint tracks the unconstrained `QQN` baseline (`3.138e-1`) on training
   loss while acting as a mild regularizer that improves generalization on the
   held-out set.

### Line Search

| Variant  | Search                | iters | final_loss | `→2e-1` iter |
|----------|-----------------------|-------|------------|--------------|
| QQN      | Armijo (default)      | 446   | 3.138e-1   | —            |
| QQN-BT   | backtracking          | 401   | 3.438e-1   | —            |
| QQN-S    | Armijo + spline       | 237   | 5.918e-1   | —            |
| QQN-BT-S | backtracking + spline | 268   | 5.601e-1   | —            |

The **backtracking / Armijo family is the robust efficiency winner** —
backtracking and Armijo land at comparable final losses (`3.438e-1` vs
`3.138e-1`). On this run the **spline refinement is net-negative**: the
spline-augmented variants (`QQN-S`, `QQN-BT-S`) finish at substantially higher
loss (`5.918e-1`, `5.601e-1`) because the higher per-iteration cost (≈64 and
≈56 ms/it) lets them complete far fewer iterations (237, 268) within the budget,
and the extra probes do not buy enough per-step progress to compensate. The
strong Wolfe search is not included in this run; on the prior convex benchmark
it over-restricted the quadratic-path step and failed to converge.

### Spline Refinement (Orthogonal Augmentation)

The spline (`spline=True`) **wraps** any line search, reusing every probe as a
cubic Hermite control point and probing the spline's stationary points:

| Variant  | Stack                 | iters | final_loss | ms/it |
|----------|-----------------------|-------|------------|-------|
| QQN      | Armijo                | 446   | 3.138e-1   | 33.75 |
| QQN-S    | Armijo + spline       | 237   | 5.918e-1   | 63.50 |
| QQN-BT   | backtracking          | 401   | 3.438e-1   | 37.52 |
| QQN-BT-S | backtracking + spline | 268   | 5.601e-1   | 56.19 |

On this non-convex benchmark the spline refinement **hurts**: it nearly
doubles per-iteration cost (≈64 ms/it vs ≈34 ms/it) while the cubic Hermite
model is too inaccurate on this rugged surface to recover that cost in
per-step progress. In a fixed wall-clock budget the spline variants simply run
fewer, more-expensive iterations and finish higher. The spline's benefit is
therefore strongly problem-dependent — beneficial on the smooth convex softmax
benchmark, counterproductive on this mixed-activation MLP.

### Performance: Best-of-Breed Stack (QQN-Fast)

The `QQN-Fast` variant combines deep L-BFGS memory (history=50), warm-started
backtracking (`init_step=2.5`, `shrink=0.65`, `c1=1e-3`, `max_iter=40`), and a
fixed trust-region (`r=2.0`):

| Variant  | Config                                      | iters | final_loss | test_acc |
|----------|---------------------------------------------|-------|------------|----------|
| QQN-Fast | L50 + BT(init=2.5, shrink=0.65) + TR(r=2.0) | 389   | 2.014e-1   | 85.4%    |

`QQN-Fast` descends to `2.014e-1` (test accuracy 85.4%) — better than the bare
`QQN` baseline but **worse** than bare `QQN-L50` (`1.043e-1`). On this harder
instance the warm-started backtracking + fixed trust-region combination is
net-negative relative to deep memory alone: the aggressive warm-start and the
fixed radius interact poorly with the mixed-activation curvature, leaving
`QQN-Fast` short of what `QQN-L50` achieves with a plain Armijo search.

### Performance: Maximal Robust Stack (QQN-Max / QQN-MaxP)

The `QQN-Max` variant stacks **all** the documented winning levers without
collapsing the diversity of the sweep: a `Fallback([L-BFGS-50, Anderson])`
oracle (deep curvature with a residual-solve safety net), warm-started
backtracking (`init_step=2.5`, `shrink=0.65`, `c1=1e-3`, `max_iter=40`), a
fixed trust-region (`r=2.0`), **and** spline refinement (`spline=True`).

| Variant  | Stack                                                       | iters | final_loss | test_acc |
|----------|-------------------------------------------------------------|-------|------------|----------|
| QQN-Max  | Fallback([L50,And]) + warm BT + fixed TR(r=2) + spline       | 220   | 2.594e-1   | 85.9%    |
| QQN-MaxP | QQN-Max + `feed_probes_to_oracle=True`                       | 107   | 2.302e+0   | 9.85%    |

`QQN-Max` descends to `2.594e-1` (test accuracy 85.9% — the **highest of any
variant in the run**) but, like `QQN-Fast`, does **not** beat bare `QQN-L50` on
training loss: the spline overhead (≈68 ms/it) and the warm-BT + fixed-TR
interaction throttle its iteration count to 220, leaving it short of L50's
`1.043e-1`. Stacking levers does *not* compound benefits on this instance.

`QQN-MaxP` (the probe-fed version) **diverges identically to `QQN-L50P`**
(`2.302e+0`, chance accuracy): probe-feeding dominates and destroys the stack
regardless of the other levers. This confirms probe-feeding as the decisive
failure mode on this configuration.

---

## Leaderboards

### Final-Loss Ranking (no variant reached the target; ranked by final loss)

Because the `→target` columns are empty for every variant on this run, the
operative ranking is by **final loss at the 15 s budget**:

```
L-BFGS        final=5.1075e-02  iters=2671  ms/it= 5.62  test_acc=85.15%  AUC=-1.03
QQN-L50       final=1.0426e-01  iters= 426  ms/it=35.30  test_acc=85.60%  AUC=-0.47
QQN-L50And    final=1.1540e-01  iters= 395  ms/it=38.12  test_acc=85.80%  AUC=-0.43
Adam          final=1.1886e-01  iters=6838  ms/it= 2.19  test_acc=84.65%  AUC=-0.73
SGD           final=1.3079e-01  iters=6986  ms/it= 2.15  test_acc=86.85%  AUC=-0.54
QQN-L20       final=1.6380e-01  iters= 408  ms/it=36.86  test_acc=85.45%  AUC=-0.37
QQN-Fast      final=2.0140e-01  iters= 389  ms/it=38.71  test_acc=85.35%  AUC=-0.34
QQN-Max       final=2.5944e-01  iters= 220  ms/it=68.47  test_acc=85.90%  AUC=-0.23
QQN           final=3.1376e-01  iters= 446  ms/it=33.75  test_acc=85.40%  AUC=-0.24
QQN-Box       final=3.2319e-01  iters= 395  ms/it=38.13  test_acc=86.75%  AUC=-0.22
QQN-BT        final=3.4377e-01  iters= 401  ms/it=37.52  test_acc=85.80%  AUC=-0.21
QQN-Sec       final=4.1544e-01  iters= 409  ms/it=36.83  test_acc=85.10%  AUC=-0.17
QQN-TR        final=4.7964e-01  iters= 399  ms/it=37.67  test_acc=83.80%  AUC=-0.13
QQN-And       final=5.2410e-01  iters= 332  ms/it=45.34  test_acc=81.75%  AUC=-0.03
QQN-BT-S      final=5.6009e-01  iters= 268  ms/it=56.19  test_acc=83.60%  AUC=-0.08
QQN-S         final=5.9178e-01  iters= 237  ms/it=63.50  test_acc=82.25%  AUC=-0.06
QQN-Mom       final=1.0836e+00  iters= 435  ms/it=34.57  test_acc=63.85%  AUC= 0.22
QQN-Mom-S-BT  final=1.4444e+00  iters= 261  ms/it=57.70  test_acc=54.15%  AUC= 0.30
QQN-Mom-S     final=1.4481e+00  iters= 260  ms/it=57.80  test_acc=53.95%  AUC= 0.30
QQN-L50P      final=2.3024e+00  iters= 190  ms/it=79.22  test_acc= 9.85%  AUC= 0.36
QQN-MaxP      final=2.3024e+00  iters= 107  ms/it=141.07 test_acc= 9.85%  AUC= 0.36
```

### Iteration-Efficiency & Cost-Aware Leaderboards

**Empty on this run.** Both leaderboards are gated on reaching `f_target`, and
no variant did, so both print no entries. The cost-aware **evals-to-target**
figures are likewise unavailable (`—` in the table above).

### Trajectory-AUC (lower = faster overall descent, selected variants)

```
L-BFGS         AUC=-1.03  final=5.1075e-02  time=15.000s
Adam           AUC=-0.73  final=1.1886e-01  time=15.002s
SGD            AUC=-0.54  final=1.3079e-01  time=15.001s
QQN-L50        AUC=-0.47  final=1.0426e-01  time=15.038s
QQN-L50And     AUC=-0.43  final=1.1540e-01  time=15.059s
QQN-L20        AUC=-0.37  final=1.6380e-01  time=15.038s
```

On this run **L-BFGS leads on AUC** (−1.03) as well as final loss, reflecting
both fast early descent *and* the deepest final loss — it dominates the
trajectory metric outright. Among QQN variants, deep-memory `QQN-L50` has the
best AUC (−0.47). Note that AUC here correlates with final loss because no
method reaches a flat asymptote within the budget.

---

## Cautionary Findings (Stall Report)

On this run **every optimizer stalled** (none reached the shared target), all
classified as "time-budget exhausted." The most instructive failures:

| Variant      | final_loss   | cause / interpretation                                   |
|--------------|--------------|----------------------------------------------------------|
| L-BFGS       | 5.108e-2     | budget exhausted *just above* target (best-in-class)     |
| QQN-L50      | 1.043e-1     | budget exhausted (best QQN; per-iter cost limits iters)  |
| Adam         | 1.189e-1     | budget exhausted (cheap steps, shallow final descent)    |
| SGD          | 1.308e-1     | budget exhausted                                         |
| QQN-TR       | 4.796e-1     | adaptive radius over-conservative on mixed activations   |
| QQN-And      | 5.241e-1     | Anderson degenerates on non-convex surface               |
| QQN-Mom*     | 1.08–1.45e+0 | first-order momentum plateaus (no curvature)             |
| QQN-L50P     | 2.302e+0     | **probe-feeding corrupts L-BFGS memory → divergence**    |
| QQN-MaxP     | 2.302e+0     | **probe-feeding dominates and destroys the max stack**   |

These are **first-class experimental findings**, not failures to hide:

1. **Probe-feeding is harmful here.** The two probe-fed variants are the *only*
    configurations that fail to descend *at all* (chance-level accuracy). This
    is the single most important new finding of this run: a lever previously
    documented as "free curvature" is configuration-dependent and can be
    catastrophic on ill-conditioned non-convex surfaces.
2. **First-order momentum** plateaus badly (loss > 1.0, accuracy 54–64%) — even
    worse than the curvature oracles. The mixed-activation landscape amplifies
    momentum's inability to capture curvature.
3. **Anderson and Secant oracles** stall on the non-convex landscape, with
    Anderson the weakest curvature oracle (`5.241e-1`).
4. **The spline and stacked variants do not compound** — `QQN-Fast`, `QQN-Max`,
    `QQN-S`, and `QQN-BT-S` all finish *worse* than bare `QQN-L50`, because
    their higher per-iteration cost reduces the iteration count affordable in
    the fixed budget.

---

## Summary of Design-Claim Validation

| QQN Design Claim                                                 | Empirical Verdict (this non-convex MLP run)                                      |
|------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Gradient + oracle blending via the quadratic path converges fast | ⚠️ QQN descends, but does **not** beat classical L-BFGS here (5.1e-2 vs 1.0e-1)   |
| The oracle is freely swappable                                   | ✅ L-BFGS, Momentum, Secant, Anderson, Fallback all run                          |
| Deep curvature memory accelerates convergence                    | ✅ L10→L20→L50 monotone in final loss (3.1e-1 → 1.6e-1 → 1.0e-1)                  |
| The line search trades wall-time, not convergence speed          | ⚠️ Spline overhead *reduces* budget-limited progress (higher final loss)         |
| Regions are low-overhead safeguards                              | ⚠️ Box improves test acc; adaptive TR is over-conservative (worse final loss)     |
| The spline reuses information to sharpen trajectories            | ❌ Net-negative here: nearly 2× cost, cubic model inaccurate on rugged surface    |
| Warm-started backtracking + fixed TR (QQN-Fast)                  | ⚠️ Converges (2.0e-1) but worse than bare QQN-L50                                 |
| Maximal robust stack (QQN-Max)                                   | ⚠️ Runs; best test acc (85.9%) but worse training loss than QQN-L50              |
| Probe-feeding enriches curvature for free (QQN-L50P / QQN-MaxP)   | ❌ **Catastrophic divergence** (loss 2.30e+0, ~10% acc) — harmful on this surface |
| Cost-aware (evals-to-target) metric reported                     | ✅ Reported (empty here — no variant reached target)                             |
| Target-sensitivity profile reported                              | ✅ Reported; only `2e-1` reachable by QQN on this instance                       |

The central thesis — that coherently blending gradient and oracle along the
quadratic path yields a fast, modular optimizer — is **only partially supported
by this run**. QQN's deep-memory stack (`QQN-L50`) is the strongest QQN
configuration and descends respectably, but on this harder mixed-activation
instance it does **not** overtake classical L-BFGS, whose cheap per-iteration
cost wins the fixed-budget race decisively. Two important negative findings
stand out: **probe-feeding is actively harmful** here, and **the spline / stacked
variants do not compound** — both reduce budget-limited progress relative to a
plain deep-memory Armijo search.

Key findings on this non-convex MLP run:
- **L-BFGS** reaches the lowest loss (`5.108e-2`) and best AUC (−1.03) — the
   clear overall winner on this instance.
- **QQN-L50 / QQN-L50And** are the strongest QQN variants (`1.043e-1` /
   `1.154e-1`), but neither beats L-BFGS within the budget.
- **Adam / SGD** descend respectably (`1.19e-1` / `1.31e-1`) on the strength of
   cheap per-step cost and high iteration counts.
- **Probe-fed variants (`QQN-L50P`, `QQN-MaxP`) diverge** to chance-level
   accuracy — probe-feeding is the decisive failure mode on this configuration.
- **No optimizer reached the `5.0e-2` target** within the 15 s budget.

See [`algorithm.md`](algorithm.md) for the conceptual treatment and
[`../results/fashion_mnist_mlp_comparison_20260622_142624.log`](../results/fashion_mnist_mlp_comparison_20260622_142624.log)
for the full raw output.

> **Re-run caveat.** The tables and leaderboards above are point estimates from
> a single run under the configuration recorded in the log
> (`activation=sigmoid,relu,gaussian`, `DEPTH=4`, `HIDDEN=64`,
> `N_TRAIN=15000`, `N_TEST=2000`, `f_target=5.0e-2`, `time_budget=15.0 s`).
> The **committed** `examples/fashion_mnist_mlp_comparison.py` now defaults to
> a *different, larger* configuration (`N_TRAIN=20000`, `N_TEST=3000`,
> `DEPTH=5`, `f_target=4.0e-2`, `time_budget=20.0 s`) and adds `QQN-L80`,
> `QQN-L80P`, and `QQN-UltraP` variants. Re-running the script as committed
> will therefore shift every absolute number and change the variant roster.
> Re-run `python examples/fashion_mnist_mlp_comparison.py` to regenerate the
> tables and refresh the referenced log file. Treat the rankings — and
> especially the probe-feeding failure — as **configuration-specific** until
> re-validated.
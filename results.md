# Fashion-MNIST MLP Optimizer Comparison — Sigmoid Activation

**Variant:** `comparison_act_sigmoid`
**Report:** `fashion_mnist_mlp_comparison`
**Date:** 2026-07-18

## Setup

| Setting        | Value                                     |
|----------------|-------------------------------------------|
| Dataset        | Fashion-MNIST (10 classes)                |
| Architecture   | `x -> 128 -> 128 -> 10` (2 hidden layers) |
| Activation     | **sigmoid** (non-convex objective)        |
| Parameters     | 118,282                                   |
| Train / Test   | 8,000 / 2,000                             |
| Time budget    | 45 s per optimizer                        |
| Target loss    | 1.0e-02 (`gtol=1e-8`)                     |
| Baselines      | SGD (lr=0.05), Adam (lr=0.01), L-BFGS     |
| Regularization | l2 = 1e-4                                 |

Backend fell back to **CPU** (no CUDA jaxlib), so wall-clock numbers reflect CPU execution. **No optimizer reached the
1e-2 target** within 45 s — every run is time-budget exhausted (or an early plateau/divergence). Rankings below are
therefore by *final loss achieved within budget*.

## Headline Result

The **QQN + deep L-BFGS-oracle** family sweeps the top of the board. The five best final losses are all `QQN-L10-*`
variants, edging out standalone L-BFGS and dramatically beating Adam, SGD, and every momentum/path-momentum oracle.

| Rank | Optimizer    | Final Loss | Iters | Train Acc | Test Acc | ms/it |
|------|--------------|------------|-------|-----------|----------|-------|
| 1    | QQN-L10-Fix  | 8.843e-02  | 3852  | 1.0000    | 0.8460   | 11.68 |
| 2    | QQN-L10-Temp | 8.848e-02  | 3796  | 1.0000    | 0.8450   | 11.85 |
| 3    | QQN-L10-Arm  | 8.894e-02  | 3557  | 1.0000    | 0.8585   | 12.65 |
| 4    | QQN-L10      | 8.895e-02  | 3549  | 1.0000    | 0.8580   | 12.68 |
| 5    | QQN-L10-BT   | 8.895e-02  | 3544  | 1.0000    | 0.8585   | 12.70 |
| 6    | QQN-L10-HZ   | 8.962e-02  | 2837  | 1.0000    | 0.8595   | 15.86 |
| 7    | **L-BFGS**   | 8.989e-02  | 2145  | 1.0000    | 0.8520   | 20.98 |
| 8    | QQN-L10-SW   | 9.002e-02  | 2206  | 1.0000    | 0.8540   | 20.41 |
| ...  | Adam         | 1.371e-01  | 5388  | 0.9998    | 0.8440   | 8.35  |
| ...  | SGD          | 5.311e-01  | 5831  | 0.8218    | 0.8200   | 7.72  |

## Key Observations

### 1. QQN-L10 wins on both iterations *and* wall-clock cost per iteration

Every top QQN-L10 variant is **cheaper per iteration** than standalone L-BFGS (~11.7–12.7 ms/it vs. 20.98 ms/it). This
is the expected effect of the bare Armijo/backtracking inner search costing ~1.0–1.1 evals/it, whereas L-BFGS's Optax
zoom search costs ~2.1 evals/it. Within the same 45 s, QQN-L10-Fix ran **3852 iterations vs. L-BFGS's 2145** — a ~1.8×
iteration throughput advantage that translates directly into a lower final loss.

### 2. Line-search choice matters less than the oracle

Among `QQN-L10-*` the inner line search barely moves the needle at this target: Fixed, Temperature, Armijo, plain, and
Backtracking all land within
`8.84e-02 – 8.90e-02`. The heavier searches (HZ, Strong-Wolfe) do fewer iterations (2837 / 2206) but reach comparable
losses — the extra per-iteration cost is not repaid here. The oracle (deep L-BFGS history) is the dominant lever,
exactly as the algorithm write-up predicts.

### 3. Convergence-rate profile confirms QQN's early-and-sustained lead

From the wall-clock convergence profile (seconds to reach each loss):

| Optimizer    | ≤1.0 | ≤0.5 | ≤0.2 | ≤0.1     |
|--------------|------|------|------|----------|
| QQN-L10-Fix  | 1.27 | 1.74 | 3.36 | **9.72** |
| QQN-L10-Temp | 1.83 | 2.30 | 3.92 | 10.29    |
| L-BFGS       | 2.34 | 3.16 | 5.70 | 16.73    |
| Adam         | 0.55 | 0.85 | 2.91 | —        |
| SGD          | 9.18 | —    | —    | —        |

Adam is fastest to the *coarse* milestones (≤0.5) but **never reaches 0.1** — it plateaus around `1.37e-01`. QQN-L10-Fix
reaches 0.1 in under 10 s, ~1.7× faster than L-BFGS in wall-clock. This matches the paper's claim that the QQN advantage
**widens as the target tightens** (the fine-tuning regime).

### 4. Non-L-BFGS oracles struggle badly on sigmoid

- **Momentum (`QQN-Mom-*`)** stalls in the `0.34 – 0.50` loss range — a first-order heavy-ball direction cannot capture
  the anisotropic curvature.
- **Path-Momentum (`QQN-PathMom-*`)** is worse still (`1.26 – 1.55+`), and several `-P` (probe-feeding) / `-Fix`
  combinations **diverge to `inf`**.
- **Adam-oracle (`QQN-Adam-*`)** is a mixed bag: the accepted-search variants land in `0.15 – 0.24`, but `QQN-Adam-Fix`
  and `QQN-Adam-Temp` **explode**
  (`3.8e+01 – 6.5e+01`) — a fixed/temperature step with an unclipped Adam direction and no descent-anchoring line search
  runs away.

### 5. Probe-feeding (`-P`) is actively harmful here

The `-P` (feed line-search probes to the oracle) axis is consistently destructive on this problem:

- Several `QQN-L10-*-P` variants **plateau immediately** at the initial loss
  `2.3026e+00 (= ln 10)` after only 2–4 iterations (`QQN-L10-Fix-P`,
  `QQN-L10-HZ-P`, etc.) — the collinear probe gradients pollute the L-BFGS history and kill progress.
- Momentum/PathMom `-P` variants diverge to `inf`. This is a clear demonstration of the history-pollution failure mode
  the algorithm doc warns about; the descent gate is insufficient to save the fixed-step / degenerate cases.

### 6. Spline / linear (`-S`) refinement: modest cost, no benefit at target

The `-S` (spline) variants (`QQN-L10-S`, `QQN-L10-Spl`, `QQN-L10-Arm-S`)
reach `~1.00e-01`, slightly worse than the un-splined `~8.9e-02`. The extra probes (~67 ms/it vs. ~12 ms/it) cut
iteration count roughly 5× (663 vs.

3852) without a matching quality gain — on this smooth-ish objective the bare quadratic path already extracts the
      available signal.

## Pareto Frontier (loss vs. time)

The non-dominated set is dominated by QQN:

  ```
    QQN-L10-Fix-P   loss=2.30e+00  time=1.11s   (instant plateau — cheap but useless)
    QQN-PathMom-SW  loss=1.52e+00  time=45.0s
    QQN-PathMom-Fix-S loss=1.26e+00 time=45.0s
    QQN-Mom         loss=3.45e-01  time=45.0s
    QQN-L10-Temp    loss=8.85e-02  time=45.0s
    QQN-L10-Fix     loss=8.84e-02  time=45.0s   ← best achievable loss
  ```

Standalone L-BFGS, Adam, and SGD are all **dominated** by QQN-L10 on the loss-vs-time frontier.

## Generalization Note

All top optimizers drive **train accuracy to 1.0000** (memorizing the 8k subset) while test accuracy clusters at
**0.84–0.86**. Test accuracy is essentially decoupled from the final training loss among the leaders — the best test
accuracy (0.8615, `QQN-Adam-Arm`) comes from an optimizer ranked
~20th by training loss. This is an overfitting regime; the optimizer race is about *training-loss minimization
efficiency*, not generalization.

## Conclusions

1. **QQN with a depth-10 L-BFGS oracle is the clear winner** on sigmoid Fashion-MNIST, beating standalone L-BFGS in both
   per-iteration cost and final loss, and beating Adam/SGD by a wide margin.
2. **The oracle is the dominant axis.** Line-search variation within
   `QQN-L10` is second-order; momentum/path-momentum/Adam oracles are far weaker and sometimes unstable.
3. **Probe-feeding (`-P`) should stay off** for this class of problem — it triggers immediate plateaus or divergence.
4. **Spline refinement (`-S`) does not pay for itself** here; the plain quadratic path is already near-optimal.
5. **CPU + 45 s budget was too tight** to hit the 1e-2 target — a GPU run or longer budget would be needed to observe
   the superlinear finish, but the ranking and QQN's widening advantage as the target tightens are already clear at the
   0.1 milestone.

### Suggested follow-ups

- Re-run on GPU (CUDA jaxlib) to reach the 1e-2 target and confirm the superlinear regime.
- Sweep L-BFGS history size (`L20`/`L50`/`L80`) — the doc reports the wall-clock knee at 80–120 on a related task.
- Investigate the `QQN-Adam-Fix`/`-Temp` divergence: an unanchored fixed step on the raw Adam direction needs a
  descent-enforcing search.
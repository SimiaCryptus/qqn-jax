# Results

## 1. Overview

We organize the empirical results as a focused tour through the questions
a practitioner is most likely to ask. We begin by contrasting the
canonical *reference* optimizers (SGD, Adam, L-BFGS) against the
best-of-breed QQN configurations discovered by our cross-product search
(Section 2). We then isolate individual QQN axes: the effect of the
Adam oracle under temperature and against an oracle baseline (Section 3),
the impact of the line-search strategy (Section 4), and finally a
detailed case study on the sine and rolling-sine activations, where
L-BFGS-oracle QQN shows its most decisive advantage and where the
PSD-region effect becomes visible (Section 5). We close with a
discussion of measurement artifacts and open questions (Section 6).

All results obey the fairness invariants of the methodology
(identical initialization, shared termination, genuine eval accounting,
a shared loss closure, and warmup exclusion). Unless stated otherwise,
we report the three complementary measures established there:
iterations (unrepresentative but familiar), wall-clock time to target
(honest but hardware-sensitive), and cumulative value-and-gradient
evaluations (the most reproducible cross-optimizer currency).

## 2. Reference Optimizers vs. Best-of-Breed QQN

Our headline comparison places the four reference optimizers against the
strongest QQN configuration found by the cross-product search. On a
4-layer `tanh,gelu,tanh` MLP (335k parameters) trained full-batch on
Fashion-MNIST, QQN with a deep L-BFGS oracle (history size 80) and
backtracking line search decisively wins both the iteration race and
wall-clock time to the `2e-2` loss target:

- **Iterations to target.** QQN-L80-BT reaches the target in
2.64× fewer iterations than standalone Optax L-BFGS.
- **Per-iteration cost.** QQN is also *cheaper* per iteration
(16.08 ms/it vs. 20.71 ms/it), because bare Armijo backtracking
averages ~1.0–1.1 evaluations per iteration whereas the Optax zoom
line search inside standalone L-BFGS costs ~2.1 evaluations per
iteration.
- **Pareto dominance.** The loss-vs-wall-time Pareto frontier is
populated *entirely* by QQN variants; standalone L-BFGS is dominated,
and SGD/Adam trail on the fine-tuning tail.

The advantage *widens as the target tightens*. First-order baselines
(SGD, Adam) exhaust their budget in the fine-tuning regime, while the
second-order information carried by the L-BFGS oracle lets QQN keep
descending. This is the compounding value proposition: the tighter the
target, the larger QQN's margin.

## 3. Adam-Oracle Comparisons: Temperature and Oracle

To isolate the oracle axis we hold the path (quadratic) and line search
(backtracking) fixed and vary only the oracle. The Adam oracle — which
supplies an adaptive-moment direction at the `t = 1` endpoint — provides
a natural first-order-flavored comparison point against the L-BFGS
oracle and against a plain Adam baseline.

Two observations stand out. First, wrapping Adam's direction in the QQN
quadratic path improves on the standalone Adam baseline: the
gradient-tangent anchor at `t = 0` recovers robustness on iterations
where the raw adaptive step overshoots, and the line search over `t`
discovers a better-scaled blend than Adam's fixed learning rate.
Second, the Adam oracle nonetheless cannot match the deep L-BFGS oracle
on the anisotropic Hessians of these MLPs — momentum and adaptive
rescaling lack the dominant-subspace capture that a deep curvature
history provides.

**Temperature.** Layering the Metropolis-style stochastic acceptance
(`temperature > 0`) on top of the Adam-oracle configuration permits
controlled uphill moves early in training. On the more multi-modal
landscapes this helps escape early plateaus, but as the temperature
cools the behavior converges to the `temperature = 0` Armijo baseline.
The net effect is a modest early-iteration benefit that does not alter
the final-loss ordering, confirming that temperature is an orthogonal
exploration knob rather than a source of the core advantage.

## 4. Line-Search Comparison

Holding the oracle (L-BFGS) and path (quadratic) fixed, we vary the line
search across backtracking/Armijo, strong Wolfe, Hager-Zhang, and the
fixed-step baseline. Because the line search traverses the path
parameter `t` directly, its behavior directly bounds overall quality.

- **Backtracking / Armijo** is the robust default and the overall
winner on wall-clock time to target: it accepts steps cheaply
(~1.0–1.1 evals/iteration) and rarely over-restricts the
quadratic-path step.
- **Strong Wolfe** keeps the L-BFGS curvature pairs `(s, y)`
well-conditioned but *over-restricts* the quadratic-path step on
several benchmarks, spending extra evaluations enforcing the strong
curvature condition without a commensurate iteration reduction.
- **Hager-Zhang** behaves comparably to backtracking as a robust
approximate-Wolfe scheme.
- **Fixed** is included as a debugging baseline; its anomalous
behavior is discussed in Section 6.

The takeaway is that the cheapest sufficient-decrease search that still
admits the full quadratic step is preferable: on the QQN path the
gradient tangent already guarantees a valid descent step exists, so the
extra curvature enforcement of strong Wolfe is more insurance than the
path needs.

## 5. Case Study: Sine and Rolling-Sine Activations

The activation sweep surfaces a regime where the L-BFGS-oracle QQN shows
especially strong performance: the periodic `sine` activation and its
windowed variant `rolling_sin`. These nonlinearities induce a loss
landscape with many locally-quadratic basins whose curvature is sharply
anisotropic — exactly the setting in which a deep curvature history pays
off.

- **L-BFGS-oracle superiority.** On both `sine` and `rolling_sin`,
QQN with the deep L-BFGS oracle reaches the target in substantially
fewer iterations than the first-order oracles (momentum, Adam) and
than the matrix-free secant oracle. The gap is larger here than on the
smoother `tanh`/`gelu` topologies, reflecting the periodic
landscape's stronger curvature structure.
- **PSD-region effect.** Enabling the PSD-secant region on these
activations produces a visible stabilization: by constraining steps to
the region where the secant curvature estimate remains
positive-definite, the optimizer avoids the spurious near-degenerate
curvature the periodic landscape occasionally produces. The effect is
most pronounced in the fine-tuning tail, where an unconstrained oracle
can otherwise emit an occasional uphill quasi-Newton direction that the
quadratic path's gradient tangent must repair at the cost of a shrunk
step.

Together these two effects illustrate the framework's central claim in
miniature: the oracle supplies aggressive curvature-aware direction, the
quadratic path's tangent anchor guarantees safety, and the region
projection cleans up the pathological corners — each an independently
swappable axis, each contributing a measurable slice of the result.

## 6. Discussion and Open Questions

Several measurement artifacts and anomalies warrant explicit discussion
in keeping with our fairness-first methodology.

### 6.1 First-Iteration Compilation Cost

Both L-BFGS and QQN exhibit an extra ~2.6 s on their first *timed*
iteration. Although the harness excludes the initial JIT trace/compile
via a warmup step (blocking on `jax.block_until_ready`), a residual
compilation cost remains attributable to JAX specializing the
line-search control flow on its first real invocation. We report this
explicitly rather than smoothing it away; it inflates the total
wall-clock of short runs but is amortized over longer ones, and it does
not affect the evaluation-count or iteration measures.

### 6.2 L-BFGS Evaluation Accounting

The Optax L-BFGS implementation exposes iteration counts but not a
direct per-iteration line-search evaluation count. We therefore
*estimate* its evaluation total: when the optimizer state exposes a
`num_linesearch_steps` field we read it and fold it into a running
average, and otherwise fall back to that adaptive running average rather
than a fixed constant. This adaptive timing-based estimation is a known
limitation of the cross-optimizer eval comparison for L-BFGS, and we
flag any evaluation-axis result involving standalone L-BFGS as
approximate. QQN's own counts, by contrast, are tracked exactly through
every line-search probe, spline probe, aux recompute, and probe-value
recovery.

### 6.3 Anomalous "Fix" Strategy Performance

The `fixed` line-search strategy, intended only as a debugging baseline,
performs surprisingly well on several activation/topology combinations,
occasionally winning against the more principled variants. We do not yet
have a satisfying explanation. One hypothesis is that on these
landscapes the quadratic path's endpoint (`t = 1`) is frequently a good
step, so a strategy that simply commits to it avoids the backtracking
overhead the adaptive searches incur. Consistent with our
fairness-first stance, we flag this as an open question for further
study rather than tuning it away, and we caution against reading the
`fixed` strategy's occasional wins as evidence that line search is
unnecessary — its performance is brittle and activation-dependent.
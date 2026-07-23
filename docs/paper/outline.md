# QQN: A Pluggable Optimizer Framework Built on Quadratic Path Interpolation

## Paper Structure

The paper is organized as four major parts, moving from a concrete,
minimal demonstration of the core idea toward the full theoretical
framework, the empirical methodology, and finally the results and
conclusions.

---

### Part I — Brief
*A minimal, self-contained demonstration of the core idea.*

Goal: convince the reader that a single, simple change to L-BFGS — a
permissive line search along a quadratic path connecting the gradient and
the quasi-Newton direction — is enough to outperform both existing methods
and every more elaborate QQN variant we tried. This part is deliberately
short and results-forward: it motivates the reframing ("where on the
curve?" rather than "which direction?"), presents the one-equation path,
and shows the headline win, deferring all generalization to later parts.

- Motivation: the classical direction dilemma (robust gradient vs. fast,
  fragile quasi-Newton) and the "pick one, then line-search" compromise.
- The minimal enhancement: the quadratic path
  `d(t) = t(1-t)(-∇f) + t²(-H∇f)` plus a permissive line search over `t`.
- Why it works in one paragraph: gradient-tangent globalization,
  oracle-endpoint speed, blend discovered automatically.
- Headline result: the simple variant beats standalone L-BFGS, SGD, and
  Adam — and beats the elaborate QQN variants too.
- Forward pointer: everything else in the paper is the generalization and
  justification of this one move.

---

### Part II — Theory
*The full framework: the four-axis factoring and its components.*

Goal: establish the conceptual basis. Reframe optimization as four
orthogonal, independently swappable axes — **gradient**, **oracle**,
**search**, **region** — and show that classical methods (L-BFGS, Newton,
momentum, OWL-QN, trust region, CG) are special cases. Derive the
quadratic path from its endpoint/tangent constraints, then present the
component library (oracles, path/spline strategies, line searches,
regions) and the solver loop.

- Orientation: the anatomy of an optimizer (the four questions).
- QQN innovations: quadratic path, strategy pattern, geometric
  principledness, provable convergence.
- Component axes:
  - Oracle strategies (gradient, momentum, Adam, L-BFGS, secant,
    Shampoo, Anderson; combinators; partitioning; probe feeding).
  - Path strategies (linear, quadratic, spline).
  - Line-search strategies (backtracking/Armijo, strong Wolfe,
    Hager-Zhang, fixed, null; temperature; wrapping; recursive descent).
  - Regions (orthant, trust region, no-decrease, box/quantization;
    composition).
- The solver loop and state threading (pure-functional JAX).
- Theoretical guarantees (global, superlinear, descent; C⁰ sufficiency).
- Related work and equivalences.

---

### Part III — Benchmarking Methodology
*How we measure fairly and reproducibly.*

Goal: describe the experimental harness in enough detail that the results
are trustworthy and reproducible. Emphasize *fairness invariants* so that
measured differences are attributable to the optimizers, not the harness.

- Experimental design: objective/models (flat MLP), datasets
  (MNIST/Fashion-MNIST), optimizers under test.
- Fairness invariants: identical init, shared termination, genuine eval
  accounting, identical loss/data/regularization, warmup exclusion.
- QQN configuration space: the cross-product of orthogonal axes and the
  profile-naming convention.
- Metrics: loss quality, accuracy, iterations, wall time/ms-iter,
  trajectory AUC, time/iters/evals to target, per-milestone crossings.
- Pareto analysis: loss vs. time; per-milestone (time, evals) frontiers.
- Reproducibility infrastructure: config binding, driver pipeline, JSON
  artifacts, manifest, viewer, batch runner, activation sweeps.
- Threats to validity: JIT-compilation attribution, L-BFGS eval-count
  estimation, anomalous line-search behavior.

- JAX specifics and complications: `jit`/`vmap` constraints, warmup and
  compilation cost, deterministic PRNG use, block-until-ready timing.

---

### Part IV — Results Discussion
*What the experiments show.*

Goal: present and interpret the empirical findings across problems and
activation functions.

- Headline: quadratic path + deep L-BFGS oracle wins both iteration race
  and wall-clock time to target on smooth, ill-conditioned objectives.
- Axis-by-axis analysis: which oracle, line search, path, and region
  choices matter, and by how much.
- Milestone/target-sensitivity behavior: the speedup widening as the
  target tightens (fine-tuning regime).
- Pareto frontiers: QQN dominance vs. standalone baselines.
- Activation-function sweep: geometry-dependent relative advantage.
- Anomalies and open questions (e.g. `Fix` line-search surprises).

---

### Part V — Conclusions
*What it means.*

Goal: state the architectural takeaway and future directions.

- The contribution is architectural: factoring, not a single algorithm.
- The parabola "falls out" of the decomposition; the straight line was a
  hard-coded default.
- The value proposition compounds as the target tightens.
- Future work: richer oracles, recursive descent, region composition,
  stochastic/mini-batch regimes.
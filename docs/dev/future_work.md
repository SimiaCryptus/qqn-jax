---
documents:
  - results.md
related:
  - conclusions.md
  - algorithm.md
  - oracles.md
  - regions.md
  - spline_search.md
---

# Future Work: Proposed Benchmarks

The [`conclusions.md`](../conclusions.md) evaluation rests on a single smooth,
deterministic, full-batch convex benchmark (softmax MNIST). Each caveat in
that document is a testable hypothesis. The benchmarks below are chosen to be
both **small-scale-compute** (laptop / single-GPU, seconds-to-minutes) and to
probe the **non-convex, stochastic, ill-conditioned, and structured** regimes
of interest to modern research.

## 1. Non-smooth & Non-convex (caveat: "smoothness flatters cheap searches")

- **2-layer ReLU MLP on (Fashion-)MNIST** (~10–50K params). ReLU kinks make
  the objective non-smooth; minimal step into non-convexity. Hypothesis:
  Wolfe / Hager-Zhang / spline searches finally earn their ~2–3× cost.
- **Classic non-convex functions** (Rosenbrock, Rastrigin, Ackley, Beale).
  Effectively free. Stress-tests the steepest-descent anchor against
  ill-conditioned valleys and many local minima.

## 2. Ill-conditioning (caveat: "well-conditioned objective")

- **Synthetic quadratics with controlled `κ ∈ {1e1, 1e3, 1e6}`.** Exact and
  cheap; isolates whether deep L-BFGS memory and stronger searches earn their
  cost as conditioning worsens.
- **Colinear-feature logistic regression.** Convex but ill-conditioned;
  separates "non-smooth" from "ill-conditioned" effects.

## 3. Stochastic / Mini-batch (caveat: "full-batch, deterministic")

- **Mini-batch softmax / MLP MNIST** with batch sizes `{32, 128, full}`.
  The largest gap for modern DL: curvature estimates and line searches behave
  differently under gradient noise. Hypothesis: the steepest-descent anchor
  still yields robust convergence, but the line-search cost/benefit ranking
  shifts toward cheaper searches and toward Adam/SGD.

## 4. Structured Parameters (caveat: "structured parameters change the ranking")

- **Small CNN on MNIST / CIFAR-10 subset** (<100K params). Genuinely
  matrix-shaped weights give the Shampoo oracle its intended structure;
  tests whether its dense inverse-root cost amortizes better.
- **Tiny char-level Transformer** (1–2 layers, tiny-shakespeare). The modern
  domain of interest; matrix-shaped attention/MLP blocks plus realistic
  landscape pathology at low cost.

## 5. Sparsity & Regularization (extends the orthant-region finding)

- **L1-regularized logistic regression.** Promotes the orthant region's
  incidental `0.0056` sparsity result to a primary metric on a genuinely
  kinked objective where the region axis should matter.

## Metrics to track across all of the above

- Final train loss **and** generalization (test acc) — to revisit the
  "generalization was not the differentiator" observation off-MNIST.
- Wall-time and iterations-to-`f_target`.
- Per-axis A/B deltas, to confirm modularity still holds off the convex case.


# Suggested tests, organized by the caveat they attack

### 1. Non-smooth / non-convex but tiny (attacks "Smoothness flatters cheap searches")
- **2-layer MLP on MNIST/Fashion-MNIST with ReLU** (~10–50K params). The ReLU kinks make the objective non-smooth; this is the minimal step into non-convexity and directly tests whether Wolfe/Hager-Zhang/spline finally pay off.
- **Classic non-convex test functions**: Rosenbrock (ill-conditioned valley), Rastrigin/Ackley (many local minima), Beale. These are *free* compute-wise and are the standard way to expose oracle/region behavior on pathological curvature. They directly test your "global convergence via steepest-descent anchor" claim against many local minima.

### 2. Ill-conditioning (attacks "well-conditioned objective where curvature is easy to exploit")
- **Synthetic quadratics with controlled condition number** `κ ∈ {10, 10³, 10⁶}`. Cheap, exact, and isolates whether deep L-BFGS memory and the stronger searches earn their cost as κ grows.
- **Logistic regression on a deliberately correlated/colinear feature set** — keeps it convex but ill-conditioned, separating "non-smooth" effects from "ill-conditioned" effects.

### 3. Stochastic / mini-batch (attacks the "full-batch, deterministic" assumption — arguably the biggest gap for modern DL)
- **Mini-batch softmax/MLP MNIST** with batch sizes `{32, 128, full}`. This is the single most important missing regime: L-BFGS curvature estimates and line searches behave very differently under gradient noise, and it's where Adam/SGD normally win. Tests whether the steepest-descent anchor still guarantees robustness when `∇f` is noisy.

### 4. Structured parameters (attacks "structured parameters change the oracle ranking")
- **Small CNN on CIFAR-10 subset / MNIST** (a couple conv layers, <100K params). The conv/linear weight tensors are genuinely matrix-shaped, giving Shampoo its intended structure and testing your explicit hypothesis that "its dense inverse-root cost may amortize better."
- **Tiny Transformer block / char-level language model** (e.g. tiny-shakespeare, a 1–2 layer model). This is the modern-research domain par excellence; matrix-shaped attention/MLP blocks again favor structure-aware preconditioners and introduce realistic loss-landscape pathology cheaply.

### 5. Sparsity / regularization regimes (extends your orthant-region sparsity finding)
- **L1-regularized logistic regression** to make the orthant region's `0.0056` sparsity result a *primary* metric rather than an incidental observation, and to introduce a genuinely non-smooth (kinked) objective where the region matters.

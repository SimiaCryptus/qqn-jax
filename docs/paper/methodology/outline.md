# Methodology — Outline

## 1. Overview
- Goal: a fair, reproducible empirical comparison of QQN against
standard first- and second-order baselines (SGD, Adam, L-BFGS).
- Objective class: non-convex MLP training on image classification
(MNIST, Fashion-MNIST) across a battery of activation functions.
- Emphasis on *fairness invariants* so measured differences reflect the
optimizers, not the harness.

## 2. Experimental Design
### 2.1 Objective and Models
- Flat MLP (`FlatMLP`) with configurable depth/width and per-layer
activation.
- Cross-entropy loss with L2 regularization; single shared loss closure.
### 2.2 Datasets
- MNIST / Fashion-MNIST subsets; balanced sampling; fixed subset seed.
- Train/test split sizes as configuration knobs.
### 2.3 Optimizers Under Test
- QQN (quadratic path + oracle + line search).
- Baselines: SGD, Adam, Optax L-BFGS (zoom line search).

## 3. Fairness Invariants
1. Identical initialization (shared PRNG seed).
2. Shared termination criteria (target loss, gradient tolerance, time
budget).
3. Genuine eval accounting (forward/backward counts).
4. Identical loss / data / regularization across variants.
5. JIT-warmup excluded from timed regions.

## 4. QQN Configuration Space
- Orthogonal axes: oracle, line search, spline/path strategy, trust
region, probe feeding, step-size memory, parametric bound, partitioning,
temperature.
- Cross-product profile generation and naming convention.

## 5. Metrics
- Final / best loss, iterations, train/test accuracy.
- Wall time, ms/iter, trajectory AUC.
- Milestone tracking: iterations / time / evals to reach loss levels.
- Iterations/time/evals to target; reached flag.

## 6. Pareto Analysis
- Loss vs. wall-time frontier.
- Per-milestone (time, evals) frontiers.

## 7. Reproducibility Infrastructure
- `ExperimentConfig` + environment binding.
- Driver pipeline: config → runs → enrichment → reports.
- JSON artifacts, manifest generation, interactive viewer.
- Variant runner (`run_reports.js`) and activation sweeps.

## 8. Threats to Validity
- JIT-compilation cost attribution.
- L-BFGS eval-count estimation (adaptive workaround).
- Anomalous line-search behaviors as open questions.
# Fashion-MNIST / MNIST MLP Optimizer Comparison

A full-batch benchmark that pits **QQN** (Quadratic-path Quasi-Newton) against
three established baselines — **SGD**, **Adam**, and **Optax L-BFGS** — on a
configurable, *non-convex* multi-layer perceptron trained on (Fashion-)MNIST.

> Run it with:
>
> ```bash
> python examples/fashion_mnist_mlp_comparison.py
> ```

---

## 1. Overview

Unlike the linear softmax classifier in `mnist_comparison.py`, this script
trains a **two-or-more-layer fully-connected network with nonlinear hidden
activations**. The hidden nonlinearity makes the loss surface genuinely
non-convex — introducing saddle points, flat regions, and non-unique minima.
This is a much sterner test for the curvature-aware methods (QQN and L-BFGS).

The objective is framed as a **full-batch, deterministic** cross-entropy loss
(with optional L2 regularization). Keeping it full-batch makes the comparison
apples-to-apples for second-order methods: every optimizer sees the exact same
objective, the same initial parameters, and the same termination criteria.

### Why full-batch and non-convex?

A larger full-batch objective has a **richer, more anisotropic Hessian** —
precisely the regime where QQN's gradient + curvature-oracle blending along the
quadratic path is most competitive against L-BFGS. The deeper / wider the
network, the more ill-conditioned and compositional the curvature becomes,
widening the gap where second-order information pays off.

### Headline findings (from the design notes)

- QQN wins decisively on **iterations-to-target** (up to ~1.9x vs L-BFGS), and
  the speedup **widens monotonically as the target tightens**.
- The historical per-iteration cost penalty was *resolved* by moving to a
  richer, more anisotropic Hessian (larger batch + deeper network), which
  shrank the deep-oracle ms/it gap below the iteration advantage — converting
  the iteration win into a **wall-clock** win.
- The decisive enabler is the **deep L-BFGS oracle** itself (with a plain
  Armijo line search), *not* a cheaper line search.

### Hard-won negative lessons (documented as negative controls)

1. **Warm-started backtracking backfires** on this smooth surface — it *raises*
   the iteration count for little-to-no ms/it saving, a net wall-clock loss.
   The cheap-probe variants are retained only as documented negative controls
   with *tamed* (gentle) warm-starts.
2. **Spline (cubic-Hermite) variants diverge** to the chance solution on the
   `tanh,gelu,tanh` surface — the cubic model's stationary-point probes are
   untrustworthy near init. Spline variants are quarantined behind a
   gentle-Armijo guard and clearly marked as negative controls.

---

## 2. Model architecture

The network maps a flattened image vector through one or more hidden layers to
class logits:

```
x -> [W1, b1] -> act -> ... -> [Wk, bk] -> act -> [Wout, bout] -> logits
```

- Hidden layers apply the configured **activation function**.
- The **output layer is always linear** (produces logits).
- All parameters are stored in a **single flat vector** (laid out as
  `W_1, b_1, W_2, b_2, ..., W_L, b_L`) so they slot directly into both QQN and
  Optax, which operate on flat arrays.

Weight initialization is activation-aware:

- **He init** (`std = sqrt(2 / fan_in)`) for `relu` layers.
- **Glorot/Xavier-style init** (`std = sqrt(1 / fan_in)`) otherwise.

The default architecture is **width 256 × depth 3** hidden layers with mixed
`tanh,gelu` activations on a 10-class problem.

---

## 3. Configuration guide

All configuration is via **environment variables**. Defaults are chosen to
reproduce the headline experiment described in the script.

### 3.1 Dataset selection

| Variable  | Values                        | Default          | Description                          |
| --------- | ----------------------------- | ---------------- | ------------------------------------ |
| `DATASET` | `mnist`, `fashion_mnist`      | `fashion_mnist`  | Which corpus to train on.            |

```bash
DATASET=fashion_mnist python examples/fashion_mnist_mlp_comparison.py
```

> An unknown value falls back to `mnist` with a warning.

### 3.2 Dataset size

| Variable  | Type | Default | Description                                       |
| --------- | ---- | ------- | ------------------------------------------------- |
| `N_TRAIN` | int  | `25000` | Full-batch **training** subset size.              |
| `N_TEST`  | int  | `5000`  | Full-batch **test** subset size (for accuracy).   |

A larger full-batch objective has a richer, more anisotropic Hessian — the
regime where QQN's gradient + oracle blending is most competitive.

> **VRAM note:** the deep L-BFGS history materializes
> `f32[history, n_train, width]` JVP tensors, which can OOM a ~6.5 GiB GPU at
> very large batch sizes. The default 25k is chosen to keep evaluation
> dominant while staying VRAM-safe with the width-256 × depth-3 network.
> Lower `N_TRAIN` if you have less VRAM.

The subset is drawn as a **reproducible, class-balanced random sample** (seed
`0`) rather than the first-N examples, giving a better-conditioned and more
representative Hessian.

### 3.3 Network topology

Precedence (highest first):

1. `HIDDEN_SIZES` — explicit comma-separated widths.
2. `DEPTH` × `HIDDEN` — uniform-width network.
3. Default: width 256 × depth 3.

| Variable       | Type            | Default | Description                                                       |
| -------------- | --------------- | ------- | ----------------------------------------------------------------- |
| `HIDDEN_SIZES` | comma-int list  | (unset) | Explicit per-layer widths, e.g. `256,128,64`. Takes precedence.   |
| `HIDDEN`       | int             | `256`   | Width of each hidden layer (uniform-width mode).                  |
| `DEPTH`        | int             | `3`     | Number of hidden layers (uniform-width mode).                     |

Examples:

```bash
# 3-layer MLP with two hidden layers of width 128 and 64
HIDDEN_SIZES=128,64 python examples/fashion_mnist_mlp_comparison.py

# Uniform 4 hidden layers of width 128
DEPTH=4 HIDDEN=128 python examples/fashion_mnist_mlp_comparison.py

# Deep, tapering network
HIDDEN_SIZES=256,128,64 python examples/fashion_mnist_mlp_comparison.py
```

> Invalid `HIDDEN_SIZES` (non-positive or non-numeric) falls back to
> `DEPTH`/`HIDDEN`; invalid `HIDDEN`/`DEPTH` falls back to `[64]`.

### 3.4 Activation function(s)

| Variable     | Values (single or comma-list) | Default     | Description                       |
| ------------ | ----------------------------- | ----------- | --------------------------------- |
| `ACTIVATION` | see table below               | `tanh,gelu` | Hidden-layer activation(s).       |

Supported activations:

| Name        | Definition                              | Notes                                     |
| ----------- | --------------------------------------- | ----------------------------------------- |
| `relu`      | `max(0, x)`                             | Triggers He init.                         |
| `sigmoid`   | `1 / (1 + e^-x)`                        | Default fallback for unknown names.       |
| `sine`      | `sin(x)`                                | Periodic.                                 |
| `gaussian`  | `exp(-x^2)`                             | Localized, RBF-like bump.                 |
| `triangle`  | periodic triangle wave in `[-1, 1]`     | Piecewise-linear, periodic.               |
| `sawtooth`  | periodic ramp in `[-1, 1)`              | Periodic.                                 |
| `logabs`    | `sign(x) * ln(|x| + 1)`                 | Heavy-tailed, odd.                        |
| `tanh`      | `tanh(x)`                               | Bounded squashing.                        |
| `gelu`      | Gaussian Error Linear Unit              | Smooth ReLU-like.                         |
| `swish`     | `x * sigmoid(x)`                        | Smooth, non-monotonic (SiLU).             |
| `softplus`  | `ln(1 + e^x)`                           | Smooth ReLU approximation.                |
| `abs`       | `|x|`                                   | V-shaped, even.                           |
| `identity`  | `x`                                     | Linear (useful in mixes).                 |

**Single activation** (applied to every hidden layer):

```bash
ACTIVATION=relu python examples/fashion_mnist_mlp_comparison.py
```

**Mixed activations** — a comma-separated list assigns activations per hidden
layer. The list is **cycled** if shorter than the number of hidden layers (and
truncated if longer):

```bash
# layer 1: relu, layer 2: sine, layer 3: gaussian
ACTIVATION=relu,sine,gaussian python examples/fashion_mnist_mlp_comparison.py

# 4 hidden layers, activations cycle: tanh, gaussian, tanh, gaussian
ACTIVATION=tanh,gaussian DEPTH=4 python examples/fashion_mnist_mlp_comparison.py
```

> Unknown activation names fall back to `sigmoid` with a warning. The output
> layer is always linear regardless of `ACTIVATION`.

### 3.5 GPU / XLA tuning (set automatically)

The script sets these *before* importing JAX to avoid speculative
multi-GiB workspace allocations that can OOM small GPUs:

| Variable           | Default value         | Purpose                                              |
| ------------------ | --------------------- | ---------------------------------------------------- |
| `XLA_FLAGS`        | `--xla_gpu_autotune_level=0` | Disables the cuBLAS-Lt autotuner's parallel probing. |
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async`   | Falls back to host memory instead of hard OOM.       |

Both use `setdefault`, so you can override them in your environment if needed.

---

## 4. Data loading

The script attempts to load a real corpus, in order:

1. **TensorFlow / Keras** (`tensorflow.keras.datasets`).
2. **torchvision** (`torchvision.datasets`).
3. **Synthetic fallback** — Gaussian-blob "MNIST-like" data so the experiment
   always runs.

Install one of the backends to use real data:

```bash
# Option A — TensorFlow / Keras (ships both MNIST + Fashion-MNIST)
pip install tensorflow

# Option B — torchvision (ships both MNIST + Fashion-MNIST)
pip install torch torchvision
```

Images are flattened to shape `(N, 784)` and scaled to `float32` in `[0, 1]`.

---

## 5. Termination criteria (shared by every optimizer)

All optimizers stop under the **same** conditions, so the comparison is fair:

| Key           | Value     | Meaning                                                |
| ------------- | --------- | ------------------------------------------------------ |
| `f_target`    | `2.0e-2`  | Headline target loss. First crossing is the win point. |
| `gtol`        | `1.0e-8`  | Gradient-norm convergence tolerance.                   |
| `time_budget` | `150.0` s | Wall-clock cap (so deep stacks aren't truncated early).|
| `milestones`  | `1e0, 5e-1, 2e-1, 1e-1` | Loss levels for the convergence-rate profile. |

The headline `f_target` is deliberately pushed tighter (`2e-2`) into the regime
where QQN's curvature blend dominates hardest, while staying reachable within
the time budget.

### Target-sensitivity profile

To address selection-bias concerns, the speedup is reported as a *curve* across
multiple targets rather than a single point:

```
target_profile = (2.0e-1, 1.0e-1, 6.0e-2, 4.0e-2, 2.0e-2)
```

---

## 6. Optimizer variants

The script runs a large suite of QQN configurations plus the three baselines.
Highlights:

### Baselines

| Name     | Configuration                                  |
| -------- | ---------------------------------------------- |
| `SGD`    | `optax.sgd(learning_rate=0.05)`                |
| `Adam`   | `optax.adam(learning_rate=0.01)`               |
| `L-BFGS` | `optax.lbfgs()` with zoom line search          |

### Headline QQN winners

| Name        | Configuration                                              | Role                                  |
| ----------- | ---------------------------------------------------------- | ------------------------------------- |
| `QQN`       | L-BFGS oracle, Armijo line search                          | Baseline QQN.                         |
| `QQN-L80`   | L-BFGS history = 80                                         | Empirical sweet-spot / Pareto winner. |
| `QQN-L120`  | L-BFGS history = 120                                        | Co-headline; often fastest wall-clock.|
| `QQN-Lean`  | L-BFGS history = 80, bare Armijo only                      | Lean expression of the winning recipe.|
| `QQN-Champ` | L-BFGS history = 120, bare Armijo only                     | Minimal pure-oracle wall-clock contender. |

### Curvature-memory sweep (the decisive lever)

`QQN-L20`, `QQN-L50`, `QQN-L80`, `QQN-L120`, `QQN-L160` sweep the L-BFGS history
size to confirm whether the deep-memory lever is still monotone and unsaturated.

### Alternative oracles

| Name        | Oracle                                          |
| ----------- | ----------------------------------------------- |
| `QQN-Mom`   | `MomentumOracle(beta=0.9)`                      |
| `QQN-Sec`   | `SecantOracle()` (matrix-free)                  |
| `QQN-And`   | `AndersonOracle(window=5)`                      |
| `QQN-L50And`| `Fallback([LBFGS(50), Anderson(5)])`            |
| `QQN-L80And`| `Fallback([LBFGS(80), Anderson(5)])`            |

### Region-constrained variants

| Name      | Region                                            |
| --------- | ------------------------------------------------- |
| `QQN-TR`  | `TrustRegion(radius=1.0, adaptive=True)`          |
| `QQN-Box` | `BoxRegion(lo=-2.0, hi=2.0)`                       |
| `QQN-Fast`| L-BFGS(120) + `TrustRegion(radius=2.0, fixed)`    |
| `QQN-Max` | `Fallback([LBFGS(80), Anderson(5)])` + fixed TR   |

### Probe-feeding variants

`QQN-L50P` (and historically `QQN-MaxP`) set `feed_probes_to_oracle=True`,
forwarding every gradient evaluated *during the line search* into the oracle's
curvature memory — enriching the L-BFGS Hessian approximation essentially for
free, since those gradients were already computed.

> **Quarantined:** on this surface probe-feeding into a deep history *stalls*
> despite descent-gated admission; `QQN-L50P` is retained as a documented
> negative control, **not** as a "free boost".

### Negative controls (documented failures)

| Name         | Why it's a negative control                                         |
| ------------ | ------------------------------------------------------------------- |
| `QQN-S`, `QQN-BT-S`, `QQN-Mom-S`, `QQN-Smooth`, `QQN-MaxS` | **Spline** variants diverge to chance on `tanh,gelu,tanh`. |
| `QQN-Cheap`, `QQN-L80-BT` | **Warm-started backtracking** raises iterations for no ms/it saving (tamed warm-starts retained). |
| `QQN-L50P`   | **Probe-feeding** pollutes the deep history and stalls.             |

---

## 7. Output / reports

The script prints a rich set of tables and profiles:

1. **Summary table** — final loss, iterations, train/test accuracy, wall-time,
   ms/it, iterations-to-target, time-to-target, vs-LBFGS speedup, estimated
   evals, and trajectory AUC.
2. **Pareto frontier** — non-dominated `(loss, wall-time)` variants.
3. **Iteration-efficiency leaderboard** — fewest iterations to target.
4. **Cost-aware leaderboard** — estimated function/grad **evals** to target
   (addresses the metric caveat that iterations are not cost-neutral; QQN's
   line-search probes issue several evaluations per accepted iteration).
5. **Target-sensitivity profile** — iterations to each target in
   `target_profile`, plus **vs-LBFGS speedup stability** for the key deep-memory
   stacks.
6. **Convergence-rate profiles** — first iteration / wall-time / eval count to
   reach each milestone, plus an **inter-milestone cost breakdown**
   (Δtime / Δevals between consecutive milestones).
7. **Stall report** — non-converging variants with a diagnosed cause
   (time-budget exhausted / plateau / slow).
8. **Loss trajectory** — compact ASCII view at log10 scale.

### Plots (optional)

If `matplotlib` is installed, two timestamped PNGs are saved under `results/`:

- `<dataset>_mlp_comparison_vs_iter_<timestamp>.png` — loss vs. iteration.
- `<dataset>_mlp_comparison_vs_time_<timestamp>.png` — loss vs. wall-clock time
  (captures per-iteration cost differences).

Baselines (`SGD`, `Adam`, `L-BFGS`) are drawn with dashed lines for contrast.

---

## 8. Cost-aware metrics — the evaluation-counting caveat

Iterations are **not cost-neutral**: QQN's line-search iterations issue several
function/gradient evaluations each, so "iterations-to-target" understates the
true work done. The script attaches a **conservative analytic estimate** of
evaluations-per-iteration to every method and reports
**evaluations-to-target** alongside iterations:

| Method          | Estimated evals / accepted iteration                         |
| --------------- | ------------------------------------------------------------ |
| SGD / Adam      | `1.0` (1 value + 1 grad)                                      |
| L-BFGS          | `~3.0` (zoom line-search probes)                             |
| QQN (armijo/BT) | `1 + min(max_iter, 4)` probes                                |
| QQN (Wolfe)     | `1 + min(max_iter, 6)` probes                                |
| QQN (Hager-Zhang)| `1 + min(max_iter, 5)` probes                               |
| QQN (fixed)     | `1 + 1`                                                       |
| + spline        | `+2.0` (stationary-point probes)                             |
| + probe-feeding | `+0` (reuses gradients already computed)                     |

These estimates are explicitly approximate but make cross-method cost
comparisons far fairer than raw iteration counts.

---

## 9. Reproducibility

- Initial parameters use `jax.random.PRNGKey(42)`, so **every optimizer starts
  from identical weights**.
- The class-balanced data subset uses `np.random.default_rng(0)`.
- The synthetic fallback uses seed `0`.

---

## 10. Quick reference — common invocations

```bash
# Default headline experiment (Fashion-MNIST, 256x3, tanh,gelu)
python examples/fashion_mnist_mlp_comparison.py

# MNIST instead of Fashion-MNIST
DATASET=mnist python examples/fashion_mnist_mlp_comparison.py

# Deeper, narrower ReLU network
DEPTH=5 HIDDEN=128 ACTIVATION=relu python examples/fashion_mnist_mlp_comparison.py

# Explicit tapering topology with mixed activations
HIDDEN_SIZES=256,128,64 ACTIVATION=tanh,gelu,gaussian \
    python examples/fashion_mnist_mlp_comparison.py

# Smaller problem for a low-VRAM GPU
N_TRAIN=8000 N_TEST=2000 HIDDEN=128 DEPTH=2 \
    python examples/fashion_mnist_mlp_comparison.py
```
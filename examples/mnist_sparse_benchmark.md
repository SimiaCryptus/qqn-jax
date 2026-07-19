# Sparse & Precision-Optimized MNIST Benchmark

A lightweight benchmark that trains a small multi-layer perceptron (MLP)
on MNIST (or Fashion-MNIST) using the **QQN** (Quadratic-Quasi-Newton)
optimizer, with an emphasis on inducing **weight sparsity** (via the
`OrthantRegion` / OWL-QN-style projection and L1 penalties) and
**precision optimization** (via quantization-delta penalties and the
geometric `QuantizationRegion`).

---

## 1. Overview

### Purpose

This experiment demonstrates and compares how different **region**
strategies and **regularizers** affect four key outcomes when training a
neural network with QQN:

1. **Final training loss** — convergence quality.
2. **Test accuracy** — generalization.
3. **Weight sparsity** — fraction of near-zero parameters.
4. **Quantization error** — mean rounding delta to a `bits`-level grid
   (lower means the model is closer to being losslessly quantizable).

The benchmark is intentionally small (a compact MLP on a subset of
MNIST) so it runs quickly on CPU while still exercising the full region
machinery through `jit`/`vmap`-compatible code paths.

### Two-phase structure

The driver runs in two phases:

1. **Phase 1 — raw (cold-start) training** of a set of *base* models
   from random initialization. Each base run produces a trained
   parameter vector.
2. **Phase 2 — quantization polishing**: every *polish* variant is
   applied as a short warm-started run on top of *every* base model
   (a cross-product). Polishing only nudges an already-trained model
   onto the quantization grid, so it uses a much smaller iteration
   budget (`POLISH_MAXITER`, default `MAXITER // 10`).

### What is `OrthantRegion`?

The `OrthantRegion` implements an **OWL-QN-style** orthant projection.
During each update it:

- Constrains the step to remain within the orthant defined by the
  current weights' signs.
- Zeros out coordinates that would otherwise cross zero.

This encourages **L1-like sparsity**, driving many weights to exactly
zero — useful for model compression and interpretability.

### What is `QuantizationRegion`?

The `QuantizationRegion` confines each step to the **rounding cell** of
each weight on a `bits`-level symmetric grid over `[lo, hi]`. This is a
*geometric* form of quantization-aware training: rather than penalizing
off-grid weights, it constrains the optimizer so weights stay near
representable grid values.

### Compared Configurations

**Base models (Phase 1):**

| # | Name                            | Region          | Regularizer |
|---|---------------------------------|-----------------|-------------|
| 1 | `baseline (dense)`              | `None`          | `None`      |
| 2 | `orthant (sparse)`              | `OrthantRegion` | `None`      |
| 3 | `l1-orthant-penalty (sparse)`   | `OrthantRegion` | `l1_reg`    |
| 4 | `l1-penalty (sparse)`           | `None`          | `l1_reg`    |

**Polish variants (Phase 2, applied to every base model):**

| Suffix                          | Region                | Regularizer |
|---------------------------------|-----------------------|-------------|
| `quant-penalty (prec)`          | `None`                | `quant_reg` |
| `quant-region (prec)`           | `QuantizationRegion`  | `None`      |
| `quant-region-penalty (prec)`   | `QuantizationRegion`  | `quant_reg` |

The line search defaults to `strong_wolfe` throughout (override via the
`LINE_SEARCH` env var).

---

## 2. Running the Benchmark

```bash
python -m examples.mnist_sparse_benchmark
```

Or via the report runner (with timestamped logs):

```bash
node run_reports.js sparse_default
node run_reports.js --report mnist_sparse_benchmark   # all sparse variants
```

### Requirements

- `jax`, `jaxlib`
- `numpy`
- **One** of the following for real data (optional — falls back to
  synthetic data otherwise):
    - `tensorflow` (uses `tensorflow.keras.datasets`)
    - `tensorflow-datasets`
    - `torchvision`
- `matplotlib` (optional — for the convergence plot)

### Output

The script prints per-configuration metrics, a final summary table, a
**Pareto frontier** (test accuracy vs. sparsity), and the **best
precision config** (lowest quantization error):

```
==========================================================================================
config                                            iters      loss  test_acc  sparsity  quant_err   time(s)
------------------------------------------------------------------------------------------
baseline (dense)                                    ...    0.XXXX    0.XXXX     0.XXX     0.XXXX     XX.XX
...
==========================================================================================
```

It also saves a log-scale convergence plot to `convergence.png`.

---

## 3. Configuration Guide

Every major knob is overridable via an **environment variable** so the
benchmark can be re-tuned without editing the source.

### 3.1 Data Configuration

| Variable   | Default   | Description                                        |
|------------|-----------|----------------------------------------------------|
| `DATASET`  | `mnist`   | `mnist` or `fashion_mnist`.                        |
| `N_TRAIN`  | `10000`   | Number of training samples to subsample.           |
| `N_TEST`   | `5000`    | Number of test samples to subsample.               |
| `SEED`     | `0`       | RNG seed for reproducible subsampling.             |

**Data source resolution order:**
1. `tensorflow.keras.datasets`
2. `tensorflow_datasets`
3. `torchvision`
4. **Synthetic fallback** (random data) — ensures the example always runs.

Images are flattened to `(n, 784)` `float32` in `[0, 1]`; labels are
integers.

### 3.2 Model Configuration

The MLP architecture is controlled by environment variables, resolved in
the following precedence order:

1. `HIDDEN_SIZES` — comma-separated hidden-layer widths, e.g. `128,64`.
2. `DEPTH` × `HIDDEN` — `DEPTH` hidden layers each of width `HIDDEN`.
3. Default: `[64, 64]`.

The input (`784`) and output (`n_classes`, inferred from the labels)
dimensions are added automatically, giving e.g. `784 -> 64 -> 64 -> 10`.

| Variable       | Default | Description                                    |
|----------------|---------|------------------------------------------------|
| `HIDDEN_SIZES` | —       | Comma-separated hidden widths (highest prec.). |
| `HIDDEN`       | `64`    | Width of each hidden layer.                    |
| `DEPTH`        | `2`     | Number of hidden layers.                       |
| `ACTIVATION`   | `tanh`  | Hidden activation(s); see below.               |

**Activation details:**
- Available: `relu`, `tanh`, `sigmoid`, `sine`, `gaussian`, `gelu`,
  `swish`, `softplus`, `abs`, `identity`.
- A comma-separated list mixes activations per hidden layer (cycled if
  shorter than the number of hidden layers), e.g. `ACTIVATION=tanh,gelu`.
- Output is always linear logits (softmax applied inside the loss).
- Weight init: He init for `relu` layers (`std = sqrt(2 / fan_in)`),
  Glorot/Xavier-style otherwise (`std = 1 / sqrt(fan_in)`); biases zero.

### 3.3 Loss & Regularizer Configuration

The base loss is softmax cross-entropy (NLL) plus a squared-L2 weight
decay. Additional sparsity/precision pressure is supplied per-config via
a `params -> scalar` regularizer that is *added* to the loss.

| Variable      | Default | Description                                         |
|---------------|---------|-----------------------------------------------------|
| `L2`          | `1e-4`  | Squared-L2 weight-decay coefficient (always on).    |
| `L1_SCALE`    | `1e-5`  | L1 sparsity-penalty scale (`l1_reg`).               |
| `QUANT_SCALE` | `1e-4`  | Quantization-delta penalty scale (`quant_reg`).     |
| `QBITS`       | `4`     | Quantization grid bit-depth.                        |

The quantization grid is a symmetric `QBITS`-level grid over `[-1, 1]`,
used by both the `quant_reg` penalty and the `QuantizationRegion`.

### 3.4 Solver Configuration (`run_config` → `QQN`)

| Variable         | Default        | Description                                     |
|------------------|----------------|-------------------------------------------------|
| `MAXITER`        | `50000`        | Raw-training iteration budget.                  |
| `POLISH_MAXITER` | `MAXITER // 10`| Polishing iteration budget.                     |
| `HISTORY_SIZE`   | `10`           | L-BFGS memory (number of stored (s, y) pairs).  |
| `LINE_SEARCH`    | `strong_wolfe` | Line search type.                               |
| `tol`            | `1e-6` (fixed) | Convergence tolerance.                          |

> **Flattening note:** The L-BFGS oracle (`qqn_jax.lbfgs`) operates on a
> **flat** 1-D parameter vector. `run_config` uses
> `jax.flatten_util.ravel_pytree` to flatten the MLP pytree and an
> `unravel` closure to reconstruct structured params inside the loss.
> For polishing runs, the base model's flat vector and `unravel` are
> passed via `init_flat` / `unravel_fn` so structure matches the
> warm-start.

### 3.5 Sparsity & Quantization Metrics

- **`sparsity(params, threshold=1e-6)`** — fraction of weight entries
  (biases excluded) with magnitude below `threshold`.
- **`mean_quant_error(params, bits=4, lo=-1, hi=1)`** — mean absolute
  rounding delta of weights to the grid. A precision-optimized network
  drives this toward zero.

---

## 4. Extending the Benchmark

### Add a new base configuration

Append a tuple `(name, region, line_search, regularizer)` to the
`base_configs` list in `main()`:

```python
base_configs = [
    # ... existing entries ...
    ("orthant strong (l1=0.5)", OrthantRegion(), "strong_wolfe", l1_reg),
]
```

### Add a new polish variant

Append a tuple `(suffix, region, line_search, regularizer)` to
`polish_configs`. It will automatically be applied to every base model:

```python
polish_configs = [
    # ... existing entries ...
    ("quant8-region (prec)",
     QuantizationRegion(bits=8, lo=QLO, hi=QHI), "strong_wolfe", None),
]
```

Each entry is automatically run, recorded, summarized, and plotted.

### Change the convergence plot

`plot_convergence(results, fname="convergence.png")` produces a
log-scale loss-vs-evaluation curve. It degrades gracefully (prints a
message and skips) if `matplotlib` is unavailable. Loss values are
captured per-evaluation via a `jax.debug.callback` host hook, which is
`jit`-compatible.

---

## 5. Interpreting Results

| Metric       | Goal      | Notes                                                       |
|--------------|-----------|-------------------------------------------------------------|
| `iters`      | —         | Iterations until convergence or `maxiter`.                  |
| `loss`       | lower     | Final training loss (NLL + L2 + regularizer).               |
| `test_acc`   | higher    | Held-out accuracy; watch for over-sparsification.           |
| `sparsity`   | higher*   | Fraction of zeroed weights. *Balance against `test_acc`.    |
| `quant_err`  | lower     | Mean rounding delta to the grid (precision quality).        |
| `time_s`     | lower     | Wall-clock (includes JIT compile + `block_until_ready`).    |

The driver surfaces two derived summaries:

- **Pareto frontier** — non-dominated configs trading off test accuracy
  against sparsity (higher of both is better).
- **Best precision config** — the `(prec)` config with the lowest
  `quant_err`.

A successful sparse configuration achieves **high sparsity** while
keeping **test accuracy** close to the dense baseline. A successful
precision configuration drives **`quant_err` toward zero** so the model
can be quantized to `QBITS` bits with minimal loss.

---

## 6. References

- `docs/conclusions.md` — adaptive trust-region stall analysis.
- `results.md` — spline gating vs. adaptive trust-region interaction.
- OWL-QN: Andrew & Gao, *"Scalable Training of L1-Regularized
  Log-Linear Models"* (ICML 2007).
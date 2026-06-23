# Sparse MNIST Classification Benchmark

    A lightweight benchmark that trains a small multi-layer perceptron (MLP)
    on MNIST using the **QQN** (Quadratic-Quasi-Newton) optimizer, with an
    emphasis on inducing **weight sparsity** via the `OrthantRegion`
    (OWL-QN style) projection.

    ---

    ## 1. Overview

    ### Purpose

    This experiment demonstrates and compares how different **region**
    strategies and **line-search** methods affect three key outcomes when
    training a neural network with QQN:

    1. **Final training loss** — convergence quality.
    2. **Test accuracy** — generalization.
    3. **Weight sparsity** — fraction of near-zero parameters.

    The benchmark is intentionally small (a compact MLP on a subset of
    MNIST) so it runs quickly on CPU while still exercising the full region
    machinery through `jit`/`vmap`-compatible code paths.

    ### What is `OrthantRegion`?

    The `OrthantRegion` implements an **OWL-QN-style** orthant projection.
    During each update it:

    - Constrains the step to remain within the orthant defined by the
      current weights' signs.
    - Zeros out coordinates that would otherwise cross zero.

    This encourages **L1-like sparsity**, driving many weights to exactly
    zero — useful for model compression and interpretability.

    ### Compared Configurations

    | # | Name                        | Region                                          | Line Search    |
    |---|-----------------------------|-------------------------------------------------|----------------|
    | 1 | `baseline (dense)`          | `None`                                          | `strong_wolfe` |
    | 2 | `baseline (spline)`         | `None`                                          | `spline`       |
    | 3 | `orthant (sparse)`          | `OrthantRegion(l1=1e-1)`                        | `strong_wolfe` |
    | 4 | `orthant (spline)`          | `OrthantRegion(l1=1e-1)`                        | `spline`       |
    | 5 | `orthant + trust`           | `Sequential([Orthant, TrustRegion(fixed)])`     | `strong_wolfe` |
    | 6 | `orthant + trust (spline)`  | `Sequential([Orthant, TrustRegion(adaptive)])`  | `spline`       |

    > **Note on trust-region adaptivity:** Per `docs/conclusions.md`, the
    > *adaptive* trust-region stalls on this class of smooth problem when
    > paired with `strong_wolfe`. Configuration 5 therefore uses a **fixed**
    > radius as the robust safeguard. Configuration 6 re-enables adaptivity
    > because the spline's monotone gating neutralizes the stall (see
    > `results.md`).

    ---

    ## 2. Running the Benchmark

    ```bash
    python -m examples.mnist_sparse_benchmark
    ```

    ### Requirements

    - `jax`, `jaxlib`
    - `numpy`
    - **One** of the following for real MNIST data (optional — falls back to
      synthetic data otherwise):
        - `tensorflow` (uses `tensorflow.keras.datasets`)
        - `tensorflow-datasets`
        - `torchvision`
    - `matplotlib` (optional — for the convergence plot)

    ### Output

    The script prints per-configuration metrics and a final summary table:

    ```
    ==============================================================================
    config                iters      loss  test_acc  sparsity   time(s)
    ------------------------------------------------------------------------------
    baseline (dense)        ...    0.XXXX    0.XXXX     0.XXX     XX.XX
    ...
    ==============================================================================
    ```

    It also saves a log-scale convergence plot to `convergence.png`.

    ---

    ## 3. Configuration Guide

    All tunable parameters live in two places: the `main()` function (global
    experiment setup) and `run_config()` (per-run solver setup).

    ### 3.1 Data Configuration (`load_mnist`)

    | Parameter | Default (in `main`) | Description                                      |
    |-----------|---------------------|--------------------------------------------------|
    | `n_train` | `10000`             | Number of training samples to subsample.         |
    | `n_test`  | `5000`              | Number of test samples to subsample.             |
    | `seed`    | `0`                 | RNG seed for reproducible subsampling.           |

    **Data source resolution order:**
    1. `tensorflow.keras.datasets.mnist`
    2. `tensorflow_datasets`
    3. `torchvision`
    4. **Synthetic fallback** (random data) — ensures the example always runs.

    Images are flattened to `(n, 784)` `float32` in `[0, 1]`; labels are
    integers.

    ### 3.2 Model Configuration

    The MLP architecture is controlled by the `sizes` list in `main()`:

    ```python
    sizes = [784, 64, 64, 10]
    ```

    - **First element** (`784`): input dimension (fixed by flattened MNIST).
    - **Middle elements**: hidden layer widths (here, two `64`-unit layers).
    - **Last element** (`10`): output classes (fixed by MNIST).

    **Architecture details:**
    - Hidden activations: `tanh` (see `mlp_forward`).
    - Output: linear logits (softmax applied inside the loss).
    - Weight init: scaled Gaussian, `scale = 1 / sqrt(n_in)`; biases zero.

    To experiment with a deeper/wider network, edit `sizes`, e.g.:

    ```python
    sizes = [784, 128, 128, 64, 10]   # deeper, wider
    ```

    ### 3.3 Loss Configuration (`cross_entropy_loss`)

    | Parameter | Default | Description                                      |
    |-----------|---------|--------------------------------------------------|
    | `l2`      | `1e-4`  | L2 regularization coefficient on all weights.    |

    The loss is softmax cross-entropy (negative log-likelihood) plus the L2
    penalty. The sparsity-inducing pressure comes from `OrthantRegion`, not
    from this L2 term.

    ### 3.4 Solver Configuration (`run_config` → `QQN`)

    | Parameter      | Default        | Description                                            |
    |----------------|----------------|--------------------------------------------------------|
    | `maxiter`      | `5000`         | Maximum optimizer iterations.                          |
    | `tol`          | `1e-6`         | Convergence tolerance.                                 |
    | `history_size` | `10`           | L-BFGS memory (number of stored (s, y) pairs).         |
    | `line_search`  | varies         | `"strong_wolfe"` or `"spline"`.                        |
    | `region`       | varies         | `None`, `OrthantRegion`, or `Sequential([...])`.       |

    > **Flattening note:** The L-BFGS oracle (`qqn_jax.lbfgs`) operates on a
    > **flat** 1-D parameter vector. `run_config` uses
    > `jax.flatten_util.ravel_pytree` to flatten the MLP pytree and an
    > `unravel` closure to reconstruct structured params inside the loss.

    ### 3.5 Region Configuration

    #### `OrthantRegion`

    ```python
    OrthantRegion(l1=1e-1)
    ```

    - `l1`: L1 strength controlling sparsity pressure. **Higher values →
      more zeros** (more aggressive sparsification), at potential cost to
      accuracy. Tune in the `1e-2` to `1e0` range.

    #### `TrustRegion`

    ```python
    TrustRegion(radius=1.0, adaptive=False)   # fixed-radius safeguard
    TrustRegion(radius=1.0)                    # adaptive (default)
    ```

    - `radius`: trust-region radius (step-size cap).
    - `adaptive`: whether the radius adjusts dynamically. **Use `False`
      with `strong_wolfe`** on smooth problems to avoid stalling; adaptive
      is safe when paired with the `spline` line search.

    #### `Sequential`

    Composes regions in order — each region's projection is applied in
    sequence:

    ```python
    Sequential([
        OrthantRegion(l1=1e-1),               # sparsity first
        TrustRegion(radius=1.0, adaptive=False),  # then step control
    ])
    ```

    ### 3.6 Line-Search Configuration

    | Method          | When to use                                                            |
    |-----------------|------------------------------------------------------------------------|
    | `strong_wolfe`  | Standard, robust choice. Pair with **fixed** trust radius.             |
    | `spline`        | Monotone-gated refinement. Neutralizes adaptive trust-region stalls.   |

    ### 3.7 Sparsity Metric (`sparsity`)

    | Parameter   | Default | Description                                          |
    |-------------|---------|------------------------------------------------------|
    | `threshold` | `1e-6`  | Magnitude below which a weight counts as "zero".     |

    Returns the fraction of weight entries (biases excluded) with magnitude
    below `threshold`.

    ---

    ## 4. Extending the Benchmark

    ### Add a new configuration

    Append a tuple `(name, region, line_search)` to the `configs` list in
    `main()`:

    ```python
    configs = [
        # ... existing entries ...
        ("orthant strong (l1=0.5)", OrthantRegion(l1=5e-1), "strong_wolfe"),
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
    | `loss`       | lower     | Final training loss (NLL + L2).                             |
    | `test_acc`   | higher    | Held-out accuracy; watch for over-sparsification.           |
    | `sparsity`   | higher*   | Fraction of zeroed weights. *Balance against `test_acc`.    |
    | `time_s`     | lower     | Wall-clock (includes JIT compile + `block_until_ready`).    |

    A successful sparse configuration achieves **high sparsity** while
    keeping **test accuracy** close to the dense baseline. The orthant
    configurations should show markedly higher `sparsity` than the dense
    baselines, ideally with minimal accuracy degradation.

    ---

    ## 6. References

    - `docs/conclusions.md` — adaptive trust-region stall analysis.
    - `results.md` — spline gating vs. adaptive trust-region interaction.
    - OWL-QN: Andrew & Gao, *"Scalable Training of L1-Regularized
      Log-Linear Models"* (ICML 2007).
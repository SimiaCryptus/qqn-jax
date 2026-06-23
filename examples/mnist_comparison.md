# MNIST Optimizer Comparison Experiment

  A validation experiment that benchmarks **QQN** (Quadratic Quasi-Newton) and
  its many configurable variants against three standard baselines — **SGD**,
  **Adam**, and **Optax's L-BFGS** — on a softmax (multinomial logistic
  regression) classifier trained on MNIST.

  ---

  ## 1. Overview

  ### Purpose

  This experiment provides an *apples-to-apples* convergence comparison between
  QQN and common first-/second-order optimizers. The optimization is framed as
  a **full-batch deterministic** problem so that second-order methods (QQN and
  L-BFGS), which assume a smooth deterministic objective, are evaluated fairly
  against the stochastic baselines.

  ### Model

  - **Task:** 10-class classification on MNIST digits.
  - **Model:** Softmax / multinomial logistic regression.
    - Parameters are stored as a single flat vector: a weight matrix
      `W` of shape `(dim × n_classes)` flattened, followed by a bias `b` of
      shape `(n_classes,)`.
    - `dim = 784` for real MNIST (28×28 flattened images).
  - **Loss:** Full-batch cross-entropy with L2 regularization
    (`l2 = 1e-4` by default).

  ### Data Loading

  The script attempts to load real MNIST data, with graceful fallbacks:

  1. **TensorFlow / Keras** — `tensorflow.keras.datasets.mnist`.
  2. **torchvision** — `torchvision.datasets.MNIST` (downloads to
     `./_mnist_data`).
  3. **Synthetic fallback** — if neither library is installed, a synthetic
     Gaussian-blob "MNIST-like" dataset is generated so the experiment always
     runs.

  Images are flattened to shape `(N, 784)` and normalized to `float32` in
  `[0, 1]`.

  ### What It Measures

  For each optimizer the experiment records:

  | Metric            | Description                                             |
  |-------------------|---------------------------------------------------------|
  | `final_loss`      | Loss at the last recorded iteration.                    |
  | `best_loss`       | Minimum loss observed across the trajectory.            |
  | `iters`           | Number of iterations performed.                         |
  | `train_acc`       | Training-set accuracy of final parameters.              |
  | `test_acc`        | Test-set accuracy of final parameters.                  |
  | `sparsity`        | Fraction of (near-)zero weights (illuminating for orthant region). |
  | `wall`            | Total wall-clock time (seconds).                        |
  | `ms_per_iter`     | Mean wall-clock cost per accepted iteration.            |
  | `iters_to_target` | First iteration reaching the shared loss/gradient target. |
  | `time_to_target`  | Wall-clock time at which the target was first reached.  |
  | `milestone_hits`  | First iteration/time crossing each intermediate loss milestone. |
  | `traj_auc`        | Trajectory AUC: trapezoid integral of `log10(loss)` over the normalized iteration axis (lower = faster overall descent). |

  ---

  ## 2. Running the Experiment

  ```bash
  python examples/mnist_comparison.py
  ```

  ### Dependencies

  - `jax`, `jax.numpy`
  - `numpy`
  - `optax`
  - `qqn_jax` (the library under test)
  - *(optional)* `tensorflow` **or** `torchvision` for real MNIST data
  - *(optional)* `matplotlib` for convergence plots

  If `matplotlib` is available, two plots are saved into a `results/`
  directory (created automatically):

  - `mnist_comparison_<timestamp>.png` — loss vs. iteration (log-scale).
  - `mnist_comparison_time_<timestamp>.png` — loss vs. wall-clock time.

  ---

  ## 3. Configuration Guide

  All top-level configuration lives in `main()`.

  ### 3.1 Problem Configuration

  | Variable    | Default | Meaning                                        |
  |-------------|---------|------------------------------------------------|
  | `n_classes` | `10`    | Number of digit classes to keep.               |
  | `n_train`   | `5000`  | Training subset size.                           |
  | `n_test`    | `1000`  | Test subset size.                               |
  | `maxiter`   | `500`   | Maximum iterations per optimizer.               |

  ### 3.2 Shared Termination Bounds (`stop`)

  These bounds are applied **uniformly to every optimizer** to keep the
  comparison fair — each method races to the same loss threshold under the same
  time limit and the same stationarity tolerance.

  ```python
  stop = {
      "f_target": 1.1e-1,    # stop once full-batch loss <= this value
      "gtol": 1.0e-4,        # stop once ||grad|| <= this value (stationarity)
      "time_budget": 15.0,   # hard wall-clock cap (seconds) per optimizer
      "milestones": (5.0e-1, 2.0e-1, 1.5e-1, 1.2e-1),  # convergence-rate probes
  }
  ```

  | Key           | Type    | Meaning                                                            |
  |---------------|---------|--------------------------------------------------------------------|
  | `f_target`    | float   | Loss threshold; reaching it counts as "converged".                 |
  | `gtol`        | float   | Gradient-norm tolerance for stationarity-based termination.        |
  | `time_budget` | float   | Per-optimizer wall-clock cap (prevents runaway methods).           |
  | `milestones`  | tuple   | Descending loss thresholds for the convergence-rate profile.       |

  **Notes on choosing these values:**

  - `f_target` should be *reachable but demanding* so the `->target` /
    `t->tgt` columns become informative. A value below every method's reach
    leaves those columns empty.
  - The `milestones` measure *convergence rate* (early- vs. late-phase descent)
    rather than just the final target. Recording the iteration/time at which
    each method first crosses each threshold is far more discriminating than a
    single time-to-target.
  - `time_budget` caps expensive methods (e.g., a dense Shampoo refresh can
    exhaust the budget; a *blocked* Shampoo plus a modest budget keeps the
    comparison meaningful).

  ### 3.3 Loss Configuration

  ```python
  loss_fn = make_loss(X_train, y_train, dim, n_classes, l2=1e-4)
  ```

  - `l2` — L2 regularization strength (default `1e-4`).

  ### 3.4 Shared Initialization

  ```python
  params0 = init_params(dim, n_classes, jax.random.PRNGKey(42))
  ```

  Every optimizer starts from **identical** parameters (fixed PRNG seed `42`),
  so differences are attributable solely to the optimizer.

  ---

  ## 4. QQN Configuration

  QQN's behavior is controlled via `_run_qqn_configured()`, which exposes
  QQN's swappable components:

  | Component          | Parameter             | Role                                          |
  |--------------------|-----------------------|-----------------------------------------------|
  | **Oracle**         | `oracle`              | Curvature source / descent direction.         |
  | **Line search**    | `line_search`         | Step-size selection strategy.                 |
  | **Line search opts** | `line_search_options` | Per-search tuning (`init_step`, `shrink`, `max_iter`). |
  | **Region**         | `region`              | Projective constraint on the step.            |
  | **Spline**         | `spline`              | Cubic Hermite refinement of the search path.  |

  ### 4.1 Oracles (`qqn_jax.oracles`)

  | Oracle                            | Description                                                        |
  |-----------------------------------|--------------------------------------------------------------------|
  | `LBFGSOracle(history_size=k)`     | L-BFGS two-loop recursion with `k` stored curvature pairs.         |
  | `MomentumOracle(beta=b)`          | First-order momentum accelerator with damping `b`.                 |
  | `ShampooOracle(block_size, update_freq)` | Kronecker-factored structure-aware preconditioner (blocked for cost). |
  | `SecantOracle()`                  | Matrix-free Barzilai–Borwein single-step secant curvature (O(n) memory). |
  | `AndersonOracle(window=m, beta=...)` | Anderson acceleration over recent residual differences (O(m²n)/step). |
  | `Fallback([primary, backup, ...])` | Use the first oracle that yields a valid direction; cascade otherwise. |

  **History-size sweep:** L-BFGS history (`5 → 10 → 20 → 50 → 100`) trades
  memory/curvature richness against per-step cost; deeper memory generally
  converges faster up to diminishing returns.

  **Momentum-beta sweep:** `0.01 → 0.1 → 0.5 → 0.9` — lower `beta` collapses
  toward steepest descent.

  ### 4.2 Line Searches (`line_search`)

  | Value            | Description                                              |
  |------------------|----------------------------------------------------------|
  | `"armijo"`       | Armijo sufficient-decrease (default).                    |
  | `"backtracking"` | Cheap, robust backtracking.                              |
  | `"strong_wolfe"` | Strong-Wolfe (tighter curvature condition).              |
  | `"hager_zhang"`  | Hager–Zhang efficient Wolfe line search.                 |

  **Line search options** (`line_search_options`), most useful with
  backtracking:

  | Option       | Meaning                                                       |
  |--------------|---------------------------------------------------------------|
  | `init_step`  | Initial step size; warm-starting beyond `α=1` lets deep-memory steps stretch into the superlinear regime. |
  | `shrink`     | Backtracking contraction factor (gentler shrink = larger probes). |
  | `max_iter`   | Maximum inner line-search iterations.                         |

  ### 4.3 Regions (`qqn_jax.regions`)

  | Region                                       | Description                                          |
  |----------------------------------------------|------------------------------------------------------|
  | `BoxRegion(lo, hi)`                          | Bound each parameter to `[lo, hi]`.                  |
  | `TrustRegion(radius, adaptive, ...)`         | Spherical trust region; optionally adaptive radius.  |
  | `OrthantRegion()`                            | OWL-QN-style orthant projection (induces sparsity).  |
  | `Sequential([region1, region2, ...])`        | Compose projections in order.                        |

  **`TrustRegion` parameters:**

  | Parameter  | Meaning                                                             |
  |------------|---------------------------------------------------------------------|
  | `radius`   | Initial trust-region radius.                                        |
  | `adaptive` | If `True`, the radius adapts to step quality (`ρ`).                 |
  | `shrink`   | Radius shrink factor when the step is poor.                         |
  | `expand`   | Radius expansion factor when the step is good.                      |
  | `eta_lo`   | Lower acceptance threshold (curvature-consistent band).             |
  | `eta_hi`   | Upper acceptance threshold for expansion.                           |

  > **Robustness note:** The naive *adaptive* trust-region tends to
  > over-shrink under the honest prediction model (comparing chord-length
  > against arc-length on a curved path), which can stall convergence.
  > **Fixed-radius** (`adaptive=False`) and **spline** stacks are the robust
  > fast path. A *curvature-consistent* variant (gentle shrink, wide stable
  > band `eta_lo=0.1, eta_hi=0.75`) can recover convergence where the naive
  > adaptive region collapsed.

  ### 4.4 Spline Refinement (`spline=True`)

  Enables cubic Hermite refinement: every probe along the search path becomes a
  spline control point, sharpening the trajectory. It raises `ms/it` (extra
  probes) but typically lowers the iteration count.

  ---

  ## 5. Defining Optimizer Variants

  Optimizers are registered in the `runners` dict in `main()`. Each entry is a
  zero-argument lambda returning the standard 7-tuple result.

  ### Adding a QQN variant

  ```python
  "QQN-MyVariant": lambda: _run_qqn_configured(
      loss_fn,
      params0,
      maxiter,
      line_search="backtracking",
      line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
      oracle=LBFGSOracle(history_size=50),
      spline=True,
      region=TrustRegion(radius=1.0, adaptive=False),
      stop=stop,
  ),
  ```

  ### Adding an Optax baseline

  ```python
  "MyOpt": lambda: run_optax(
      loss_fn, params0, optax.adamw(learning_rate=0.01), maxiter, stop=stop
  ),
  ```

  ### Built-in baselines

  | Name      | Configuration                                       |
  |-----------|-----------------------------------------------------|
  | `SGD`     | `optax.sgd(learning_rate=0.5)`                      |
  | `Adam`    | `optax.adam(learning_rate=0.05)`                   |
  | `L-BFGS`  | `optax.lbfgs()` with zoom line search.             |

  ### Notable pre-defined QQN stacks

  | Variant         | Purpose                                                          |
  |-----------------|------------------------------------------------------------------|
  | `QQN`           | Baseline: L-BFGS oracle + Armijo line search.                    |
  | `QQN-L50`       | Deep L-BFGS memory (size 50).                                    |
  | `QQN-L50Spln`   | Deep memory + cubic Hermite spline (fewest-iteration converger). |
  | `QQN-L50WS+`    | Warm-started backtracking + fixed trust-region (cheap per-step). |
  | `QQN-Champion`  | Best-on-both-axes stack (no spline, no adaptive radius).         |
  | `QQN-Apex`      | Data-driven optimum: deep robust memory + spline + fixed TR.     |
  | `QQN-Fast`      | Strongest *robust* speed stack (L100 + warm start + fixed TR).   |

  ---

  ## 6. Output Reports

  The experiment prints a sequence of analysis tables to stdout:

  1. **Summary table** — all metrics, sorted by `final_loss` (ascending), with a
     `vs LBFGS` speedup column (iterations-to-target relative to L-BFGS).
  2. **Pareto frontier** — non-dominated variants on the (loss, wall-time)
     trade-off.
  3. **Composite efficiency score** — single geometric-mean rank fusing
     iterations-to-target, time-to-target, and final loss (converging variants).
  4. **Iteration-efficiency leaderboard** — converging variants ranked by
     fewest iterations to target.
  5. **Trajectory-AUC leaderboard** — ranked by overall descent quality (lower
     is better).
  6. **Convergence-rate profile** — iteration at which each method first crosses
     each loss milestone.
  7. **Stall report** — variants that exhausted their budget without reaching
     the target, with a classified cause (time-budget exhausted / plateau /
     slow).
  8. **Loss trajectory** — compact ASCII sample of `log10(loss)`.
  9. **Per-step cost decomposition** — isolates `ms/it` vs. iters trade-offs
     (spline cost, warm-start cost, fusion cost).
  10. **A/B controlled comparisons** — each pair isolates a single variable
      (oracle depth, region radius, line search, etc.) against a named baseline
      so the effect is causal.

  ---

  ## 7. Implementation Notes

  - **JIT compilation:** Each optimizer's update/step function is JIT-compiled
    via `jax.jit`. The first iteration therefore includes compilation overhead;
    `ms/it` is averaged across all accepted iterations.
  - **Trajectory recording:** Optimizers are run one update at a time to record
    the full per-iteration loss and timing trajectory.
  - **Convergence test** (`_converged`): returns `True` if the loss reaches
    `f_target` **or** the gradient norm reaches `gtol`.
  - **Milestone tracking** (`_update_milestones`): records the first
    iteration/time each descending loss threshold is crossed.

  ---

  ## 8. Interpreting Results

  - A method that descends fast early but stalls late (e.g., momentum) will
    cross the large-loss milestones quickly but never reach the tight ones.
  - A method that accelerates near the optimum (e.g., deep-memory QQN) will lag
    early but cross the tight milestones first.
  - The **trajectory AUC** is the best single-scalar summary of *overall*
    convergence quality, rewarding both fast early descent and deep late
    refinement.
  - The **Pareto frontier** and **composite efficiency score** identify the
    genuine best-on-all-axes configurations rather than per-axis leaders.
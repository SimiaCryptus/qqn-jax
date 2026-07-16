# Benchmark Results

This directory contains raw benchmark logs (and generated plots) from the
`qqn-jax` experiment suite. Each `.log` file captures one complete run of an
example script, including the exact command used, the environment banner, a
per-configuration results table, several analysis profiles, and the final
exit code / elapsed time.

## File Naming Convention

Log files follow the pattern:

```
<report>_<variant>_<YYYYMMDD>_<HHMMSS>.log
```

| Field       | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| `report`    | The benchmark/example that produced the log.                   |
| `variant`   | The named configuration (activation, width, depth, target, …). |
| `YYYYMMDD`  | Run date (UTC).                                                 |
| `HHMMSS`    | Run start time.                                                 |

Generated plots share the same timestamped stem with a `.png` suffix, e.g.
`fashion_mnist_mlp_comparison_vs_time_20260715-070335.png`.

## Log Anatomy

Every log begins with a metadata header:

```
# variant:  <name>
# report:   <benchmark>
# desc:     <one-line description of the experimental intent>
# started:  <ISO-8601 timestamp>
# command:  <exact shell command, including env-var overrides>
```

and ends with a footer:

```
# exit code: <n>  elapsed: <seconds>s
```

In between you will find:

- A **run banner** (architecture, dataset sizes, shared stopping criteria).
- The **main results table** (`final_loss`, `iters`, `train_acc`,
  `test_acc`, `time(s)`, `ms/it`, evals, AUC, …).
- A **Pareto frontier** (loss vs. time — non-dominated configurations).
- **Iteration-efficiency** and **cost-aware** leaderboards.
- **Target-sensitivity** and **convergence-rate** profiles.
- A **stall report** (configs that never reached the shared target).
- A sampled **loss trajectory** in `log10`.

> **Note:** All runs in this directory were executed on **CPU** — the banner
> line *"a CUDA-enabled jaxlib is not installed. Falling back to cpu."*
> confirms this. Wall-clock numbers should be read accordingly.

## Reports

### `fashion_mnist_mlp_comparison`

Head-to-head comparison of **QQN** (and its variants `QQN-Temp`,
`QQN-Adam`, `QQN-PathMom`) against **Adam** and **L-BFGS** on MLP
classifiers, mostly on Fashion-MNIST (one MNIST profiling run).

| Variant                        | Arch / Activation                | Notes                                              |
|--------------------------------|----------------------------------|----------------------------------------------------|
| `fashion_default`              | 256×3, `tanh,gelu,tanh`          | Headline experiment.                               |
| `fashion_qqn_wide`             | 512×3, `tanh,gelu,tanh`          | Wider net → richer curvature (2nd-order advantage).|
| `fashion_qqn_deep_hessian`     | 256×4, `tanh,gelu`               | Deep, anisotropic Hessian — QQN's strongest regime.|
| `fashion_alt_linear`           | 128×2, `identity`                | Linear (convex) hidden layers; 1st-order friendly. |
| `fashion_rolling_sin`          | 256×3, `rolling_sin`             | Rolling-sine activation.                           |
| `fashion_rolling_sin_control` | 256×3, `sine`                    | Plain-sine control for the rolling variant.        |
| `fashion_rolling_atan2`        | 256×2, `rolling_atan2`           | Rolling-atan2 activation (hard, all configs stall).|
| `fashion_profile_simple_fast` | 64×2 MNIST, `sine`               | Small/fast run with JAX+Perfetto profiling.        |

### `mnist_sparse_benchmark`

Sparse / quantization-aware MNIST benchmark exploring `OrthantRegion`
(OWL-QN-style L1), quantization penalties/regions, and two-stage
*train → polish* pipelines. Reports a Pareto frontier of **test-loss vs.
sparsity** plus a best-precision (lowest `quant_loss`) config.

| Variant           | Description                                        |
|-------------------|----------------------------------------------------|
| `sparse_default`  | 784→64→64→10 `tanh`, strong-Wolfe line search.     |

## Highlights

A few takeaways visible directly in these logs:

- **QQN wins the iteration race** on smooth, full-batch, ill-conditioned
  MLPs. In `fashion_qqn_deep_hessian` the `QQN`/`QQN-Temp` variants reach
  the lowest final losses and dominate the loss-vs-time Pareto frontier.
- **Best per-iteration cost.** On the small, fast profiling run
  (`fashion_profile_simple_fast`) QQN reaches the target in fewer/comparable
  iterations to L-BFGS while spending far fewer function evaluations
  (~1.0/it vs ~2.1/it for the Optax zoom search inside L-BFGS).
- **First-order methods stall in the fine-tuning regime.** Adam typically
  exhausts its time budget well short of the tight loss targets that QQN and
  L-BFGS reach.
- **`QQN-PathMom` and `QQN-Adam` under-perform** the L-BFGS-oracle QQN
  variants on these anisotropic surfaces, consistent with the claim that a
  deep L-BFGS oracle best captures the dominant curvature subspace.
- **Hard activations expose limits.** In `fashion_rolling_atan2` *every*
  optimizer stalls (no target reached), a useful negative result.
- **Sparse/quant trade-offs.** In `sparse_default`, the `l1-penalty` and
  `l1-orthant-penalty` pipelines occupy the test-loss/sparsity Pareto
  frontier; a quant-penalty polish yields the lowest quantization loss.

## Reproducing a Run

Each log's `# command:` line is copy-pasteable. For example:

```bash
# Headline Fashion-MNIST comparison
python3 examples/fashion_mnist_mlp_comparison.py

# Wider network variant (env-var overrides)
DATASET="fashion_mnist" N_TRAIN="25000" N_TEST="5000" \
  HIDDEN="512" DEPTH="3" ACTIVATION="tanh,gelu,tanh" \
  python3 examples/fashion_mnist_mlp_comparison.py

# Sparse / quantization benchmark
python3 examples/mnist_sparse_benchmark.py
```

Results will vary with hardware, jaxlib build (CPU vs. CUDA), and any
wall-clock-based stopping criteria (`TIME_BUDGET`).

## Related Documentation

- [`docs/results.md`](../docs/results.md) — narrative analysis of the MNIST
  benchmark, baselines, and component sweeps.
- [`docs/conclusions.md`](../docs/conclusions.md) — synthesis of findings.
- [`../README.md`](../README.md) — project overview and the QQN algorithm.
"""2-layer ReLU MLP benchmark: QQN vs SGD vs Adam vs L-BFGS.

Thin entry point. All shared machinery now lives in the ``experiments``
package (see ``experiments/plan.md``). The hard-won experimental lessons that
used to live inline here are now in ``experiments/NOTES.md`` (anchors:
eval-dominance, deep-memory, warm-start, spline-divergence).

Configuration is entirely env-var driven (preserved contract for
``run_reports.js``):

  DATASET           mnist | fashion_mnist (default fashion_mnist)
  HIDDEN_SIZES      comma-separated explicit widths (overrides HIDDEN/DEPTH)
  HIDDEN, DEPTH     uniform-width topology (defaults 256 x 3)
  ACTIVATION        single name or comma-separated mix (default tanh,gelu)
  N_CLASSES, N_TRAIN, N_TEST, L2, SEED, SUBSET_SEED, SYNTH_DIM, MAXITER
  F_TARGET, GTOL, TIME_BUDGET, MILESTONES, TARGET_PROFILE
  SGD_LR, ADAM_LR

Enabled optimizer variants live in
``experiments.optimizers.profiles.ENABLED``.

Run with:  python examples/fashion_mnist_mlp_comparison.py
"""

import os

# The cuBLAS-Lt autotuner profiles candidate algorithms concurrently, each
# allocating a multi-GiB workspace for the large full-batch JVP matmuls; on a
# ~6.5GiB GPU that profiling itself OOMs before the solve starts. Disable
# autotuning and prefer async host fallback. Set BEFORE importing jax.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

from qqn_jax.profiling import profile_session, device_memory_report

from experiments.config import ExperimentConfig
from experiments.driver import run_experiment


def main():
    config = ExperimentConfig.from_env(
        dataset="fashion_mnist",
        activation_default="tanh,gelu",
        hidden_default_hidden=256,
        hidden_default_depth=3,
        n_train=25000,
        n_test=5000,
        f_target=2e-2,
        time_budget=150.0,
        milestones=(1e0, 5e-1, 2e-1, 1e-1),
        target_profile=(2e-1, 1e-1, 6e-2, 4e-2, 2e-2),
    )
    run_experiment(config)


if __name__ == "__main__":
    with profile_session("fashion_mnist_mlp_comparison"):
        main()
        mem = device_memory_report()
        if mem is not None:
            print("\n[profile] Device memory at end of run:\n" + mem)

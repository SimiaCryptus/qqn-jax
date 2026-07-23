"""2-layer ReLU MLP benchmark: QQN vs SGD vs Adam vs L-BFGS.

Thin entry point. See ``experiments/plan.md`` for shared machinery and
``experiments/NOTES.md`` for experimental lessons.

Optimizer profiles now live alongside this report in
``reports/reference/profiles.py`` and are passed explicitly into the
driver (rather than the driver reaching into ``experiments.optimizers``).
"""

import os
import sys


os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

# Make sibling module (profiles.py) importable when run by file path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qqn_jax.profiling import profile_session, device_memory_report

from experiments.config import ExperimentConfig
from experiments.driver import run_experiment

import profiles as _profiles


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
        milestones=(1e0, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2),
        target_profile=(2e-1, 1e-1, 6e-2, 4e-2, 2e-2),
    )
    run_experiment(
        config,
        enabled=_profiles.ENABLED,
        build_runners=_profiles.build_runners,
    )


if __name__ == "__main__":
    with profile_session("fashion_mnist_mlp_comparison"):
        main()
        mem = device_memory_report()
        if mem is not None:
            print("\n[profile] Device memory at end of run:\n" + mem)
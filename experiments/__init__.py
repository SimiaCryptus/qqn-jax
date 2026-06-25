"""Experiment harness package: declarative benchmarks for qqn_jax.

See ``plan.md`` for the architecture and ``NOTES.md`` for the lab notebook
of hard-won experimental lessons. The public surface is intentionally small:

    from experiments import ExperimentConfig, run_experiment

Everything else (data loaders, models, runners, reporting) is reachable
through the subpackages but is considered internal-ish plumbing.
"""

from experiments.config import ExperimentConfig
from experiments.metrics.result import RunResult
from experiments.driver import run_experiment
from experiments.sparse_config import SparseConfig
from experiments.sparse_driver import run_sparse_experiment

__all__ = [
    "ExperimentConfig",
    "RunResult",
    "run_experiment",
    "SparseConfig",
    "run_sparse_experiment",
]

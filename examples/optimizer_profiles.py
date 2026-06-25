"""Backward-compat shim.

The optimizer profile registry now lives in
``experiments.optimizers.profiles``. This module re-exports the public
surface so any code (or ``run_reports.js`` path) importing
``optimizer_profiles`` keeps working during/after migration.
"""

from experiments.optimizers.profiles import ENABLED, build_runners

__all__ = ["ENABLED", "build_runners"]

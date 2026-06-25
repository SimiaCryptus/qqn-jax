"""RunResult: the dataclass replacing the fragile positional 8-tuple.

Runners return a ``RunResult`` carrying the *measured* outputs; the driver
later enriches it in place with derived quantities (final/best loss, train/
test accuracy, ms/iter, trajectory AUC, evals/iter, per-target iterations,
reached flag).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["RunResult"]


@dataclass
class RunResult:
    # --- measured by the runner ---
    params: Any
    history: list
    times: list
    wall: float
    iters_to_target: Optional[int]
    time_to_target: Optional[float]
    milestone_hits: dict  # {milestone: (iter, wall_time, evals) | None}
    evals_to_target: Optional[int]

    # --- derived (filled by the driver.enrich) ---
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    iters: Optional[int] = None
    train_acc: Optional[float] = None
    test_acc: Optional[float] = None
    ms_per_iter: Optional[float] = None
    traj_auc: Optional[float] = None
    evals_per_iter: Optional[float] = None
    target_iters: dict = field(default_factory=dict)
    reached: bool = False

    def as_tuple(self):
        """Backward-compat shim: the legacy positional 8-tuple.

        Retained during migration so any not-yet-ported driver code can keep
        unpacking the old contract. Delete once all callers use the fields.
        """
        return (
            self.params,
            self.history,
            self.wall,
            self.times,
            self.iters_to_target,
            self.time_to_target,
            self.milestone_hits,
            self.evals_to_target,
        )

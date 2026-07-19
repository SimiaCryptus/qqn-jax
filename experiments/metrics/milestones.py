"""Shared milestone tracking + convergence test (centralized fairness).

These are the cross-cutting termination semantics that MUST be identical for
every optimizer (plan.md §8.2). One ``converged`` and one
``update_milestones`` for the whole suite.
"""

__all__ = ["converged", "update_milestones"]


def converged(value, gnorm, f_target, gtol):
    """Shared convergence test: target loss reached OR gradient ~ 0."""
    if f_target is not None and value <= f_target:
        return True
    if gtol is not None and gnorm <= gtol:
        return True
    return False


def update_milestones(milestones, hit, value, it, now, evals=None, fwd=None, bwd=None):
    """Record the first iteration/time/evals each loss milestone is crossed.

    Each recorded hit is a tuple ``(iteration, wall_time, evals, fwd, bwd)``
    so the convergence-rate profile can report not just *when* but *how long*
    and *how much work* (combined value+grad calls, plus the separate forward
    value evals and backward gradient evals) it took to first cross each loss
    level.
    """
    if not milestones:
        return
    for m in milestones:
        if hit.get(m) is None and value <= m:
            hit[m] = (it, now, evals, fwd, bwd)

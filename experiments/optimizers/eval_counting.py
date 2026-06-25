"""Genuine eval accounting + a display-only heuristic fallback.

The headline cost-aware results use GENUINE counts (``QQNState.num_evals``,
Optax line-search probe counts, 1/step for SGD/Adam) — see ``runners.py``.
``estimate_evals_per_iter`` is retained ONLY to populate the per-iteration
display column for variants that never reached the target (and thus have no
measured evals-to-target).
"""

__all__ = ["estimate_evals_per_iter"]


def estimate_evals_per_iter(method, qqn_kwargs=None):
    """Heuristic evaluation multiplicity per accepted iteration (display only).

    - First-order (SGD/Adam): 1 value + 1 grad per step.
    - L-BFGS (Optax zoom): ~1 value/grad per step + a few line-search probes.
    - QQN: 1 value/grad to form the path + the line-search probe count
      (+ spline probes when enabled).
    """
    qqn_kwargs = qqn_kwargs or {}
    if method in ("SGD", "Adam"):
        return 1.0
    if method == "L-BFGS":
        return 3.0
    ls = qqn_kwargs.get("line_search", "armijo")
    ls_opts = qqn_kwargs.get("line_search_options", {}) or {}
    if ls in ("armijo", "backtracking"):
        probes = min(ls_opts.get("max_iter", 30), 4)
    elif ls == "strong_wolfe":
        probes = min(ls_opts.get("max_iter", 10), 6)
    elif ls == "hager_zhang":
        probes = min(ls_opts.get("max_iter", 30), 5)
    else:  # fixed
        probes = 1
    base = 1.0 + float(probes)
    if qqn_kwargs.get("spline", False):
        base += 2.0
    return base

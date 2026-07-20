"""Pareto frontier helper (loss vs. wall-time)."""

__all__ = ["pareto_frontier", "pareto_frontier_2d", "milestone_pareto_frontiers"]


def pareto_frontier(results, loss_key="final_loss", time_key="wall"):
    """Return the non-dominated ``(name, result)`` pairs.

    A variant is dominated if another is at least as good on both loss and
    time and strictly better on at least one.
    """
    pareto = []
    for name, r in results.items():
        dominated = any(
            (
                getattr_or_item(o, loss_key) <= getattr_or_item(r, loss_key)
                and getattr_or_item(o, time_key) < getattr_or_item(r, time_key)
            )
            or (
                getattr_or_item(o, loss_key) < getattr_or_item(r, loss_key)
                and getattr_or_item(o, time_key) <= getattr_or_item(r, time_key)
            )
            for on, o in results.items()
            if on != name
        )
        if not dominated:
            pareto.append((name, r))
    return pareto


def pareto_frontier_2d(points):
    """Generic 2D min-min Pareto frontier over ``(name, x, y)`` triples.
    Both ``x`` and ``y`` are minimized (lower is better). A point is
    dominated if another is <= on both axes and strictly < on at least one.
    Points with a ``None`` on either axis are skipped entirely.
    Returns the list of non-dominated ``(name, x, y)`` triples.
    """
    usable = [(n, x, y) for (n, x, y) in points if x is not None and y is not None]
    frontier = []
    for name, x, y in usable:
        dominated = any(
            (ox <= x and oy < y) or (ox < x and oy <= y)
            for on, ox, oy in usable
            if on != name
        )
        if not dominated:
            frontier.append((name, x, y))
    return frontier


def _milestone_time_evals(r, m):
    """Return ``(time, total_evals)`` for the first crossing of milestone ``m``.
    ``total_evals`` prefers the split fwd+bwd counters, falling back to the
    combined ``evals`` field. Either component may be ``None`` when the run
    never reached the milestone or the counters weren't populated.
    """
    hit = r.milestone_hits.get(m)
    if hit is None:
        return (None, None)
    t = hit[1]
    fwd = hit[3] if len(hit) >= 4 else None
    bwd = hit[4] if len(hit) >= 5 else None
    if fwd is not None and bwd is not None:
        tot = int(fwd) + int(bwd)
    elif len(hit) >= 3 and hit[2] is not None:
        tot = int(hit[2])
    else:
        tot = None
    return (t, tot)


def milestone_pareto_frontiers(results, milestones):
    """Per-milestone Pareto frontiers on (wall-time, total evals) to reach.
    For each milestone we build the min-min Pareto frontier over the two
    *honest cross-optimizer* cost axes — wall-clock time and total fwd+bwd
    evals to *first* reach that loss level. This credits fast early-milestone
    winners (e.g. Adam) that later fall behind, since domination is evaluated
    independently at each loss level.
    Returns ``{milestone: [(name, time, total_evals), ...]}`` (only milestones
    that at least one run reached are included).
    """
    out = {}
    for m in milestones:
        points = [(name, *_milestone_time_evals(r, m)) for name, r in results.items()]
        frontier = pareto_frontier_2d(points)
        if frontier:
            out[m] = frontier
    return out


def getattr_or_item(obj, key):
    """Access ``key`` on either a dataclass/object or a dict."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)

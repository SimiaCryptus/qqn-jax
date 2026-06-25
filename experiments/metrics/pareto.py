"""Pareto frontier helper (loss vs. wall-time)."""

__all__ = ["pareto_frontier"]


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


def getattr_or_item(obj, key):
    """Access ``key`` on either a dataclass/object or a dict."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)

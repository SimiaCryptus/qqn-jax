"""Per-axis analysis of QQN configuration profiles.

Each QQN profile name is the hyphenation of its non-empty axis tokens (see
``experiments.optimizers.profiles``): ``QQN-<oracle>-<line_search>-<temp>-...``.
This module recovers, for each axis, which token each profile carried, then
aggregates *normalized* per-metric statistics across every QQN run that
shares a given token value. The normalization is min-max across all QQN
profiles (per metric, lower-is-better oriented) so different metrics are
comparable and the aggregate per-axis-value mean is a dimensionless score
in ``[0, 1]`` (0 = best observed, 1 = worst observed).
"""

import numpy as np

from experiments.optimizers.profiles import _AXES

__all__ = ["axis_analysis", "report_axis_analysis"]


_METRICS = [
    ("final_loss", False),
    ("iters_to_target", False),
    ("time_to_target", False),
    ("evals_to_target", False),
    ("traj_auc", False),
    ("wall", False),
    ("train_acc", True),
    ("test_acc", True),
]


def _axis_token_sets():
    """Return ``[(axis_index, {token, ...}), ...]`` for the non-empty tokens
    of each axis, preserving the fixed axis order used to build names."""
    axis_tokens = []
    for idx, axis_fn in enumerate(_AXES):
        tokens = {tok for tok in axis_fn().keys() if tok}
        axis_tokens.append((idx, tokens))
    return axis_tokens


def _parse_profile_tokens(name):
    """Split a ``QQN-...`` profile name into its ordered token list."""
    if name != "QQN" and not name.startswith("QQN-"):
        return None
    parts = name.split("-")[1:]
    return parts


def _token_for_axis(tokens, axis_token_set):
    """Which token from ``axis_token_set`` (if any) this profile carries."""
    for tok in tokens:
        if tok in axis_token_set:
            return tok
    return None


def _metric_value(r, key):
    """Extract a metric from a RunResult, returning None when unavailable."""
    val = getattr(r, key, None)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def axis_analysis(results):
    """Compute normalized per-axis-value aggregate statistics.

    Returns a dict:
        {
          axis_index: {
            "axis_name": str,
            "values": {
                token_or_"<default>": {
                    "count": int,
                    "score": float | None,      # mean normalized score
                    "metrics": {metric: {"mean_norm", "mean_raw"}},
                    "profiles": [names...],
                }, ...
            }
          }, ...
        }
    """
    qqn = {n: r for n, r in results.items() if _parse_profile_tokens(n) is not None}
    if not qqn:
        return {}

    norm_ranges = {}
    for key, higher_better in _METRICS:
        vals = [
            (-v if higher_better else v)
            for v in (_metric_value(r, key) for r in qqn.values())
            if v is not None
        ]
        if vals:
            norm_ranges[key] = (min(vals), max(vals), higher_better)

    def _normalized(r, key):
        info = norm_ranges.get(key)
        if info is None:
            return None
        lo, hi, higher_better = info
        v = _metric_value(r, key)
        if v is None:
            return None
        v = -v if higher_better else v
        if hi <= lo:
            return 0.0
        return (v - lo) / (hi - lo)

    axis_tokens = _axis_token_sets()
    analysis = {}
    for axis_idx, token_set in axis_tokens:
        axis_name = _AXES[axis_idx].__name__.lstrip("_").replace("_axis", "")
        buckets = {}
        for name, r in qqn.items():
            tokens = _parse_profile_tokens(name)
            tok = _token_for_axis(tokens, token_set)
            key = tok if tok is not None else "<default>"
            buckets.setdefault(key, []).append((name, r))

        if len(buckets) <= 1:
            continue

        values = {}
        for tok, members in buckets.items():
            metric_stats = {}
            all_norms = []
            for key, _hb in _METRICS:
                norms = [
                    n
                    for n in (_normalized(r, key) for _n, r in members)
                    if n is not None
                ]
                raws = [
                    v
                    for v in (_metric_value(r, key) for _n, r in members)
                    if v is not None
                ]
                if norms:
                    metric_stats[key] = {
                        "mean_norm": float(np.mean(norms)),
                        "mean_raw": float(np.mean(raws)) if raws else None,
                    }
                    all_norms.extend(norms)
            values[tok] = {
                "count": len(members),
                "score": float(np.mean(all_norms)) if all_norms else None,
                "metrics": metric_stats,
                "profiles": [n for n, _r in members],
            }
        analysis[axis_idx] = {"axis_name": axis_name, "values": values}
    return analysis


def report_axis_analysis(results):
    """Print the per-axis normalized aggregate statistics table."""
    analysis = axis_analysis(results)
    if not analysis:
        return analysis
    print("\n" + "=" * 90)
    print("QQN per-axis analysis (normalized aggregate scores; 0=best, 1=worst)")
    print("=" * 90)
    metric_keys = [k for k, _ in _METRICS]
    for axis_idx in sorted(analysis):
        entry = analysis[axis_idx]
        print(f"\nAxis: {entry['axis_name']}")
        header = f"  {'value':<12}{'n':>4}{'score':>9}" + "".join(
            f"{k[:9]:>11}" for k in metric_keys
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        ordered = sorted(
            entry["values"].items(),
            key=lambda kv: kv[1]["score"] if kv[1]["score"] is not None else 9e9,
        )
        for tok, stats in ordered:
            score = "—" if stats["score"] is None else f"{stats['score']:.3f}"
            cells = []
            for k in metric_keys:
                ms = stats["metrics"].get(k)
                cells.append("—" if ms is None else f"{ms['mean_norm']:.3f}")
            print(
                f"  {tok:<12}{stats['count']:>4}{score:>9}"
                + "".join(f"{c:>11}" for c in cells)
            )
    return analysis

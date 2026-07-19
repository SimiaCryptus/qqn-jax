"""Detailed JSON export of a full experiment run.

Serializes the resolved ``ExperimentConfig`` plus every ``RunResult``
(measured + derived fields, full loss / time / eval trajectories, milestone
hits, and per-target iterations) to a timestamped file under ``results/``.
Everything is coerced to plain JSON-safe Python types so the artifact is a
complete, reload-able record of the experiment.
"""

import dataclasses
import json
import os
import time

import numpy as np

__all__ = ["write_results_json"]


def _jsonify(obj):
    """Recursively coerce ``obj`` into JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, dict):
        return {str(_jsonify(k)): _jsonify(v) for k, v in obj.items()}
    # Fall back to a string representation for anything exotic.
    return str(obj)


def _config_to_dict(config):
    """Serialize an ExperimentConfig, dropping non-serializable callables."""
    try:
        raw = dataclasses.asdict(config)
    except Exception:
        raw = {k: getattr(config, k) for k in vars(config)}
    # ``activation_fn`` is a live callable (and ``activation_name`` may be a
    # list of names) — keep the names, drop the function objects.
    raw.pop("activation_fn", None)
    return _jsonify(raw)


def _result_to_dict(name, r):
    """Serialize one RunResult (measured + derived) to a plain dict."""
    # ``milestone_hits`` maps a float milestone -> tuple|None; stringify keys.
    milestone_hits = {
        f"{m:.6e}": (None if hit is None else list(hit))
        for m, hit in (r.milestone_hits or {}).items()
    }
    target_iters = {
        f"{t:.6e}": v for t, v in (r.target_iters or {}).items()
    }
    return _jsonify(
        {
            "name": name,
            "final_loss": r.final_loss,
            "best_loss": r.best_loss,
            "iters": r.iters,
            "train_acc": r.train_acc,
            "test_acc": r.test_acc,
            "wall": r.wall,
            "ms_per_iter": r.ms_per_iter,
            "traj_auc": r.traj_auc,
            "iters_to_target": r.iters_to_target,
            "time_to_target": r.time_to_target,
            "evals_to_target": r.evals_to_target,
            "evals_per_iter": r.evals_per_iter,
            "reached": r.reached,
            "history": r.history,
            "times": r.times,
            "milestone_hits": milestone_hits,
            "target_iters": target_iters,
        }
    )


def write_results_json(results, config, *, extra=None, results_dir="results"):
    """Write a timestamped JSON artifact with all experimental data.

    Args:
        results: ``{name: RunResult}`` mapping from the driver.
        config: the resolved ``ExperimentConfig``.
        extra: optional dict of additional top-level metadata (e.g. the
            per-axis analysis) merged into the artifact.
        results_dir: output directory (created if missing).

    Returns:
        The path of the written JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(results_dir, f"{config.dataset}_experiment_{timestamp}.json")

    payload = {
        "timestamp": timestamp,
        "config": _config_to_dict(config),
        "results": {
            name: _result_to_dict(name, r) for name, r in results.items()
        },
    }
    if extra:
        payload.update(_jsonify(extra))

    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[json] Wrote detailed experiment data to {out}")
    return out
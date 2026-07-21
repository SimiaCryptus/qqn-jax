"""Detailed JSON export of experiment runs.

Two entry points:

- ``write_run_json`` writes ONE optimizer's ``RunResult`` to its own
  timestamped, self-describing artifact (dataset, network topology,
  optimizer details, timing, and full fitness / time / eval trajectories).
  The driver calls this as each individual optimizer run completes so that
  partial results are persisted incrementally.

- ``write_results_json`` writes the legacy aggregate artifact for the whole
  run (kept for backward compatibility + the axis analysis summary).

Everything is coerced to plain JSON-safe Python types so the artifacts are
complete, reload-able records. A matching TypeScript schema lives in
``experiments/reporting/schema.ts``.
"""

import dataclasses
import json
import os
import time

import numpy as np

__all__ = ["write_run_json", "write_results_json"]

SCHEMA_VERSION = "1.0.0"


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

    return str(obj)


def _config_to_dict(config):
    """Serialize an ExperimentConfig, dropping non-serializable callables."""
    try:
        raw = dataclasses.asdict(config)
    except Exception:
        raw = {k: getattr(config, k) for k in vars(config)}

    raw.pop("activation_fn", None)
    return _jsonify(raw)


def _topology_from_config(config):
    """Extract a self-describing network-topology descriptor."""
    hidden = list(getattr(config, "hidden_sizes", []) or [])
    activation = getattr(config, "activation_name", None)
    if isinstance(activation, (list, tuple)):
        activation = list(activation)
    arch = ["x"] + hidden + [getattr(config, "n_classes", None)]
    return _jsonify(
        {
            "hidden_sizes": hidden,
            "n_hidden_layers": len(hidden),
            "n_classes": getattr(config, "n_classes", None),
            "activation": activation,
            "arch": "->".join(str(s) for s in arch),
            "l2": getattr(config, "l2", None),
        }
    )


def _dataset_from_config(config):
    """Extract a self-describing dataset descriptor."""
    return _jsonify(
        {
            "name": getattr(config, "dataset", None),
            "n_train": getattr(config, "n_train", None),
            "n_test": getattr(config, "n_test", None),
            "n_classes": getattr(config, "n_classes", None),
            "balanced": getattr(config, "balanced", None),
            "subset_seed": getattr(config, "subset_seed", None),
            "synth_dim": getattr(config, "synth_dim", None),
        }
    )


def _optimizer_descriptor(name, config, extra=None):
    """Build a self-describing optimizer descriptor from name + config."""
    desc = {"name": name}
    n = name.lower()
    if n == "sgd":
        desc["type"] = "sgd"
        desc["learning_rate"] = getattr(config, "sgd_lr", None)
    elif n == "adam":
        desc["type"] = "adam"
        desc["learning_rate"] = getattr(config, "adam_lr", None)
    elif n == "l-bfgs":
        desc["type"] = "lbfgs"
        desc["memory_size"] = getattr(config, "lbfgs_memory_size", None)
    elif n.startswith("qqn"):
        desc["type"] = "qqn"
    else:
        desc["type"] = n
    if extra:
        desc.update(extra)
    return _jsonify(desc)


def _result_to_dict(name, r):
    """Serialize one RunResult (measured + derived) to a plain dict."""

    milestone_hits = {
        f"{m:.6e}": (None if hit is None else list(hit))
        for m, hit in (r.milestone_hits or {}).items()
    }
    target_iters = {f"{t:.6e}": v for t, v in (r.target_iters or {}).items()}
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
            "eval_counts": getattr(r, "eval_counts", None),
            "fwd_counts": getattr(r, "fwd_counts", None),
            "bwd_counts": getattr(r, "bwd_counts", None),
            "milestone_hits": milestone_hits,
            "target_iters": target_iters,
        }
    )


def _sanitize_filename(name):
    """Make ``name`` safe for use inside a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def write_run_json(
    name,
    result,
    config,
    *,
    optimizer_extra=None,
    results_dir="results",
):
    """Write ONE optimizer run to its own self-describing JSON artifact.

    Intended to be called by the driver immediately after a single
    optimizer's run has completed and been enriched, so results are
    persisted incrementally.

    Args:
        name: the optimizer variant name (e.g. ``"Adam"``, ``"QQN"``).
        result: the (enriched) ``RunResult`` for this run.
        config: the resolved ``ExperimentConfig``.
        optimizer_extra: optional dict of extra optimizer hyperparameters
            (e.g. QQN-specific kwargs) merged into the optimizer descriptor.
        results_dir: output directory (created if missing).

    Returns:
        The path of the written JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = _sanitize_filename(name)
    out = os.path.join(
        results_dir,
        f"{config.dataset}_{safe_name}_{timestamp}.json",
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "optimizer_run",
        "timestamp": timestamp,
        "dataset": _dataset_from_config(config),
        "topology": _topology_from_config(config),
        "optimizer": _optimizer_descriptor(name, config, extra=optimizer_extra),
        "stop": _jsonify(getattr(config, "stop", None)),
        "maxiter": getattr(config, "maxiter", None),
        "seed": getattr(config, "seed", None),
        "result": _result_to_dict(name, result),
    }

    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[json] Wrote run artifact for {name} to {out}")
    return out


def write_results_json(results, config, *, extra=None, results_dir="results"):
    """Write a timestamped aggregate JSON artifact with all experimental data.

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
        "schema_version": SCHEMA_VERSION,
        "kind": "experiment",
        "timestamp": timestamp,
        "dataset": _dataset_from_config(config),
        "topology": _topology_from_config(config),
        "config": _config_to_dict(config),
        "results": {name: _result_to_dict(name, r) for name, r in results.items()},
    }
    if extra:
        extra_jsonified = _jsonify(extra)
        if isinstance(extra_jsonified, dict):
            payload.update(extra_jsonified)

    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[json] Wrote detailed experiment data to {out}")
    return out

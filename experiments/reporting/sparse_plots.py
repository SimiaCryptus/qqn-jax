"""Sparse-benchmark plots: convergence, Pareto, per-config metrics bars.

All matplotlib-safe (degrade gracefully when matplotlib is unavailable).
Moved verbatim from the example driver, parameterized by the list of
per-config result dicts produced by ``sparse_driver.run_config``.
"""

from typing import Any, Dict, List

__all__ = ["plot_convergence", "plot_pareto", "plot_metrics_bar"]


def plot_convergence(results: List[Dict[str, Any]], fname: str = "convergence.png"):
    """Plot loss-vs-evaluation convergence curves for all configs."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot_convergence] matplotlib unavailable ({exc!r}); skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        history = r.get("loss_history", [])
        if not history:
            continue
        ax.plot(range(len(history)), history, label=r["name"], linewidth=1.5)
    ax.set_xlabel("loss evaluation")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("QQN convergence: sparse MNIST")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"\nSaved convergence plot to {fname!r}")
    try:
        plt.show()
    except Exception:
        pass


def plot_pareto(results: List[Dict[str, Any]], fname: str = "pareto.png"):
    """Scatter test_loss vs. sparsity, highlighting the Pareto frontier."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot_pareto] matplotlib unavailable ({exc!r}); skipping plot.")
        return
    if not results:
        return
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, o in enumerate(results):
            if i == j:
                continue
            no_worse = (
                o["test_loss"] <= r["test_loss"] and o["sparsity"] >= r["sparsity"]
            )
            strictly_better = (
                o["test_loss"] < r["test_loss"] or o["sparsity"] > r["sparsity"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [r["sparsity"] for r in results]
    ys = [r["test_loss"] for r in results]
    ax.scatter(xs, ys, c="lightgray", s=30, label="dominated", zorder=1)
    px = [r["sparsity"] for r in pareto]
    py = [r["test_loss"] for r in pareto]
    ax.scatter(px, py, c="crimson", s=60, label="Pareto frontier", zorder=3)
    frontier = sorted(pareto, key=lambda d: d["sparsity"])
    ax.plot(
        [r["sparsity"] for r in frontier],
        [r["test_loss"] for r in frontier],
        c="crimson",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )
    for r in frontier:
        ax.annotate(
            r["name"],
            (r["sparsity"], r["test_loss"]),
            fontsize=6,
            textcoords="offset points",
            xytext=(4, 4),
        )
    ax.set_xlabel("weight sparsity (fraction near-zero)")
    ax.set_ylabel("test loss")
    ax.set_title("Accuracy vs. sparsity trade-off: sparse MNIST")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"Saved Pareto plot to {fname!r}")
    try:
        plt.show()
    except Exception:
        pass


def plot_metrics_bar(results: List[Dict[str, Any]], fname: str = "metrics_bar.png"):
    """Grouped bar chart comparing key metrics across all configurations."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot_metrics_bar] matplotlib unavailable ({exc!r}); skipping plot.")
        return
    if not results:
        return
    names = [r["name"] for r in results]
    test_losses = [r["test_loss"] for r in results]
    sparsities = [r["sparsity"] for r in results]
    quant_losses = [r.get("quant_loss", 0.0) for r in results]
    n = len(results)
    y = _np.arange(n)
    height = 0.25
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * n)))
    ax.barh(y - height, test_losses, height, label="test_loss", color="steelblue")
    ax.barh(y, sparsities, height, label="sparsity", color="seagreen")
    ax.barh(y + height, quant_losses, height, label="quant_loss", color="indianred")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("value")
    ax.set_title("Per-config metrics: sparse MNIST")
    ax.legend()
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"Saved metrics bar chart to {fname!r}")
    try:
        plt.show()
    except Exception:
        pass

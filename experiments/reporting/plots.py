"""Convergence plotting (matplotlib-safe, degrades gracefully)."""

import os
import time

__all__ = ["save_plots"]


def save_plots(results, config, *, arch_str=None):
    """Save loss-vs-iteration and loss-vs-time convergence plots.

    Degrades gracefully when matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[plot] matplotlib not available; skipping plot.")
        return

    baselines = {"SGD", "Adam", "L-BFGS"}
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    activation_str = (
        ",".join(config.activation_name)
        if isinstance(config.activation_name, (list, tuple))
        else str(config.activation_name)
    )
    n_hidden_layers = len(config.hidden_sizes)
    if arch_str is None:
        arch_str = "->".join(
            str(s) for s in (["x"] + list(config.hidden_sizes) + [config.n_classes])
        )
    config_title = (
        f"{n_hidden_layers + 1}-layer MLP {arch_str} on {config.dataset}\n"
        f"activation={activation_str}  classes={config.n_classes}  "
        f"n_train={config.n_train}  maxiter={config.maxiter}  "
        "(QQN variants vs baselines)"
    )

    def _draw(x_key, x_label, file_suffix):
        plt.figure(figsize=(7, 5))
        for name, r in results.items():
            xs = r.times if x_key == "times" else range(len(r.history))
            if name in baselines:
                plt.semilogy(xs, r.history, label=name, linestyle="--", linewidth=2)
            else:
                plt.semilogy(xs, r.history, label=name, alpha=0.85)
        plt.xlabel(x_label)
        plt.ylabel("full-batch loss")
        plt.title(config_title)
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        out = os.path.join(
            results_dir,
            f"{config.dataset}_mlp_comparison_{file_suffix}_{timestamp}.png",
        )
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved {file_suffix} convergence plot to {out}")

    print()
    _draw("iteration", "iteration", "vs_iter")
    _draw("times", "wall-clock time (s)", "vs_time")

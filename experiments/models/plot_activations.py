"""Plotting utility for the canonical activation registry.

Generates one PNG per activation function in ``ACTIVATIONS``, plotting the
function over a representative input range. Periodic (e.g. sine, hermite
wave/pulse) and non-periodic activations alike are sampled densely so the
shape is clear.

Usage:
    python -m experiments.models.plot_activations [--outdir DIR] [--xmin X]
        [--xmax X] [--num N]
"""

import argparse
import os

import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from experiments.models.activations import ACTIVATIONS, UNIVARIATE_ACTIVATIONS


def _sanitize(name):
    """Turn an activation name into a filesystem-safe stem."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def plot_activation(name, fn, outdir, *, xmin=-6.0, xmax=6.0, num=1000):
    """Render a single activation to ``<outdir>/<name>.png``.

    Args:
        name: activation name (used for the title and filename).
        fn: the activation callable.
        outdir: directory to write the PNG into.
        xmin: left edge of the sampled input range.
        xmax: right edge of the sampled input range.
        num: number of sample points.

    Returns:
        The path of the written PNG, or ``None`` if evaluation failed.
    """
    x = jnp.linspace(xmin, xmax, num)
    try:
        y = np.asarray(fn(x))
    except Exception as exc:  # noqa: BLE001 - report and skip bad activations
        print(f"[plot] Skipping {name!r}: evaluation failed ({exc}).")
        return None

    x_np = np.asarray(x)
    if y.shape != x_np.shape:
        # Some activations (e.g. rolling windows) may reduce/reshape; try to
        # broadcast or flatten to a 1-D curve aligned with x.
        y = np.reshape(y, (-1,))
        if y.shape[0] != x_np.shape[0]:
            print(
                f"[plot] Skipping {name!r}: output shape {y.shape} "
                f"incompatible with input {x_np.shape}."
            )
            return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_np, y, color="tab:blue", linewidth=2)
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel(f"{name}(x)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{_sanitize(name)}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_all_activations(outdir="../../reports/activation_plots", *, xmin=-6.0, xmax=6.0, num=1000):
    """Plot every activation in the registry, one PNG per function.

    Args:
        outdir: directory to write PNGs into.
        xmin: left edge of the sampled input range.
        xmax: right edge of the sampled input range.
        num: number of sample points.

    Returns:
        A list of the written PNG paths.
    """
    written = []
    for name, fn in sorted(UNIVARIATE_ACTIVATIONS.items()):
        path = plot_activation(name, fn, outdir, xmin=xmin, xmax=xmax, num=num)
        if path is not None:
            written.append(path)
            print(f"[plot] Wrote {path}")
    print(f"[plot] Done: {len(written)} plot(s) in {outdir!r}.")
    return written


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="../../reports/activation_plots",
        help="Directory to write PNGs into (default: ../../reports/activation_plots).",
    )
    parser.add_argument("--xmin", type=float, default=-6.0, help="Min input value.")
    parser.add_argument("--xmax", type=float, default=6.0, help="Max input value.")
    parser.add_argument(
        "--num", type=int, default=1000, help="Number of sample points."
    )
    args = parser.parse_args()
    plot_all_activations(
        args.outdir, xmin=args.xmin, xmax=args.xmax, num=args.num
    )


if __name__ == "__main__":
    _main()
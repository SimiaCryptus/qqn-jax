"""Plotting utility for rolling-window base activation functions.

The rolling-window activations couple neighbouring units via a small base
function ``g(x_0, ..., x_{w-1})``. Applying such a base directly to a 1-D
sweep (as ``plot_activations`` does) is misleading because the second (and
third) argument is a *shifted* copy of the same signal, not an independent
axis. This utility instead visualises the base functions on their own
terms:

  * 2-input bases (window=2) are rendered as 2-D heatmaps over ``(x, y)``,
    with the ``y = x`` diagonal (the "flat signal" locus) overlaid.
  * 3-input bases (window=3) are rendered as a row of 2-D heatmaps over
    ``(a, c)`` with the middle argument ``b`` held at several fixed slices.

Usage:
    python -m experiments.models.plot_rolling_activations \
        [--outdir DIR] [--vmin V] [--vmax V] [--num N]
"""

import argparse
import os

import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from experiments.models.rolling_window_activation import (
    _sin_diff,
    _euclidean,
    _tan_ratio,
    _xlog_share,
    _harmonic,
    _softmin_diff,
    _sum_over_prod,
    _parallel,
    _ratio,
    _soft_xor,
    _soft_and,
    _soft_or,
    _atan2_ramp,
)


# Registry of base functions with their window width. Keyed by a display
# name; each value is ``(base_fn, window)``.
BASE_FUNCTIONS = {
    "sin_diff": (_sin_diff, 2),
    "euclidean": (_euclidean, 2),
    "tan_ratio": (_tan_ratio, 2),
    "xlog_share": (_xlog_share, 2),
    "harmonic": (_harmonic, 2),
    "softmin": (_softmin_diff, 2),
    "sum_over_prod": (_sum_over_prod, 2),
    "parallel": (_parallel, 2),
    "ratio": (_ratio, 2),
    "soft_xor": (_soft_xor, 2),
    "soft_and": (_soft_and, 2),
    "soft_or": (_soft_or, 2),
    "atan2_ramp": (_atan2_ramp, 3),
}


def _sanitize(name):
    """Turn a base-function name into a filesystem-safe stem."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _finite_clip(z, vmin, vmax):
    """Replace non-finite values with NaN and clip to a robust range.

    Several bases (ratio, sum_over_prod, tan_ratio, ...) can spike near
    poles even with the small floors. For a readable heatmap we clip to a
    symmetric percentile-based range unless explicit limits are supplied.
    """
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z), z, np.nan)
    if vmin is None or vmax is None:
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            lo, hi = -1.0, 1.0
        else:
            lo = np.nanpercentile(finite, 1.0)
            hi = np.nanpercentile(finite, 99.0)
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    return z, vmin, vmax


def plot_base_2d(name, fn, outdir, *, span=6.0, num=400, vmin=None, vmax=None):
    """Render a 2-input base function as a heatmap over ``(x, y)``.

    Args:
        name: base-function name (title + filename).
        fn: callable ``fn(x, y)`` broadcasting over arrays.
        outdir: output directory.
        span: inputs are sampled on ``[-span, span]`` in both axes.
        num: grid resolution per axis.
        vmin/vmax: optional colour limits (auto-ranged when ``None``).

    Returns:
        The written PNG path, or ``None`` on failure.
    """
    axis = jnp.linspace(-span, span, num)
    xx, yy = jnp.meshgrid(axis, axis, indexing="xy")
    try:
        zz = np.asarray(fn(xx, yy))
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] Skipping {name!r}: evaluation failed ({exc}).")
        return None

    zz, lo, hi = _finite_clip(zz, vmin, vmax)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.pcolormesh(
        np.asarray(axis),
        np.asarray(axis),
        zz,
        cmap="twilight",
        shading="auto",
        vmin=lo,
        vmax=hi,
    )
    # The y = x diagonal is the "flat signal" locus (unit == its neighbour).
    ax.plot([-span, span], [-span, span], color="white", linewidth=0.8,
            alpha=0.6, linestyle="--")
    ax.set_title(f"{name}(x, y)")
    ax.set_xlabel("x  (h_i)")
    ax.set_ylabel("y  (h_{i+1})")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{_sanitize(name)}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_base_3d(
    name, fn, outdir, *, span=6.0, num=300, slices=(-3.0, 0.0, 3.0),
    vmin=None, vmax=None,
):
    """Render a 3-input base as a row of ``(a, c)`` heatmaps per ``b`` slice.

    Args:
        name: base-function name (title + filename).
        fn: callable ``fn(a, b, c)`` broadcasting over arrays.
        outdir: output directory.
        span: inputs sampled on ``[-span, span]`` for the ``a`` and ``c`` axes.
        num: grid resolution per axis.
        slices: fixed values of the middle argument ``b``.
        vmin/vmax: optional shared colour limits (auto-ranged when ``None``).

    Returns:
        The written PNG path, or ``None`` on failure.
    """
    axis = jnp.linspace(-span, span, num)
    aa, cc = jnp.meshgrid(axis, axis, indexing="xy")

    panels = []
    try:
        for b in slices:
            zz = np.asarray(fn(aa, jnp.full_like(aa, b), cc))
            panels.append(zz)
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] Skipping {name!r}: evaluation failed ({exc}).")
        return None

    # Compute a shared colour range across all slices for comparability.
    stacked = np.stack([np.asarray(p, dtype=float) for p in panels])
    stacked, lo, hi = _finite_clip(stacked, vmin, vmax)

    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    im = None
    for ax, b, zz in zip(axes[0], slices, stacked):
        im = ax.pcolormesh(
            np.asarray(axis),
            np.asarray(axis),
            zz,
            cmap="twilight",
            shading="auto",
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(f"{name}(a, b={b:g}, c)")
        ax.set_xlabel("a  (h_i)")
        ax.set_ylabel("c  (h_{i+2})")
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{_sanitize(name)}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_all_rolling_bases(
    outdir="rolling_activation_plots", *, span=6.0, num=400,
    slices=(-3.0, 0.0, 3.0), vmin=None, vmax=None,
):
    """Plot every registered rolling-window base function.

    Dispatches to the 2-D or 3-D renderer based on each base's window width.

    Returns:
        A list of the written PNG paths.
    """
    written = []
    for name, (fn, window) in sorted(BASE_FUNCTIONS.items()):
        if window == 2:
            path = plot_base_2d(
                name, fn, outdir, span=span, num=num, vmin=vmin, vmax=vmax
            )
        elif window == 3:
            path = plot_base_3d(
                name, fn, outdir, span=span, num=min(num, 300),
                slices=slices, vmin=vmin, vmax=vmax,
            )
        else:
            print(f"[plot] Skipping {name!r}: unsupported window={window}.")
            path = None
        if path is not None:
            written.append(path)
            print(f"[plot] Wrote {path}")
    print(f"[plot] Done: {len(written)} plot(s) in {outdir!r}.")
    return written


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="rolling_activation_plots",
        help="Directory to write PNGs into (default: rolling_activation_plots).",
    )
    parser.add_argument(
        "--span", type=float, default=6.0,
        help="Inputs sampled on [-span, span] per axis (default: 6.0).",
    )
    parser.add_argument(
        "--num", type=int, default=400,
        help="Grid resolution per axis (default: 400).",
    )
    parser.add_argument(
        "--slices", type=float, nargs="+", default=[-3.0, 0.0, 3.0],
        help="Fixed middle-argument values for 3-input bases.",
    )
    parser.add_argument(
        "--vmin", type=float, default=None, help="Colour-scale minimum.",
    )
    parser.add_argument(
        "--vmax", type=float, default=None, help="Colour-scale maximum.",
    )
    args = parser.parse_args()
    plot_all_rolling_bases(
        args.outdir,
        span=args.span,
        num=args.num,
        slices=tuple(args.slices),
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    _main()
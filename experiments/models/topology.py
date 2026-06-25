"""Hidden-layer topology parsing (HIDDEN_SIZES / DEPTH / HIDDEN).

Consolidates the two divergent ``_parse_hidden_sizes`` copies. The defaults
are passed in by the caller so each benchmark keeps its own narrative
defaults (fashion: width 256 x depth 3; sparse can override).
"""

import os

__all__ = ["parse_hidden_sizes"]


def parse_hidden_sizes(*, default_hidden=256, default_depth=3, default=None):
    """Resolve the hidden-layer topology from environment variables.

    Precedence:
      1. ``HIDDEN_SIZES`` (comma-separated explicit widths).
      2. ``DEPTH`` x ``HIDDEN`` (uniform-width network).
      3. ``default`` (an explicit list) if provided, else
         ``[default_hidden] * default_depth``.

    Returns:
        A list of positive ints (may be empty for a pure linear model).
    """
    fallback = (
        list(default) if default is not None else [default_hidden] * default_depth
    )

    raw = os.environ.get("HIDDEN_SIZES")
    if raw is not None and raw.strip() != "":
        try:
            sizes = [int(tok) for tok in raw.split(",") if tok.strip() != ""]
            if any(s <= 0 for s in sizes):
                raise ValueError("hidden sizes must be positive")
            return sizes
        except ValueError as exc:
            print(
                f"[config] Invalid HIDDEN_SIZES={raw!r} ({exc}); "
                "falling back to DEPTH/HIDDEN."
            )
    try:
        hidden = int(os.environ.get("HIDDEN", str(default_hidden)))
        depth = int(os.environ.get("DEPTH", str(default_depth)))
        if hidden <= 0 or depth < 0:
            raise ValueError
    except ValueError:
        print(f"[config] Invalid HIDDEN/DEPTH; using default {fallback}.")
        return fallback
    return [hidden] * depth

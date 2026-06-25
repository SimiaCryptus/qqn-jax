"""Canonical environment-variable parsing helpers.

Every numeric / list / string parameter of an experiment is overridable via
an environment variable so a headline run can be re-tuned without editing
source. Each helper validates its input and falls back to the documented
default (with a warning) on malformed values.

This is the single source of truth replacing the per-script copies of
``_env_int`` / ``_env_float`` / ``_env_str`` / ``_env_float_list``.
"""

import os

__all__ = ["env_int", "env_float", "env_str", "env_float_list", "env_int_list"]


def env_str(name, default):
    """Parse a string environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def env_float(name, default):
    """Parse a float environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[config] Invalid {name}={raw!r}; using default {default}.")
        return default


def env_int(name, default):
    """Parse an int environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] Invalid {name}={raw!r}; using default {default}.")
        return default


def env_float_list(name, default):
    """Parse a comma-separated float list env var, falling back to ``default``.

    ``default`` should be a tuple/list of floats. An empty/unset variable
    returns the default; a malformed entry triggers a warning and the default.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return tuple(default)
    try:
        vals = tuple(float(tok) for tok in raw.split(",") if tok.strip() != "")
        if not vals:
            raise ValueError("empty list")
        return vals
    except ValueError as exc:
        print(f"[config] Invalid {name}={raw!r} ({exc}); using default {default}.")
        return tuple(default)


def env_int_list(name, default):
    """Parse a comma-separated int list env var, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return tuple(default)
    try:
        vals = tuple(int(tok) for tok in raw.split(",") if tok.strip() != "")
        if not vals:
            raise ValueError("empty list")
        return vals
    except ValueError as exc:
        print(f"[config] Invalid {name}={raw!r} ({exc}); using default {default}.")
        return tuple(default)

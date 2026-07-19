"""Optimizer profile registry (moved + generalized from optimizer_profiles.py).

``build_runners(ctx)`` constructs the ``{name: runner_lambda}`` map and the
companion ``{name: qqn_kwargs}`` map (used purely for the eval-cost display
estimate). Only profiles whose names appear in ``ENABLED`` are returned.

``ctx`` is duck-typed: it must expose ``loss_fn``, ``params0``, ``maxiter``,
``stop``, ``sgd_lr``, ``adam_lr`` and the three runner helpers
``run_qqn`` / ``run_optax`` / ``run_optax_lbfgs``. The driver passes the
canonical runners, so by default this uses ``experiments.optimizers.runners``.

QQN profiles are generated as the *cross product* of a handful of orthogonal
axes (oracle, line search, spline, region, probe-feeding). Each axis is a
``{token: kwargs}`` map defined by one of the ``_*_axis`` functions below:

    * ``""`` is always the axis's default and contributes nothing to the
      generated profile name.
    * every other entry ``token: kwargs`` means "when this axis takes this
      value, merge ``kwargs`` into the ``ctx.run_qqn(...)`` call, and append
      ``token`` to the profile name".
    * to disable a variant, simply comment out its line -- ``_qqn_registry``
      only ever sees the entries that are still present in the dict.

The cross product of every axis's *currently enabled* entries is registered
under a name built by hyphenating the non-empty tokens (in fixed axis order:
oracle, line_search, spline, region, probes) after a leading ``"QQN"``, e.g.
``oracle="L80"`` + ``line_search="BT"`` => ``"QQN-L80-BT"``. Only names that
also appear in ``ENABLED`` are actually built.
"""

import itertools

import optax

__all__ = ["ENABLED", "build_runners"]

from qqn_jax import AdamOracle, LBFGSOracle
from qqn_jax.oracles import AnchoredMultiSecantOracle

ENABLED = [
    "QQN",
    "Adam",
]


def _oracle_axis():
    """Oracle axis: token -> ``run_qqn`` kwargs selecting the oracle."""
    return {
        "Adam": {"oracle": AdamOracle()},
        "AMS": {"oracle": AnchoredMultiSecantOracle(window=10)},
        "L10": {"oracle": LBFGSOracle(history_size=10)},
    }


def _line_search_axis():
    """Line-search axis: token -> ``run_qqn`` kwargs selecting the search.

    The line search's role in QQN is *usually permissive*: the quadratic path
    ``d(t)`` already encodes the curvature, so the search only needs to pick a
    step that makes sufficient (Armijo) progress along the curve — it is *not*
    meant to solve the 1-D subproblem to optimality. The tokens below make
    that spectrum explicit, from most permissive to most exacting:

      * ``Null`` — the *maximally* permissive extreme: unconditionally accept
        the ``t = 1`` oracle endpoint (no acceptance test at all).
      * ``Arm`` (default) — permissive Armijo backtracking: accept the first
        step meeting sufficient decrease. This is the robust efficiency winner
        on smooth full-batch problems.
      * ``ArmLoose`` / ``ArmTight`` — the *same* permissive Armijo search with
        an explicitly loosened / tightened sufficient-decrease constant ``c1``,
        via ``line_search_options``. ``ArmLoose`` (tiny ``c1``, few shrinks)
        underlines "just take a reasonable step"; ``ArmTight`` (larger ``c1``,
        more shrinks) demands more decrease before accepting.
      * ``SW`` — strong Wolfe: enforces the curvature condition too, a stricter
        (less permissive) accept.
      * ``Bisect`` — the *special-case* exacting extreme: a bisection search
        that drives the along-path directional derivative to zero, locating a
        genuine 1-D minimum. Reserve this for problems where an accurate
        along-path minimizer is worth the extra gradient evaluations.
    """
    return {
        "BT": {"line_search": "backtracking"},
        "AW": {"line_search": "armijo_wolfe"},
        "Fix": {"line_search": "fixed"},
        "SW": {"line_search": "strong_wolfe"},
    }


def _region_axis():
    """Region axis: token -> ``run_qqn`` kwargs selecting the trust region."""
    return {
        "": {},
    }


def _spline_axis():
    """Spline/linear axis: token -> ``run_qqn`` kwargs toggling the path
    refinement.

    ``"S"`` enables the cubic Hermite *spline* refinement (reuses every
    probe's gradient as a control point). ``"L"`` selects the *linear*
    refinement — the deliberate opposite of the spline: it interpolates
    value-only between the origin and the oracle point, throwing out the
    gradient information entirely (falling back to the gradient ray only
    when there is no genuine oracle point).
    """
    return {
        "": {},
    }


def _probes_axis():
    """Probe-feeding axis: token -> ``run_qqn`` kwargs toggling probe replay."""
    return {
        "": {},
    }


def _partition_axis():
    """Partition axis: token -> ``run_qqn`` kwargs selecting per-layer
    partitioning of the flat parameter vector.
    ``""`` (default) leaves the solver unpartitioned (a single oracle drives
    the whole flat parameter vector). ``"Part"`` requests per-layer
    partitioning: each weight/bias block gets its own oracle curvature
    history so incompatible per-layer scales never mix.
    Because the concrete segment sizes depend on the model geometry (only
    known at runtime), the ``"Part"`` entry carries a sentinel marker rather
    than a literal ``partition_sizes`` tuple; the factory resolves the real
    sizes from ``ctx.partition_sizes`` when the runner is built.
    """
    return {
        "": {},
        "Part": {"_per_layer": True},
    }


def _temperature_axis():
    """Temperature axis: token -> ``run_qqn`` kwargs enabling a Metropolis-style
    stochastic line-search acceptance.
    ``""`` (default) leaves ``temperature`` at its per-line-search default
    (``0.0`` for the plain backtracking/Armijo family, i.e. no stochastic
    uphill moves). Every non-empty entry threads a ``temperature`` (and,
    optionally, ``cooling``/``seed``) into the chosen line search's
    ``line_search_options`` so the simulated-annealing acceptance path is
    activated. This composes with any line search that honours ``temperature``
    (the backtracking/Armijo family); searches that ignore it (strong Wolfe,
    Hager-Zhang, fixed, null) are unaffected.
    """
    return {
        "": {},
    }


_AXES = [
    _oracle_axis,
    _line_search_axis,
    _temperature_axis,
    _spline_axis,
    _region_axis,
    _probes_axis,
    _partition_axis,
]


_DISPLAY_KWARG_KEYS = ("line_search", "line_search_options", "spline")


def _qqn_registry():
    """Build the ``{name: factory}`` registry for every enabled QQN axis
    combination (the cross product of each axis's currently enabled entries).
    """
    axes = [list(axis().items()) for axis in _AXES]
    registry = {}
    for combo in itertools.product(*axes):
        tokens = [token for token, _kwargs in combo if token]
        name = "-".join(["QQN", *tokens])
        kwargs = {}
        for _token, axis_kwargs in combo:
            for key, val in axis_kwargs.items():
                if key == "line_search_options" and isinstance(val, dict):
                    merged = dict(kwargs.get("line_search_options", {}))
                    merged.update(val)
                    kwargs["line_search_options"] = merged
                else:
                    kwargs[key] = val
        display_kwargs = {k: v for k, v in kwargs.items() if k in _DISPLAY_KWARG_KEYS}

        def factory(ctx, _kwargs=kwargs, _display=display_kwargs):

            run_kwargs = dict(_kwargs)
            if run_kwargs.pop("_per_layer", False):
                partition_sizes = getattr(ctx, "partition_sizes", None)
                if not partition_sizes:
                    raise ValueError(
                        "Profile requested per-layer partitioning but "
                        "ctx.partition_sizes is missing/empty. The driver must "
                        "expose the flat per-layer block sizes on ctx."
                    )
                run_kwargs["partition_sizes"] = tuple(partition_sizes)
            return (
                lambda: ctx.run_qqn(
                    ctx.loss_fn,
                    ctx.params0,
                    ctx.maxiter,
                    stop=ctx.stop,
                    **run_kwargs,
                ),
                _display,
            )

        registry[name] = factory
    return registry


def _baseline_profiles():
    """Non-QQN baselines: these don't participate in the QQN axis cross
    product since they have no oracle / line-search / region axes."""

    def SGD(ctx):
        return (
            lambda: ctx.run_optax(
                ctx.loss_fn,
                ctx.params0,
                optax.sgd(learning_rate=ctx.sgd_lr),
                ctx.maxiter,
                stop=ctx.stop,
            ),
            {},
        )

    def Adam(ctx):
        return (
            lambda: ctx.run_optax(
                ctx.loss_fn,
                ctx.params0,
                optax.adam(learning_rate=ctx.adam_lr),
                ctx.maxiter,
                stop=ctx.stop,
            ),
            {},
        )

    def LBFGS(ctx):
        return (
            lambda: ctx.run_optax_lbfgs(
                ctx.loss_fn, ctx.params0, ctx.maxiter, stop=ctx.stop
            ),
            {},
        )

    return {
        "SGD": SGD,
        "Adam": Adam,
        "L-BFGS": LBFGS,
    }


def _profiles():
    """Return the ``{name: factory}`` registry: the QQN axis cross product
    plus the fixed non-QQN baselines."""
    registry = _qqn_registry()
    registry.update(_baseline_profiles())
    return registry


def build_runners(ctx, enabled=None):
    """Build ``(runners, qqn_kwarg_map)`` for every enabled profile.

    Args:
        ctx: namespace carrying shared experiment objects + runner helpers.
        enabled: optional override of the ``ENABLED`` list (e.g. supplied by
            an ``ExperimentConfig``). Defaults to module-level ``ENABLED``.

    Returns:
        ``(runners, qqn_kwarg_map)`` ordered dicts keyed by profile name.
    """
    registry = _profiles()
    names = enabled if enabled is not None else ENABLED
    runners = {}
    qqn_kwarg_map = {}

    qqn_names = sorted(
        name for name in registry if name == "QQN" or name.startswith("QQN-")
    )
    resolved = []
    for name in names:
        if name == "QQN":
            if not qqn_names:
                raise KeyError(
                    "Enabled 'QQN' group produced no profiles; every QQN axis "
                    "must have at least one enabled entry."
                )
            resolved.extend(qqn_names)
        else:
            resolved.append(name)

    seen = set()
    for name in resolved:
        if name in seen:
            continue
        seen.add(name)
        if name not in registry:
            raise KeyError(f"Enabled profile {name!r} has no factory.")
        runner, kwargs = registry[name](ctx)
        runners[name] = runner
        qqn_kwarg_map[name] = kwargs
    return runners, qqn_kwarg_map

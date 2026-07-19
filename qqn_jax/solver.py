"""QQN (Quadratic Quasi-Newton) solver.

QQN constructs the quadratic interpolation path

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

blending the steepest-descent direction (``-∇f``) with the L-BFGS direction
(``-H∇f``), and performs a line search over this path.

The solver follows the JAXopt-style ``init_state`` / ``update`` / ``run``
interface and keeps all state in JIT-compatible NamedTuples.

Note on parameterization:
    The line search traverses the path parameter ``t`` directly. The points
    ``x + d(t)`` along the curve are *states*, not directions to be
    re-scaled by a separate inner line search. Importantly, rescaling the
    gradient (or the oracle direction) does **not** change the geometric
    path traced by ``d(t)`` — it only distorts the parameterization (i.e.
    how ``t`` maps onto arc length along the curve). The curve itself, and
    therefore the set of candidate states, is invariant to such rescaling.
"""

from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp

from qqn_jax.oracles.strategy import resolve_oracle
from qqn_jax.oracles.oracle import OracleInfo
from qqn_jax.line_search import LINE_SEARCHES
from qqn_jax.paths import (
    PathStrategy,
    QUADRATIC_PATH,
    LINEAR_PATH,
)
from qqn_jax.regions.strategy import RegionInfo, resolve_region
from qqn_jax.utils import (
    make_value_and_grad,
    tree_l2_norm,
    tree_negative,
    tree_vdot,
)


class QQNState(NamedTuple):
    """Immutable state container for QQN.

    Attributes:
        iter: iteration counter.
        value: current objective value.
        grad: current gradient.
         oracle_state: state of the oracle (e.g. L-BFGS history).
        step_size: last accepted step size ``α`` (the path parameter ``t``).
        error: gradient norm (convergence metric).
        done: whether convergence has been reached.
        aux: optional auxiliary output of the objective.
         region_state: optional state for the projective region.
    """

    iter: jnp.ndarray
    value: jnp.ndarray
    grad: jnp.ndarray
    oracle_state: Any
    step_size: jnp.ndarray
    error: jnp.ndarray
    done: jnp.ndarray
    aux: Any = None
    region_state: Any = ()
    num_evals: jnp.ndarray = jnp.asarray(0, jnp.int32)
    qn_slope: jnp.ndarray = jnp.asarray(0.0)
    ls_success: jnp.ndarray = jnp.asarray(True)
    last_reduction: jnp.ndarray = jnp.asarray(0.0)


class QQN:
    """Quadratic Quasi-Newton optimizer.

    Args:
        fun: objective function ``f(params, *args) -> scalar`` (or
            ``(scalar, aux)`` if ``has_aux=True``).
        maxiter: maximum number of iterations.
        tol: convergence tolerance on the gradient L2 norm.
        history_size: L-BFGS memory size ``m``.
        line_search: name of the line-search strategy. One of
             ``"armijo"`` (default), ``"backtracking"``, ``"strong_wolfe"``,
             ``"hager_zhang"``, ``"fixed"``, ``"null"`` or ``"bisection"``.
             The line search's role in QQN is *generally permissive*: since the
             quadratic path ``d(t)`` already encodes the curvature, the search
             only needs to pick a step that makes sufficient progress along the
             curve rather than solve the 1-D subproblem exactly. ``"null"`` is
             the maximally permissive extreme (accept ``t = 1`` unconditionally)
             and ``"bisection"`` is the exacting special case that drives the
             along-path slope to zero to find a *true* minimum. Empirically (see
             ``docs/results.md``) the backtracking/Armijo family is the robust
             efficiency winner on smooth full-batch problems; ``"strong_wolfe"``
             can over-restrict the quadratic-path step and fail to converge.
        line_search_options: optional dict of keyword arguments forwarded to
             the chosen line-search function (e.g. ``c1``, ``c2``, ``max_iter``,
             ``init_step``, ``shrink``, ``step_size``). These override the
             line-search defaults.
        spline: when ``True``, enable the cubic Hermite spline refinement. This
             is orthogonal to ``line_search``: every probe along the (consistent)
             path is reused as a control point and the spline's stationary points
             guide the search. It composes with any chosen line search.
        linear: when ``True``, wrap the chosen line search with the value-only
             linear-chord refinement (see ``qqn_jax.paths.linear``). Mutually
             exclusive with ``spline``.
        path: optional explicit ``PathStrategy`` (see ``qqn_jax.paths.base``)
             overriding the curve the solver traverses. Defaults to the
             canonical quadratic path (``qqn_jax.paths.quadratic.QUADRATIC_PATH``)
             or, when ``linear=True``, the straight chord
             (``qqn_jax.paths.linear.LINEAR_PATH``). This is the single source
             of truth threaded through the *selected* line search itself
             (as an explicit ``path`` keyword, first-class, unconditionally
             — not only when ``spline``/``linear`` is enabled), the
             line-search refinements (``spline``/``linear``) and the
             along-path predicted-reduction model used by the trust-region
             update, so none of the three can silently drift out of sync
             with the curve actually traversed.
        has_aux: whether ``fun`` returns auxiliary data.

    Note:
        The line search traverses the path parameter ``t ∈ [0, 1]`` directly.
        Each evaluated point ``x + d(t)`` is a *state* on the quadratic curve,
        not a direction to be independently re-scaled. Rescaling the gradient
        does not change the path geometry — only its parameterization.
    """

    def __init__(
        self,
        fun: Callable,
        maxiter: int = 100,
        tol: float = 1e-5,
        history_size: int = 10,
        line_search: str = "backtracking",
        line_search_options: Optional[Dict[str, Any]] = None,
        spline: bool = False,
        linear: bool = False,
        path: Optional[PathStrategy] = None,
        has_aux: bool = False,
        region=None,
        oracle="lbfgs",
        feed_probes_to_oracle: bool = False,
        probe_descent_gate: bool = True,
        max_probes: int = 32,
        max_t: float = 1000.0,
        partition_sizes: Optional[tuple[int, ...]] = None,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.tol = tol
        self.history_size = history_size
        self.line_search = line_search
        self.line_search_options = dict(line_search_options or {})
        self.spline = spline
        self.linear = linear
        # The path strategy is the shared component (``qqn_jax.paths.base.
        # PathStrategy``) that remaps the scalar line-search parameter ``t``
        # into the probe offset ``d(t)`` and its velocity ``d'(t)``. It is a
        # first-class, explicit solver attribute — used both to seed the
        # ``spline``/``linear`` refinements below *and* to derive the
        # along-path predicted-reduction model in ``update`` — rather than an
        # assumption re-derived by hand (and potentially inconsistently) in
        # each consumer.
        self.path: PathStrategy = (
            path if path is not None else (LINEAR_PATH if linear else QUADRATIC_PATH)
        )
        self.has_aux = has_aux
        self._value_and_grad = make_value_and_grad(fun, has_aux=has_aux)
        self.region = resolve_region(region)
        self.oracle = resolve_oracle(oracle, history_size=history_size)
        # Per-layer partitioning is a *cross-cutting solver* concern: the
        # single oracle above stays completely unaware of it. When
        # ``partition_sizes`` is supplied, the solver splits the flat
        # params/grad (and probe buffers) into contiguous segments and drives
        # the oracle independently on each segment, holding a tuple of
        # per-segment oracle states and concatenating the per-segment t=1
        # endpoints back into the full direction. This matters for QN oracles
        # (each layer keeps its own curvature history, so incompatible
        # per-layer scales never mix) and is a harmless no-op for first-order
        # oracles (their per-coordinate moments are identical either way).
        self.partition_sizes = (
            tuple(int(s) for s in partition_sizes)
            if partition_sizes is not None
            else None
        )
        if self.partition_sizes is not None:
            # Static cumulative offsets delimiting each segment; kept as plain
            # Python ints so all slicing stays jit/vmap/grad friendly.
            self._partition_offsets = tuple(
                int(o) for o in jnp.cumsum(jnp.asarray((0,) + self.partition_sizes))
            )
        else:
            self._partition_offsets = None
        self.feed_probes_to_oracle = feed_probes_to_oracle
        # When feeding probes, only admit those that (a) strictly *decrease* the
        # objective relative to the current iterate and (b) lie on the accepted
        # side of the path (their step does not overshoot the accepted step).
        # The prior benchmark (docs/...144249.analysis.md, §4) showed that
        # feeding *rejected* line-search probes injects non-representative
        # (s, y) curvature pairs that pollute the L-BFGS history and cause
        # catastrophic stalls. Gating on descent is the documented fix: only
        # genuinely improving probes enrich the curvature memory.
        self.probe_descent_gate = probe_descent_gate
        self.max_probes = max_probes
        # Maximum path parameter ``t`` the line search may explore. Values
        # greater than 1 enable *extrapolation* past the oracle endpoint. The
        # backtracking/Armijo family grows the step (capped here) before
        # shrinking; the bisection search caps its bracket expansion here.
        self.max_t = max_t

        if line_search not in LINE_SEARCHES:
            raise ValueError(
                f"Unknown line_search: {line_search!r}. "
                f"Available: {sorted(LINE_SEARCHES)}."
            )
        if self.spline and self.linear:
            raise ValueError(
                "spline and linear are mutually exclusive path refinements; "
                "enable at most one."
            )
        # The spline refinement is orthogonal to the chosen line search: rather
        # than replacing it, it *wraps* it. The spline is an expanded definition
        # of the curve — it reuses every probe (with its gradient) as a control
        # point of a cubic Hermite spline along the consistent path, then tries
        # to improve on the inner search's accepted point. It composes with any
        # line search.
        base_ls = LINE_SEARCHES[line_search]
        opts = self.line_search_options
        # The path strategy is a first-class, explicit choice threaded into
        # the inner line search *unconditionally* — not only when a spline/
        # linear refinement wraps it. This makes the quadratic path an
        # alternative form of path construction on equal footing with linear
        # and spline, rather than an implicit default the line search happens
        # to fall back to. (Fixes the long-standing TODO: the quadratic path
        # is NOT a "default to be wrapped".)
        opts = {**opts, "path": self.path}
        # Line searches that support extrapolation (t > 1) accept ``max_step``.
        # Only inject it for those to avoid passing an unexpected kwarg to
        # searches that don't accept it. The user may still override via
        # ``line_search_options``.
        if "max_step" not in opts:
            opts = {**opts, "max_step": self.max_t}
        # When feeding probes to the oracle, size the line-search probe buffers
        # to ``max_probes`` so they match the oracle's replay capacity.
        if self.feed_probes_to_oracle:
            opts = {**opts, "max_probes": self.max_probes}
        else:
            # Probes are unused downstream; disable recording so the inner
            # line-search ``while_loop`` skips the (max_probes, n) scratch.
            opts = {**opts, "record_probes": False}
        if opts:
            base_ls = partial(base_ls, **opts)
        # The linear refinement is, like the spline, an orthogonal *path*
        # choice that wraps whatever inner line search was selected: it
        # samples the straight chord to the oracle endpoint (throwing out
        # gradient/curvature information) and keeps the best feasible point.
        # Each branch selects a *path construction*, not a base-plus-optional-
        # refinement. The quadratic path is a first-class primitive that needs
        # no wrapper: the inner line search already traverses ``self.path``
        # (bound above), so there is nothing to refine it *toward*. The spline
        # and linear cases are genuinely different constructions that reuse the
        # inner search as a baseline and then probe additional points on the
        # same curve.
        if self.spline:
            from qqn_jax.paths.spline import spline_wrap
            self._ls = spline_wrap(base_ls, path=self.path)
        elif self.linear:
            from qqn_jax.paths.linear import linear_wrap
            self._ls = linear_wrap(base_ls, path=self.path)
        else:
            # Direct quadratic (or explicitly-supplied) path construction: the
            # line search itself traverses ``self.path`` via the ``path`` kwarg
            # bound into ``base_ls`` above. No wrapping is needed or sensible —
            # this is the primitive, not a default awaiting refinement.
            self._ls = base_ls

    # --- Internal helpers -------------------------------------------------

    def _eval(self, params, *args):
        """Evaluate value and grad, splitting off aux if present."""
        if self.has_aux:
            (value, aux), grad = self._value_and_grad(params, *args)
        else:
            value, grad = self._value_and_grad(params, *args)
            aux = None
        return value, grad, aux

    # --- Per-layer partitioning helpers -----------------------------------
    def _segments(self, x):
        """Split a flat ``(n,)`` array into the configured contiguous
        segments (static offsets -> jit/vmap/grad safe)."""
        assert self.partition_sizes is not None
        assert self._partition_offsets is not None
        off = self._partition_offsets
        return [x[off[i] : off[i + 1]] for i in range(len(self.partition_sizes))]

    def _oracle_init(self, params):
        """Initialize the oracle state, respecting partitioning.
        Returns a single oracle state when unpartitioned, or a tuple of
        per-segment oracle states when ``partition_sizes`` is set."""
        if self.partition_sizes is None:
            return self.oracle.init(params)
        return tuple(self.oracle.init(seg) for seg in self._segments(params))

    def _oracle_direction(self, params, grad, oracle_state):
        """Compute the oracle's t=1 endpoint, respecting partitioning.
        When partitioned, the oracle is driven independently on each segment
        and the per-segment endpoints are concatenated back into the full
        direction. The returned oracle state mirrors the input structure."""
        if self.partition_sizes is None:
            return self.oracle.direction(params, grad, oracle_state)
        p_segs = self._segments(params)
        g_segs = self._segments(grad)
        dirs = []
        new_states = []
        for p_i, g_i, s_i in zip(p_segs, g_segs, oracle_state):
            d_i, ns_i = self.oracle.direction(p_i, g_i, s_i)
            dirs.append(d_i)
            new_states.append(ns_i)
        return jnp.concatenate(dirs, axis=0), tuple(new_states)

    def _slice_oracle_info(self, info, i):
        """Project an ``OracleInfo`` onto segment ``i``.
        Flat per-iterate fields (params/new_params/grad/new_grad) are sliced
        to the segment; probe buffers (shape ``(k, n)``) are sliced along
        their parameter axis. Scalar / mask fields (t, step_size,
        probe_valid, probe_alphas) are shared verbatim."""
        assert self._partition_offsets is not None
        off = self._partition_offsets
        lo, hi = off[i], off[i + 1]

        def seg(v):
            return None if v is None else v[lo:hi]

        def seg_probe(v):
            return None if v is None else v[:, lo:hi]

        return OracleInfo(
            params=seg(info.params),
            new_params=seg(info.new_params),
            grad=seg(info.grad),
            new_grad=seg(info.new_grad),
            t=info.t,
            step_size=info.step_size,
            probe_params=seg_probe(info.probe_params),
            probe_grads=seg_probe(info.probe_grads),
            probe_valid=info.probe_valid,
            probe_alphas=info.probe_alphas,
        )

    def _oracle_update(self, oracle_state, info):
        """Update the oracle state, respecting partitioning."""
        if self.partition_sizes is None:
            return self.oracle.update(oracle_state, info)
        return tuple(
            self.oracle.update(s_i, self._slice_oracle_info(info, i))
            for i, s_i in enumerate(oracle_state)
        )

    def _plain_value_and_grad(self, params, *args):
        """Value-and-grad returning only ``(value, grad)`` for line search."""
        if self.has_aux:
            (value, _aux), grad = self._value_and_grad(params, *args)
        else:
            value, grad = self._value_and_grad(params, *args)
        return value, grad

    # --- JAXopt-style interface ------------------------------------------

    def init_state(self, params, *args) -> QQNState:
        """Initialize solver state at ``params``."""
        value, grad, aux = self._eval(params, *args)
        oracle_state = self._oracle_init(params)
        error = tree_l2_norm(grad)
        region_state = self.region.init(params)
        return QQNState(
            iter=jnp.asarray(0, jnp.int32),
            value=value,
            grad=grad,
            oracle_state=oracle_state,
            step_size=jnp.asarray(1.0),
            error=error,
            done=error <= self.tol,
            aux=aux,
            region_state=region_state,
            # init_state performs exactly one value-and-grad evaluation.
            num_evals=jnp.asarray(1, jnp.int32),
            qn_slope=jnp.asarray(0.0, dtype=value.dtype),
            ls_success=jnp.asarray(True),
            last_reduction=jnp.asarray(0.0, dtype=value.dtype),
        )

    def update(self, params, state: QQNState, *args):
        """Perform a single QQN iteration.

        A *single* line search traverses the quadratic path ``d(t)`` over the
        parameter ``t ∈ [0, 1]``. The points along the path are states, not
        directions to be re-searched: the search selects one ``t`` (the step
        size along the curve) and the corresponding state ``x + d(t)`` is the
        accepted iterate.

        Returns ``(new_params, new_state)``.
        """
        grad = state.grad

        # 1. Oracle: L-BFGS direction (-H∇f), the t=1 endpoint of the path.
        qn_dir, _ = self._oracle_direction(params, grad, state.oracle_state)
        # Diagnostic: directional derivative along the oracle direction. A
        # non-negative value means the oracle handed back a non-descent
        # direction at t=1 (degenerate curvature) — worth surfacing.
        qn_slope = jnp.asarray(tree_vdot(grad, qn_dir), dtype=state.value.dtype)

        # 2. Gradient: steepest descent direction (-∇f), the path's tangent.
        # Note: grad_dir = -grad is never materialized; the only consumer is
        # the directional-derivative model below, where ⟨∇f, grad_dir⟩ = -‖∇f‖².
        # Optional probe recorder: wrap the value-and-grad handed to the line
        # search so every evaluated (params, grad) is captured into a
        # fixed-size circular scratch buffer. We thread the buffer through a
        # Python list closure over a JAX-carried state to stay JIT-safe: the
        # buffer itself is built as outputs and re-derived deterministically.
        #
        # Because line searches are jitted ``while_loop``s, a Python-mutating
        # closure won't work. Instead we re-evaluate the probe points after the
        # search using the alphas the search reports is not generally possible
        # (only the accepted alpha is returned). We therefore record probes via
        # a stateful host-side buffer is not jit-safe either. The robust,
        # JIT-compatible route is to have the recording wrapper write into a
        # ref-like carry — which JAX lacks — so we instead reconstruct probes
        # from the spline/inner contract below.

        # 3. Single line search along the quadratic path.
        #    The "direction" handed to the line search is the path itself,
        #    parameterized so that step size ``t`` traces d(t). The search
        #    walks the curve directly; each probe ``x + d(t)`` is a state.

        res = self._ls(
            self._plain_value_and_grad,
            params,
            qn_dir,
            state.value,
            grad,
            *args,
            region=self.region,
            region_state=state.region_state,
        )

        new_params = res.new_params
        new_value = res.new_value
        new_grad = res.new_grad
        step_size = res.step_size
        best_t = step_size

        # Recompute aux at the accepted point if needed.
        if self.has_aux:
            # We already have new_value/new_grad from the line search; only the
            # aux is missing. Call ``fun`` directly (no grad) to avoid a second
            # backward pass per iteration.
            _, aux = self.fun(new_params, *args)
        else:
            aux = None

        # Update the oracle state (e.g. L-BFGS curvature pair, momentum).
        # When enabled, forward every (params, grad) evaluated *during* the
        # line search into the oracle's curvature memory — not just the
        # accepted point. The probe buffers are fixed-size and fully JIT/vmap
        # compatible (see LineSearchResult.probe_*).
        # ``extra_recovery_evals`` accounts for any forward passes spent
        # recovering probe objective values for the descent gate. It MUST be
        # bound on every control-flow path (it is summed into step_evals
        # below), so initialize it to zero here.
        extra_recovery_evals = jnp.asarray(0, jnp.int32)
        if self.feed_probes_to_oracle and res.probe_params is not None:
            probe_valid = res.probe_valid
            # Enforce the "accepted side" rule promised in the docstring: a
            # probe must not overshoot the accepted step. Probes beyond
            # ``step_size`` sit on a rejected stretch of the ray and inject
            # non-representative curvature. Gate them out here (the oracle then
            # only replays a small, well-spaced, *closer* subset).
            if res.probe_alphas is not None:
                on_accepted_side = res.probe_alphas <= step_size
                probe_valid = jnp.logical_and(probe_valid, on_accepted_side)
            if self.probe_descent_gate and res.probe_values is not None:
                # Descent gate: only admit probes whose objective value strictly
                # improves on the *current* iterate. The line search already
                # evaluated f at every probe (``res.probe_values``), so the gate
                # is free — no extra forward passes. (The previous code paid an
                # extra ``vmap`` of ``max_probes`` forward evaluations per
                # iteration to recover values the line search had thrown away.)
                descends = res.probe_values < state.value
                probe_valid = jnp.logical_and(probe_valid, descends)
            elif self.probe_descent_gate:
                # The (e.g. spline-wrapped) line search recorded probe params and
                # grads but not their objective values. Recover the values with a
                # single vmapped forward pass so the descent gate can still admit
                # only genuinely-improving probes (the documented fix against
                # history-polluting rejected probes).
                probe_values = jax.vmap(
                    lambda p: self._plain_value_and_grad(p, *args)[0]
                )(res.probe_params)
                descends = probe_values < state.value
                probe_valid = jnp.logical_and(probe_valid, descends)
                # Override the zero default: we spent one forward pass per probe
                # slot recovering values the line search did not retain.
                extra_recovery_evals = jnp.asarray(res.probe_params.shape[0], jnp.int32)
            oracle_info = OracleInfo(
                params=params,
                new_params=new_params,
                grad=grad,
                new_grad=new_grad,
                t=best_t,
                step_size=step_size,
                probe_params=res.probe_params,
                probe_grads=res.probe_grads,
                probe_valid=probe_valid,
                probe_alphas=res.probe_alphas,
            )
        else:
            oracle_info = OracleInfo(
                params=params,
                new_params=new_params,
                grad=grad,
                new_grad=new_grad,
                t=best_t,
                step_size=step_size,
            )
        new_oracle_state = self._oracle_update(state.oracle_state, oracle_info)
        # Update region state (e.g. adaptive trust-region radius).
        actual_reduction = state.value - new_value
        # Honest predicted reduction from the along-path model.
        #
        # For *any* differentiable path ``d(t)`` with ``d(0) = 0``, the model
        # that holds the gradient fixed at its ``t = 0`` value gives
        #   pred(t) = -∫₀ᵗ ⟨∇f, d'(τ)⟩ dτ = -⟨∇f, d(t)⟩
        # by the fundamental theorem of calculus — independent of the
        # specific curve. Rather than re-deriving (and hand-expanding) this
        # per path, as before, we materialize ``d(t)`` through
        # ``self.path.offset`` — the same first-class ``PathStrategy``
        # component (``qqn_jax.paths.quadratic.QUADRATIC_PATH`` by default)
        # that built every probe the line search actually evaluated. This
        # keeps the trust-region model exactly in sync with whichever curve
        # the solver is configured to traverse, instead of hard-coding the
        # quadratic path's closed form here (which silently diverged from
        # the accepted point whenever ``linear=True`` routed the step
        # through a different curve).
        grad_dir = tree_negative(grad)
        d_t = self.path.offset(best_t, grad_dir, qn_dir)
        pred_reduction = jnp.asarray(-tree_vdot(grad, d_t))
        # The model reduction is non-negative whenever the step descends along
        # the path (which the line search guarantees via sufficient decrease).
        # A tiny positive epsilon avoids a 0/0 ρ when the step is degenerate.
        eps_pred = jnp.asarray(1e-16, dtype=pred_reduction.dtype)
        pred_reduction = jnp.maximum(pred_reduction, eps_pred)
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=pred_reduction,
            actual_reduction=actual_reduction,
            t=best_t,
            step_size=step_size,
        )
        new_region_state = self.region.update(state.region_state, info)

        error = tree_l2_norm(new_grad)
        # Terminate (rather than spin to maxiter) if the iterate diverges to a
        # non-finite state — a single bad start in a vmap batch otherwise wastes
        # the whole batch's remaining iterations on NaN arithmetic.
        finite = jnp.logical_and(jnp.isfinite(new_value), jnp.isfinite(error))
        done = jnp.logical_or(error <= self.tol, jnp.logical_not(finite))
        # --- Eval accounting -------------------------------------------------
        ls_evals = res.num_evals
        if ls_evals is None:
            # Defensive: a custom line search may not report evals. Assume the
            # minimum (one accepted-point evaluation) so totals never decrease.
            ls_evals = jnp.asarray(1, jnp.int32)
        aux_evals = (
            jnp.asarray(1, jnp.int32) if self.has_aux else jnp.asarray(0, jnp.int32)
        )
        step_evals = ls_evals + aux_evals + extra_recovery_evals
        new_num_evals = state.num_evals + step_evals

        new_state = QQNState(
            iter=state.iter + 1,
            value=new_value,
            grad=new_grad,
            oracle_state=new_oracle_state,
            step_size=step_size,
            error=error,
            done=done,
            aux=aux,
            region_state=new_region_state,
            num_evals=new_num_evals,
            qn_slope=qn_slope,
            ls_success=res.done,
            last_reduction=actual_reduction,
        )
        return new_params, new_state

    def run(self, init_params, *args):
        """Run QQN to convergence (or ``maxiter``).

        Uses ``lax.while_loop`` so the whole optimization is JIT/vmap
        compatible.
        """
        state = self.init_state(init_params, *args)

        def cond(carry):
            params, state = carry
            not_converged = jnp.logical_not(state.done)
            not_maxiter = state.iter < self.maxiter
            return jnp.logical_and(not_converged, not_maxiter)

        def body(carry):
            params, state = carry
            new_params, new_state = self.update(params, state, *args)
            return new_params, new_state

        final_params, final_state = jax.lax.while_loop(cond, body, (init_params, state))
        return final_params, final_state
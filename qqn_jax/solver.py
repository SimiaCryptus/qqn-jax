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
from qqn_jax.line_search.util import make_scalar_problem


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

        self.path: PathStrategy = (
            path if path is not None else (LINEAR_PATH if linear else QUADRATIC_PATH)
        )
        self.has_aux = has_aux
        self._value_and_grad = make_value_and_grad(fun, has_aux=has_aux)
        self.region = resolve_region(region)
        self.oracle = resolve_oracle(oracle, history_size=history_size)

        self.partition_sizes = (
            tuple(int(s) for s in partition_sizes)
            if partition_sizes is not None
            else None
        )
        if self.partition_sizes is not None:
            self._partition_offsets = tuple(
                int(o) for o in jnp.cumsum(jnp.asarray((0,) + self.partition_sizes))
            )
        else:
            self._partition_offsets = None

        self.feed_probes_to_oracle = feed_probes_to_oracle
        self.probe_descent_gate = probe_descent_gate
        self.max_probes = max_probes
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

        base_ls = LINE_SEARCHES[line_search]
        opts = self.line_search_options

        if "max_step" not in opts:
            opts = {**opts, "max_step": self.max_t}

        if self.feed_probes_to_oracle:
            opts = {**opts, "max_probes": self.max_probes}
        else:
            opts = {**opts, "record_probes": False}

        if self.spline:
            from qqn_jax.paths.spline import spline_wrap

            # Wrappers take the multidimensional signature and build the
            # scalar problem themselves; forward opts to the bare inner
            # search only (path is threaded by the wrapper).
            inner = partial(base_ls, **opts) if opts else base_ls
            self._ls = spline_wrap(inner, path=self.path)
            self._ls_is_wrapped = True
        elif self.linear:
            from qqn_jax.paths.linear import linear_wrap

            inner = partial(base_ls, **opts) if opts else base_ls
            self._ls = linear_wrap(inner, path=self.path)
            self._ls_is_wrapped = True
        else:
            self._ls = partial(base_ls, **opts) if opts else base_ls
            self._ls_is_wrapped = False

    def _eval(self, params, *args):
        """Evaluate value and grad, splitting off aux if present."""
        if self.has_aux:
            (value, aux), grad = self._value_and_grad(params, *args)
        else:
            value, grad = self._value_and_grad(params, *args)
            aux = None
        return value, grad, aux

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

        qn_dir, _ = self._oracle_direction(params, grad, state.oracle_state)

        qn_slope = jnp.asarray(tree_vdot(grad, qn_dir), dtype=state.value.dtype)

        if self._ls_is_wrapped:
            # Path wrappers still take the multidimensional interface and
            # build the scalar problem internally.
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
        else:
            # Bare line searches receive *only* the prepared 1-D problem:
            # the solver folds the path + region into ``eval_at`` here, so
            # the line search is entirely path-agnostic.
            eval_at, slope0 = make_scalar_problem(
                self._plain_value_and_grad,
                params,
                grad,
                qn_dir,
                self.region,
                state.region_state,
                self.path,
                *args,
            )
            res = self._ls(
                eval_at,
                params,
                state.value,
                grad,
                slope0,
            )

        new_params = res.new_params
        new_value = res.new_value
        new_grad = res.new_grad
        step_size = res.step_size
        best_t = step_size

        if self.has_aux:
            _, aux = self.fun(new_params, *args)
        else:
            aux = None

        extra_recovery_evals = jnp.asarray(0, jnp.int32)
        if self.feed_probes_to_oracle and res.probe_params is not None:
            probe_valid = res.probe_valid

            if res.probe_alphas is not None:
                on_accepted_side = res.probe_alphas <= step_size
                probe_valid = jnp.logical_and(probe_valid, on_accepted_side)
            if self.probe_descent_gate and res.probe_values is not None:
                descends = res.probe_values < state.value
                probe_valid = jnp.logical_and(probe_valid, descends)
            elif self.probe_descent_gate:
                probe_values = jax.vmap(
                    lambda p: self._plain_value_and_grad(p, *args)[0]
                )(res.probe_params)
                descends = probe_values < state.value
                probe_valid = jnp.logical_and(probe_valid, descends)

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

        actual_reduction = state.value - new_value

        grad_dir = tree_negative(grad)
        d_t = self.path.offset(best_t, grad_dir, qn_dir)
        pred_reduction = jnp.asarray(-tree_vdot(grad, d_t))

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

        finite = jnp.logical_and(jnp.isfinite(new_value), jnp.isfinite(error))
        done = jnp.logical_or(error <= self.tol, jnp.logical_not(finite))

        ls_evals = res.num_evals
        if ls_evals is None:
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

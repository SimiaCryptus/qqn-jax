"""QQN (Quadratic Quasi-Newton) solver.

QQN constructs the quadratic interpolation path

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

blending the steepest-descent direction (``-∇f``) with the L-BFGS direction
(``-H∇f``), and performs a line search over this path.

The solver follows the JAXopt-style ``init_state`` / ``update`` / ``run``
interface and keeps all state in JIT-compatible NamedTuples.
"""

from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional
import inspect
import random


import jax
import jax.numpy as jnp

from qqn_jax.oracles.strategy import resolve_oracle
from qqn_jax.oracles.oracle import OracleInfo
from qqn_jax.line_search import LINE_SEARCHES
from qqn_jax.paths import SPLINE_PATH
from qqn_jax.paths.spline import spline_refine
from qqn_jax.paths.base import make_evaluator
from qqn_jax.paths.linear import LINEAR_PATH, linear_refine
from qqn_jax.paths.quadratic import QUADRATIC_PATH
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
        num_evals: cumulative number of objective/gradient evaluations.
        qn_slope: directional derivative ``∇f·(-H∇f)`` of the quasi-Newton
            direction at the current iterate.
        ls_success: whether the last line search reported success.
        last_reduction: objective reduction achieved by the last accepted step.
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
        line_search: name of the line-search strategy.
             The line search's role in QQN is *generally permissive*: since the
             quadratic path ``d(t)`` already encodes the curvature, the search
             only needs to pick a step that makes sufficient progress along the
             curve rather than solve the 1-D subproblem exactly.
        line_search_options: optional dict of keyword arguments forwarded to
             the chosen line-search function (e.g. ``c1``, ``c2``, ``max_iter``,
             ``init_step``, ``shrink``, ``step_size``). These override the
             line-search defaults.
        path_strategy: string selector for the curve/refinement the solver
             traverses. One of:
               * ``"quadratic"`` (default) — the canonical quadratic path
                 (``qqn_jax.paths.quadratic.QUADRATIC_PATH``) with no extra
                 line-search refinement.
               * ``"linear"`` — the straight chord
                 (``qqn_jax.paths.linear.LINEAR_PATH``); the chosen line search
                 is wrapped with the value-only linear-chord refinement (see
                 ``qqn_jax.paths.linear``).
               * ``"spline"`` — the quadratic path with the cubic Hermite
                 spline refinement enabled: every probe along the (consistent)
                 path is reused as a control point and the spline's stationary
                 points guide the search. It composes with any chosen line
                 search.
        spline_max_control_points: hard upper bound (minimum 2) on the number
             of control points the spline refinement may accumulate. Bounding
             the spline's complexity prevents a fractal / Zeno's-paradox effect
             in which ever-finer subdivision of the interval stalls line-search
             progress. When the seed control points (origin, probes, endpoint)
             plus ``spline_refine_rounds`` would exceed this bound, the seed
             buffer is trimmed to its lowest-fitness points and the number of
             refinement rounds is capped accordingly. Defaults to ``32``.

        region: optional trust-region / projective-region strategy selector.
             Passed to ``qqn_jax.regions.strategy.resolve_region``. When
             ``None`` an identity (no-op) region is used. The region can
             rescale or project the path offset before evaluation and is
             updated each iteration from the predicted vs. actual reduction.
        oracle: string selector (or object) for the quasi-Newton oracle that
             supplies the ``-H∇f`` endpoint of the path. Defaults to
             ``"lbfgs"``. Resolved via
             ``qqn_jax.oracles.strategy.resolve_oracle`` using
             ``history_size`` as the L-BFGS memory.
        feed_probes_to_oracle: when ``True`` the line-search probe points
             (and their gradients) along the path are recorded and fed back to
             the oracle's ``update``, enriching its curvature history beyond
             the single accepted step. Only probes on the accepted side of the
             step (``alpha <= step_size``) are marked valid.
        max_probes: maximum number of probe points buffered per iteration when
             ``feed_probes_to_oracle`` is enabled.
        max_t: upper bound on the path/line-search parameter ``t`` (forwarded
             to the line search as ``max_step``). Defaults to ``1000.0``.
        partition_sizes: optional tuple of contiguous segment sizes that
             partition a flat ``(n,)`` parameter vector. When set, the oracle
             is driven independently on each segment (a block-diagonal
             quasi-Newton approximation) and the per-segment endpoints are
             concatenated back into the full direction.
         remember_step_size: when ``True`` the line search's ``init_step`` for
              each iteration is set to the previous iteration's accepted step
              size (the path parameter ``t``). This can warm-start the search
              and reduce the number of probes when successive steps have
              similar magnitude. Defaults to ``False``.
        seed: base RNG seed forwarded to line searches that expose a
             ``seed`` argument (e.g. ``strong_wolfe_search``, whose
             stochastic/Metropolis acceptance depends on it). The seed is
             rotated deterministically every iteration (``seed + iter``) so
             each iteration's stochastic decisions use a distinct, but fully
             reproducible, key. When ``None`` (the default) a random base
             seed is drawn once at construction time via :mod:`random`.
             Line searches that do not accept a ``seed`` keyword are
             unaffected.
        has_aux: whether ``fun`` returns auxiliary data.
    """

    def __init__(
        self,
        fun: Callable,
        maxiter: int = 100,
        tol: float = 1e-5,
        history_size: int = 10,
        line_search: str = "backtracking",
        line_search_options: Optional[Dict[str, Any]] = None,
        path_strategy: str = "quadratic",
        has_aux: bool = False,
        region=None,
        oracle=None,
        feed_probes_to_oracle: bool = False,
        max_probes: int = 32,
        max_t: float = 1000.0,
        partition_sizes: Optional[tuple[int, ...]] = None,
        spline_refine_rounds: int = 4,
        spline_max_control_points: int = 32,
        remember_step_size: bool = False,
        seed: Optional[int] = None,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.tol = tol
        self.history_size = history_size
        self.line_search = line_search
        self.line_search_options = dict(line_search_options or {})
        self.path_strategy = path_strategy
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
        self.max_probes = max_probes
        self.max_t = max_t
        self.spline_refine_rounds = int(spline_refine_rounds)
        self.spline_max_control_points = max(2, int(spline_max_control_points))
        self.remember_step_size = bool(remember_step_size)
        self.seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        if line_search not in LINE_SEARCHES:
            raise ValueError(
                f"Unknown line_search: {line_search!r}. "
                f"Available: {sorted(LINE_SEARCHES)}."
            )
        base_ls = LINE_SEARCHES[line_search]
        self._ls_supports_seed = "seed" in inspect.signature(base_ls).parameters
        opts = self.line_search_options
        if "max_step" not in opts:
            opts = {**opts, "max_step": self.max_t}

        need_probes = self.feed_probes_to_oracle or path_strategy == "spline"
        if need_probes:
            opts = {**opts, "max_probes": self.max_probes}
        else:
            opts = {**opts, "record_probes": False}
        self._base_ls = base_ls
        self._ls_opts = opts
        inner = partial(base_ls, **opts) if opts else base_ls
        self._inner_search = inner

        self._spline = False
        if path_strategy == "linear":
            self.path = LINEAR_PATH
            self._refine = True
        elif path_strategy == "quadratic":
            self.path = QUADRATIC_PATH
            self._refine = False
        elif path_strategy == "spline":
            self.path = SPLINE_PATH
            self._refine = False
            self._spline = True
        else:
            raise ValueError(f"Unknown path_strategy: {path_strategy!r}. ")

    def _eval(self, params, *args):
        """Evaluate objective value, gradient and auxiliary output.
        Returns ``(value, grad, aux)`` where ``aux`` is ``None`` when the
        solver was constructed with ``has_aux=False``.
        """
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
        """Value-and-grad returning only ``(value, grad)``.
        Drops any auxiliary output so the callable matches the signature
        expected by the line search / path evaluator.
        """
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
        The step proceeds as follows:
          1. Query the oracle for the quasi-Newton endpoint ``-H∇f`` (the
             ``t=1`` point of the path), respecting any partitioning.
          2. Build the scalar 1-D subproblem ``φ(t) = f(x + d(t))`` via the
             configured path and (optional) region projection.
          3. Run the inner line search over ``t``; if ``path_strategy`` is
             ``"linear"`` apply the value-only chord refinement.
          4. Update the oracle (optionally with recorded probe points) and the
             region state from the predicted vs. actual reduction.
          5. Recompute the gradient norm and convergence / finiteness flags.
        Args:
            params: current iterate.
            state: current :class:`QQNState`.
            *args: extra positional arguments forwarded to ``fun``.


        Returns ``(new_params, new_state)``.
        """
        grad = state.grad

        qn_dir, _ = self._oracle_direction(params, grad, state.oracle_state)

        qn_slope = jnp.asarray(tree_vdot(grad, qn_dir), dtype=state.value.dtype)

        dtype = state.value.dtype
        eval_at = make_evaluator(
            self._plain_value_and_grad,
            params,
            grad,
            qn_dir,
            self.region,
            state.region_state,
            self.path,
            *args,
        )
        grad_dir = tree_negative(grad)
        slope0 = tree_vdot(
            grad,
            self.path.velocity(jnp.asarray(0.0, dtype=dtype), grad_dir, qn_dir),
        )
        if self.remember_step_size:
            opts = {**self._ls_opts, "init_step": state.step_size}
            inner_search = partial(self._base_ls, **opts)
        else:
            inner_search = self._inner_search
        if self._ls_supports_seed:
            # Rotate the RNG seed deterministically with the iteration
            # counter so stochastic acceptance decisions (e.g. Metropolis
            # criterion in ``strong_wolfe_search``) differ each iteration
            # while remaining fully reproducible from ``self.seed``.
            iter_seed = jnp.asarray(self.seed, dtype=jnp.uint32) + jnp.asarray(
                state.iter, dtype=jnp.uint32
            )
            inner_search = partial(inner_search, seed=iter_seed)
        res = inner_search(
            eval_at,
            params,
            state.value,
            grad,
            slope0,
        )
        if self._refine:
            res = linear_refine(res, eval_at, dtype)
        elif self._spline:
            res = spline_refine(
                res,
                eval_at,
                self.path,
                grad_dir,
                qn_dir,
                state.value,
                slope0,
                dtype,
                rounds=self.spline_refine_rounds,
                max_control_points=self.spline_max_control_points,
            )

        new_params = res.new_params
        new_value = res.new_value
        new_grad = res.new_grad
        step_size = res.step_size
        best_t = step_size

        step_finite = jnp.logical_and(
            jnp.isfinite(new_value),
            jnp.all(jnp.isfinite(new_grad)),
        )
        # Honor the line search's own acceptance criterion (`res.done`), which
        # already encodes stochastic (temperature-based) acceptance of
        # non-improving steps. Without this, a step accepted by the
        # Metropolis rule (value may be worse) would be discarded here as
        # "not improving", marking the iteration as stalled and terminating
        # the run early instead of producing the expected fuzzy/spiky
        # convergence curve at high temperature.
        accept = jnp.logical_and(
           step_finite, jnp.logical_or(new_value <= state.value, res.done)
        )

        gnorm_sq = tree_vdot(grad, grad)
        safe_scale = jnp.asarray(1.0, dtype=dtype) / (
            jnp.asarray(1.0, dtype=dtype) + gnorm_sq
        )
        fb_params = params + safe_scale * grad_dir
        fb_value, fb_grad = self._plain_value_and_grad(fb_params, *args)
        fb_finite = jnp.logical_and(
            jnp.isfinite(fb_value), jnp.all(jnp.isfinite(fb_grad))
        )
        fb_accept = jnp.logical_and(
            jnp.logical_not(accept),
            jnp.logical_and(fb_finite, fb_value < state.value),
        )
        new_params = jnp.where(fb_accept, fb_params, new_params)
        new_value = jnp.where(fb_accept, fb_value, new_value)
        new_grad = jnp.where(fb_accept, fb_grad, new_grad)
        accept = jnp.logical_or(accept, fb_accept)
        step_size = jnp.where(fb_accept, safe_scale, step_size)
        new_params = jnp.where(accept, new_params, params)
        new_value = jnp.where(accept, new_value, state.value)
        new_grad = jnp.where(accept, new_grad, grad)
        step_size = jnp.where(
            accept, step_size, jnp.asarray(0.0, dtype=step_size.dtype)
        )
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
            grad=grad,
            new_grad=new_grad,
        )
        new_region_state = self.region.update(state.region_state, info)

        error = tree_l2_norm(new_grad)

        finite = jnp.logical_and(jnp.isfinite(new_value), jnp.isfinite(error))
        converged = error <= self.tol

        stalled = jnp.logical_not(accept)
        done = jnp.logical_or(
            converged,
            jnp.logical_or(jnp.logical_not(finite), stalled),
        )

        ls_evals = res.num_evals
        if ls_evals is None:
            ls_evals = jnp.asarray(1, jnp.int32)
        aux_evals = (
            jnp.asarray(1, jnp.int32) if self.has_aux else jnp.asarray(0, jnp.int32)
        )
        step_evals = (
            ls_evals + aux_evals + extra_recovery_evals + jnp.asarray(1, jnp.int32)
        )
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
        Args:
            init_params: starting iterate.
            *args: extra positional arguments forwarded to ``fun``.
        Returns:
            ``(final_params, final_state)`` where ``final_state`` is the
            terminal :class:`QQNState`. Iteration stops when the gradient norm
            falls below ``tol`` (or a non-finite value/error is encountered),
            or after ``maxiter`` iterations.
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
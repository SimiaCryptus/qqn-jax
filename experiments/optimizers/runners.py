"""Canonical optimizer runners returning ``RunResult``.

The three runners (QQN, generic Optax, Optax L-BFGS) share one termination
loop semantics and one milestone tracker (plan.md §8). The genuine
eval-counting logic lives here, once.
"""

import time

import jax
import jax.numpy as jnp
import optax

from qqn_jax import QQN

from experiments.metrics.milestones import converged, update_milestones
from experiments.metrics.result import RunResult

__all__ = ["run_qqn", "run_optax", "run_optax_lbfgs"]


def run_qqn(loss_fn, params0, maxiter, stop=None, **qqn_kwargs):
    """Run a configurable QQN variant; returns a ``RunResult``."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    solver = QQN(loss_fn, maxiter=maxiter, **qqn_kwargs)
    state = solver.init_state(params0)
    params = params0
    history = [float(state.value)]
    times = [0.0]
    # ``state.num_evals`` is the TRUE cumulative evaluation count (line-search
    # probes, spline probes, recovery evals, plus the init_state eval).
    eval_counts = [int(state.num_evals)]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    update_milestones(
        milestones, milestone_hits, history[-1], 0, 0.0, int(state.num_evals)
    )
    t0 = time.perf_counter()
    update = jax.jit(solver.update)
    for it in range(maxiter):
        params, state = update(params, state)
        history.append(float(state.value))
        now = time.perf_counter() - t0
        times.append(now)
        gnorm = float(state.error)
        cum_evals = int(state.num_evals)
        eval_counts.append(cum_evals)
        update_milestones(
            milestones, milestone_hits, history[-1], it + 1, now, cum_evals
        )
        if iters_to_target is None and converged(history[-1], gnorm, f_target, gtol):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
        if bool(state.done):
            break
    wall = time.perf_counter() - t0
    evals_to_target = None if iters_to_target is None else eval_counts[iters_to_target]
    return RunResult(
        params=params,
        history=history,
        times=times,
        wall=wall,
        iters_to_target=iters_to_target,
        time_to_target=time_to_target,
        milestone_hits=milestone_hits,
        evals_to_target=evals_to_target,
    )


def run_optax(loss_fn, params0, optimizer, maxiter, stop=None):
    """Run a generic Optax optimizer; returns a ``RunResult``."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    opt_state = optimizer.init(params0)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, jnp.linalg.norm(grad)

    params = params0
    history = [float(loss_fn(params))]
    times = [0.0]
    eval_counts = [0]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    update_milestones(milestones, milestone_hits, history[-1], 0, 0.0, 0)
    t0 = time.perf_counter()
    for it in range(maxiter):
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)
        cum_evals = it + 1  # one value+grad call per step
        eval_counts.append(cum_evals)
        update_milestones(
            milestones, milestone_hits, history[-1], it + 1, now, cum_evals
        )
        if iters_to_target is None and converged(
            history[-1], float(gnorm), f_target, gtol
        ):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
    wall = time.perf_counter() - t0
    evals_to_target = None if iters_to_target is None else eval_counts[iters_to_target]
    return RunResult(
        params=params,
        history=history,
        times=times,
        wall=wall,
        iters_to_target=iters_to_target,
        time_to_target=time_to_target,
        milestone_hits=milestone_hits,
        evals_to_target=evals_to_target,
    )


def _extract_ls_evals(opt_state):
    """Walk the (nested) optax state for a ``num_linesearch_steps`` field."""
    for leaf in jax.tree_util.tree_leaves_with_path(opt_state):
        path, val = leaf
        name = path[-1].name if path and hasattr(path[-1], "name") else ""
        if "linesearch" in str(name).lower() and "step" in str(name).lower():
            try:
                return int(val)
            except Exception:
                return None
    return None


def run_optax_lbfgs(loss_fn, params0, maxiter, stop=None):
    """Run Optax's L-BFGS (with zoom line search); returns a ``RunResult``."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(params0)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, jnp.linalg.norm(grad)

    params = params0
    history = [float(loss_fn(params))]
    times = [0.0]
    eval_counts = [0]
    cum_evals = 0
    _ls_unavailable = False
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    update_milestones(milestones, milestone_hits, history[-1], 0, 0.0, 0)
    t0 = time.perf_counter()
    for it in range(maxiter):
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)
        ls_steps = None if _ls_unavailable else _extract_ls_evals(opt_state)
        if ls_steps is None:
            _ls_unavailable = True
            cum_evals += 3  # conservative fallback: ~2 probes/step + base call
        else:
            cum_evals += 1 + max(ls_steps, 0)
        eval_counts.append(cum_evals)
        update_milestones(
            milestones, milestone_hits, history[-1], it + 1, now, cum_evals
        )
        if iters_to_target is None and converged(
            history[-1], float(gnorm), f_target, gtol
        ):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
    wall = time.perf_counter() - t0
    evals_to_target = None if iters_to_target is None else eval_counts[iters_to_target]
    return RunResult(
        params=params,
        history=history,
        times=times,
        wall=wall,
        iters_to_target=iters_to_target,
        time_to_target=time_to_target,
        milestone_hits=milestone_hits,
        evals_to_target=evals_to_target,
    )

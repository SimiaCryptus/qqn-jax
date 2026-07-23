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

    eval_counts = [int(state.num_evals)]

    fwd_counts = [int(state.num_evals)]
    bwd_counts = [int(state.num_evals)]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    update_milestones(
        milestones,
        milestone_hits,
        history[-1],
        0,
        0.0,
        int(state.num_evals),
        fwd=int(state.num_evals),
        bwd=int(state.num_evals),
    )
    update = jax.jit(solver.update)
    # Warmup: trigger JIT compilation of `update` *outside* the timed region
    # so the (potentially multi-second) trace/compile cost is not attributed
    # to the first iteration's wall time. We run one real step, block until
    # it is materialized, then start the clock.
    params, state = update(params, state)
    jax.block_until_ready((params, state))
    history.append(float(state.value))
    cum_evals = int(state.num_evals)
    eval_counts.append(cum_evals)
    fwd_counts.append(cum_evals)
    bwd_counts.append(cum_evals)
    t0 = time.perf_counter()
    times.append(0.0)
    gnorm = float(state.error)
    update_milestones(
        milestones,
        milestone_hits,
        history[-1],
        1,
        0.0,
        cum_evals,
        fwd=cum_evals,
        bwd=cum_evals,
    )
    if iters_to_target is None and converged(history[-1], gnorm, f_target, gtol):
        iters_to_target = 1
        time_to_target = 0.0
    _warm_done = bool(state.done)
    for it in range(maxiter):
        if it == 0:
            # First real loop slot already consumed by the warmup step above.
            if _warm_done or iters_to_target is not None:
                break
            continue
        params, state = update(params, state)
        history.append(float(state.value))
        now = time.perf_counter() - t0
        times.append(now)
        gnorm = float(state.error)
        cum_evals = int(state.num_evals)
        eval_counts.append(cum_evals)
        fwd_counts.append(cum_evals)
        bwd_counts.append(cum_evals)
        update_milestones(
            milestones,
            milestone_hits,
            history[-1],
            it + 1,
            now,
            cum_evals,
            fwd=cum_evals,
            bwd=cum_evals,
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
        eval_counts=eval_counts,
        fwd_counts=fwd_counts,
        bwd_counts=bwd_counts,
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

    fwd_counts = [0]
    bwd_counts = [0]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    update_milestones(milestones, milestone_hits, history[-1], 0, 0.0, 0, fwd=0, bwd=0)
    # Warmup: compile `step` outside the timed region (JIT compile cost must
    # not be charged to iteration 1's wall time). One real step is taken.
    params, opt_state, value, gnorm = step(params, opt_state)
    jax.block_until_ready((params, opt_state, value, gnorm))
    history.append(float(value))
    eval_counts.append(1)
    fwd_counts.append(1)
    bwd_counts.append(1)
    t0 = time.perf_counter()
    times.append(0.0)
    update_milestones(
        milestones, milestone_hits, history[-1], 1, 0.0, 1, fwd=1, bwd=1
    )
    if iters_to_target is None and converged(
        history[-1], float(gnorm), f_target, gtol
    ):
        iters_to_target = 1
        time_to_target = 0.0
    for it in range(maxiter):
        if it == 0:
            # First loop slot consumed by the warmup step above.
            if iters_to_target is not None:
                break
            continue
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)
        cum_evals = it + 1
        eval_counts.append(cum_evals)
        fwd_counts.append(cum_evals)
        bwd_counts.append(cum_evals)
        update_milestones(
            milestones,
            milestone_hits,
            history[-1],
            it + 1,
            now,
            cum_evals,
            fwd=cum_evals,
            bwd=cum_evals,
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
        eval_counts=eval_counts,
        fwd_counts=fwd_counts,
        bwd_counts=bwd_counts,
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


def run_optax_lbfgs(loss_fn, params0, maxiter, stop=None, memory_size=10):
    """Run Optax's L-BFGS (with zoom line search); returns a ``RunResult``."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer = optax.lbfgs(memory_size=memory_size)
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

    fwd_counts = [0]
    bwd_counts = [0]
    cum_fwd = 0
    cum_bwd = 0
    cum_evals = 0
    iters_to_target = None
    time_to_target = None
    # Adaptive line-search-step estimation. When the optax state exposes a
    # `num_linesearch_steps` field we read it directly and fold it into a
    # running average. When it is *unavailable* (private layout changed, or
    # field absent) we fall back to that running average instead of a fixed
    # constant, so the eval count adapts to the problem's observed behaviour.
    _ls_obs_sum = 0.0
    _ls_obs_count = 0
    _ls_default = 2.0  # prior mean line-search steps before any observation
    def _est_ls_steps(opt_state):
        nonlocal _ls_obs_sum, _ls_obs_count
        observed = _extract_ls_evals(opt_state)
        if observed is not None:
            _ls_obs_sum += max(observed, 0)
            _ls_obs_count += 1
            return max(observed, 0)
        # Adaptive fallback: running average of observed steps, else prior.
        if _ls_obs_count > 0:
            return _ls_obs_sum / _ls_obs_count
        return _ls_default

    milestone_hits = {m: None for m in milestones}
    update_milestones(milestones, milestone_hits, history[-1], 0, 0.0, 0, fwd=0, bwd=0)
    # Warmup: compile `step` outside the timed region.
    params, opt_state, value, gnorm = step(params, opt_state)
    jax.block_until_ready((params, opt_state, value, gnorm))
    history.append(float(value))
    _ls0 = _est_ls_steps(opt_state)
    cum_evals += 1 + _ls0
    cum_fwd += 1 + _ls0
    cum_bwd += 1
    eval_counts.append(int(round(cum_evals)))
    fwd_counts.append(int(round(cum_fwd)))
    bwd_counts.append(int(round(cum_bwd)))
    t0 = time.perf_counter()
    times.append(0.0)
    update_milestones(
        milestones,
        milestone_hits,
        history[-1],
        1,
        0.0,
        int(round(cum_evals)),
        fwd=int(round(cum_fwd)),
        bwd=int(round(cum_bwd)),
    )
    if iters_to_target is None and converged(
        history[-1], float(gnorm), f_target, gtol
    ):
        iters_to_target = 1
        time_to_target = 0.0
    for it in range(maxiter):
        if it == 0:
            # First loop slot consumed by the warmup step above.
            if iters_to_target is not None:
                break
            continue
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)

        ls_steps = _est_ls_steps(opt_state)
        cum_evals += 1 + ls_steps
        cum_fwd += 1 + ls_steps
        cum_bwd += 1
        eval_counts.append(int(round(cum_evals)))
        fwd_counts.append(int(round(cum_fwd)))
        bwd_counts.append(int(round(cum_bwd)))
        update_milestones(
            milestones,
            milestone_hits,
            history[-1],
            it + 1,
            now,
            int(round(cum_evals)),
            fwd=int(round(cum_fwd)),
            bwd=int(round(cum_bwd)),
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
"""The experiment driver: config -> {name: RunResult}, with reporting.

Encapsulates the fairness invariants (plan.md §8): identical init, shared
termination, genuine eval accounting, same loss/data/regularization, and a
profiling span per variant.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qqn_jax.profiling import profile_region

from experiments.data.loaders import load_image_dataset
from experiments.models.mlp import FlatMLP
from experiments.optimizers import runners as _runners
from experiments.optimizers import profiles as _profiles
from experiments.reporting.tables import report_tables
from experiments.reporting.plots import save_plots
from experiments.reporting.axis_analysis import report_axis_analysis
from experiments.reporting.json_export import write_results_json, write_run_json

__all__ = ["run_experiment", "build_model", "enrich"]


class _Ctx:
    """Lightweight context bundle passed to the profile factories."""

    loss_fn: Any
    params0: Any
    maxiter: int
    stop: dict
    sgd_lr: float
    adam_lr: float
    lbfgs_memory_size: int
    run_qqn: Any
    run_optax: Any
    run_optax_lbfgs: Any
    partition_sizes: Any


def build_model(config, dim):
    """Construct the FlatMLP from a config + observed feature dim."""
    return FlatMLP(
        dim=dim,
        hidden_sizes=config.hidden_sizes,
        n_classes=config.n_classes,
        activation_fn=config.activation_fn,
        activation_name=config.activation_name,
    )


def enrich(result, model, data, config):
    """Fill the derived RunResult fields (acc, AUC, ms/it, target iters)."""
    X_train, y_train, X_test, y_test = data
    result.train_acc = float(model.accuracy(result.params, X_train, y_train))
    result.test_acc = float(model.accuracy(result.params, X_test, y_test))
    result.final_loss = result.history[-1]
    result.best_loss = min(result.history)
    result.iters = len(result.history) - 1
    result.reached = result.iters_to_target is not None

    n_iters = max(len(result.history) - 1, 1)
    result.ms_per_iter = (result.wall / n_iters) * 1e3

    log_hist = np.log10(np.maximum(np.asarray(result.history), 1e-12))
    if len(log_hist) > 1:
        x_axis = np.linspace(0.0, 1.0, len(log_hist))
        result.traj_auc = float(np.trapezoid(log_hist, x_axis))
    else:
        result.traj_auc = float(log_hist[-1])

    target_iters = {}
    for tgt in config.target_profile:
        hit_it = None
        for i, v in enumerate(result.history):
            if v <= tgt:
                hit_it = i
                break
        target_iters[tgt] = hit_it
    result.target_iters = target_iters


def run_experiment(config, *, enabled=None, do_plots=True):
    """Run every enabled optimizer variant and print/plot the report."""
    n_hidden_layers = len(config.hidden_sizes)
    arch_str = "->".join(
        str(s) for s in (["x"] + list(config.hidden_sizes) + [config.n_classes])
    )
    activation_str = (
        ",".join(config.activation_name)
        if isinstance(config.activation_name, (list, tuple))
        else str(config.activation_name)
    )
    print(
        f"=== {n_hidden_layers + 1}-layer MLP comparison: "
        "QQN vs SGD vs Adam vs L-BFGS ==="
    )
    print(
        f"    dataset={config.dataset}  hidden_sizes={config.hidden_sizes}  "
        f"arch={arch_str}  activation={activation_str}  (non-convex objective)"
    )
    print(
        f"  classes={config.n_classes}  n_train={config.n_train}  "
        f"n_test={config.n_test}  maxiter={config.maxiter}\n"
    )
    stop = config.stop
    print(
        f"  shared stop: f_target={stop['f_target']:.1e}  "
        f"gtol={stop['gtol']:.1e}  time_budget={stop['time_budget']:.1f}s\n"
    )

    xtr, ytr, xte, yte = load_image_dataset(
        config.dataset,
        config.n_train,
        config.n_test,
        config.n_classes,
        seed=config.subset_seed,
        balanced=config.balanced,
        synth_dim=config.synth_dim,
    )
    dim = xtr.shape[1]
    X_train, y_train = jnp.asarray(xtr), jnp.asarray(ytr)
    X_test, y_test = jnp.asarray(xte), jnp.asarray(yte)
    data = (X_train, y_train, X_test, y_test)

    model = build_model(config, dim)
    loss_fn = model.make_loss(X_train, y_train, l2=config.l2)

    params0 = model.init_params(jax.random.PRNGKey(config.seed))
    print(
        f"  l2={config.l2:.1e}  sgd_lr={config.sgd_lr}  "
        f"adam_lr={config.adam_lr}  seed={config.seed}\n"
        f"  n_classes={config.n_classes}  subset_seed={config.subset_seed}\n"
        f"  milestones={config.milestones}  "
        f"target_profile={config.target_profile}\n"
    )
    print(f"  model parameters: {int(params0.shape[0])}\n")

    ctx = _Ctx()
    ctx.loss_fn = loss_fn
    ctx.params0 = params0
    ctx.maxiter = config.maxiter
    ctx.stop = stop
    ctx.sgd_lr = config.sgd_lr
    ctx.adam_lr = config.adam_lr
    ctx.lbfgs_memory_size = getattr(config, "lbfgs_memory_size", 10)
    ctx.run_qqn = _runners.run_qqn
    ctx.run_optax = _runners.run_optax
    ctx.run_optax_lbfgs = _runners.run_optax_lbfgs

    ctx.partition_sizes = getattr(model, "partition_sizes", None)
    if callable(ctx.partition_sizes):
        ctx.partition_sizes = ctx.partition_sizes()
    runners, qqn_kwarg_map = _profiles.build_runners(ctx, enabled=enabled)

    results = {}
    for name, runner in runners.items():
        with profile_region(name):
            result = runner()
        enrich(result, model, data, config)

        if result.evals_to_target is not None and result.iters_to_target:
            result.evals_per_iter = result.evals_to_target / result.iters_to_target
        else:
            if name in ("SGD", "Adam"):
                result.evals_per_iter = 1.0
            elif name == "L-BFGS":
                result.evals_per_iter = 3.0
            else:
                result.evals_per_iter = -1.0
        results[name] = result
        optimizer_extra = qqn_kwarg_map.get(name) if qqn_kwarg_map else None
        write_run_json(name, result, config, optimizer_extra=optimizer_extra)

    report_tables(results, config)
    axis_stats = report_axis_analysis(results)
    write_results_json(results, config, extra={"axis_analysis": axis_stats})
    if do_plots:
        save_plots(results, config, arch_str=arch_str)
    return results

"""Sparse / quantization benchmark driver (ported from the example script).

Trains a pytree MLP with QQN under various sparsity (OrthantRegion, L1) and
quantization (QuantizationRegion, quant-delta penalty) configurations, then
cross-products a set of quantization "polishing" phases on top of every
base-trained model. Reports a summary table, the accuracy-vs-sparsity Pareto
frontier, and the best precision config; saves convergence / Pareto / metrics
plots.

Driven by ``SparseConfig`` (env-var contract preserved). The flat-array
L-BFGS oracle operates on a raveled parameter vector; the loss closure
unflattens before evaluating the pytree network.
"""

import time
from typing import Any, Dict, List, Sequence, Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qqn_jax.regions.quantization import QuantizationRegion
from qqn_jax.regions.sequence import Sequential
from qqn_jax.solver import QQN
from qqn_jax import OrthantRegion
from qqn_jax.regularizers import l1_penalty, quantization_delta_penalty

from experiments.data.loaders import load_image_dataset
from experiments.models.pytree_mlp import (
    init_params,
    cross_entropy_loss,
    test_loss,
    sparsity,
    round_params_to_grid,
)
from experiments.reporting.sparse_plots import (
    plot_convergence,
    plot_pareto,
    plot_metrics_bar,
)

__all__ = ["run_sparse_experiment", "run_config"]


def run_config(
    name: str,
    region,
    x_train,
    y_train,
    x_test,
    y_test,
    sizes: List[int],
    maxiter: int = 100,
    seed: int = 0,
    line_search: str = "strong_wolfe",
    regularizer=None,
    quant_bits: int = 4,
    init_flat=None,
    unravel_fn=None,
    activation=jnp.tanh,
    activation_names: Union[str, Sequence[str]] = "tanh",
    l2: float = 1e-4,
    history_size: int = 10,
    quant_lo: float = -1.0,
    quant_hi: float = 1.0,
) -> Dict[str, Any]:
    """Train one configuration and collect metrics.

    The MLP parameter pytree is flattened to a single 1-D vector so it is
    compatible with the flat-array L-BFGS oracle. ``init_flat`` (optional)
    warm-starts the optimizer from a previously trained flat parameter vector
    (the polishing phase); ``unravel_fn`` must then match its structure.
    """
    key = jax.random.PRNGKey(seed)

    if init_flat is not None and unravel_fn is not None:
        flat_params0 = init_flat
        unravel = unravel_fn
    else:
        params0_tree = init_params(key, sizes, activation=activation_names)
        flat_params0, unravel = ravel_pytree(params0_tree)

    def loss_fn(flat_params):
        params = unravel(flat_params)
        return cross_entropy_loss(
            params,
            x_train,
            y_train,
            l2=l2,
            regularizer=regularizer,
            activation=activation,
        )

    loss_history: List[float] = []

    def _record(val):
        loss_history.append(float(val))

    def loss_fn_recorded(flat_params):
        val = loss_fn(flat_params)
        jax.debug.callback(_record, val)
        return val

    solver = QQN(
        loss_fn_recorded,
        maxiter=maxiter,
        tol=1e-6,
        history_size=history_size,
        line_search=line_search,
        region=region,
    )

    run = jax.jit(solver.run)

    t0 = time.perf_counter()
    final_flat, final_state = run(flat_params0)
    jax.block_until_ready(final_flat)
    elapsed = time.perf_counter() - t0

    final_params = unravel(final_flat)
    final_loss = float(final_state.value)
    test_loss_val = float(test_loss(final_params, x_test, y_test, activation))
    spars = sparsity(final_params)
    quant_params = round_params_to_grid(
        final_params, bits=quant_bits, lo=quant_lo, hi=quant_hi
    )
    quant_loss_val = float(test_loss(quant_params, x_test, y_test, activation))
    quant_sparsity_val = sparsity(quant_params)

    return {
        "name": name,
        "iters": int(final_state.iter),
        "loss": final_loss,
        "test_loss": test_loss_val,
        "sparsity": spars,
        "quant_loss": quant_loss_val,
        "quant_sparsity": quant_sparsity_val,
        "time_s": elapsed,
        "loss_history": loss_history,
        "final_flat": final_flat,
        "unravel": unravel,
    }


def _build_regularizers(config):
    """Build the L1 / quant / joint penalty callables from the config."""

    def l1_reg(params):
        return l1_penalty(params, scale=config.l1_scale, weights_only=True)

    def quant_reg(params):
        return quantization_delta_penalty(
            params,
            scale=config.quant_scale,
            bits=config.qbits,
            lo=config.quant_lo,
            hi=config.quant_hi,
            weights_only=True,
        )

    def quant_l1_reg(params):
        return quantization_delta_penalty(
            params,
            scale=config.quant_scale,
            bits=config.qbits,
            lo=config.quant_lo,
            hi=config.quant_hi,
            weights_only=True,
        ) + l1_penalty(params, scale=config.l1_scale, weights_only=True)

    return l1_reg, quant_reg, quant_l1_reg


def _base_configs(l1_reg):
    return [
        ("baseline (dense)", None, "strong_wolfe", None),
        ("orthant (sparse)", OrthantRegion(), "strong_wolfe", None),
        ("l1-orthant-penalty (sparse)", OrthantRegion(), "strong_wolfe", l1_reg),
        ("l1-penalty (sparse)", None, "strong_wolfe", l1_reg),
    ]


def _polish_configs(config, quant_reg, quant_l1_reg):
    QBITS, QLO, QHI = config.qbits, config.quant_lo, config.quant_hi
    return [
        ("quant-penalty (prec)", None, "strong_wolfe", quant_reg),
        (
            "quant-region (prec)",
            QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI),
            "strong_wolfe",
            None,
        ),
        (
            "quant-region-penalty (prec)",
            QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI),
            "strong_wolfe",
            quant_reg,
        ),
        (
            "orthant+quant-region (sparse+prec)",
            Sequential(
                [QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI), OrthantRegion()]
            ),
            "strong_wolfe",
            None,
        ),
        (
            "orthant+quant-region+quant-l1 (sparse+prec)",
            Sequential(
                [QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI), OrthantRegion()]
            ),
            "strong_wolfe",
            quant_l1_reg,
        ),
        ("quant+l1-penalty (sparse+prec)", None, "strong_wolfe", quant_l1_reg),
        (
            "quant-region+quant-l1 (sparse+prec)",
            QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI),
            "strong_wolfe",
            quant_l1_reg,
        ),
    ]


def _report(results):
    """Summary table + Pareto frontier + best precision config."""
    print("\n" + "=" * 90)
    header = (
        f"{'config':<48}{'iters':>7}{'loss':>10}"
        f"{'test_loss':>11}{'sparsity':>10}{'quant_loss':>11}{'time(s)':>10}"
    )
    print(header)
    print("-" * 114)
    for r in results:
        print(
            f"{r['name']:<48}{r['iters']:>7}{r['loss']:>10.4f}"
            f"{r['test_loss']:>11.4f}{r['sparsity']:>10.3f}"
            f"{r.get('quant_loss', 0.0):>11.4f}{r['time_s']:>10.2f}"
        )
    print("=" * 114)

    print("\nPareto frontier (test_loss vs. sparsity — non-dominated configs):")
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, o in enumerate(results):
            if i == j:
                continue
            no_worse = (
                o["test_loss"] <= r["test_loss"] and o["sparsity"] >= r["sparsity"]
            )
            strictly_better = (
                o["test_loss"] < r["test_loss"] or o["sparsity"] > r["sparsity"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    for r in sorted(pareto, key=lambda d: d["sparsity"], reverse=True):
        print(
            f"  {r['name']:<48} test_loss={r['test_loss']:.4f}  "
            f"sparsity={r['sparsity']:.3f}"
        )

    prec = [r for r in results if "prec" in r["name"]]
    if prec:
        best = min(prec, key=lambda d: d["quant_loss"])
        print(
            f"\nBest precision config (lowest quant_loss): {best['name']}\n"
            f"  quant_loss={best['quant_loss']:.4f}  "
            f"test_loss={best['test_loss']:.4f}  "
            f"sparsity={best['sparsity']:.3f}"
        )


def run_sparse_experiment(config, *, do_plots=True):
    """Run the full base->polish sparse/quantization benchmark."""
    print(f"Loading {config.dataset} subset...")
    x_train, y_train, x_test, y_test = load_image_dataset(
        config.dataset,
        config.n_train,
        config.n_test,
        10,  # use all 10 classes (permutation subsetting)
        seed=config.seed,
        balanced=False,
        synth_dim=784,
    )
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    dim = x_train.shape[1]
    n_classes = int(max(int(y_train.max()), int(y_test.max())) + 1)
    sizes = [dim, *config.hidden_sizes, n_classes]
    arch_str = "->".join(str(s) for s in sizes)
    activation_str = (
        ",".join(config.activation_name)
        if isinstance(config.activation_name, (list, tuple))
        else str(config.activation_name)
    )
    print(
        f"  arch={arch_str}  activation={activation_str}  "
        f"line_search={config.line_search}  history_size={config.history_size}"
    )
    print(
        f"  maxiter={config.maxiter}  l2={config.l2:.1e}  "
        f"l1_scale={config.l1_scale:.1e}  quant_scale={config.quant_scale:.1e}  "
        f"qbits={config.qbits}  seed={config.seed}\n"
    )

    l1_reg, quant_reg, quant_l1_reg = _build_regularizers(config)
    base_configs = _base_configs(l1_reg)
    polish_configs = _polish_configs(config, quant_reg, quant_l1_reg)

    results = []
    base_results = []
    for name, region, line_search, regularizer in base_configs:
        print(f"\n=== [base] Running: {name} ===")
        res = run_config(
            name,
            region,
            x_train,
            y_train,
            x_test,
            y_test,
            sizes,
            maxiter=config.maxiter,
            line_search=line_search,
            regularizer=regularizer,
            quant_bits=config.qbits,
            seed=config.seed,
            activation=config.activation_fn,
            activation_names=config.activation_name,
            l2=config.l2,
            history_size=config.history_size,
            quant_lo=config.quant_lo,
            quant_hi=config.quant_hi,
        )
        results.append(res)
        base_results.append(res)
        print(
            f"  iters={res['iters']:3d}  loss={res['loss']:.4f}  "
            f"test_loss={res['test_loss']:.4f}  "
            f"sparsity={res['sparsity']:.3f}  "
            f"q_sparsity={res.get('quant_sparsity', 0.0):.3f}  "
            f"quant_loss={res['quant_loss']:.4f}  time={res['time_s']:.2f}s"
        )

    for base in base_results:
        for suffix, region, line_search, regularizer in polish_configs:
            name = f"{base['name']} -> {suffix}"
            print(f"\n=== [polish] Running: {name} ===")
            res = run_config(
                name,
                region,
                x_train,
                y_train,
                x_test,
                y_test,
                sizes,
                maxiter=config.polish_maxiter,
                line_search=line_search,
                regularizer=regularizer,
                quant_bits=config.qbits,
                init_flat=base["final_flat"],
                unravel_fn=base["unravel"],
                seed=config.seed,
                activation=config.activation_fn,
                activation_names=config.activation_name,
                l2=config.l2,
                history_size=config.history_size,
                quant_lo=config.quant_lo,
                quant_hi=config.quant_hi,
            )
            results.append(res)
            print(
                f"  iters={res['iters']:3d}  loss={res['loss']:.4f}  "
                f"test_loss={res['test_loss']:.4f}  "
                f"sparsity={res['sparsity']:.3f}  "
                f"quant_loss={res['quant_loss']:.4f}  time={res['time_s']:.2f}s"
            )

    # Release warm-start vectors held only for polishing.
    for base in base_results:
        base.pop("final_flat", None)
        base.pop("unravel", None)

    _report(results)

    if do_plots:
        plot_convergence(results)
        plot_pareto(results)
        plot_metrics_bar(results)
    return results

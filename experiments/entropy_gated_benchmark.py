"""EG-PTGP ablation (docs/entropy_gated_regions.md §7.4) on (Fashion-)MNIST.

Runs the four cells that decide whether the idea has content:

    ERM          plain cross-entropy, no region
    Gate-Obj     entropy-hinge objective only (α-weighted CE), no region
    Region-Only  plain CE + EntropyGatedRegion (constraints only)
    EG-PTGP      gated objective + EntropyGatedRegion (the full method)

and logs the §7.1 instrumentation per iteration: |M| / N, test prediction
churn (H1), ECE (H2), train/test accuracy, and the accepted-step norm.

Env vars: DATASET, N_TRAIN, N_TEST, HIDDEN, DEPTH, ACTIVATION, MAXITER,
TIME_BUDGET, H_MEM, TAU, KAPPA, ZETA, POLICY, NUM_REGIONS, MAX_CONSTRAINTS,
WARMUP, LINE_SEARCH, L2, SEED.
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experiments import env
from experiments.data.loaders import load_image_dataset
from experiments.models import mlp
from experiments.models.activations import parse_activation
from experiments.models.topology import parse_hidden_sizes
from qqn_jax import QQN
from qqn_jax.regions.entropy_gated import (
    EntropyGatedRegion,
    gate_statistics,
    make_gated_loss,
)


def _ece(probs, labels, n_bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(ece)


def run_variant(name, loss_fn, region, params0, logits_fn, data, cfg):
    X_train, y_train, X_test, y_test = data
    solver = QQN(
        loss_fn,
        maxiter=cfg["maxiter"],
        tol=1e-8,
        line_search=cfg["line_search"],
        region=region,
    )
    update = jax.jit(solver.update)

    @jax.jit
    def diagnostics(params):
        tr = logits_fn(params, X_train)
        te = logits_fn(params, X_test)
        stats = gate_statistics(tr, y_train, h_mem=cfg["h_mem"], beta=1.0 / cfg["tau"])
        n_mem = jnp.sum(stats.stability > 0.05)
        train_acc = jnp.mean(stats.correct)
        te_probs = jax.nn.softmax(te, axis=-1)
        test_acc = jnp.mean(jnp.argmax(te, -1) == y_test)
        return n_mem, train_acc, test_acc, te_probs, jnp.mean(stats.entropy)

    state = solver.init_state(params0)
    params = params0
    n_mem, tr_acc, te_acc, te_probs, H = diagnostics(params)
    prev_pred = np.argmax(np.asarray(te_probs), axis=1)
    rows = []
    t0 = time.perf_counter()
    churn_total = 0.0
    for it in range(cfg["maxiter"]):
        params, state = update(params, state)
        n_mem, tr_acc, te_acc, te_probs, H = diagnostics(params)
        te_probs_np = np.asarray(te_probs)
        pred = np.argmax(te_probs_np, axis=1)
        churn = float(np.mean(pred != prev_pred))
        churn_total += churn
        prev_pred = pred
        now = time.perf_counter() - t0
        rows.append(
            {
                "it": it + 1,
                "value": float(state.value),
                "frac_mem": float(n_mem) / X_train.shape[0],
                "train_acc": float(tr_acc),
                "test_acc": float(te_acc),
                "churn": churn,
                "ece": _ece(te_probs_np, np.asarray(y_test)),
                "entropy": float(H),
                "step": float(state.step_size),
                "time": now,
            }
        )
        if (it + 1) % cfg["log_every"] == 0 or bool(state.done):
            r = rows[-1]
            print(
                f"  [{name:<11}] it={r['it']:4d} f={r['value']:.4f} "
                f"|M|/N={r['frac_mem']:.3f} acc(tr/te)={r['train_acc']:.4f}/"
                f"{r['test_acc']:.4f} churn={r['churn']:.4f} ece={r['ece']:.4f} "
                f"H={r['entropy']:.3f} t={r['step']:.3g} ({r['time']:.1f}s)"
            )
        if bool(state.done) or now >= cfg["time_budget"]:
            break
    summary = dict(rows[-1])
    summary["name"] = name
    summary["mean_churn"] = churn_total / max(len(rows), 1)
    summary["iters"] = len(rows)
    return summary, rows


def main():
    dataset = env.env_str("DATASET", "mnist").lower()
    n_train = env.env_int("N_TRAIN", 5000)
    n_test = env.env_int("N_TEST", 2000)
    n_classes = 10
    hidden = parse_hidden_sizes(default_hidden=128, default_depth=1)
    act_name, act_fn = parse_activation(len(hidden), default="tanh")
    cfg: dict[str, Any] = {
        "maxiter": env.env_int("MAXITER", 300),
        "time_budget": env.env_float("TIME_BUDGET", 120.0),
        "h_mem": env.env_float("H_MEM", 0.5),
        "tau": env.env_float("TAU", 0.1),
        "kappa": env.env_float("KAPPA", 0.1),
        "zeta": env.env_float("ZETA", 0.0),
        "policy": env.env_str("POLICY", "pair"),
        "num_regions": env.env_int("NUM_REGIONS", 8),
        "max_constraints": env.env_int("MAX_CONSTRAINTS", 256),
        "warmup": env.env_int("WARMUP", 5),
        "line_search": env.env_str("LINE_SEARCH", "backtracking"),
        "l2": env.env_float("L2", 1e-4),
        "seed": env.env_int("SEED", 0),
        "log_every": env.env_int("LOG_EVERY", 10),
    }
    print("=== EG-PTGP ablation ===")
    print(
        f"  dataset={dataset} n_train={n_train} n_test={n_test} hidden={hidden} act={act_name}"
    )
    print(
        f"  H_mem={cfg['h_mem']} tau={cfg['tau']} kappa={cfg['kappa']} zeta={cfg['zeta']} "
        f"policy={cfg['policy']} k={cfg['num_regions']} max_constraints={cfg['max_constraints']} "
        f"warmup={cfg['warmup']} line_search={cfg['line_search']}\n"
    )

    xtr, ytr, xte, yte = load_image_dataset(
        dataset, n_train, n_test, n_classes, seed=cfg["seed"], balanced=True
    )
    X_train, y_train = jnp.asarray(xtr), jnp.asarray(ytr)
    X_test, y_test = jnp.asarray(xte), jnp.asarray(yte)
    data = (X_train, y_train, X_test, y_test)
    dim = X_train.shape[1]

    model = mlp.FlatMLP(dim, hidden, n_classes, act_fn, act_name)
    params0 = model.init_params(jax.random.PRNGKey(cfg["seed"]))
    print(f"  model parameters: {int(params0.shape[0])}\n")

    def logits_fn(params, x):
        return mlp.forward(params, x, dim, hidden, n_classes, act_fn)

    erm_loss = model.make_loss(X_train, y_train, l2=cfg["l2"])
    gated_loss = make_gated_loss(
        logits_fn,
        X_train,
        y_train,
        h_mem=cfg["h_mem"],
        beta=1.0 / cfg["tau"],
        l2=cfg["l2"],
    )

    def make_region():
        return EntropyGatedRegion(
            logits_fn,
            X_train,
            y_train,
            h_mem=cfg["h_mem"],
            tau=cfg["tau"],
            kappa=cfg["kappa"],
            zeta=cfg["zeta"],
            policy=cfg["policy"],
            num_regions=cfg["num_regions"],
            max_constraints=cfg["max_constraints"],
            warmup_steps=cfg["warmup"],
            dead_zone_ratio=0.05,
            seed=cfg["seed"],
        )

    variants = [
        ("ERM", erm_loss, None),
        ("Gate-Obj", gated_loss, None),
        ("Region-Only", erm_loss, make_region()),
        ("EG-PTGP", gated_loss, make_region()),
    ]

    summaries = []
    for name, loss_fn, region in variants:
        print(f"--- {name} ---")
        summary, _rows = run_variant(
            name, loss_fn, region, params0, logits_fn, data, cfg
        )
        summaries.append(summary)

    print("\n" + "=" * 100)
    print(
        f"{'variant':<13}{'iters':>6}{'train_acc':>11}{'test_acc':>10}"
        f"{'ECE':>8}{'mean_churn':>12}{'|M|/N':>8}{'H':>7}{'time(s)':>9}"
    )
    print("-" * 100)
    for s in summaries:
        print(
            f"{s['name']:<13}{s['iters']:>6}{s['train_acc']:>11.4f}{s['test_acc']:>10.4f}"
            f"{s['ece']:>8.4f}{s['mean_churn']:>12.5f}{s['frac_mem']:>8.3f}"
            f"{s['entropy']:>7.3f}{s['time']:>9.1f}"
        )
    print("=" * 100)
    print(
        "\nH1: EG-PTGP mean_churn should be ≥30% below ERM at matched accuracy.\n"
        "H2: EG-PTGP ECE should be below ERM; compare with Gate-Obj to isolate the hinge.\n"
        "H8: watch |M|/N -> 1 with the loss stalling: that is the dead zone."
    )


if __name__ == "__main__":
    main()

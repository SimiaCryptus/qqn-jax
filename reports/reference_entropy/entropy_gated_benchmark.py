"""EG-PTGP ablation (docs/entropy_gated_regions.md §7.4) on (Fashion-)MNIST.

Runs the four cells that decide whether the idea has content:

    ERM          plain cross-entropy, no region
    Gate-Obj     entropy-hinge objective only (α-weighted CE), no region
    Region-Only  plain CE + EntropyGatedRegion (constraints only)
    EG-PTGP      gated objective + EntropyGatedRegion (the full method)

Changes relative to the first sweep (see reports/reference_entropy/notes.md):

  * **Iteration budget, not wall clock.** ``TIME_BUDGET`` is a safety net
    (``<= 0`` disables it). Churn/ECE are additionally re-scored at the
    *matched* iteration count shared by every cell, so a truncated cell can
    no longer win by averaging a transient over fewer steps.
  * **Multiple seeds.** ``SEEDS=0,1,2,3,4`` gives error bars; the data split
    is pinned by ``DATA_SEED`` so only the initialization varies.
  * **H1 measured properly.** ``mean_churn`` (a convergence diagnostic) is
    kept for continuity but the headline numbers are *tail churn* (last
    ``TAIL_ITERS``) and *cross-seed churn* (test-set disagreement between
    independently seeded runs) — the deployment-relevant quantity.
  * **H2 de-confounded.** Signed calibration error (mean confidence −
    accuracy) exposes the over/under-confidence sign flip, and every gated
    cell is scored against ERM *interpolated to the same train accuracy*.
  * **H8 tell fixed.** The dead zone is ``t = 0 ∧ Δf ≈ 0`` (``|M|/N`` does
    *not* go to 1); detected, reported, and optionally used to stop early.
  * ``f`` printed at full precision and ``‖Δθ‖`` logged per iteration, so
    "step is zero but accuracy drifts" is no longer ambiguous.
  * Per-projection region instrumentation (``REGION_DEBUG=1``).

Env vars:
   data/model  DATASET, N_TRAIN, N_TEST, HIDDEN, DEPTH, ACTIVATION,
               SEED, SEEDS, DATA_SEED
   budget      MAXITER, TIME_BUDGET, LOG_EVERY, TAIL_ITERS,
               FREEZE_PATIENCE, FREEZE_TOL
   gate        H_MEM, TAU, KAPPA, ZETA
   region      POLICY, NUM_REGIONS, MAX_CONSTRAINTS, GRAD_CHUNK, WARMUP,
               REGION_DEBUG
   optimizer   LINE_SEARCH, LBFGS_MEMORY, ADAM_LR, L2
   selection   VARIANTS (comma-separated subset of the four cells)
"""

import os

# --- Memory hygiene (must run BEFORE jax / tensorflow are imported) -------
# The region takes a per-sample gradient pass, so peak device memory is
# bursty. Pre-allocating a fixed arena makes those bursts fail; let XLA
# grow on demand instead. TensorFlow (used only to download the dataset)
# is additionally kept off the GPU in experiments/data/loaders.py.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from experiments import env
from experiments.data.loaders import load_image_dataset
from experiments.models import mlp
from experiments.models.activations import parse_activation
from experiments.models.topology import parse_hidden_sizes
from qqn_jax import AdamOracle, Fallback, LBFGSOracle, QQN
from qqn_jax.regions.entropy_gated import (
    EntropyGatedRegion,
    ProjectionRecorder,
    gate_statistics,
    make_gated_loss,
)

# Standard line-search options per search name (mirrors the canonical QQN
# profile in reports/reference/profiles.py).
_LINE_SEARCH_OPTIONS = {
    "armijo_wolfe": {"c1": 1e-9, "c2": 0.7, "max_iter": 10},
}

_TRUTHY = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _calibration(probs, labels, n_bins=15):
    """Return ``(ECE, signed_CE)``.

    ``signed_CE = mean(confidence) − accuracy`` is positive for an
    overconfident model (ERM's failure) and negative for an underconfident
    one (what a too-large ``H_mem`` produces). Unsigned ECE hides that flip
    and makes the two look equally good/bad.
    """
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(ece), float(conf.mean() - acc.mean())


def _mean_std(values):
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))],
                     dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _pm(values, p=4):
    mean, std = _mean_std(values)
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{p}f}±{std:.{p}f}"


def _table(header, rows):
    cols = list(zip(*([list(header)] + [list(r) for r in rows]))) if rows else None
    widths = (
        [max(len(str(c)) for c in col) for col in cols]
        if cols
        else [len(h) for h in header]
    )

    def fmt(cells):
        return "  ".join(
            str(c).ljust(w) if i == 0 else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    line = fmt(header)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(fmt(r))


@jax.jit
def _delta_norm(a, b):
    fa, _ = ravel_pytree(a)
    fb, _ = ravel_pytree(b)
    return jnp.linalg.norm(fa - fb)


def _cross_seed_churn(preds):
    """Mean pairwise test-set disagreement between independently seeded runs.

    This is the H1 quantity: "if I retrain, how many predictions move?".
    Iterate-to-iterate churn during optimization is a convergence
    diagnostic and is trivially minimized by not moving at all.
    """
    if len(preds) < 2:
        return float("nan")
    total, n = 0.0, 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            total += float(np.mean(preds[i] != preds[j]))
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _make_solver(loss_fn, region, cfg):
    """The *standard* QQN solver, identical for every ablation cell.

    This deliberately does not sweep optimizer variants: the oracle,
    line search and step-size policy are pinned to the canonical QQN
    configuration so all observed differences come from the entropy gate.
    """
    oracle = Fallback(
        [
            LBFGSOracle(history_size=cfg["lbfgs_memory"]),
            AdamOracle(learning_rate=cfg["adam_lr"]),
        ]
    )
    return QQN(
        loss_fn,
        maxiter=cfg["maxiter"],
        tol=1e-8,
        oracle=oracle,
        line_search=cfg["line_search"],
        line_search_options=cfg["line_search_options"],
        remember_step_size=False,
        region=region,
    )


def run_variant(name, loss_fn, region, params0, logits_fn, data, cfg, seed):
    X_train, y_train, X_test, y_test = data
    y_test_np = np.asarray(y_test)
    solver = _make_solver(loss_fn, region, cfg)
    update = jax.jit(solver.update)

    @jax.jit
    def diagnostics(params):
        tr = logits_fn(params, X_train)
        te = logits_fn(params, X_test)
        stats = gate_statistics(
            tr, y_train, h_mem=cfg["h_mem"], beta=1.0 / cfg["tau"]
        )
        n_mem = jnp.sum(stats.stability > 0.05)
        train_acc = jnp.mean(stats.correct)
        te_probs = jax.nn.softmax(te, axis=-1)
        test_acc = jnp.mean(jnp.argmax(te, -1) == y_test)
        return n_mem, train_acc, test_acc, te_probs, jnp.mean(stats.entropy)

    def _host_diagnostics(params):
        """Pull the diagnostics to the host and immediately free the
        device buffers so nothing accumulates across iterations."""
        n_mem, tr_acc, te_acc, te_probs, H = diagnostics(params)
        out = (
            float(n_mem),
            float(tr_acc),
            float(te_acc),
            np.asarray(te_probs),
            float(H),
        )
        del n_mem, tr_acc, te_acc, te_probs, H
        return out

    state = solver.init_state(params0)
    params = params0
    _n_mem, _tr, _te, te_probs_np, _H = _host_diagnostics(params)
    prev_pred = np.argmax(te_probs_np, axis=1)
    prev_value = float(state.value)
    rows = []
    churn_total = 0.0
    stall = 0
    stop_reason = "maxiter"
    warned_step0 = False
    t0 = time.perf_counter()

    for it in range(cfg["maxiter"]):
        prev_params = params
        params, state = update(params, state)
        dtheta = float(_delta_norm(params, prev_params))
        n_mem, tr_acc, te_acc, te_probs_np, H = _host_diagnostics(params)
        pred = np.argmax(te_probs_np, axis=1)
        churn = float(np.mean(pred != prev_pred))
        churn_total += churn
        prev_pred = pred
        value = float(state.value)
        d_f = abs(value - prev_value)
        prev_value = value
        step = float(state.step_size)
        ece, signed_ce = _calibration(te_probs_np, y_test_np)
        now = time.perf_counter() - t0
        rows.append(
            {
                "it": it + 1,
                "value": value,
                "d_f": d_f,
                "dtheta": dtheta,
                "frac_mem": n_mem / X_train.shape[0],
                "train_acc": tr_acc,
                "test_acc": te_acc,
                "churn": churn,
                "ece": ece,
                "signed_ce": signed_ce,
                "entropy": H,
                "step": step,
                "time": now,
            }
        )

        # H8 tell: accepted step exactly zero *and* the objective is not
        # moving. (|M|/N -> 1 is the *inert*-gate signature, a different
        # regime; it is reported separately in the summary.)
        frozen_now = step == 0.0 and dtheta == 0.0 and d_f <= cfg["freeze_tol"]
        stall = stall + 1 if frozen_now else 0
        if step == 0.0 and dtheta > 0.0 and not warned_step0:
            warned_step0 = True
            print(
                f"    (note: state.step_size==0 while ||dtheta||={dtheta:.3e} "
                f"at it={it + 1}: step_size is reporting a line-search trial, "
                "not the accepted step)"
            )

        done = bool(state.done)
        budget = cfg["time_budget"] > 0.0 and now >= cfg["time_budget"]
        freeze = cfg["freeze_patience"] > 0 and stall >= cfg["freeze_patience"]
        last = done or budget or freeze or (it + 1) == cfg["maxiter"]
        if (it + 1) % cfg["log_every"] == 0 or last:
            r = rows[-1]
            print(
                f"  [{name:<11}|s{seed}] it={r['it']:4d} f={r['value']:.8g} "
                f"|M|/N={r['frac_mem']:.3f} acc(tr/te)={r['train_acc']:.4f}/"
                f"{r['test_acc']:.4f} churn={r['churn']:.4f} ece={r['ece']:.4f} "
                f"sece={r['signed_ce']:+.4f} H={r['entropy']:.3f} "
                f"t={r['step']:.3g} |dθ|={r['dtheta']:.3e} ({r['time']:.1f}s)"
            )
        if done:
            stop_reason = "converged"
            break
        if freeze:
            stop_reason = "frozen"
            print(
                f"    !! dead zone: {stall} consecutive iterations with "
                f"t=0 and Δf<={cfg['freeze_tol']:g} — stopping early."
            )
            break
        if budget:
            stop_reason = "time"
            print(
                "    !! wall-clock budget hit: this cell is NOT "
                "iteration-matched with the others."
            )
            break

    # Terminal freeze detection (independent of the early-stop patience).
    frozen_at = -1
    for i in range(len(rows) - 1, -1, -1):
        if rows[i]["step"] == 0.0 and rows[i]["dtheta"] == 0.0:
            frozen_at = i + 1
        else:
            break

    tail_n = max(1, min(cfg["tail_iters"], len(rows)))
    tail = rows[-tail_n:]
    report_n = max(1, min(20, len(rows)))
    report = rows[-report_n:]

    summary = dict(rows[-1])
    summary.update(
        name=name,
        seed=seed,
        iters=len(rows),
        stop=stop_reason,
        mean_churn=churn_total / max(len(rows), 1),
        tail_churn=float(np.mean([r["churn"] for r in tail])),
        tail_n=tail_n,
        # Final-iterate ECE jitters by ±0.005; average the last few logged
        # iterations instead of reporting a single sample.
        ece=float(np.mean([r["ece"] for r in report])),
        ece_final=rows[-1]["ece"],
        signed_ce=float(np.mean([r["signed_ce"] for r in report])),
        frozen_at=frozen_at,
        inert=bool(rows[-1]["frac_mem"] > 0.99 and rows[-1]["entropy"] < 0.05),
        churn_seq=[r["churn"] for r in rows],
    )
    return summary, rows, prev_pred


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    dataset = env.env_str("DATASET", "mnist").lower()
    n_train = env.env_int("N_TRAIN", 5000)
    n_test = env.env_int("N_TEST", 2000)
    n_classes = 10
    hidden = parse_hidden_sizes(default_hidden=128, default_depth=1)
    act_name, act_fn = parse_activation(len(hidden), default="tanh")
    cfg: dict[str, Any] = {
        "maxiter": env.env_int("MAXITER", 300),
        # <= 0 disables the wall-clock stop. Prefer that: a time budget
        # silently truncates cells and turns every averaged metric into a
        # measure of how many iterations the cell managed to run.
        "time_budget": env.env_float("TIME_BUDGET", 0.0),
        "tail_iters": env.env_int("TAIL_ITERS", 100),
        "freeze_patience": env.env_int("FREEZE_PATIENCE", 25),
        "freeze_tol": env.env_float("FREEZE_TOL", 1e-9),
        "h_mem": env.env_float("H_MEM", 0.5),
        "tau": env.env_float("TAU", 0.1),
        "kappa": env.env_float("KAPPA", 0.1),
        "zeta": env.env_float("ZETA", 0.0),
        "policy": env.env_str("POLICY", "mean"),
        "num_regions": env.env_int("NUM_REGIONS", 8),
        "max_constraints": env.env_int("MAX_CONSTRAINTS", 256),
        "grad_chunk": env.env_int("GRAD_CHUNK", 64),
        "warmup": env.env_int("WARMUP", 5),
        "line_search": env.env_str("LINE_SEARCH", "armijo_wolfe"),
        "lbfgs_memory": env.env_int("LBFGS_MEMORY", 50),
        "adam_lr": env.env_float("ADAM_LR", 1e-3),
        "l2": env.env_float("L2", 1e-4),
        "seed": env.env_int("SEED", 0),
        "log_every": env.env_int("LOG_EVERY", 10),
    }
    cfg["line_search_options"] = dict(
        _LINE_SEARCH_OPTIONS.get(cfg["line_search"], {})
    )
    seeds_raw = env.env_str("SEEDS", "").strip()
    seeds = (
        [int(s) for s in seeds_raw.split(",") if s.strip()]
        if seeds_raw
        else [cfg["seed"]]
    )
    # The data split is held fixed so cross-seed churn measures *training*
    # nondeterminism, not a different test set.
    data_seed = env.env_int("DATA_SEED", seeds[0])
    region_debug = env.env_str("REGION_DEBUG", "1").lower() in _TRUTHY

    print("=== EG-PTGP ablation ===")
    print(
        f"  dataset={dataset} n_train={n_train} n_test={n_test} "
        f"hidden={hidden} act={act_name} data_seed={data_seed}"
    )
    print(
        f"  H_mem={cfg['h_mem']} tau={cfg['tau']} kappa={cfg['kappa']} "
        f"zeta={cfg['zeta']} policy={cfg['policy']} k={cfg['num_regions']} "
        f"max_constraints={cfg['max_constraints']} "
        f"grad_chunk={cfg['grad_chunk']} warmup={cfg['warmup']} "
        f"line_search={cfg['line_search']}"
    )
    budget_desc = (
        f"{cfg['time_budget']:.0f}s (safety net)"
        if cfg["time_budget"] > 0
        else "disabled"
    )
    print(
        f"  budget: maxiter={cfg['maxiter']} time_budget={budget_desc} "
        f"freeze_patience={cfg['freeze_patience']} "
        f"tail_iters={cfg['tail_iters']} seeds={seeds}"
    )
    print(
        "  optimizer=QQN(standard): oracle=Fallback[LBFGS("
        f"{cfg['lbfgs_memory']}), Adam({cfg['adam_lr']:g})] "
        f"line_search={cfg['line_search']}{cfg['line_search_options']} "
        "remember_step_size=False\n"
    )

    xtr, ytr, xte, yte = load_image_dataset(
        dataset, n_train, n_test, n_classes, seed=data_seed, balanced=True
    )
    X_train, y_train = jnp.asarray(xtr), jnp.asarray(ytr)
    X_test, y_test = jnp.asarray(xte), jnp.asarray(yte)
    data = (X_train, y_train, X_test, y_test)
    dim = X_train.shape[1]

    model = mlp.FlatMLP(dim, hidden, n_classes, act_fn, act_name)
    print(
        "  model parameters: "
        f"{int(model.init_params(jax.random.PRNGKey(0)).shape[0])}\n"
    )

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

    def make_region(seed, recorder):
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
            grad_chunk=cfg["grad_chunk"],
            warmup_steps=cfg["warmup"],
            dead_zone_ratio=0.05,
            dead_zone_floor=True,
            recorder=recorder,
            seed=seed,
        )

    def no_region(seed, recorder):
        return None

    # Regions are built lazily, one at a time, so two copies of the
    # constraint machinery (and their compiled projections) are never
    # resident on the device simultaneously.
    variants = [
        ("ERM", erm_loss, no_region),
        ("Gate-Obj", gated_loss, no_region),
        ("Region-Only", erm_loss, make_region),
        ("EG-PTGP", gated_loss, make_region),
    ]
    wanted = env.env_str("VARIANTS", "")
    if wanted.strip():
        keep = {w.strip().lower() for w in wanted.split(",") if w.strip()}
        variants = [v for v in variants if v[0].lower() in keep]
        if not variants:
            raise SystemExit(f"VARIANTS={wanted!r} selected no known cells.")

    runs: dict[str, list[dict]] = {}
    preds: dict[str, list[np.ndarray]] = {}
    for name, loss_fn, region_factory in variants:
        runs[name] = []
        preds[name] = []
        for seed in seeds:
            print(f"--- {name} (seed={seed}) ---")
            params0 = model.init_params(jax.random.PRNGKey(seed))
            uses_region = region_factory is make_region
            recorder = (
                ProjectionRecorder() if (uses_region and region_debug) else None
            )
            region = region_factory(seed, recorder)
            summary, rows, pred = run_variant(
                name, loss_fn, region, params0, logits_fn, data, cfg, seed
            )
            summary["rows"] = rows
            summary["region"] = recorder.summary() if recorder else None
            runs[name].append(summary)
            preds[name].append(pred)
            del region, recorder, params0
            # Drop this run's compiled executables/buffers before the next
            # one traces its own (each region compiles a distinct projection).
            jax.clear_caches()

    # --- matched-iteration re-scoring -----------------------------------
    all_runs = [s for lst in runs.values() for s in lst]
    common_iters = min(s["iters"] for s in all_runs)
    for s in all_runs:
        seq = s["churn_seq"][:common_iters]
        s["churn_matched"] = float(np.mean(seq)) if seq else float("nan")

    # --- matched-accuracy ERM calibration reference ---------------------
    erm_pts = sorted(
        (r["train_acc"], r["ece"]) for s in runs.get("ERM", []) for r in s["rows"]
    )
    if erm_pts:
        erm_acc = np.asarray([p[0] for p in erm_pts])
        erm_ece = np.asarray([p[1] for p in erm_pts])
        for s in all_runs:
            s["ece_matched"] = float(np.interp(s["train_acc"], erm_acc, erm_ece))
            s["ece_rel"] = (s["ece"] - s["ece_matched"]) / max(
                s["ece_matched"], 1e-12
            )
    else:
        for s in all_runs:
            s["ece_matched"] = float("nan")
            s["ece_rel"] = float("nan")

    cross = {name: _cross_seed_churn(preds[name]) for name in runs}

    # --- tables ----------------------------------------------------------
    def col(name, key, p=4):
        return _pm([s[key] for s in runs[name]], p)

    print("\n" + "=" * 118)
    print(f"ACCURACY / COST   (mean±std over {len(seeds)} seed(s), "
          f"matched iterations = {common_iters})")
    _table(
        ("variant", "iters", "stop", "train_acc", "test_acc", "|M|/N", "H", "time(s)"),
        [
            (
                name,
                _pm([s["iters"] for s in runs[name]], 0),
                ",".join(sorted({s["stop"] for s in runs[name]})),
                col(name, "train_acc"),
                col(name, "test_acc"),
                col(name, "frac_mem", 3),
                col(name, "entropy", 3),
                _pm([s["time"] for s in runs[name]], 1),
            )
            for name in runs
        ],
    )

    print("\nCALIBRATION (H2)  — sECE = mean confidence − accuracy "
          "(+ overconfident, − underconfident)")
    _table(
        ("variant", "ECE", "sECE", "ERM@same train_acc", "relative"),
        [
            (
                name,
                col(name, "ece"),
                _pm([s["signed_ce"] for s in runs[name]], 4),
                col(name, "ece_matched"),
                _pm([100.0 * s["ece_rel"] for s in runs[name]], 1) + " %",
            )
            for name in runs
        ],
    )

    print("\nCHURN (H1)        — 'run' is a convergence diagnostic; "
          "'cross-seed' is the deployment quantity")
    _table(
        ("variant", "run mean", f"run mean@{common_iters}",
         f"tail({cfg['tail_iters']})", "cross-seed"),
        [
            (
                name,
                col(name, "mean_churn", 5),
                col(name, "churn_matched", 5),
                col(name, "tail_churn", 5),
                "n/a" if math.isnan(cross[name]) else f"{cross[name]:.5f}",
            )
            for name in runs
        ],
    )

    region_rows = [
        (
            name,
            f"{np.mean([s['region']['frac_binding'] for s in runs[name]]):.3f}",
            f"{np.mean([s['region']['ratio'] for s in runs[name]]):.4f}",
            f"{np.mean([s['region']['cos'] for s in runs[name]]):.4f}",
            f"{np.mean([s['region']['n_active'] for s in runs[name]]):.2f}",
            f"{np.mean([s['region']['n_valid'] for s in runs[name]]):.2f}",
            f"{np.mean([s['region']['lam_max'] for s in runs[name]]):.3e}",
            f"{np.mean([s['region']['frac_guarded'] for s in runs[name]]):.3f}",
            f"{np.mean([s['region']['violation'] for s in runs[name]]):.2e}",
            f"{np.mean([s['region']['vbar_norm'] for s in runs[name]]):.2e}",
        )
        for name in runs
        if runs[name] and runs[name][0]["region"] is not None
    ]
    if region_rows:
        print("\nREGION PROJECTION — if 'binding' is ~0 the constraint never "
              "fires and the cell is ERM with extra cost")
        _table(
            ("variant", "binding", "|s*|/|step|", "cos", "#λ>0", "#valid",
             "λmax", "guard", "resid", "|v̄|"),
            region_rows,
        )

    # --- machine-readable rows -------------------------------------------
    sweep_keys = (
        "h_mem", "tau", "kappa", "zeta", "policy", "num_regions",
        "max_constraints", "warmup",
    )
    tag = " ".join(f"{k}={cfg[k]}" for k in sweep_keys)
    print()
    for name in runs:
        for s in runs[name]:
            reg = s["region"] or {}
            print(
                f"[row] variant={s['name']} dataset={dataset} act={act_name} "
                f"{tag} seed={s['seed']} train_acc={s['train_acc']:.4f} "
                f"test_acc={s['test_acc']:.4f} ece={s['ece']:.4f} "
                f"ece_matched={s['ece_matched']:.4f} "
                f"signed_ce={s['signed_ce']:+.4f} "
                f"mean_churn={s['mean_churn']:.5f} "
                f"churn_matched={s['churn_matched']:.5f} "
                f"tail_churn={s['tail_churn']:.5f} "
                f"cross_seed_churn={cross[name]:.5f} "
                f"frac_mem={s['frac_mem']:.3f} entropy={s['entropy']:.3f} "
                f"iters={s['iters']} stop={s['stop']} "
                f"frozen_at={s['frozen_at']} inert={int(s['inert'])} "
                f"binding={reg.get('frac_binding', float('nan')):.3f} "
                f"proj_ratio={reg.get('ratio', float('nan')):.4f} "
                f"time={s['time']:.1f}"
            )

    # --- verdicts ---------------------------------------------------------
    print("\n=== Hypotheses ===")

    def verdict(tag_, ok, detail):
        mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  {mark}  {tag_}: {detail}")

    if "ERM" in runs and "EG-PTGP" in runs:
        c_erm, c_eg = cross["ERM"], cross["EG-PTGP"]
        if math.isnan(c_erm) or math.isnan(c_eg):
            verdict("H1", None, "cross-seed churn needs >=2 seeds (SEEDS=0,1,2).")
        else:
            rel = (c_eg - c_erm) / max(c_erm, 1e-12)
            verdict(
                "H1",
                rel <= -0.30,
                f"cross-seed churn {c_eg:.4f} vs ERM {c_erm:.4f} "
                f"({100 * rel:+.1f}%; target ≤ −30%)",
            )
        e_eg, _ = _mean_std([s["ece"] for s in runs["EG-PTGP"]])
        m_eg, _ = _mean_std([s["ece_matched"] for s in runs["EG-PTGP"]])
        s_eg, _ = _mean_std([s["signed_ce"] for s in runs["EG-PTGP"]])
        verdict(
            "H2",
            e_eg < m_eg,
            f"ECE {e_eg:.4f} vs matched-accuracy ERM {m_eg:.4f} "
            f"({100 * (e_eg - m_eg) / max(m_eg, 1e-12):+.1f}%); "
            f"sECE={s_eg:+.4f} "
            f"({'over' if s_eg > 0 else 'under'}confident)",
        )
    frozen = [
        f"{name}(seed {s['seed']} @ it {s['frozen_at']})"
        for name in runs
        for s in runs[name]
        if s["frozen_at"] > 0
    ]
    inert = [
        f"{name}(seed {s['seed']})" for name in runs for s in runs[name] if s["inert"]
    ]
    verdict(
        "H8",
        bool(frozen or inert),
        "dead zone via t=0 ∧ Δf≈0: "
        + (", ".join(frozen) if frozen else "none")
        + " | inert gate (|M|/N→1, H→0): "
        + (", ".join(inert) if inert else "none"),
    )
    print(
        "\nNote: the H8 tell is `t = 0 ∧ Δf ≈ 0`, NOT `|M|/N → 1` — "
        "|M|/N saturates around 0.70 in the frozen regime.\n"
        "Note: ERM early-stopped at matched train accuracy is the only "
        "fair calibration baseline; the ECE column above is scored "
        "against it."
    )


if __name__ == "__main__":
    main()
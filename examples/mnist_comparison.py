"""MNIST validation experiment: QQN vs SGD vs Adam vs L-BFGS.

Trains a small softmax (logistic regression) classifier on a subset of
MNIST and compares the convergence behaviour of QQN against three
common baselines: SGD, Adam, and Optax's L-BFGS.

The optimization is framed as a *full-batch* deterministic problem so
that the comparison is apples-to-apples for the second-order methods
(QQN and L-BFGS), which assume a smooth, deterministic objective.

Data loading:
    The script tries to load MNIST via ``torchvision`` or ``tensorflow``
    if available. If neither is installed, it falls back to a synthetic
    Gaussian-blob "MNIST-like" dataset so the experiment always runs.

Run with:  python examples/mnist_comparison.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any

from qqn_jax import QQN, AndersonOracle
from qqn_jax.oracles import (
    LBFGSOracle,
    MomentumOracle,
    ShampooOracle,
    SecantOracle,
    Fallback,
)
from qqn_jax.regions import (
    BoxRegion,
    TrustRegion,
    OrthantRegion,
    Sequential,
)


def _load_mnist_numpy(n_train: int, n_test: int, n_classes: int):
    """Try to load a real MNIST subset; fall back to synthetic data.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays with images
        flattened to shape (N, 784) and float32 in [0, 1].
    """

    try:
        from tensorflow.keras.datasets import mnist  # type: ignore

        (xtr, ytr), (xte, yte) = mnist.load_data()
        xtr = xtr.reshape(xtr.shape[0], -1).astype(np.float32) / 255.0
        xte = xte.reshape(xte.shape[0], -1).astype(np.float32) / 255.0
        return _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes)
    except Exception:
        pass

    try:
        from torchvision import datasets

        train = datasets.MNIST(root="./_mnist_data", train=True, download=True)
        test = datasets.MNIST(root="./_mnist_data", train=False, download=True)
        xtr = train.data.numpy().reshape(len(train), -1).astype(np.float32) / 255.0
        ytr = train.targets.numpy()
        xte = test.data.numpy().reshape(len(test), -1).astype(np.float32) / 255.0
        yte = test.targets.numpy()
        return _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes)
    except Exception:
        pass

    print("[data] Real MNIST unavailable; using synthetic Gaussian blobs.")
    return _synthetic(n_train, n_test, n_classes, dim=784)


def _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes):
    """Keep only the first ``n_classes`` classes and subsample."""
    train_mask = ytr < n_classes
    test_mask = yte < n_classes
    xtr, ytr = xtr[train_mask][:n_train], ytr[train_mask][:n_train]
    xte, yte = xte[test_mask][:n_test], yte[test_mask][:n_test]
    return xtr, ytr.astype(np.int32), xte, yte.astype(np.int32)


def _synthetic(n_train, n_test, n_classes, dim):
    """Generate a linearly-separable-ish synthetic classification set."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=3.0, size=(n_classes, dim)).astype(np.float32)

    def make(n):
        y = rng.integers(0, n_classes, size=n).astype(np.int32)
        x = centers[y] + rng.normal(scale=1.0, size=(n, dim)).astype(np.float32)
        return x.astype(np.float32), y

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte


def init_params(dim: int, n_classes: int, key) -> jnp.ndarray:
    """Flat parameter vector: W (dim x n_classes) followed by b (n_classes)."""
    w = 0.01 * jax.random.normal(key, (dim * n_classes,))
    b = jnp.zeros((n_classes,))
    return jnp.concatenate([w, b])


def _unpack(params, dim, n_classes):
    w = params[: dim * n_classes].reshape(dim, n_classes)
    b = params[dim * n_classes :]
    return w, b


def make_loss(X, y, dim, n_classes, l2: float = 1e-4):
    """Build a full-batch cross-entropy loss ``f(params) -> scalar``."""
    Y = jax.nn.one_hot(y, n_classes)

    def loss(params):
        w, b = _unpack(params, dim, n_classes)
        logits = X @ w + b
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.mean(jnp.sum(Y * log_probs, axis=-1))
        reg = 0.5 * l2 * jnp.sum(params**2)
        return ce + reg

    return loss


def accuracy(params, X, y, dim, n_classes):
    w, b = _unpack(params, dim, n_classes)
    logits = X @ w + b
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == y).astype(jnp.float32))


def _grad_norm(loss_fn, params):
    """Compute the L2 norm of the gradient at ``params``."""
    g = jax.grad(loss_fn)(params)
    return float(jnp.linalg.norm(g))


def _converged(value, gnorm, f_target, gtol):
    """Shared convergence test: target loss reached OR gradient ~ 0."""
    if f_target is not None and value <= f_target:
        return True
    if gtol is not None and gnorm <= gtol:
        return True
    return False


def _update_milestones(milestones, hit, value, it, now):
    """Record the first iteration/time each loss milestone is crossed.
    ``milestones`` is a tuple of descending loss thresholds; ``hit`` is a
    mutable dict mapping each threshold to ``(iter, time)`` (or ``None``).
    This lets us report a full *convergence-rate profile* per optimizer
    rather than a single time-to-target, which is far more discriminating
    for separating early- from late-phase convergence behaviour.
    """
    if not milestones:
        return
    for m in milestones:
        if hit.get(m) is None and value <= m:
            hit[m] = (it, now)


def run_qqn(loss_fn, params0, maxiter, stop=None):
    """Run QQN and return (final_params, history_of_losses, wall_time)."""
    return _run_qqn_configured(loss_fn, params0, maxiter, stop=stop)


def _run_qqn_configured(
    loss_fn,
    params0,
    maxiter,
    line_search: str = "armijo",
    line_search_options=None,
    oracle: Any = "lbfgs",
    region=None,
    spline: bool = False,
    stop=None,
):
    """Run a configurable QQN variant.

    Exposes QQN's swappable components — the *oracle* (curvature source),
    the *line search* (step-size selection), and the *region* (projective
      constraint) — so we can benchmark several QQN flavours side-by-side.
     ``stop`` is a dict with shared termination bounds applied uniformly to
     every optimizer: ``f_target`` (loss threshold), ``gtol`` (gradient-norm
     tolerance), and ``time_budget`` (wall-clock seconds).
    """
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    solver = QQN(
        loss_fn,
        maxiter=maxiter,
        line_search=line_search,
        line_search_options=line_search_options,
        oracle=oracle,
        region=region,
    )

    state = solver.init_state(params0)
    params = params0
    history = [float(state.value)]
    times = [0.0]

    iters_to_target = None
    time_to_target = None

    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    update = jax.jit(solver.update)
    for it in range(maxiter):
        params, state = update(params, state)
        history.append(float(state.value))
        now = time.perf_counter() - t0
        times.append(now)

        gnorm = _grad_norm(loss_fn, params)
        _update_milestones(milestones, milestone_hits, history[-1], it + 1, now)
        if iters_to_target is None and _converged(history[-1], gnorm, f_target, gtol):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
        if bool(state.done):
            break
    wall = time.perf_counter() - t0
    return (
        params,
        history,
        wall,
        times,
        iters_to_target,
        time_to_target,
        milestone_hits,
    )


def run_optax(loss_fn, params0, optimizer, maxiter, stop=None):
    """Run a generic Optax optimizer; returns (params, history, wall, times)."""
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
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    for it in range(maxiter):
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)

        _update_milestones(milestones, milestone_hits, history[-1], it + 1, now)
        if iters_to_target is None and _converged(
            history[-1], float(gnorm), f_target, gtol
        ):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
    wall = time.perf_counter() - t0
    return (
        params,
        history,
        wall,
        times,
        iters_to_target,
        time_to_target,
        milestone_hits,
    )


def run_optax_lbfgs(loss_fn, params0, maxiter, stop=None):
    """Run Optax's L-BFGS (with zoom line search) on the full-batch loss."""
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
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, jnp.linalg.norm(grad)

    params = params0
    history = [float(loss_fn(params))]
    times = [0.0]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    for it in range(maxiter):
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)

        _update_milestones(milestones, milestone_hits, history[-1], it + 1, now)
        if iters_to_target is None and _converged(
            history[-1], float(gnorm), f_target, gtol
        ):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
    wall = time.perf_counter() - t0
    return (
        params,
        history,
        wall,
        times,
        iters_to_target,
        time_to_target,
        milestone_hits,
    )


def main():

    n_classes = 10
    n_train = 5000
    n_test = 1000
    maxiter = 500

    stop = {
        "f_target": 1.1e-1,
        "gtol": 1.0e-4,
        "time_budget": 15.0,
        "milestones": (5.0e-1, 2.0e-1, 1.5e-1, 1.2e-1),
    }

    print("=== MNIST optimizer comparison: QQN vs SGD vs Adam vs L-BFGS ===")
    print("    (QQN variants: line search / oracle / region)")
    print("    Robustness note: the adaptive trust-region over-shrinks under")
    print("    the honest pred model; fixed-radius / spline stacks are the")
    print("    robust fast path (see the iteration-efficiency leaderboard).")
    print(
        f"  classes={n_classes}  n_train={n_train}  n_test={n_test}  "
        f"maxiter={maxiter}\n"
    )
    print(
        f"  shared stop: f_target={stop['f_target']:.1e}  "
        f"gtol={stop['gtol']:.1e}  time_budget={stop['time_budget']:.1f}s\n"
    )

    xtr, ytr, xte, yte = _load_mnist_numpy(n_train, n_test, n_classes)
    dim = xtr.shape[1]

    X_train = jnp.asarray(xtr)
    y_train = jnp.asarray(ytr)
    X_test = jnp.asarray(xte)
    y_test = jnp.asarray(yte)

    loss_fn = make_loss(X_train, y_train, dim, n_classes)

    params0 = init_params(dim, n_classes, jax.random.PRNGKey(42))

    runners = {
        "QQN": lambda: run_qqn(loss_fn, params0, maxiter, stop=stop),
        "QQN-SW": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="strong_wolfe",
            stop=stop,
        ),
        "QQN-BT": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            stop=stop,
        ),
        "QQN-Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            spline=True,
            stop=stop,
        ),
        "QQN-L50Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            stop=stop,
        ),
        "QQN-SplnTR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            spline=True,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L50SplnTR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L100Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
            spline=True,
            stop=stop,
        ),
        "QQN-BTSpln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            spline=True,
            stop=stop,
        ),
        "QQN-Mom": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.9),
            stop=stop,
        ),
        "QQN-Sec": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=SecantOracle(),
            stop=stop,
        ),
        "QQN-And": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=AndersonOracle(window=5),
            stop=stop,
        ),
        "QQN-L50And": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=50), AndersonOracle(window=5)]),
            stop=stop,
        ),
        "QQN-L50Sec": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=50), SecantOracle()]),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-Mom50": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.5),
            stop=stop,
        ),
        "QQN-Mom10": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.1),
            stop=stop,
        ),
        "QQN-Mom01": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.01),
            stop=stop,
        ),
        "QQN-Sh": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=ShampooOracle(update_freq=25),
            stop=stop,
        ),
        "QQN-L5": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=5),
            stop=stop,
        ),
        "QQN-L20": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
            stop=stop,
        ),
        "QQN-L50": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            stop=stop,
        ),
        "QQN-L100": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
            stop=stop,
        ),
        "QQN-Fall": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=10), MomentumOracle()]),
            stop=stop,
        ),
        "QQN-Box": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=BoxRegion(lo=-2.0, hi=2.0),
            stop=stop,
        ),
        "QQN-TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-TR025": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=0.25, adaptive=True),
            stop=stop,
        ),
        "QQN-TR2": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=2.0, adaptive=True),
            stop=stop,
        ),
        "QQN-TRfix": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=False),
            stop=stop,
        ),
        "QQN-Orth": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=OrthantRegion(),
            stop=stop,
        ),
        "QQN-Seq": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=Sequential(
                [BoxRegion(lo=-2.0, hi=2.0), TrustRegion(radius=1.0, adaptive=True)]
            ),
            stop=stop,
        ),
        "QQN-SW+TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="strong_wolfe",
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L20HZ": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="hager_zhang",
            oracle=LBFGSOracle(history_size=20),
            stop=stop,
        ),
        "QQN-L50TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L50TRcc": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(
                radius=1.0,
                adaptive=True,
                shrink=0.5,
                expand=2.0,
                eta_lo=0.1,
                eta_hi=0.75,
            ),
            stop=stop,
        ),
        "QQN-And2": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=AndersonOracle(window=5, beta=1.5),
            stop=stop,
        ),
        "QQN-L100TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L50TR2": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=2.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L50BTTR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-L50WS": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 4.0, "shrink": 0.8, "max_iter": 50},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.5, adaptive=False),
            stop=stop,
        ),
        "QQN-L50Endpt": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=False),
            stop=stop,
        ),
        "QQN-L50WS+": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=False),
            stop=stop,
        ),
        "QQN-AndWS": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.5, "shrink": 0.7, "max_iter": 45},
            oracle=Fallback(
                [LBFGSOracle(history_size=50), AndersonOracle(window=5, beta=1.5)]
            ),
            region=TrustRegion(radius=1.5, adaptive=False),
            stop=stop,
        ),
        "QQN-SplnWS": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            region=TrustRegion(radius=1.5, adaptive=False),
            stop=stop,
        ),
        "QQN-L50TRfix": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=False),
            stop=stop,
        ),
        "QQN-Fast": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=100),
            region=TrustRegion(radius=1.0, adaptive=False),
            stop=stop,
        ),
        "QQN-Best": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        "QQN-Champion": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 3.0, "shrink": 0.75, "max_iter": 45},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.5, adaptive=False),
            stop=stop,
        ),
        "QQN-Apex": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 1.5, "shrink": 0.75, "max_iter": 40},
            oracle=LBFGSOracle(history_size=100),
            spline=True,
            region=TrustRegion(radius=2.0, adaptive=False),
            stop=stop,
        ),
        "QQN-L20Box": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
            region=BoxRegion(lo=-2.0, hi=2.0),
            stop=stop,
        ),
        "SGD": lambda: run_optax(
            loss_fn, params0, optax.sgd(learning_rate=0.5), maxiter, stop=stop
        ),
        "Adam": lambda: run_optax(
            loss_fn, params0, optax.adam(learning_rate=0.05), maxiter, stop=stop
        ),
        "L-BFGS": lambda: run_optax_lbfgs(loss_fn, params0, maxiter, stop=stop),
    }

    results = {}
    for name, runner in runners.items():
        (
            params,
            history,
            wall,
            times,
            iters_to_target,
            time_to_target,
            milestone_hits,
        ) = runner()
        train_acc = float(accuracy(params, X_train, y_train, dim, n_classes))
        test_acc = float(accuracy(params, X_test, y_test, dim, n_classes))

        sparsity = float(jnp.mean((jnp.abs(params) < 1e-6).astype(jnp.float32)))

        reached = iters_to_target is not None

        n_iters = max(len(history) - 1, 1)
        ms_per_iter = (wall / n_iters) * 1e3

        log_hist = np.log10(np.maximum(np.asarray(history), 1e-12))
        if len(log_hist) > 1:
            x_axis = np.linspace(0.0, 1.0, len(log_hist))
            traj_auc = float(np.trapezoid(log_hist, x_axis))
        else:
            traj_auc = float(log_hist[-1])
        results[name] = {
            "final_loss": history[-1],
            "best_loss": min(history),
            "iters": len(history) - 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "wall": wall,
            "sparsity": sparsity,
            "history": history,
            "times": times,
            "reached": reached,
            "iters_to_target": iters_to_target,
            "time_to_target": time_to_target,
            "milestone_hits": milestone_hits,
            "ms_per_iter": ms_per_iter,
            "traj_auc": traj_auc,
        }

    ordered = sorted(results.items(), key=lambda kv: kv[1]["final_loss"])

    lbfgs_ref = results.get("L-BFGS", {}).get("iters_to_target")
    print(
        f"{'optimizer':<10}{'final_loss':>14}{'iters':>8}"
        f"{'train_acc':>12}{'test_acc':>11}{'sparsity':>10}{'time(s)':>10}"
        f"{'ms/it':>8}{'->target':>10}{'t->tgt':>9}{'vs LBFGS':>10}{'AUC':>8}"
    )
    print("-" * 120)
    for name, r in ordered:
        it_tgt = "—" if r["iters_to_target"] is None else f"{r['iters_to_target']}"
        t_tgt = "—" if r["time_to_target"] is None else f"{r['time_to_target']:.3f}"

        if lbfgs_ref is not None and r["iters_to_target"] is not None:
            spd = f"{lbfgs_ref / r['iters_to_target']:.2f}x"
        else:
            spd = "—"
        print(
            f"{name:<10}{r['final_loss']:>14.6e}{r['iters']:>8}"
            f"{r['train_acc']:>12.4f}{r['test_acc']:>11.4f}"
            f"{r['sparsity']:>10.4f}{r['wall']:>10.3f}"
            f"{r['ms_per_iter']:>8.2f}{it_tgt:>10}{t_tgt:>9}{spd:>10}"
            f"{r['traj_auc']:>8.2f}"
        )

    print("\nPareto frontier (loss vs. time — non-dominated variants):")
    pareto = []
    for name, r in ordered:
        dominated = any(
            (o["final_loss"] <= r["final_loss"] and o["wall"] < r["wall"])
            or (o["final_loss"] < r["final_loss"] and o["wall"] <= r["wall"])
            for on, o in results.items()
            if on != name
        )
        if not dominated:
            pareto.append((name, r))
    for name, r in sorted(pareto, key=lambda kv: kv[1]["wall"]):
        print(f"  {name:<12} loss={r['final_loss']:.4e}  time={r['wall']:.3f}s")

    print("\nComposite efficiency score (lower = better; converging only):")
    conv = [
        (name, r) for name, r in results.items() if r["iters_to_target"] is not None
    ]
    if conv:
        best_iters = min(r["iters_to_target"] for _, r in conv)
        best_time = min(r["time_to_target"] for _, r in conv)
        best_loss = min(r["final_loss"] for _, r in conv)
        scored = []
        for name, r in conv:
            ni = r["iters_to_target"] / best_iters
            nt = r["time_to_target"] / best_time
            nl = r["final_loss"] / best_loss

            score = float((ni * nt * nl) ** (1.0 / 3.0))
            scored.append((name, r, score))
        scored.sort(key=lambda x: x[2])
        for name, r, score in scored[:12]:
            print(
                f"  {name:<14} score={score:.3f}  "
                f"iters={r['iters_to_target']:>3}  time={r['time_to_target']:.3f}s  "
                f"final={r['final_loss']:.4e}"
            )

    print("\nIteration-efficiency leaderboard (target reached, fewest iters):")
    converged = [
        (name, r) for name, r in results.items() if r["iters_to_target"] is not None
    ]
    converged.sort(key=lambda kv: (kv[1]["iters_to_target"], kv[1]["wall"]))
    for name, r in converged[:12]:
        spd = (
            f"{lbfgs_ref / r['iters_to_target']:.2f}x" if lbfgs_ref is not None else "—"
        )
        print(
            f"  {name:<14} iters={r['iters_to_target']:>4}  "
            f"time={r['time_to_target']:.3f}s  vs_LBFGS={spd:>6}  "
            f"final={r['final_loss']:.4e}"
        )

    print("\nTrajectory-AUC leaderboard (lower = faster overall descent):")
    auc_ranked = sorted(results.items(), key=lambda kv: kv[1]["traj_auc"])
    for name, r in auc_ranked[:12]:
        print(
            f"  {name:<14} AUC={r['traj_auc']:+.3f}  "
            f"final={r['final_loss']:.4e}  time={r['wall']:.3f}s"
        )

    milestones = stop.get("milestones", ())
    if milestones:
        print("\nConvergence-rate profile (iteration first reaching each loss):")
        header = (
            "  "
            + f"{'optimizer':<12}"
            + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
        )
        print(header)

        tightest = milestones[-1]

        def _sort_key(kv):
            hit = kv[1]["milestone_hits"].get(tightest)
            return hit[0] if hit is not None else 10**9

        for name, r in sorted(results.items(), key=_sort_key):
            cells = []
            for m in milestones:
                hit = r["milestone_hits"].get(m)
                cells.append("—" if hit is None else f"{hit[0]}")
            print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))

    stalled = [(name, r) for name, r in results.items() if r["iters_to_target"] is None]
    if stalled:
        print("\nStall report (never reached the shared target):")
        stalled.sort(key=lambda kv: kv[1]["final_loss"])
        for name, r in stalled:
            if r["wall"] >= stop.get("time_budget", float("inf")) - 0.5:
                cause = "time-budget exhausted"
            elif r["final_loss"] > 0.5:
                cause = "stalled (plateau)"
            else:
                cause = "slow (no target in maxiter)"
            print(
                f"  {name:<14} final={r['final_loss']:.4e}  "
                f"iters={r['iters']:>3}  time={r['wall']:.3f}s  [{cause}]"
            )

    print("\nLoss trajectory (log10, sampled):")
    sample_points = 10
    for name, r in results.items():
        hist = r["history"]
        idxs = np.linspace(0, len(hist) - 1, sample_points).astype(int)
        vals = [f"{np.log10(max(hist[i], 1e-12)):6.2f}" for i in idxs]
        print(f"  {name:<8} " + " ".join(vals))

    cost_pairs = [
        ("spline cost", "QQN-L50", "QQN-L50Spln"),
        ("warm-start cost", "QQN-L50TRfix", "QQN-L50WS+"),
        ("fusion cost", "QQN-L50WS+", "QQN-AndWS", "QQN-SplnWS", "QQN-Apex"),
    ]
    print("\nPer-step cost decomposition (ms/it vs iters trade-off):")
    for title, *variants in cost_pairs:
        present = [v for v in variants if v in results]
        if len(present) < 2:
            continue
        print(f"  [{title}]")
        for v in present:
            r = results[v]
            it_tgt = r["iters_to_target"]
            it_str = "—" if it_tgt is None else f"{it_tgt}"
            print(
                f"    {v:<13} ms/it={r['ms_per_iter']:>7.2f}  "
                f"->target={it_str:>4}  total={r['wall']:.3f}s"
            )

    ab_pairs = [
        (
            "oracle: L-BFGS history",
            "QQN-L5",
            "QQN",
            "QQN-L20",
            "QQN-L50",
            "QQN-L100",
        ),
        (
            "oracle: momentum beta",
            "QQN-Mom01",
            "QQN-Mom10",
            "QQN-Mom50",
            "QQN-Mom",
        ),
        (
            "oracle: accelerator class (Mom vs Shampoo)",
            "QQN-Mom10",
            "QQN-Sh",
        ),
        (
            "region: trust radius",
            "QQN-TR025",
            "QQN-TR",
            "QQN-TR2",
        ),
        ("region: trust adaptivity", "QQN-TRfix", "QQN-TR"),
        (
            "region: combinator (Sequential box+TR)",
            "QQN-TR",
            "QQN-Box",
            "QQN-Seq",
        ),
        (
            "search: line search (oracle=L-BFGS-10)",
            "QQN",
            "QQN-BT",
            "QQN-SW",
            "QQN-Spln",
        ),
        (
            "spline: best-of-breed refinement",
            "QQN-Spln",
            "QQN-BTSpln",
            "QQN-L50Spln",
            "QQN-L100Spln",
            "QQN-SplnTR",
            "QQN-L50SplnTR",
        ),
        (
            "best-of-breed: L50 region (adaptive vs fixed)",
            "QQN-L50",
            "QQN-L50TR",
            "QQN-L50TRfix",
            "QQN-L50TR2",
            "QQN-L50BTTR",
        ),
        (
            "best-of-breed: L100 combos",
            "QQN-L100",
            "QQN-L100TR",
        ),
        (
            "performance: warm-start (fixed TR, the robust speed lever)",
            "QQN-L50BTTR",
            "QQN-L50WS+",
            "QQN-L50WS",
        ),
        (
            "performance: endpoint / fixed-TR robustness",
            "QQN-L50TRfix",
            "QQN-L50Endpt",
        ),
        (
            "champion: diversity-preserving best stack",
            "QQN-L50BTTR",
            "QQN-L50TRfix",
            "QQN-Fast",
            "QQN-Champion",
            "QQN-Apex",
        ),
        (
            "performance: warm-start fusion (oracle + spline levers)",
            "QQN-L50WS+",
            "QQN-AndWS",
            "QQN-SplnWS",
            "QQN-Apex",
        ),
        (
            "best-of-breed: full stack",
            "QQN-L50BTTR",
            "QQN-L50SplnTR",
            "QQN-Best",
        ),
    ]

    print("\nA/B controlled comparisons (vs first column = baseline):")

    for title, *variants in ab_pairs:
        present = [v for v in variants if v in results]
        if len(present) < 2:
            continue
        base = results[present[0]]
        print(f"  [{title}]")
        for v in present:
            r = results[v]
            d_iters = r["iters"] - base["iters"]
            d_wall = r["wall"] - base["wall"]
            marker = " (baseline)" if v == present[0] else ""
            print(
                f"    {v:<11} iters={r['iters']:>3} (Δ{d_iters:+d})"
                f"  loss={r['final_loss']:.3e}"
                f"  time={r['wall']:.3f}s (Δ{d_wall:+.3f}){marker}"
            )

    try:
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure(figsize=(7, 5))
        baselines = {"SGD", "Adam", "L-BFGS"}
        for name, r in results.items():
            if name in baselines:
                plt.semilogy(r["history"], label=name, linestyle="--", linewidth=2)
            else:
                plt.semilogy(r["history"], label=name, alpha=0.85)
        plt.xlabel("iteration")
        plt.ylabel("full-batch loss")
        plt.title("MNIST optimizer comparison (QQN variants vs baselines)")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        out = os.path.join(results_dir, f"mnist_comparison_{timestamp}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n[plot] Saved convergence plot to {out}")

        plt.figure(figsize=(7, 5))
        for name, r in results.items():
            if name in baselines:
                plt.semilogy(
                    r["times"], r["history"], label=name, linestyle="--", linewidth=2
                )
            else:
                plt.semilogy(r["times"], r["history"], label=name, alpha=0.85)
        plt.xlabel("wall-clock time (s)")
        plt.ylabel("full-batch loss")
        plt.title("MNIST optimizer comparison vs time (QQN variants vs baselines)")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        out_time = os.path.join(results_dir, f"mnist_comparison_time_{timestamp}.png")
        plt.savefig(out_time, dpi=120, bbox_inches="tight")
        print(f"[plot] Saved time-based convergence plot to {out_time}")
    except Exception:
        print("\n[plot] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()

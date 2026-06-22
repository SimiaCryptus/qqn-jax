"""Smoke tests for the MNIST comparison experiment.

These tests use the synthetic-data fallback so they never require a
network download, and they verify that each optimizer path runs and
produces a finite, decreasing-ish loss trajectory.
"""

import importlib.util
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Import the example module by path (examples/ is not a package).
_HERE = os.path.dirname(__file__)
_EXAMPLE_PATH = os.path.join(_HERE, "..", "examples", "mnist_comparison.py")
_spec = importlib.util.spec_from_file_location("mnist_comparison", _EXAMPLE_PATH)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


@pytest.fixture(scope="module")
def problem():
    """A tiny synthetic classification problem and its loss."""
    n_classes = 3
    xtr, ytr, xte, yte = mc._synthetic(
        n_train=200, n_test=100, n_classes=n_classes, dim=20
    )
    dim = xtr.shape[1]
    X = jnp.asarray(xtr)
    y = jnp.asarray(ytr)
    loss_fn = mc.make_loss(X, y, dim, n_classes)
    params0 = mc.init_params(dim, n_classes, jax.random.PRNGKey(0))
    return loss_fn, params0, X, y, dim, n_classes


def test_loss_is_finite(problem):
    loss_fn, params0, *_ = problem
    v = float(loss_fn(params0))
    assert np.isfinite(v)
    assert v > 0.0


def test_qqn_runs_and_decreases(problem):
    loss_fn, params0, *_ = problem
    _, history, wall, *_ = mc.run_qqn(loss_fn, params0, maxiter=10)
    assert len(history) >= 2
    assert all(np.isfinite(h) for h in history)
    assert history[-1] <= history[0]
    assert wall >= 0.0


def test_optax_optimizers_run(problem):
    loss_fn, params0, *_ = problem
    for opt in (optax.sgd(0.1), optax.adam(0.05)):
        _, history, *_ = mc.run_optax(loss_fn, params0, opt, maxiter=10)
        assert all(np.isfinite(h) for h in history)
        assert history[-1] <= history[0] + 1e-6


def test_optax_lbfgs_runs(problem):
    loss_fn, params0, *_ = problem
    _, history, *_ = mc.run_optax_lbfgs(loss_fn, params0, maxiter=10)
    assert all(np.isfinite(h) for h in history)
    assert history[-1] <= history[0] + 1e-6


def test_accuracy_in_range(problem):
    loss_fn, params0, X, y, dim, n_classes = problem
    acc = float(mc.accuracy(params0, X, y, dim, n_classes))
    assert 0.0 <= acc <= 1.0


def test_synthetic_shapes_consistent():
    xtr, ytr, xte, yte = mc._synthetic(n_train=50, n_test=25, n_classes=4, dim=8)
    assert xtr.shape == (50, 8)
    assert xte.shape == (25, 8)
    assert ytr.shape == (50,)
    assert yte.shape == (25,)
    assert int(ytr.min()) >= 0
    assert int(ytr.max()) <= 3


def test_qqn_history_is_non_increasing(problem):
    loss_fn, params0, *_ = problem
    _, history, *_ = mc.run_qqn(loss_fn, params0, maxiter=15)
    # QQN should not increase the loss between the first and last point.
    assert history[-1] <= history[0] + 1e-6


def test_optax_adam_makes_progress(problem):
    loss_fn, params0, *_ = problem
    _, history, *_ = mc.run_optax(loss_fn, params0, optax.adam(0.05), maxiter=20)
    assert history[-1] <= history[0] + 1e-6
    assert all(np.isfinite(h) for h in history)


def test_wall_time_is_nonnegative(problem):
    loss_fn, params0, *_ = problem
    _, _, wall, *_ = mc.run_qqn(loss_fn, params0, maxiter=5)
    assert wall >= 0.0

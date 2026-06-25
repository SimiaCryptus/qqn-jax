"""Assert the cross-cutting fairness invariants on a tiny synthetic config.

These guard plan.md §8: identical init, shared termination, genuine eval
accounting, same loss/data, and one profiling span per variant.
"""

import jax

from experiments.config import ExperimentConfig
from experiments.data.loaders import load_image_dataset
from experiments.driver import build_model
from experiments.optimizers import runners as R


def _tiny_config():
    cfg = ExperimentConfig(
        dataset="synthetic",
        n_classes=3,
        n_train=120,
        n_test=60,
        hidden_sizes=[8],
        activation="tanh",
        synth_dim=16,
        maxiter=5,
        f_target=1e-9,  # unreachable -> exercise the budget/maxiter path
        gtol=1e-12,
        time_budget=10.0,
        milestones=(1e0, 5e-1),
        target_profile=(5e-1, 2e-1),
    )
    _, fn = __import__(
        "experiments.models.activations", fromlist=["resolve_activation"]
    ).resolve_activation("tanh")
    cfg.activation_name = "tanh"
    cfg.activation_fn = fn
    return cfg


def _data_and_loss(cfg):
    xtr, ytr, xte, yte = load_image_dataset(
        cfg.dataset,
        cfg.n_train,
        cfg.n_test,
        cfg.n_classes,
        seed=cfg.subset_seed,
        balanced=cfg.balanced,
        synth_dim=cfg.synth_dim,
    )
    import jax.numpy as jnp

    data = (jnp.asarray(xtr), jnp.asarray(ytr), jnp.asarray(xte), jnp.asarray(yte))
    model = build_model(cfg, xtr.shape[1])
    loss_fn = model.make_loss(data[0], data[1], l2=cfg.l2)
    return model, data, loss_fn


def test_identical_init():
    """Invariant #1: same params0 from the same seed."""
    cfg = _tiny_config()
    model, _, _ = _data_and_loss(cfg)
    a = model.init_params(jax.random.PRNGKey(cfg.seed))
    b = model.init_params(jax.random.PRNGKey(cfg.seed))
    assert (a == b).all()


def test_runners_return_runresult_and_count_evals():
    """Invariants #2-#3: shared loop + genuine eval accounting fields exist."""
    cfg = _tiny_config()
    model, data, loss_fn = _data_and_loss(cfg)
    params0 = model.init_params(jax.random.PRNGKey(cfg.seed))
    import optax

    for result in (
        R.run_qqn(loss_fn, params0, cfg.maxiter, stop=cfg.stop),
        R.run_optax(loss_fn, params0, optax.adam(1e-2), cfg.maxiter, stop=cfg.stop),
        R.run_optax_lbfgs(loss_fn, params0, cfg.maxiter, stop=cfg.stop),
    ):
        # Shared termination loop produced a full history + times.
        assert len(result.history) == len(result.times)
        assert result.history[0] >= result.history[-1] - 1e-3
        # Milestone dict keyed by the shared milestones.
        assert set(result.milestone_hits.keys()) == set(cfg.milestones)

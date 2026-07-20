"""ExperimentConfig: one dataclass carrying every knob + env binding.

``from_env`` performs the env binding preserving current defaults and the
warning-on-malformed behavior (via ``experiments.env``). The headline
fashion defaults live here; callers pass overrides for other benchmarks.
"""

from dataclasses import dataclass, field
from typing import Union

from experiments import env
from experiments.models.activations import parse_activation
from experiments.models.topology import parse_hidden_sizes

__all__ = ["ExperimentConfig"]


@dataclass
class ExperimentConfig:
    dataset: str = "fashion_mnist"
    n_classes: int = 10
    n_train: int = 25000
    n_test: int = 5000
    hidden_sizes: list = field(default_factory=lambda: [256, 256, 256])
    activation: Union[str, list] = "tanh,gelu"
    l2: float = 1e-4
    seed: int = 42
    subset_seed: int = 0
    synth_dim: int = 784
    maxiter: int = 1_000_000
    balanced: bool = True

    f_target: float = 2e-2
    gtol: float = 1e-8
    time_budget: float = 150.0
    milestones: tuple = (1e0, 5e-1, 2e-1, 1e-1, 5e-2, 1e-2)
    target_profile: tuple = (2e-1, 1e-1, 6e-2, 4e-2, 2e-2)

    sgd_lr: float = 0.05
    adam_lr: float = 0.01

    activation_name: object = None
    activation_fn: object = None

    @classmethod
    def from_env(cls, **defaults):
        """Build a config from env vars, layered over keyword ``defaults``.

        ``defaults`` lets each benchmark supply its own narrative defaults
        (e.g. the sparse benchmark passes ``activation_default="tanh"`` and a
        different ``hidden_default``). Recognized control keys:
            activation_default, hidden_default_hidden, hidden_default_depth,
            hidden_default (explicit list).
        """
        activation_default = defaults.pop("activation_default", "tanh,gelu")
        hidden_default_hidden = defaults.pop("hidden_default_hidden", 256)
        hidden_default_depth = defaults.pop("hidden_default_depth", 3)
        hidden_default = defaults.pop("hidden_default", None)

        base = cls(**defaults)

        dataset = env.env_str("DATASET", base.dataset).lower()
        if dataset not in ("mnist", "fashion_mnist"):
            print(f"[config] Unknown DATASET={dataset!r}; falling back to 'mnist'.")
            dataset = "mnist"

        n_classes = env.env_int("N_CLASSES", base.n_classes)
        if n_classes < 2:
            print(f"[config] N_CLASSES={n_classes} too small; using 2.")
            n_classes = 2

        hidden_sizes = parse_hidden_sizes(
            default_hidden=hidden_default_hidden,
            default_depth=hidden_default_depth,
            default=hidden_default,
        )
        activation_name, activation_fn = parse_activation(
            len(hidden_sizes), default=activation_default
        )

        cfg = cls(
            dataset=dataset,
            n_classes=n_classes,
            n_train=env.env_int("N_TRAIN", base.n_train),
            n_test=env.env_int("N_TEST", base.n_test),
            hidden_sizes=hidden_sizes,
            activation=activation_name,
            l2=env.env_float("L2", base.l2),
            seed=env.env_int("SEED", base.seed),
            subset_seed=env.env_int("SUBSET_SEED", base.subset_seed),
            synth_dim=env.env_int("SYNTH_DIM", base.synth_dim),
            maxiter=env.env_int("MAXITER", base.maxiter),
            balanced=base.balanced,
            f_target=env.env_float("F_TARGET", base.f_target),
            gtol=env.env_float("GTOL", base.gtol),
            time_budget=env.env_float("TIME_BUDGET", base.time_budget),
            milestones=env.env_float_list("MILESTONES", base.milestones),
            target_profile=env.env_float_list("TARGET_PROFILE", base.target_profile),
            sgd_lr=env.env_float("SGD_LR", base.sgd_lr),
            adam_lr=env.env_float("ADAM_LR", base.adam_lr),
        )
        cfg.activation_name = activation_name
        cfg.activation_fn = activation_fn
        return cfg

    @property
    def stop(self):
        """The shared termination dict passed to every runner."""
        return {
            "f_target": self.f_target,
            "gtol": self.gtol,
            "time_budget": self.time_budget,
            "milestones": tuple(self.milestones),
        }

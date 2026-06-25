"""SparseConfig: knobs for the sparse / quantization MNIST benchmark.

The sparse benchmark has a different narrative from the MLP-comparison
driver: a pytree MLP trained with region-heavy sparsity / quantization
machinery, a base->polish cross-product, and sparsity / quant metrics.
Its env-var contract (DATASET, N_TRAIN, N_TEST, HIDDEN_SIZES, HIDDEN,
DEPTH, ACTIVATION, MAXITER, POLISH_MAXITER, LINE_SEARCH, HISTORY_SIZE,
L2, L1_SCALE, QUANT_SCALE, QBITS, SEED) is preserved verbatim.
"""

from dataclasses import dataclass, field
from typing import Union

from experiments import env
from experiments.models.activations import parse_activation
from experiments.models.topology import parse_hidden_sizes

__all__ = ["SparseConfig"]


@dataclass
class SparseConfig:
    dataset: str = "mnist"
    n_train: int = 10000
    n_test: int = 5000
    seed: int = 0
    line_search: str = "strong_wolfe"
    history_size: int = 10
    l2: float = 1e-4
    l1_scale: float = 1e-5
    quant_scale: float = 1e-4
    qbits: int = 4
    quant_lo: float = -1.0
    quant_hi: float = 1.0
    maxiter: int = 50000
    polish_maxiter: int = 5000
    hidden_sizes: list = field(default_factory=lambda: [64, 64])
    activation: Union[str, list] = "tanh"
    # --- runtime-resolved (filled by from_env) ---
    activation_name: object = None
    activation_fn: object = None

    @classmethod
    def from_env(cls):
        """Build a sparse-benchmark config from the env-var contract."""
        base = cls()

        dataset = env.env_str("DATASET", base.dataset).lower()
        if dataset not in ("mnist", "fashion_mnist"):
            print(
                f"[config] Unknown DATASET={dataset!r}; falling back to 'mnist'. "
                "Valid values: 'mnist', 'fashion_mnist'."
            )
            dataset = "mnist"

        hidden_sizes = parse_hidden_sizes(
            default_hidden=64, default_depth=2, default=[64, 64]
        )
        activation_names, activation_fns = parse_activation(
            len(hidden_sizes), default="tanh"
        )

        maxiter = env.env_int("MAXITER", base.maxiter)
        polish_maxiter = env.env_int("POLISH_MAXITER", max(1, maxiter // 10))

        cfg = cls(
            dataset=dataset,
            n_train=env.env_int("N_TRAIN", base.n_train),
            n_test=env.env_int("N_TEST", base.n_test),
            seed=env.env_int("SEED", base.seed),
            line_search=env.env_str("LINE_SEARCH", base.line_search),
            history_size=env.env_int("HISTORY_SIZE", base.history_size),
            l2=env.env_float("L2", base.l2),
            l1_scale=env.env_float("L1_SCALE", base.l1_scale),
            quant_scale=env.env_float("QUANT_SCALE", base.quant_scale),
            qbits=env.env_int("QBITS", base.qbits),
            quant_lo=base.quant_lo,
            quant_hi=base.quant_hi,
            maxiter=maxiter,
            polish_maxiter=polish_maxiter,
            hidden_sizes=hidden_sizes,
            activation=activation_names,
        )
        cfg.activation_name = activation_names
        cfg.activation_fn = activation_fns
        return cfg

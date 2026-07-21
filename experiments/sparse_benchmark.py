"""Benchmark: sparse MNIST classification with QQN + region machinery.

Thin entry point: builds a ``SparseConfig`` from the environment and runs
the ported sparse / quantization benchmark driver. All shared machinery
(data loading, pytree MLP, regions, reporting, plots) now lives in the
``experiments`` package — see ``experiments/sparse_driver.py``.

The env-var contract is unchanged (see ``experiments/sparse_config.py``):
    DATASET, N_TRAIN, N_TEST, HIDDEN_SIZES, HIDDEN, DEPTH, ACTIVATION,
    MAXITER, POLISH_MAXITER, LINE_SEARCH, HISTORY_SIZE, L2, L1_SCALE,
    QUANT_SCALE, QBITS, SEED.

Examples::
    ACTIVATION=relu DEPTH=3 HIDDEN=128 python -m examples.mnist_sparse_benchmark
    DATASET=fashion_mnist N_TRAIN=20000 python -m examples.mnist_sparse_benchmark
    ACTIVATION=tanh,gelu HIDDEN_SIZES=128,64 python -m examples.mnist_sparse_benchmark
"""

from experiments import SparseConfig, run_sparse_experiment


def main():
    config = SparseConfig.from_env()
    run_sparse_experiment(config)


if __name__ == "__main__":
    main()

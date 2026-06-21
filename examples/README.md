# QQN-JAX Examples

## Sparse MNIST Benchmark

`mnist_sparse_benchmark.py` trains a small MLP on a subset of MNIST and
compares three QQN configurations to demonstrate **projective regions**:

| Config              | Region                                  | Goal                       |
|---------------------|-----------------------------------------|----------------------------|
| `baseline (dense)`  | `None`                                  | Reference (no constraint)  |
| `orthant (sparse)`  | `OrthantRegion(l1=1e-3)`                | Weight sparsity (OWL-QN)   |
| `orthant + trust`   | `Sequential([Orthant, TrustRegion])`    | Sparsity + step control    |

The `OrthantRegion` zeroes weight coordinates whose sign would flip during
an update, inducing sparsity in the trained network. The benchmark reports
final loss, test accuracy, the fraction of near-zero weights (sparsity),
and wall-clock time.

### Run

```bash
python -m examples.mnist_sparse_benchmark
```

### Data

The loader uses `tensorflow_datasets` if available; otherwise it falls
back to synthetic data so the example runs end-to-end in any environment.
Install MNIST support with:

```bash
pip install tensorflow-datasets
```

### Notes

- All region projections are pure JAX, so `solver.run` is wrapped in
  `jax.jit` for the benchmark.
- Increase `n_train`, `sizes`, or `maxiter` in `main()` for a heavier run.
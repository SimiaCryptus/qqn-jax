# Shampoo Oracle

A structure-aware, preconditioned optimization oracle based on the
**Shampoo** algorithm (Gupta, Koren & Singer, 2018). This document
describes the implementation found in `shampoo.py` and discusses its
design, mathematics, and practical trade-offs.

---

## 1. Overview

Shampoo is a *full-matrix-inspired* preconditioner that maintains
second-moment statistics of the gradients and uses their (matrix)
inverse roots to rescale the gradient before taking a step. Unlike
diagonal methods (e.g. Adam) that only capture per-coordinate scaling,
Shampoo captures **correlations** between coordinates through
Kronecker-factored preconditioners.

The oracle exposes the standard `Oracle` interface used throughout
`qqn-jax`:

- `init(params)` — build the initial preconditioner state.
- `direction(params, grad, state)` — return a search direction and the
  updated state.
- `update(state, info)` — post-step hook (a no-op here).

---

## 2. State

```python
class ShampooState(NamedTuple):
    L: jnp.ndarray   # left preconditioner statistics, shape (n, n)
    R: jnp.ndarray   # right preconditioner statistics, shape (1, 1)
    step: jnp.ndarray  # int32 step counter
```

Because the flat-vector setting reshapes the gradient `g` (shape
`(n,)`) into a column vector of shape `(n, 1)`, the "matrix block"
has:

- a **left** factor `L` of shape `(n, n)` accumulating `g gᵀ`, and
- a **right** factor `R` of shape `(1, 1)` accumulating `gᵀ g`
  (a scalar).

This is the natural Kronecker factorization of the full-matrix
second-moment statistic for a rank-1 (column) gradient block.

---

## 3. Mathematics

### 3.1 Statistics accumulation

At each step Shampoo accumulates outer products:

```
L ← L + g gᵀ           (n × n)
R ← R + gᵀ g           (1 × 1, a scalar)
```

These are un-decayed running sums of the gradient second moments
along each of the two "axes" of the reshaped block.

### 3.2 Inverse p-th roots

The preconditioner uses inverse fourth-roots of the two factors.
For an order-2 tensor block the exponent is `-1/(2·order) = -1/4`,
which is why `p = 4.0` is passed to `_matrix_inverse_pth_root`.

The helper computes `mat^{-1/p}` for a symmetric PSD matrix through an
eigendecomposition:

```
mat  ← mat + ε I
w, v ← eigh(mat)
w    ← max(w, ε)             # clamp for numerical safety
mat^{-1/p} = (v · w^{-1/p}) vᵀ
```

The `ε I` regularization guarantees positive-definiteness (so `eigh`
is well-behaved), and clamping the eigenvalues avoids division by
tiny or negative values from round-off.

### 3.3 Preconditioned direction

Given `Lr = L^{-1/4}` and `Rr = R^{-1/4}`, the preconditioned
gradient is:

```
precond = (Lr · g) · Rr
d       = -precond
```

which is the Shampoo update for a single 2-D block. The negative sign
makes `d` a descent direction consumed by the outer optimizer/line
search.

---

## 4. Amortized refresh cadence

Recomputing matrix inverse roots via `eigh` is `O(n³)` and expensive.
To keep the per-step cost amortized, the inverse roots are only
recomputed every `update_freq` steps:

```python
do_refresh = (state.step % update_freq) == 0
```

The branch is expressed with `jax.lax.cond` so the entire
`direction` function remains `jit`-friendly (no Python-level control
flow on traced values):

- **refresh** branch: update `L`, recompute both inverse roots, and
  return the fully preconditioned direction plus the new `L`.
- **keep** branch: return the raw (negated) gradient and leave `L`
  unchanged.

Note that `R` is updated *every* step regardless of the branch, while
`L` is only committed to on refresh steps. On non-refresh steps the
direction falls back to plain gradient descent (`d = -g`), which keeps
progress flowing between expensive refreshes.

---

## 5. Parameters

| Parameter     | Default | Meaning                                                        |
|---------------|---------|----------------------------------------------------------------|
| `block_size`  | `128`   | Reserved for block-partitioning large tensors (unused in the flat-vector path). |
| `update_freq` | `20`    | Static cadence (in steps) for recomputing inverse roots.       |
| `epsilon`     | `1e-6`  | Regularization / eigenvalue clamp for numerical stability.     |

---

## 6. Discussion

### Strengths
- Captures cross-coordinate curvature through the left factor `L`,
  unlike diagonal adaptive methods.
- The Kronecker factorization keeps the preconditioner tractable and
  the eigendecomposition is done on symmetric PSD matrices.
- Amortized refresh keeps the average per-step cost manageable and the
  code fully compatible with `jax.jit`.

### Limitations & caveats
- The current flat-vector implementation forms an `n × n` matrix `L`,
  giving `O(n²)` memory and `O(n³)` refresh cost. This does **not**
  scale to very large `n`; genuine block-partitioning (the intent of
  `block_size`) would be required for large models.
- Statistics are accumulated as un-decayed sums; there is no
  exponential moving average, so very long runs may over-accumulate.
- On non-refresh steps the method degrades to gradient descent, so the
  effective preconditioning is only as fresh as the last refresh.

---

## 7. Interface Summary

```python
oracle = ShampooOracle(block_size=128, update_freq=20, epsilon=1e-6)
state  = oracle.init(params)
d, state = oracle.direction(params, grad, state)
state  = oracle.update(state, info)
```

The returned direction `d` is a descent direction intended to be fed
into a line search or fixed step-size update in the surrounding
optimization loop.

## References

- V. Gupta, T. Koren, Y. Singer. *Shampoo: Preconditioned Stochastic
  Tensor Optimization.* ICML 2018.
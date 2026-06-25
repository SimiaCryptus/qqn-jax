"""Subsetting strategies for image datasets.

``balanced_subset`` draws a reproducible, class-balanced random subset (the
fashion strategy: a balanced full-batch objective has a better-conditioned,
more representative Hessian). ``permutation_subset`` takes a reproducible
random permutation prefix (the sparse strategy). ``synthetic`` generates a
linearly-separable-ish Gaussian-blob fallback.
"""

import numpy as np

__all__ = ["balanced_subset", "permutation_subset", "synthetic"]


def balanced_subset(xtr, ytr, xte, yte, n_train, n_test, n_classes, seed=0):
    """Keep only the first ``n_classes`` classes and class-balanced subsample."""
    rng = np.random.default_rng(seed)
    train_mask = ytr < n_classes
    test_mask = yte < n_classes
    xtr, ytr = xtr[train_mask], ytr[train_mask]
    xte, yte = xte[test_mask], yte[test_mask]

    def _balanced_indices(labels, n_total):
        n_total = min(n_total, labels.shape[0])
        per_class = max(n_total // n_classes, 1)
        idxs = []
        for c in range(n_classes):
            cls_idx = np.where(labels == c)[0]
            if cls_idx.size == 0:
                continue
            take = min(per_class, cls_idx.size)
            idxs.append(rng.choice(cls_idx, size=take, replace=False))
        idxs = (
            np.concatenate(idxs) if idxs else np.arange(min(n_total, labels.shape[0]))
        )
        rng.shuffle(idxs)
        return idxs[:n_total]

    tr_idx = _balanced_indices(ytr, n_train)
    te_idx = _balanced_indices(yte, n_test)
    xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    xte, yte = xte[te_idx], yte[te_idx]
    return xtr, ytr.astype(np.int32), xte, yte.astype(np.int32)


def permutation_subset(xtr, ytr, xte, yte, n_train, n_test, n_classes, seed=0):
    """Keep first ``n_classes`` classes and take a random permutation prefix."""
    rng = np.random.default_rng(seed)
    train_mask = ytr < n_classes
    test_mask = yte < n_classes
    xtr, ytr = xtr[train_mask], ytr[train_mask]
    xte, yte = xte[test_mask], yte[test_mask]

    tr_perm = rng.permutation(xtr.shape[0])[: min(n_train, xtr.shape[0])]
    te_perm = rng.permutation(xte.shape[0])[: min(n_test, xte.shape[0])]
    xtr, ytr = xtr[tr_perm], ytr[tr_perm]
    xte, yte = xte[te_perm], yte[te_perm]
    return xtr, ytr.astype(np.int32), xte, yte.astype(np.int32)


def synthetic(n_train, n_test, n_classes, dim=784, seed=0):
    """Generate a linearly-separable-ish synthetic classification set."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(n_classes, dim)).astype(np.float32)

    def make(n):
        y = rng.integers(0, n_classes, size=n).astype(np.int32)
        x = centers[y] + rng.normal(scale=1.0, size=(n, dim)).astype(np.float32)
        return x.astype(np.float32), y

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte

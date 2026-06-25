"""Unified image-dataset loader: keras -> tfds -> torchvision -> synthetic.

Single ``load_image_dataset`` implementing the canonical fallback chain. The
``balanced`` flag selects class-balanced (fashion) vs. permutation (sparse)
subsetting so both historical callers collapse to one code path.
"""

import numpy as np

from experiments.data.subset import balanced_subset, permutation_subset, synthetic

__all__ = ["load_image_dataset"]


_INSTALL_HINT = (
    "[data] No dataset backend found. Install ONE of the following to use a\n"
    "       real (Fashion-)MNIST corpus instead of the synthetic fallback:\n"
    "           pip install tensorflow            # Keras datasets (MNIST + Fashion)\n"
    "           pip install torch torchvision     # torchvision datasets\n"
)


def _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes, *, balanced, seed):
    strategy = balanced_subset if balanced else permutation_subset
    return strategy(xtr, ytr, xte, yte, n_train, n_test, n_classes, seed=seed)


def load_image_dataset(
    dataset,
    n_train,
    n_test,
    n_classes,
    *,
    seed=0,
    balanced=True,
    synth_dim=784,
):
    """Load a real (Fashion-)MNIST subset; fall back to synthetic.

    Args:
        dataset: ``"mnist"`` or ``"fashion_mnist"``.
        n_train, n_test: full-batch subset sizes.
        n_classes: number of leading classes to keep.
        seed: subsampling RNG seed (reproducible subset).
        balanced: class-balanced subsetting (True) vs. permutation (False).
        synth_dim: feature dim for the synthetic fallback.

    Returns:
        (X_train, y_train, X_test, y_test) numpy arrays; images flattened to
        (N, 784) float32 in [0, 1] and int32 labels.
    """
    # --- Attempt 1: tensorflow / keras ---
    try:
        if dataset == "fashion_mnist":
            from tensorflow.keras.datasets import fashion_mnist as ds  # type: ignore
        else:
            from tensorflow.keras.datasets import mnist as ds  # type: ignore

        (xtr, ytr), (xte, yte) = ds.load_data()
        xtr = xtr.reshape(xtr.shape[0], -1).astype(np.float32) / 255.0
        xte = xte.reshape(xte.shape[0], -1).astype(np.float32) / 255.0
        print(f"[data] Loaded {dataset} via tensorflow.keras.")
        return _subset(
            xtr,
            ytr,
            xte,
            yte,
            n_train,
            n_test,
            n_classes,
            balanced=balanced,
            seed=seed,
        )
    except Exception:
        pass

    # --- Attempt 2: torchvision ---
    try:
        from torchvision import datasets  # type: ignore

        if dataset == "fashion_mnist":
            cls = datasets.FashionMNIST
            root = "./_fashion_mnist_data"
        else:
            cls = datasets.MNIST
            root = "./_mnist_data"

        train = cls(root=root, train=True, download=True)
        test = cls(root=root, train=False, download=True)
        xtr = train.data.numpy().reshape(len(train), -1).astype(np.float32) / 255.0
        ytr = train.targets.numpy()
        xte = test.data.numpy().reshape(len(test), -1).astype(np.float32) / 255.0
        yte = test.targets.numpy()
        print(f"[data] Loaded {dataset} via torchvision.")
        return _subset(
            xtr,
            ytr,
            xte,
            yte,
            n_train,
            n_test,
            n_classes,
            balanced=balanced,
            seed=seed,
        )
    except Exception:
        pass

    # --- Fallback: synthetic "MNIST-like" Gaussian blobs ---
    print(_INSTALL_HINT)
    print(f"[data] Real {dataset} unavailable; using synthetic Gaussian blobs.")
    return synthetic(n_train, n_test, n_classes, dim=synth_dim, seed=seed)

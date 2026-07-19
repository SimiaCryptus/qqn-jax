# Libraries & Dependencies

This document explains how to resolve the most common setup warnings
you may encounter when running the experiment:

1. A CUDA-enabled `jaxlib` is not installed (GPU fallback to CPU).
2. The real MNIST dataset is unavailable (fallback to synthetic data).

---

## 1. Installing JAX / jaxlib

If you see a message like:

```
ERROR: An NVIDIA GPU may be present on this machine, but a CUDA-enabled
jaxlib is not installed. Falling back to cpu.
```

it means JAX is installed, but only the CPU build of `jaxlib` is present.

### CPU-only (works everywhere)

```bash
pip install --upgrade "jax[cpu]"
```

### GPU (NVIDIA / CUDA)

First confirm your CUDA and cuDNN versions:

```bash
nvidia-smi          # check the driver / CUDA version
nvcc --version      # check the CUDA toolkit version (if installed)
```

Then install the matching CUDA-enabled wheel. For recent JAX releases the
CUDA libraries can be pulled in automatically:

```bash
# CUDA 12.x (most modern setups)
pip install --upgrade "jax[cuda12]"

# CUDA 11.x
pip install --upgrade "jax[cuda11]"
```

If you manage CUDA yourself and want the wheels from Google's index:

```bash
pip install --upgrade "jax[cuda12_local]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Verifying the install

```shell
python -c "import jax; print(jax.devices())"
```

If `jax.devices()` shows only `CpuDevice`, the CPU fallback is still in
effect — recheck that your CUDA toolkit/driver versions match the
`jaxlib` build you installed.

---

## 2. Installing the real MNIST dataset

If you see:

```
[data] Real MNIST unavailable; using synthetic Gaussian blobs.
```

the script could not import `torchvision` or `tensorflow`, so it generated
a synthetic Gaussian-blob "MNIST-like" dataset instead. The experiment will
still run, but to use the **real** MNIST data install one of the following.

### Option A: via torchvision (PyTorch)

```bash
pip install torch torchvision
```

MNIST will be downloaded automatically on first use:

```python
from torchvision import datasets, transforms

datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
```

### Option B: via TensorFlow / Keras

```bash
pip install tensorflow
```

MNIST is bundled with Keras and downloaded on first use:

```python
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

### Offline / manual download

If the machine has no internet access, download the dataset elsewhere and
place the files in the cache directory the loader expects:

- `torchvision` default: `./data/MNIST/raw/`
- `tf.keras` default:    `~/.keras/datasets/mnist.npz`

The four raw MNIST files are:

```
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

---

## Quick install (CPU + real MNIST)

To get rid of both warnings on a CPU-only machine:

```bash
pip install --upgrade "jax[cpu]" torch torchvision
```

For GPU machines, swap the JAX line for the CUDA variant above.

## Data loading behavior

The script tries to load MNIST via `torchvision` or `tensorflow` if
available. If neither is installed, it falls back to a synthetic
Gaussian-blob "MNIST-like" dataset so the experiment always runs.
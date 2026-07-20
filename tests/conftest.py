"""Shared pytest fixtures and configuration for the QQN-JAX test suite."""

import jax
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def rng_key():
    """A deterministic PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def mlp_params():
    """A small list-of-dicts MLP parameter pytree used by regularizer tests."""
    import jax.numpy as jnp

    return [
        {"w": jnp.array([[0.1, -0.2], [0.3, 0.4]]), "b": jnp.array([0.0, 0.5])},
        {"w": jnp.array([[0.5], [-0.5]]), "b": jnp.array([1.0])},
    ]

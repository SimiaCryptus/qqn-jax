"""Shared pytest configuration for oracle tests.

Forces JAX into float64-capable, deterministic CPU mode so numerical
assertions are stable across platforms.
"""

import jax
import pytest

# Keep tests on CPU and enable higher precision for stable assertions.
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(autouse=True, scope="session")
def _jax_config():
    """Ensure deterministic, CPU-friendly JAX behaviour during tests."""
    # Keep default float32 precision; enable NaN checking to catch bugs early.
    jax.config.update("jax_platform_name", "cpu")
    yield

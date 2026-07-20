"""Unit tests for qqn_jax.types (smoke tests for the typed interface)."""

import jax.numpy as jnp

from qqn_jax import types as qtypes


class TestTypeExports:
    def test_all_names_present(self):
        for name in [
            "Params",
            "Grad",
            "Direction",
            "Value",
            "ObjectiveFn",
            "ValueAndGradFn",
            "Any",
            "chex",
        ]:
            assert name in qtypes.__all__

    def test_scalar_type_exists(self):
        assert hasattr(qtypes, "Scalar")

    def test_chex_reexported(self):
        assert qtypes.chex is not None


class TestTypeAnnotationsUsable:
    def test_objective_fn_callable_annotation(self):
        # A function matching the ObjectiveFn signature should be usable.
        def f(x) -> qtypes.Scalar:
            return jnp.sum(x**2)

        out = f(jnp.array([1.0, 2.0]))
        assert float(out) == 5.0

    def test_value_and_grad_fn_signature(self):
        def vg(x):
            return jnp.sum(x**2), 2 * x

        value, grad = vg(jnp.array([1.0, 2.0]))
        assert float(value) == 5.0

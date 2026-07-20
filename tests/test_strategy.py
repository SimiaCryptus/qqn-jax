"""Unit tests for the resolve_oracle strategy dispatcher."""

import pytest

from qqn_jax.oracles.strategy import resolve_oracle
from qqn_jax.oracles.oracle import Oracle


class TestResolveOracle:
    def test_none_defaults_to_lbfgs(self):
        oracle = resolve_oracle(None)
        assert isinstance(oracle, Oracle)

    def test_lbfgs_string(self):
        assert isinstance(resolve_oracle("lbfgs"), Oracle)

    @pytest.mark.parametrize(
        "name",
        [
            "momentum",
            "adam",
            "path_momentum",
            "shampoo",
            "secant",
            "anderson",
            "ams_qn",
            "anderson+secant",
            "lbfgs+secant",
        ],
    )
    def test_known_strings(self, name):
        assert isinstance(resolve_oracle(name), Oracle)

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown oracle"):
            resolve_oracle("does_not_exist")

    def test_oracle_instance_passthrough(self):
        from qqn_jax.oracles.momentum import MomentumOracle

        o = MomentumOracle()
        assert resolve_oracle(o) is o

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_oracle(12345)

    def test_history_size_forwarded(self):

        oracle = resolve_oracle("ams_qn", history_size=15)
        assert isinstance(oracle, Oracle)

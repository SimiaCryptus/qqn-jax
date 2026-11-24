"""Tests for the AdamPreconditionedLBFGSOracle hybrid strategy."""

import jax.numpy as jnp
from qqn_jax import QQN, AdamPreconditionedLBFGSOracle


def test_lbfgs_adam_string_selector_flat():
    """QQN(..., oracle='lbfgs_adam') converges on a standard flat quadratic."""

    def fun(x):
        return jnp.sum((x - jnp.array([1.0, -2.0, 0.5, 3.0])) ** 2)

    x0 = jnp.zeros((4,), dtype=jnp.float32)
    solver = QQN(
        fun,
        maxiter=30,
        history_size=5,
        oracle=AdamPreconditionedLBFGSOracle(history_size=5, learning_rate=0.2),
        line_search="backtracking",
    )
    xf, st = solver.run(x0)
    assert jnp.isfinite(st.value)
    assert float(st.value) < 1e-4

    # Also verify string shortcut resolves cleanly via resolve_oracle
    solver_str = QQN(fun, maxiter=5, history_size=5, oracle="lbfgs_adam")
    _, st_str = solver_str.run(x0)
    assert float(st_str.value) < float(fun(x0))


def test_lbfgs_adam_pytree_anisotropic_and_bf16():
    """Hybrid oracle handles 10,000x cross-leaf anisotropy and preserves bf16."""

    def aniso(p):
        s = p["slow"].astype(jnp.float32)
        f = p["fast"].astype(jnp.float32)
        return 1e-2 * jnp.sum((s - 1.0) ** 2) + 1e2 * jnp.sum((f + 0.5) ** 2)

    p0 = {
        "slow": jnp.zeros((6,), dtype=jnp.float32),
        "fast": jnp.ones((6,), dtype=jnp.float32),
    }
    solver = QQN(
        aniso,
        maxiter=20,
        oracle=AdamPreconditionedLBFGSOracle(history_size=5, learning_rate=0.2),
        line_search="backtracking",
    )
    pf, st = solver.run(p0)
    assert float(st.value) < float(aniso(p0)) * 1e-3

    p0_bf16 = {k: v.astype(jnp.bfloat16) for k, v in p0.items()}
    pf_bf16, st_bf16 = solver.run(p0_bf16)
    assert pf_bf16["slow"].dtype == jnp.bfloat16
    assert pf_bf16["fast"].dtype == jnp.bfloat16
    assert float(st_bf16.value) < float(aniso(p0_bf16))

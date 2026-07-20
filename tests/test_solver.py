"""Unit tests for qqn_jax.solver (QQN)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.solver import QQN, QQNState

jax.config.update("jax_enable_x64", True)


def quadratic_bowl(x):
    """Simple convex quadratic with minimum at origin."""
    return jnp.sum(x**2)


def shifted_quadratic(x, center):
    """Quadratic with minimum at ``center``."""
    return jnp.sum((x - center) ** 2)


def rosenbrock(x):
    """The classic Rosenbrock function (min at [1, 1])."""
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


class TestInitState:
    def test_returns_qqnstate(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        state = solver.init_state(jnp.array([1.0, 2.0]))
        assert isinstance(state, QQNState)

    def test_initial_value(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        x0 = jnp.array([3.0, 4.0])
        state = solver.init_state(x0)
        assert float(state.value) == pytest.approx(25.0)

    def test_initial_iter_zero(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        state = solver.init_state(jnp.array([1.0]))
        assert int(state.iter) == 0

    def test_initial_error_is_grad_norm(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        x0 = jnp.array([3.0, 4.0])
        state = solver.init_state(x0)

        assert float(state.error) == pytest.approx(10.0)

    def test_num_evals_starts_at_one(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        state = solver.init_state(jnp.array([1.0]))
        assert int(state.num_evals) == 1

    def test_done_when_already_converged(self):
        solver = QQN(quadratic_bowl, maxiter=10, tol=1e-3)
        state = solver.init_state(jnp.array([0.0, 0.0]))
        assert bool(state.done)

    def test_with_aux(self):
        def f(x):
            return jnp.sum(x**2), {"norm": jnp.sum(jnp.abs(x))}

        solver = QQN(f, maxiter=10, has_aux=True)
        state = solver.init_state(jnp.array([1.0, 2.0]))
        assert state.aux is not None
        assert float(state.aux["norm"]) == pytest.approx(3.0)


class TestUpdate:
    def test_single_step_reduces_value(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        x0 = jnp.array([5.0, 5.0])
        state = solver.init_state(x0)
        new_params, new_state = solver.update(x0, state)
        assert float(new_state.value) <= float(state.value)

    def test_iter_increments(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        x0 = jnp.array([2.0])
        state = solver.init_state(x0)
        _, new_state = solver.update(x0, state)
        assert int(new_state.iter) == 1

    def test_num_evals_increases(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        x0 = jnp.array([2.0, 3.0])
        state = solver.init_state(x0)
        _, new_state = solver.update(x0, state)
        assert int(new_state.num_evals) > int(state.num_evals)

    def test_returns_tuple(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        x0 = jnp.array([1.0])
        state = solver.init_state(x0)
        out = solver.update(x0, state)
        assert len(out) == 2


class TestRunConvergence:
    def test_quadratic_bowl(self):
        solver = QQN(quadratic_bowl, maxiter=50, tol=1e-6)
        x0 = jnp.array([5.0, -3.0, 2.0])
        final_params, final_state = solver.run(x0)
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(3), atol=1e-4)
        assert float(final_state.error) <= 1e-6 or bool(final_state.done)

    def test_shifted_quadratic(self):
        center = jnp.array([1.0, -2.0, 3.0])
        solver = QQN(shifted_quadratic, maxiter=50, tol=1e-6)
        x0 = jnp.zeros(3)
        final_params, final_state = solver.run(x0, center)
        np.testing.assert_allclose(
            np.asarray(final_params), np.asarray(center), atol=1e-4
        )

    def test_rosenbrock_2d(self):
        solver = QQN(rosenbrock, maxiter=500, tol=1e-6)
        x0 = jnp.array([-1.2, 1.0])
        final_params, final_state = solver.run(x0)
        np.testing.assert_allclose(
            np.asarray(final_params), np.array([1.0, 1.0]), atol=1e-2
        )

    def test_stops_at_maxiter(self):
        solver = QQN(rosenbrock, maxiter=3, tol=1e-12)
        x0 = jnp.array([-1.2, 1.0])
        _, final_state = solver.run(x0)
        assert int(final_state.iter) <= 3

    def test_already_at_minimum(self):
        solver = QQN(quadratic_bowl, maxiter=50, tol=1e-6)
        x0 = jnp.zeros(3)
        final_params, final_state = solver.run(x0)
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(3), atol=1e-6)
        assert int(final_state.iter) == 0


class TestConfiguration:
    def test_unknown_line_search_raises(self):
        with pytest.raises(ValueError):
            QQN(quadratic_bowl, line_search="does_not_exist")

    def test_unknown_path_strategy_raises(self):
        with pytest.raises(ValueError):
            QQN(quadratic_bowl, path_strategy="nonexistent")

    def test_linear_path_strategy(self):
        solver = QQN(quadratic_bowl, maxiter=50, path_strategy="linear")
        assert solver._refine is True
        final_params, _ = solver.run(jnp.array([3.0, 4.0]))
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(2), atol=1e-3)

    def test_quadratic_path_default(self):
        solver = QQN(quadratic_bowl, maxiter=10)
        assert solver.path_strategy == "quadratic"
        assert solver._refine is False
        assert solver._spline is False

    def test_spline_path_strategy(self):
        solver = QQN(quadratic_bowl, maxiter=50, path_strategy="spline")
        assert solver._spline is True
        final_params, _ = solver.run(jnp.array([3.0, 4.0]))
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(2), atol=1e-3)

    def test_history_size_stored(self):
        solver = QQN(quadratic_bowl, history_size=5)
        assert solver.history_size == 5

    def test_line_search_options_forwarded(self):
        solver = QQN(
            quadratic_bowl,
            maxiter=50,
            line_search_options={"max_iter": 20},
        )
        assert solver.line_search_options["max_iter"] == 20

    def test_max_t_stored(self):
        solver = QQN(quadratic_bowl, max_t=500.0)
        assert solver.max_t == 500.0


class TestPartitioning:
    def test_partition_offsets_computed(self):
        solver = QQN(quadratic_bowl, partition_sizes=(2, 3))
        assert solver._partition_offsets == (0, 2, 5)

    def test_no_partition(self):
        solver = QQN(quadratic_bowl)
        assert solver.partition_sizes is None
        assert solver._partition_offsets is None

    def test_segments_split(self):
        solver = QQN(quadratic_bowl, partition_sizes=(2, 3))
        x = jnp.arange(5.0)
        segs = solver._segments(x)
        assert len(segs) == 2
        np.testing.assert_allclose(np.asarray(segs[0]), np.array([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(segs[1]), np.array([2.0, 3.0, 4.0]))

    def test_partitioned_convergence(self):
        solver = QQN(quadratic_bowl, maxiter=100, tol=1e-6, partition_sizes=(2, 2))
        x0 = jnp.array([3.0, -2.0, 1.0, 4.0])
        final_params, _ = solver.run(x0)
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(4), atol=1e-3)


class TestJitVmap:
    def test_run_jittable(self):
        solver = QQN(quadratic_bowl, maxiter=50, tol=1e-6)
        run_jit = jax.jit(solver.run)
        final_params, _ = run_jit(jnp.array([5.0, -3.0]))
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(2), atol=1e-3)

    def test_run_vmappable(self):
        solver = QQN(quadratic_bowl, maxiter=50, tol=1e-6)
        x0_batch = jnp.array([[5.0, -3.0], [2.0, 4.0], [-1.0, 1.0]])
        batched_run = jax.vmap(solver.run)
        final_params, _ = batched_run(x0_batch)
        np.testing.assert_allclose(
            np.asarray(final_params), np.zeros((3, 2)), atol=1e-3
        )

    def test_shifted_quadratic_vmap_over_centers(self):
        solver = QQN(shifted_quadratic, maxiter=50, tol=1e-6)
        centers = jnp.array([[1.0, 1.0], [-2.0, 3.0]])
        x0 = jnp.zeros(2)
        run = jax.vmap(lambda c: solver.run(x0, c))
        final_params, _ = run(centers)
        np.testing.assert_allclose(
            np.asarray(final_params), np.asarray(centers), atol=1e-3
        )


class TestHasAux:
    def test_aux_available_after_run(self):
        def f(x):
            return jnp.sum(x**2), {"sum": jnp.sum(x)}

        solver = QQN(f, maxiter=50, tol=1e-6, has_aux=True)
        _, final_state = solver.run(jnp.array([3.0, 4.0]))
        assert final_state.aux is not None

    def test_aux_counts_eval(self):
        def f(x):
            return jnp.sum(x**2), None

        solver = QQN(f, maxiter=10, has_aux=True)
        x0 = jnp.array([2.0, 3.0])
        state = solver.init_state(x0)
        _, new_state = solver.update(x0, state)

        assert int(new_state.num_evals) > int(state.num_evals)


class TestRobustness:
    def test_single_variable(self):
        solver = QQN(quadratic_bowl, maxiter=50, tol=1e-6)
        final_params, _ = solver.run(jnp.array([10.0]))
        np.testing.assert_allclose(np.asarray(final_params), np.zeros(1), atol=1e-4)

    def test_high_dimensional(self):
        solver = QQN(quadratic_bowl, maxiter=100, tol=1e-6)
        x0 = jnp.linspace(-5.0, 5.0, 20)
        final_params, final_state = solver.run(x0)
        assert float(final_state.error) < 1e-3 or bool(final_state.done)

    def test_result_finite(self):
        solver = QQN(rosenbrock, maxiter=100, tol=1e-6)
        final_params, final_state = solver.run(jnp.array([-1.2, 1.0]))
        assert np.all(np.isfinite(np.asarray(final_params)))
        assert np.isfinite(float(final_state.value))

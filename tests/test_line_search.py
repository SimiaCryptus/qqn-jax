"""Comprehensive unit tests for the QQN line search module.

Tests cover:
  * Utility functions (probes, metropolis acceptance, scalar problem).
  * All line search strategies (backtracking, armijo_wolfe, bisection,
    fixed_step, null, strong_wolfe, hager_zhang).
  * JIT/vmap compatibility.
  * Edge cases (temperature, max_step clipping, non-descent directions).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.line_search.util import (
    _metropolis_accept,
    _empty_probes,
    _record_probe,
)
from qqn_jax.line_search.result import LineSearchResult
from qqn_jax.line_search import (
    backtracking_search,
    armijo_wolfe_search,
    bisection_search,
    fixed_step_search,
    null_search,
    strong_wolfe_search,
    hager_zhang_search,
    LINE_SEARCHES,
)

def make_scalar_problem(
        value_and_grad_fn,
        params,
        grad,
        direction,
        region,
        region_state,
        path,
        *args,
):
    """Prepare the 1-D differentiable problem the line search solves.
    This is the *single* place the multidimensional path/region machinery
    lives. It returns:
      * ``eval_at(t) -> (projected_params, value, grad, slope)`` where
        ``slope = φ'(t) = ⟨∇f, d'(t)⟩`` is the 1-D directional derivative
        along the path, and
      * ``slope0`` — the directional derivative at ``t = 0`` (i.e.
        ``⟨∇f, d'(0)⟩``), the ``φ'(0)`` every Armijo/Wolfe test needs.
    Line searches receive only ``eval_at`` and ``slope0``; they are entirely
    unaware of ``direction``, ``path`` or ``region``.
    """
    from qqn_jax.utils import tree_add_scaled, tree_negative, tree_vdot

    grad_dir = tree_negative(grad)

    def project(candidate):
        return region.project(params, candidate, region_state)

    def eval_at(t):
        d = path.offset(t, grad_dir, direction)
        raw = tree_add_scaled(params, 1.0, d)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        v = path.velocity(t, grad_dir, direction)
        slope = tree_vdot(g, v)
        return projected, val, g, slope

    v0 = path.velocity(jnp.asarray(0.0, dtype=grad.dtype), grad_dir, direction)
    slope0 = tree_vdot(grad, v0)
    return eval_at, slope0
def _make_projected_point(region, region_state, params):
    """Return a fn ``α -> projected(x + α·d)`` for a given direction.
    The caller curries the direction in; here we build a helper that, given
    a tentative point ``x + α·d``, projects it onto the region. When the
    region is the identity, this is a no-op (zero overhead).
    """

    def project_candidate(candidate):
        return region.project(params, candidate, region_state)

    return project_candidate

class IdentityRegion:
    """A trivial region whose projection is a no-op."""

    @staticmethod
    def project(params, candidate, region_state):
        return candidate


class SteepestDescentPath:
    """A simple straight-line path along the (negative gradient) direction.

    ``offset(t, grad_dir, direction)`` returns ``t * direction`` and
    ``velocity(t, ...)`` returns ``direction`` (constant velocity).
    """

    @staticmethod
    def offset(t, grad_dir, direction):
        return t * direction

    @staticmethod
    def velocity(t, grad_dir, direction):
        return direction


def make_quadratic(A, b):
    """Return a value_and_grad_fn for f(x) = 0.5 x^T A x - b^T x."""

    def value_and_grad_fn(x):
        val = 0.5 * x @ (A @ x) - b @ x
        g = A @ x - b
        return val, g

    return value_and_grad_fn


def make_eval_at(value_and_grad_fn, params, direction):
    """Build an ``eval_at`` and ``slope0`` for a straight line search."""
    return make_scalar_problem(
        value_and_grad_fn,
        params,
        grad=value_and_grad_fn(params)[1],
        direction=direction,
        region=IdentityRegion(),
        region_state=None,
        path=SteepestDescentPath(),
    )


def descent_setup(dim=3, seed=0):
    """Create a quadratic problem with a guaranteed descent direction."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim))
    A = jnp.asarray(M @ M.T + dim * np.eye(dim), dtype=jnp.float32)
    b = jnp.asarray(rng.standard_normal(dim), dtype=jnp.float32)
    params = jnp.asarray(rng.standard_normal(dim), dtype=jnp.float32)
    vg = make_quadratic(A, b)
    value, grad = vg(params)

    direction = -grad
    eval_at, slope0 = make_eval_at(vg, params, direction)
    return {
        "vg": vg,
        "A": A,
        "b": b,
        "params": params,
        "value": value,
        "grad": grad,
        "direction": direction,
        "eval_at": eval_at,
        "slope0": slope0,
    }


ALL_SEARCHES = [
    backtracking_search,
    armijo_wolfe_search,
    bisection_search,
    fixed_step_search,
    null_search,
    strong_wolfe_search,
    hager_zhang_search,
]

SEARCH_IDS = [s.__name__ for s in ALL_SEARCHES]


class TestEmptyProbes:
    def test_shapes_and_dtypes(self):
        params = jnp.zeros((4,), dtype=jnp.float32)
        pp, pg, pv, pval, pa = _empty_probes(params, 8)
        assert pp.shape == (8, 4)
        assert pg.shape == (8, 4)
        assert pv.shape == (8,)
        assert pval.shape == (8,)
        assert pa.shape == (8,)
        assert pv.dtype == jnp.bool_
        assert pp.dtype == jnp.float32

    def test_initial_values(self):
        params = jnp.zeros((2,), dtype=jnp.float32)
        pp, pg, pv, pval, pa = _empty_probes(params, 3)
        assert jnp.all(pp == 0.0)
        assert jnp.all(pg == 0.0)
        assert not jnp.any(pv)
        assert jnp.all(jnp.isinf(pval))
        assert jnp.all(pa == 0.0)


class TestRecordProbe:
    def test_records_in_slot(self):
        params = jnp.zeros((2,), dtype=jnp.float32)
        pp, pg, pv, pval, pa = _empty_probes(params, 4)
        p = jnp.asarray([1.0, 2.0])
        g = jnp.asarray([3.0, 4.0])
        pp, pg, pv, pval, pa = _record_probe(pp, pg, pv, pval, pa, 1, p, g, 5.0, 0.5, 4)
        np.testing.assert_allclose(pp[1], p)
        np.testing.assert_allclose(pg[1], g)
        assert bool(pv[1])
        assert float(pval[1]) == 5.0
        assert float(pa[1]) == 0.5

        assert not bool(pv[0])
        assert not bool(pv[2])

    def test_out_of_range_slot_ignored(self):
        params = jnp.zeros((2,), dtype=jnp.float32)
        pp, pg, pv, pval, pa = _empty_probes(params, 2)
        p = jnp.asarray([1.0, 2.0])
        g = jnp.asarray([3.0, 4.0])

        npp, npg, npv, npval, npa = _record_probe(
            pp, pg, pv, pval, pa, 5, p, g, 5.0, 0.5, 2
        )
        assert not jnp.any(npv)
        np.testing.assert_allclose(npp, pp)

    def test_negative_slot_ignored(self):
        params = jnp.zeros((2,), dtype=jnp.float32)
        pp, pg, pv, pval, pa = _empty_probes(params, 2)
        p = jnp.asarray([1.0, 2.0])
        g = jnp.asarray([3.0, 4.0])
        npp, npg, npv, npval, npa = _record_probe(
            pp, pg, pv, pval, pa, -1, p, g, 5.0, 0.5, 2
        )
        assert not jnp.any(npv)


class TestMetropolisAccept:
    def test_disabled_at_zero_temperature(self):
        key = jax.random.PRNGKey(0)
        accepted, new_key = _metropolis_accept(
            jnp.asarray(-1.0), jnp.asarray(0.0), key, jnp.float32
        )
        assert not bool(accepted)

    def test_disabled_negative_temperature(self):
        key = jax.random.PRNGKey(0)
        accepted, _ = _metropolis_accept(
            jnp.asarray(1.0), jnp.asarray(-0.5), key, jnp.float32
        )
        assert not bool(accepted)

    def test_downhill_high_temperature_often_accepts(self):

        n_accept = 0
        for s in range(50):
            key = jax.random.PRNGKey(s)
            accepted, _ = _metropolis_accept(
                jnp.asarray(-10.0), jnp.asarray(1.0), key, jnp.float32
            )
            n_accept += int(bool(accepted))
        assert n_accept == 50

    def test_uphill_low_temperature_rarely_accepts(self):

        n_accept = 0
        for s in range(50):
            key = jax.random.PRNGKey(s)
            accepted, _ = _metropolis_accept(
                jnp.asarray(10.0), jnp.asarray(1e-3), key, jnp.float32
            )
            n_accept += int(bool(accepted))
        assert n_accept == 0

    def test_key_advances(self):
        key = jax.random.PRNGKey(42)
        _, new_key = _metropolis_accept(
            jnp.asarray(1.0), jnp.asarray(1.0), key, jnp.float32
        )
        assert not jnp.array_equal(key, new_key)

    def test_deterministic_given_key(self):
        key = jax.random.PRNGKey(7)
        a1, _ = _metropolis_accept(jnp.asarray(0.5), jnp.asarray(1.0), key, jnp.float32)
        a2, _ = _metropolis_accept(jnp.asarray(0.5), jnp.asarray(1.0), key, jnp.float32)
        assert bool(a1) == bool(a2)


class TestMakeScalarProblem:
    def test_slope0_matches_directional_derivative(self):
        s = descent_setup()

        expected = float(jnp.dot(s["grad"], s["direction"]))
        assert np.isclose(float(s["slope0"]), expected, rtol=1e-5)

    def test_slope0_negative_for_descent(self):
        s = descent_setup()
        assert float(s["slope0"]) < 0.0

    def test_eval_at_zero_recovers_params(self):
        s = descent_setup()
        p, v, g, slope = s["eval_at"](jnp.asarray(0.0, dtype=jnp.float32))
        np.testing.assert_allclose(p, s["params"], rtol=1e-5)
        assert np.isclose(float(v), float(s["value"]), rtol=1e-5)
        np.testing.assert_allclose(g, s["grad"], rtol=1e-5)

    def test_eval_at_moves_along_direction(self):
        s = descent_setup()
        t = jnp.asarray(0.1, dtype=jnp.float32)
        p, v, g, slope = s["eval_at"](t)
        expected_p = s["params"] + t * s["direction"]
        np.testing.assert_allclose(p, expected_p, rtol=1e-5)

    def test_eval_at_decreases_value_for_descent(self):
        s = descent_setup()
        _, v_small, _, _ = s["eval_at"](jnp.asarray(0.01, dtype=jnp.float32))
        assert float(v_small) < float(s["value"])


class TestMakeProjectedPoint:
    def test_identity_projection(self):
        proj = _make_projected_point(IdentityRegion(), None, jnp.zeros(3))
        cand = jnp.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(proj(cand), cand)


class TestAllSearchesCommon:
    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_returns_line_search_result(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert isinstance(res, LineSearchResult)

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_result_fields_populated(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert res.step_size is not None
        assert res.new_value is not None
        assert res.new_grad is not None
        assert res.new_params is not None
        assert res.done is not None
        assert res.probe_params is not None
        assert res.probe_grads is not None
        assert res.probe_valid is not None
        assert res.probe_values is not None
        assert res.probe_alphas is not None
        assert res.num_evals is not None

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_new_params_consistent_with_step(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        expected = s["params"] + float(res.step_size) * s["direction"]
        np.testing.assert_allclose(
            np.asarray(res.new_params), np.asarray(expected), rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_step_size_within_max_step(self, search):
        s = descent_setup()
        res = search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            max_step=0.5,
        )
        assert float(res.step_size) <= 0.5 + 1e-5

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_num_evals_positive(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert int(res.num_evals) >= 1

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_probe_shapes(self, search):
        s = descent_setup()
        res = search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            max_probes=16,
        )
        n = s["params"].shape[0]
        assert res.probe_params.shape == (16, n)
        assert res.probe_grads.shape == (16, n)
        assert res.probe_valid.shape == (16,)
        assert res.probe_values.shape == (16,)
        assert res.probe_alphas.shape == (16,)

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_at_least_one_probe_recorded(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert int(jnp.sum(res.probe_valid)) >= 1

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_valid_probes_have_finite_values(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        valid = np.asarray(res.probe_valid)
        vals = np.asarray(res.probe_values)
        assert np.all(np.isfinite(vals[valid]))

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_jit_compatible(self, search):
        s = descent_setup()
        jitted = jax.jit(lambda p, v, g, sl: search(s["eval_at"], p, v, g, sl))
        res = jitted(s["params"], s["value"], s["grad"], s["slope0"])
        assert isinstance(res, LineSearchResult)
        assert np.isfinite(float(res.step_size))


DESCENT_SEARCHES = [
    backtracking_search,
    armijo_wolfe_search,
    bisection_search,
    strong_wolfe_search,
    hager_zhang_search,
]
DESCENT_IDS = [s.__name__ for s in DESCENT_SEARCHES]


class TestDescentQuality:
    @pytest.mark.parametrize("search", DESCENT_SEARCHES, ids=DESCENT_IDS)
    def test_reduces_objective(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert float(res.new_value) <= float(s["value"]) + 1e-4

    @pytest.mark.parametrize("search", DESCENT_SEARCHES, ids=DESCENT_IDS)
    def test_positive_step_for_descent(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert float(res.step_size) > 0.0

    @pytest.mark.parametrize("search", DESCENT_SEARCHES, ids=DESCENT_IDS)
    def test_done_true_for_descent(self, search):
        s = descent_setup()
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert bool(res.done)


class TestArmijoConditionSatisfied:
    @pytest.mark.parametrize("search", [backtracking_search, bisection_search])
    def test_armijo_holds(self, search):
        s = descent_setup()
        c1 = 1e-4
        res = search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            c1=c1,
        )
        armijo_rhs = float(s["value"]) + c1 * float(res.step_size) * float(s["slope0"])
        assert float(res.new_value) <= armijo_rhs + 1e-4


class TestBisectionStationarity:
    def test_finds_near_stationary_point(self):

        s = descent_setup(dim=4, seed=3)
        res = bisection_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            max_iter=40,
            max_step=10.0,
        )
        _, _, _, slope = s["eval_at"](res.step_size)

        assert abs(float(slope)) < 1e-1


class TestFixedStep:
    def test_uses_given_step_size(self):
        s = descent_setup()
        res = fixed_step_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            step_size=0.3,
            max_step=1.0,
        )
        assert np.isclose(float(res.step_size), 0.3)

    def test_clips_to_max_step(self):
        s = descent_setup()
        res = fixed_step_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            step_size=2.0,
            max_step=0.5,
        )
        assert np.isclose(float(res.step_size), 0.5)

    def test_done_true_zero_temperature(self):
        s = descent_setup()
        res = fixed_step_search(
            s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"]
        )
        assert bool(res.done)

    def test_single_eval(self):
        s = descent_setup()
        res = fixed_step_search(
            s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"]
        )
        assert int(res.num_evals) == 1


class TestNullSearch:
    def test_always_done(self):
        s = descent_setup()
        res = null_search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert bool(res.done)

    def test_uses_step_size(self):
        s = descent_setup()
        res = null_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            step_size=0.25,
        )
        assert np.isclose(float(res.step_size), 0.25)

    def test_clips_to_max_step(self):
        s = descent_setup()
        res = null_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            step_size=5.0,
            max_step=1.0,
        )
        assert np.isclose(float(res.step_size), 1.0)

    def test_accepts_even_uphill(self):

        s = descent_setup()
        eval_at, slope0 = make_eval_at(s["vg"], s["params"], s["grad"])
        res = null_search(eval_at, s["params"], s["value"], s["grad"], slope0)
        assert bool(res.done)


class TestTemperatureBehaviour:
    def test_fixed_step_temperature_gates_done(self):

        s = descent_setup()
        eval_at, slope0 = make_eval_at(s["vg"], s["params"], s["grad"])

        res_cold = fixed_step_search(
            eval_at,
            s["params"],
            s["value"],
            s["grad"],
            slope0,
            step_size=0.1,
            temperature=0.0,
        )

        assert bool(res_cold.done)

    @pytest.mark.parametrize(
        "search",
        [backtracking_search, armijo_wolfe_search, bisection_search],
    )
    def test_high_temperature_may_accept_uphill(self, search):
        s = descent_setup()

        eval_at, slope0 = make_eval_at(s["vg"], s["params"], s["grad"])
        res = search(
            eval_at,
            s["params"],
            s["value"],
            s["grad"],
            slope0,
            temperature=1e6,
            seed=1,
        )

        assert bool(res.done)


class TestBacktracking:
    def test_shrinks_when_initial_step_too_large(self):

        s = descent_setup()
        res = backtracking_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            init_step=1.0,
            max_step=1.0,
            shrink=0.5,
            max_iter=10,
        )
        assert float(res.new_value) <= float(s["value"]) + 1e-4

    def test_extrapolation_grows_step(self):

        s = descent_setup(dim=2, seed=5)
        res = backtracking_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            init_step=0.01,
            max_step=2.0,
            shrink=0.5,
            max_iter=10,
        )
        assert float(res.step_size) >= 0.01

    def test_record_probes_false_shrinks_buffer(self):
        s = descent_setup()
        res = backtracking_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            record_probes=False,
            max_probes=32,
        )

        assert res.probe_params.shape[0] == 1


class TestRegistry:
    def test_all_searches_registered(self):
        expected = {
            "strong_wolfe",
            "backtracking",
            "armijo_wolfe",
            "hager_zhang",
            "fixed",
            "null",
            "bisection",
        }
        assert set(LINE_SEARCHES.keys()) == expected

    @pytest.mark.parametrize("name", list(LINE_SEARCHES.keys()))
    def test_registered_search_callable(self, name):
        s = descent_setup()
        search = LINE_SEARCHES[name]
        res = search(s["eval_at"], s["params"], s["value"], s["grad"], s["slope0"])
        assert isinstance(res, LineSearchResult)


class TestVmapCompatibility:
    @pytest.mark.parametrize(
        "search",
        [
            fixed_step_search,
            null_search,
            backtracking_search,
            bisection_search,
            armijo_wolfe_search,
        ],
    )
    def test_vmap_over_params(self, search):
        dim = 3
        A = jnp.eye(dim, dtype=jnp.float32) * 2.0
        b = jnp.zeros(dim, dtype=jnp.float32)
        vg = make_quadratic(A, b)

        def run(params):
            value, grad = vg(params)
            direction = -grad
            eval_at, slope0 = make_scalar_problem(
                vg,
                params,
                grad,
                direction,
                IdentityRegion(),
                None,
                SteepestDescentPath(),
            )
            res = search(eval_at, params, value, grad, slope0)
            return res.step_size, res.new_value

        batch = jnp.asarray(
            np.random.default_rng(0).standard_normal((5, dim)),
            dtype=jnp.float32,
        )
        steps, vals = jax.vmap(run)(batch)
        assert steps.shape == (5,)
        assert vals.shape == (5,)
        assert np.all(np.isfinite(np.asarray(steps)))


class TestEdgeCases:
    def test_already_at_minimum(self):

        dim = 3
        A = jnp.eye(dim, dtype=jnp.float32) * 2.0
        b = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
        vg = make_quadratic(A, b)
        x_star = jnp.linalg.solve(A, b)
        value, grad = vg(x_star)

        direction = -grad
        eval_at, slope0 = make_eval_at(vg, x_star, direction)
        res = backtracking_search(eval_at, x_star, value, grad, slope0)

        assert float(res.new_value) <= float(value) + 1e-3

    @pytest.mark.parametrize("search", ALL_SEARCHES, ids=SEARCH_IDS)
    def test_single_dimension(self, search):
        A = jnp.asarray([[4.0]], dtype=jnp.float32)
        b = jnp.asarray([2.0], dtype=jnp.float32)
        vg = make_quadratic(A, b)
        params = jnp.asarray([5.0], dtype=jnp.float32)
        value, grad = vg(params)
        direction = -grad
        eval_at, slope0 = make_eval_at(vg, params, direction)
        res = search(eval_at, params, value, grad, slope0)
        assert np.isfinite(float(res.step_size))
        assert res.new_params.shape == (1,)

    def test_max_iter_one(self):
        s = descent_setup()
        res = backtracking_search(
            s["eval_at"],
            s["params"],
            s["value"],
            s["grad"],
            s["slope0"],
            max_iter=1,
        )
        assert np.isfinite(float(res.step_size))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

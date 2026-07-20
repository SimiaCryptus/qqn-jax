"""Comprehensive unit tests for qqn_jax.regions.

Covers every region implementation, the composition helper, region
resolution, the shared tree utilities, and cross-cutting properties
such as jit/vmap/grad compatibility and identity equivalence.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from qqn_jax.regions import (
    Region,
    RegionInfo,
    resolve_region,
    IdentityRegion,
    BoxRegion,
    OrthantRegion,
    QuantizationRegion,
    NoDecreaseRegion,
    TrustRegion,
    Sequential,
)
from qqn_jax.regions.types import _tree_add, _tree_sub
from qqn_jax.regions.trustregion import TrustRegionState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run(region, params, candidate):
    """Initialise the region on ``params`` and project ``candidate``."""
    state = region.init(params)
    return region.project(params, candidate, state)


def _assert_tree_allclose(a, b, **kw):
    a_leaves = jax.tree_util.tree_leaves(a)
    b_leaves = jax.tree_util.tree_leaves(b)
    assert len(a_leaves) == len(b_leaves)
    for x, y in zip(a_leaves, b_leaves):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), **kw)


@pytest.fixture
def vec_params():
    return jnp.array([1.0, -2.0, 3.0, 0.0])


@pytest.fixture
def tree_params():
    return {
        "w": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "b": jnp.array([-1.0, 0.5]),
    }


# --------------------------------------------------------------------------- #
# Tree utilities
# --------------------------------------------------------------------------- #


class TestTreeUtils:
    def test_tree_add_vector(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        _assert_tree_allclose(_tree_add(a, b), jnp.array([5.0, 7.0, 9.0]))

    def test_tree_sub_vector(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        _assert_tree_allclose(_tree_sub(a, b), jnp.array([-3.0, -3.0, -3.0]))

    def test_tree_add_pytree(self, tree_params):
        out = _tree_add(tree_params, tree_params)
        _assert_tree_allclose(out["w"], tree_params["w"] * 2)
        _assert_tree_allclose(out["b"], tree_params["b"] * 2)

    def test_tree_sub_pytree_zero(self, tree_params):
        out = _tree_sub(tree_params, tree_params)
        for leaf in jax.tree_util.tree_leaves(out):
            _assert_tree_allclose(leaf, jnp.zeros_like(leaf))

    def test_add_sub_inverse(self, tree_params):
        other = jax.tree_util.tree_map(lambda x: x + 0.7, tree_params)
        recovered = _tree_sub(_tree_add(tree_params, other), other)
        _assert_tree_allclose(recovered, tree_params, rtol=1e-6)


# --------------------------------------------------------------------------- #
# resolve_region
# --------------------------------------------------------------------------- #


class TestResolveRegion:
    def test_none_returns_identity(self, vec_params):
        region = resolve_region(None)
        cand = vec_params + 5.0
        _assert_tree_allclose(_run(region, vec_params, cand), cand)

    def test_passthrough(self):
        box = BoxRegion(lo=0.0, hi=1.0)
        assert resolve_region(box) is box


# --------------------------------------------------------------------------- #
# IdentityRegion
# --------------------------------------------------------------------------- #


class TestIdentityRegion:
    def test_project_is_identity(self, tree_params):
        region = IdentityRegion()
        cand = jax.tree_util.tree_map(lambda x: x * 3 + 1, tree_params)
        _assert_tree_allclose(_run(region, tree_params, cand), cand)

    def test_init_is_empty_tuple(self, vec_params):
        assert IdentityRegion().init(vec_params) == ()

    def test_update_returns_state(self):
        region = IdentityRegion()
        info = RegionInfo()
        assert region.update("state", info) == "state"


# --------------------------------------------------------------------------- #
# BoxRegion
# --------------------------------------------------------------------------- #


class TestBoxRegion:
    def test_clip_scalar_bounds(self, vec_params):
        region = BoxRegion(lo=-1.0, hi=1.0)
        cand = jnp.array([5.0, -5.0, 0.5, -0.5])
        out = _run(region, vec_params, cand)
        _assert_tree_allclose(out, jnp.array([1.0, -1.0, 0.5, -0.5]))

    def test_lo_none_is_neg_inf(self):
        region = BoxRegion(lo=None, hi=2.0)
        cand = jnp.array([-1e30, 5.0])
        out = _run(region, jnp.zeros(2), cand)
        _assert_tree_allclose(out, jnp.array([-1e30, 2.0]))

    def test_hi_none_is_pos_inf(self):
        region = BoxRegion(lo=0.0, hi=None)
        cand = jnp.array([-3.0, 1e30])
        out = _run(region, jnp.zeros(2), cand)
        _assert_tree_allclose(out, jnp.array([0.0, 1e30]))

    def test_both_none_is_identity(self):
        region = BoxRegion()
        cand = jnp.array([-100.0, 100.0])
        _assert_tree_allclose(_run(region, jnp.zeros(2), cand), cand)

    def test_pytree_bounds(self, tree_params):
        region = BoxRegion(lo=0.0, hi=1.0)
        cand = jax.tree_util.tree_map(lambda x: x + 10.0, tree_params)
        out = _run(region, tree_params, cand)
        for leaf in jax.tree_util.tree_leaves(out):
            assert jnp.all(leaf <= 1.0) and jnp.all(leaf >= 0.0)

    def test_within_bounds_unchanged(self):
        region = BoxRegion(lo=-10.0, hi=10.0)
        cand = jnp.array([1.0, -2.0, 3.0])
        _assert_tree_allclose(_run(region, jnp.zeros(3), cand), cand)


# --------------------------------------------------------------------------- #
# OrthantRegion
# --------------------------------------------------------------------------- #


class TestOrthantRegion:
    def test_positive_coord_clamped_at_zero(self):
        region = OrthantRegion()
        params = jnp.array([1.0, 2.0])
        cand = jnp.array([-0.5, 3.0])  # first tries to cross zero
        out = _run(region, params, cand)
        _assert_tree_allclose(out, jnp.array([0.0, 3.0]))

    def test_negative_coord_clamped_at_zero(self):
        region = OrthantRegion()
        params = jnp.array([-1.0, -2.0])
        cand = jnp.array([0.5, -3.0])
        out = _run(region, params, cand)
        _assert_tree_allclose(out, jnp.array([0.0, -3.0]))

    def test_zero_coord_stays_zero(self):
        region = OrthantRegion()
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([5.0, -5.0])
        out = _run(region, params, cand)
        _assert_tree_allclose(out, jnp.array([0.0, 0.0]))

    def test_sign_preserved(self):
        region = OrthantRegion()
        params = jnp.array([1.0, -1.0, 0.0, 3.0])
        cand = jnp.array([-2.0, 2.0, 1.0, 1.0])
        out = _run(region, params, cand)
        # signs must match params (or be zero)
        for p, o in zip(params, out):
            if p > 0:
                assert o >= 0
            elif p < 0:
                assert o <= 0
            else:
                assert o == 0

    def test_same_orthant_unchanged(self):
        region = OrthantRegion()
        params = jnp.array([1.0, -1.0])
        cand = jnp.array([5.0, -0.1])
        _assert_tree_allclose(_run(region, params, cand), cand)


# --------------------------------------------------------------------------- #
# QuantizationRegion
# --------------------------------------------------------------------------- #


class TestQuantizationRegion:
    def test_requires_bits_or_step(self):
        with pytest.raises(ValueError):
            QuantizationRegion()

    def test_lock_snaps_to_grid(self):
        # step=0.5 over [-1, 1] -> grid points at ..., -0.5, 0, 0.5, ...
        region = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, lock=True)
        params = jnp.array([0.2, -0.7, 0.9])
        cand = params  # ignored under lock
        out = _run(region, params, cand)
        # nearest grid points
        _assert_tree_allclose(out, jnp.array([0.0, -0.5, 1.0]))

    def test_lock_ignores_candidate(self):
        region = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, lock=True)
        params = jnp.array([0.2])
        out1 = _run(region, params, jnp.array([0.9]))
        out2 = _run(region, params, jnp.array([-0.9]))
        _assert_tree_allclose(out1, out2)

    def test_cell_confinement(self):
        region = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, lock=False)
        params = jnp.array([0.0])  # cell around grid point 0: [-0.25, 0.25]
        # candidate wants to jump to 0.9 but is walled at cell boundary
        out = _run(region, params, jnp.array([0.9]))
        assert jnp.all(out <= 0.25 + 1e-6)
        assert jnp.all(out >= -0.25 - 1e-6)

    def test_free_movement_within_cell(self):
        region = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, lock=False)
        params = jnp.array([0.0])
        cand = jnp.array([0.1])  # inside cell
        _assert_tree_allclose(_run(region, params, cand), cand)

    def test_window_tightens_cell(self):
        wide = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, window=1.0)
        narrow = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, window=0.2)
        params = jnp.array([0.0])
        cand = jnp.array([0.24])
        out_wide = _run(wide, params, cand)
        out_narrow = _run(narrow, params, cand)
        # narrow window clamps more aggressively
        assert float(out_narrow[0]) < float(out_wide[0])

    def test_bits_grid(self):
        # 1 bit -> levels = 1 -> delta = (1 - 0) / 1 = 1.0
        region = QuantizationRegion(bits=1, lo=0.0, hi=1.0, lock=True)
        params = jnp.array([0.3, 0.7])
        out = _run(region, params, params)
        _assert_tree_allclose(out, jnp.array([0.0, 1.0]))

    def test_step_overrides_bits(self):
        region = QuantizationRegion(bits=8, step=0.5, lo=-1.0, hi=1.0, lock=True)
        params = jnp.array([0.2])
        out = _run(region, params, params)
        _assert_tree_allclose(out, jnp.array([0.0]))

    def test_clipping_to_range(self):
        region = QuantizationRegion(step=0.5, lo=-1.0, hi=1.0, lock=True)
        params = jnp.array([5.0, -5.0])
        out = _run(region, params, params)
        _assert_tree_allclose(out, jnp.array([1.0, -1.0]))


# --------------------------------------------------------------------------- #
# NoDecreaseRegion
# --------------------------------------------------------------------------- #


class TestNoDecreaseRegion:
    def test_removes_increasing_component(self):
        # secondary gradient points along +x; a step along +x increases g
        grad = jnp.array([1.0, 0.0])
        region = NoDecreaseRegion(lambda p: grad)
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([1.0, 1.0])  # step (1,1); <g,s> = 1 > 0
        out = _run(region, params, cand)
        step = out - params
        # projected step must not increase g: <grad, step> <= 0
        assert float(jnp.vdot(grad, step)) <= 1e-6
        # orthogonal component (y) preserved
        np.testing.assert_allclose(float(step[1]), 1.0, rtol=1e-6)

    def test_descent_passes_through(self):
        grad = jnp.array([1.0, 0.0])
        region = NoDecreaseRegion(lambda p: grad)
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([-1.0, 2.0])  # <g,s> = -1 < 0, decreasing g
        out = _run(region, params, cand)
        _assert_tree_allclose(out, cand, rtol=1e-6)

    def test_zero_gradient_is_identity(self):
        region = NoDecreaseRegion(lambda p: jnp.zeros(2))
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([3.0, -4.0])
        _assert_tree_allclose(_run(region, params, cand), cand, rtol=1e-5)

    def test_pure_ascent_fully_removed(self):
        grad = jnp.array([1.0, 0.0])
        region = NoDecreaseRegion(lambda p: grad)
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([2.0, 0.0])  # pure ascent
        out = _run(region, params, cand)
        _assert_tree_allclose(out, params, atol=1e-6)

    def test_pytree_gradient(self, tree_params):
        grad = jax.tree_util.tree_map(jnp.ones_like, tree_params)
        region = NoDecreaseRegion(lambda p: grad)
        cand = jax.tree_util.tree_map(lambda x: x + 1.0, tree_params)
        out = _run(region, tree_params, cand)
        step = _tree_sub(out, tree_params)
        c = sum(
            jnp.vdot(gi, si)
            for gi, si in zip(
                jax.tree_util.tree_leaves(grad),
                jax.tree_util.tree_leaves(step),
            )
        )
        assert float(c) <= 1e-4


# --------------------------------------------------------------------------- #
# TrustRegion
# --------------------------------------------------------------------------- #


class TestTrustRegion:
    def test_init_state(self, vec_params):
        region = TrustRegion(radius=2.0)
        state = region.init(vec_params)
        assert isinstance(state, TrustRegionState)
        np.testing.assert_allclose(float(state.radius), 2.0)

    def test_step_within_radius_unchanged(self):
        region = TrustRegion(radius=10.0)
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([1.0, 1.0])  # norm sqrt(2) < 10
        out = _run(region, params, cand)
        _assert_tree_allclose(out, cand, rtol=1e-6)

    def test_step_clipped_to_radius(self):
        region = TrustRegion(radius=1.0)
        params = jnp.array([0.0, 0.0])
        cand = jnp.array([3.0, 4.0])  # norm 5 -> clipped to 1
        out = _run(region, params, cand)
        step = out - params
        np.testing.assert_allclose(float(jnp.linalg.norm(step)), 1.0, rtol=1e-5)
        # direction preserved
        np.testing.assert_allclose(float(step[0] / step[1]), 3.0 / 4.0, rtol=1e-5)

    def test_non_adaptive_update_noop(self, vec_params):
        region = TrustRegion(radius=1.0, adaptive=False)
        state = region.init(vec_params)
        info = RegionInfo(
            params=vec_params,
            new_params=vec_params,
            pred_reduction=jnp.asarray(1.0),
            actual_reduction=jnp.asarray(1.0),
        )
        new_state = region.update(state, info)
        np.testing.assert_allclose(float(new_state.radius), 1.0)

    def test_adaptive_shrink_on_poor_ratio(self):
        region = TrustRegion(radius=1.0, adaptive=True, shrink=0.5, eta_lo=0.1)
        state = TrustRegionState(radius=jnp.asarray(1.0))
        params = jnp.array([0.0, 0.0])
        new_params = jnp.array([0.05, 0.0])  # small step, no progress
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=jnp.asarray(1.0),
            actual_reduction=jnp.asarray(-1.0),  # rho < 0 < eta_lo
        )
        new_state = region.update(state, info)
        assert float(new_state.radius) < 1.0

    def test_adaptive_expand_on_good_ratio_at_boundary(self):
        region = TrustRegion(radius=1.0, adaptive=True, expand=2.0, eta_hi=0.75)
        state = TrustRegionState(radius=jnp.asarray(1.0))
        params = jnp.array([0.0, 0.0])
        new_params = jnp.array([1.0, 0.0])  # at boundary (norm 1)
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=jnp.asarray(1.0),
            actual_reduction=jnp.asarray(1.0),  # rho = 1 > eta_hi
        )
        new_state = region.update(state, info)
        assert float(new_state.radius) > 1.0

    def test_radius_capped_at_max(self):
        region = TrustRegion(radius=1.0, radius_max=1.5, adaptive=True, expand=10.0)
        state = TrustRegionState(radius=jnp.asarray(1.0))
        params = jnp.array([0.0, 0.0])
        new_params = jnp.array([1.0, 0.0])
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=jnp.asarray(1.0),
            actual_reduction=jnp.asarray(1.0),
        )
        new_state = region.update(state, info)
        assert float(new_state.radius) <= 1.5 + 1e-6

    def test_radius_floor_on_progress(self):
        # good progress should keep radius at least step norm
        region = TrustRegion(radius=1.0, adaptive=True, eta_hi=0.75)
        state = TrustRegionState(radius=jnp.asarray(1.0))
        params = jnp.array([0.0, 0.0])
        new_params = jnp.array([0.5, 0.0])
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=jnp.asarray(1.0),
            actual_reduction=jnp.asarray(0.5),  # progress, mid ratio
        )
        new_state = region.update(state, info)
        assert float(new_state.radius) >= 0.5 - 1e-6


# --------------------------------------------------------------------------- #
# Sequential
# --------------------------------------------------------------------------- #


class TestSequential:
    def test_empty_is_identity(self):
        region = Sequential([])
        params = jnp.array([1.0, 2.0])
        cand = jnp.array([5.0, 6.0])
        _assert_tree_allclose(_run(region, params, cand), cand)

    def test_composition_order(self):
        # Box to [0, 10] then Box to [0, 1]; final must respect both
        region = Sequential([BoxRegion(0.0, 10.0), BoxRegion(0.0, 1.0)])
        params = jnp.zeros(2)
        cand = jnp.array([5.0, -3.0])
        out = _run(region, params, cand)
        _assert_tree_allclose(out, jnp.array([1.0, 0.0]))

    def test_state_is_tuple(self, vec_params):
        region = Sequential([BoxRegion(0.0, 1.0), TrustRegion(radius=1.0)])
        state = region.init(vec_params)
        assert isinstance(state, tuple)
        assert len(state) == 2

    def test_update_fans_out(self, vec_params):
        region = Sequential([TrustRegion(radius=1.0), BoxRegion(0.0, 1.0)])
        state = region.init(vec_params)
        info = RegionInfo(
            params=vec_params,
            new_params=vec_params,
            pred_reduction=jnp.asarray(1.0),
            actual_reduction=jnp.asarray(1.0),
        )
        new_state = region.update(state, info)
        assert isinstance(new_state, tuple)
        assert len(new_state) == 2

    def test_single_region_equivalence(self):
        box = BoxRegion(0.0, 1.0)
        seq = Sequential([BoxRegion(0.0, 1.0)])
        params = jnp.zeros(2)
        cand = jnp.array([5.0, -3.0])
        _assert_tree_allclose(_run(seq, params, cand), _run(box, params, cand))


# --------------------------------------------------------------------------- #
# Cross-cutting: jit / vmap / grad compatibility
# --------------------------------------------------------------------------- #


class TestTransformCompatibility:
    @pytest.mark.parametrize(
        "region",
        [
            IdentityRegion(),
            BoxRegion(-1.0, 1.0),
            OrthantRegion(),
            QuantizationRegion(step=0.5, lo=-1.0, hi=1.0),
            NoDecreaseRegion(lambda p: jnp.ones_like(p)),
            TrustRegion(radius=1.0),
        ],
    )
    def test_jit_project(self, region):
        params = jnp.array([0.5, -0.5, 0.1])
        cand = jnp.array([2.0, -2.0, 0.3])
        state = region.init(params)
        fn = jax.jit(region.project)
        out = fn(params, cand, state)
        ref = region.project(params, cand, state)
        _assert_tree_allclose(out, ref, rtol=1e-6)

    def test_vmap_box(self):
        region = BoxRegion(-1.0, 1.0)
        params = jnp.zeros((5, 3))
        cand = jnp.linspace(-3.0, 3.0, 15).reshape(5, 3)

        def proj(p, c):
            return region.project(p, c, region.init(p))

        out = jax.vmap(proj)(params, cand)
        assert out.shape == (5, 3)
        assert jnp.all(out <= 1.0) and jnp.all(out >= -1.0)

    def test_grad_through_trust_region(self):
        region = TrustRegion(radius=1.0)

        def loss(cand):
            params = jnp.zeros_like(cand)
            out = region.project(params, cand, region.init(params))
            return jnp.sum(out**2)

        g = jax.grad(loss)(jnp.array([3.0, 4.0]))
        assert g.shape == (2,)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_no_decrease(self):
        grad_vec = jnp.array([1.0, 0.0])
        region = NoDecreaseRegion(lambda p: grad_vec)

        def loss(cand):
            params = jnp.zeros_like(cand)
            out = region.project(params, cand, region.init(params))
            return jnp.sum(out**2)

        g = jax.grad(loss)(jnp.array([1.0, 1.0]))
        assert jnp.all(jnp.isfinite(g))


# --------------------------------------------------------------------------- #
# Region protocol structural checks
# --------------------------------------------------------------------------- #


class TestRegionProtocol:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: IdentityRegion(),
            lambda: BoxRegion(0.0, 1.0),
            lambda: OrthantRegion(),
            lambda: QuantizationRegion(step=0.5),
            lambda: NoDecreaseRegion(lambda p: p),
            lambda: TrustRegion(),
            lambda: Sequential([BoxRegion(0.0, 1.0)]),
        ],
    )
    def test_is_region_with_callables(self, factory):
        region = factory()
        assert isinstance(region, Region)
        assert callable(region.init)
        assert callable(region.project)
        assert callable(region.update)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

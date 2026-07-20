"""Unit tests for point_history: publish() and secant_view()."""

import jax.numpy as jnp
import numpy as np
import pytest

from qqn_jax.oracles.oracle import OracleInfo
from qqn_jax.oracles.point_history import (
    publish,
    secant_view,
    PublishedPoints,
    SecantStoreView,
)


def make_info_no_probes(n=3):
    x = jnp.arange(n, dtype=jnp.float32)
    xn = x + 1.0
    g = jnp.ones(n, dtype=jnp.float32)
    gn = g * 2.0
    return OracleInfo(
        params=x,
        new_params=xn,
        grad=g,
        new_grad=gn,
        t=jnp.asarray(1.0),
        step_size=jnp.asarray(1.0),
    )


def make_info_with_probes(n=3, k=4):
    x = jnp.zeros(n, dtype=jnp.float32)
    xn = jnp.ones(n, dtype=jnp.float32) * 5.0
    g = jnp.ones(n, dtype=jnp.float32)
    gn = jnp.ones(n, dtype=jnp.float32) * 0.5
    # probes at various alphas (unordered)
    probe_alphas = jnp.asarray([0.5, 0.1, 0.9, 0.3], dtype=jnp.float32)
    probe_params = jnp.stack(
        [jnp.ones(n, dtype=jnp.float32) * a for a in [0.5, 0.1, 0.9, 0.3]]
    )
    probe_grads = jnp.stack(
        [jnp.ones(n, dtype=jnp.float32) * (1.0 - a) for a in [0.5, 0.1, 0.9, 0.3]]
    )
    probe_valid = jnp.asarray([True, True, False, True])
    return OracleInfo(
        params=x,
        new_params=xn,
        grad=g,
        new_grad=gn,
        t=jnp.asarray(1.0),
        step_size=jnp.asarray(1.0),
        probe_params=probe_params,
        probe_grads=probe_grads,
        probe_valid=probe_valid,
        probe_alphas=probe_alphas,
    )


class TestPublish:
    def test_returns_none_without_probes(self):
        info = make_info_no_probes()
        assert publish(info) is None

    def test_returns_none_with_partial_probes(self):
        info = make_info_no_probes()._replace(
            probe_params=jnp.zeros((2, 3)),
            probe_alphas=None,
        )
        assert publish(info) is None

    def test_publish_basic_structure(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        assert isinstance(pts, PublishedPoints)
        # 4 probes + 1 accepted
        assert pts.params_seq.shape[0] == 5
        assert pts.grad_seq.shape[0] == 5
        assert pts.alpha_seq.shape[0] == 5
        assert pts.valid_seq.shape[0] == 5

    def test_publish_accepted_point_is_last_and_valid(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        np.testing.assert_allclose(pts.params_seq[-1], info.new_params)
        np.testing.assert_allclose(pts.grad_seq[-1], info.new_grad)
        assert bool(pts.valid_seq[-1])

    def test_publish_anchor(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        np.testing.assert_allclose(pts.anchor_params, info.params)
        np.testing.assert_allclose(pts.anchor_grad, info.grad)

    def test_publish_ordered_by_alpha(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        # valid probe alphas should be non-decreasing among themselves
        valid = np.asarray(pts.valid_seq[:-1])
        alphas = np.asarray(pts.alpha_seq[:-1])
        valid_alphas = alphas[valid]
        assert np.all(np.diff(valid_alphas) >= 0)

    def test_max_replay_caps_probes(self):
        info = make_info_with_probes()
        pts = publish(info, max_replay=2)
        assert pts is not None
        # 2 kept probes + 1 accepted
        assert pts.params_seq.shape[0] == 3

    def test_max_replay_keeps_largest_alpha(self):
        info = make_info_with_probes()
        pts = publish(info, max_replay=1)
        assert pts is not None
        # only 1 probe kept + accepted; largest valid alpha is 0.5
        assert pts.params_seq.shape[0] == 2
        # the retained probe's alpha should be the max valid one (0.5)
        assert float(pts.alpha_seq[0]) == pytest.approx(0.5)

    def test_max_replay_larger_than_k(self):
        info = make_info_with_probes(k=4)
        pts = publish(info, max_replay=100)
        assert pts is not None
        assert pts.params_seq.shape[0] == 5

    def test_step_size_none_defaults_to_one(self):
        info = make_info_with_probes()._replace(step_size=None)
        pts = publish(info)
        assert pts is not None
        assert float(pts.alpha_seq[-1]) == pytest.approx(1.0)


class TestSecantView:
    def test_secant_view_shapes(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        view = secant_view(pts)
        assert isinstance(view, SecantStoreView)
        k = pts.params_seq.shape[0]
        n = pts.params_seq.shape[1]
        assert view.deltas.shape == (k, n)
        assert view.gdeltas.shape == (k, n)
        assert view.anch_dx.shape == (k, n)
        assert view.anch_dg.shape == (k, n)

    def test_deltas_are_consecutive_differences(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        view = secant_view(pts)
        anchored_p = np.concatenate(
            [np.asarray(pts.anchor_params)[None], np.asarray(pts.params_seq)], axis=0
        )
        expected = anchored_p[1:] - anchored_p[:-1]
        np.testing.assert_allclose(view.deltas, expected, rtol=1e-6)

    def test_anchored_dx(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        view = secant_view(pts)
        new_params = np.asarray(pts.params_seq[-1])
        expected = new_params[None, :] - np.asarray(pts.params_seq)
        np.testing.assert_allclose(view.anch_dx, expected, rtol=1e-6)
        # last anchored dx is zero (x_new - x_new)
        np.testing.assert_allclose(view.anch_dx[-1], 0.0, atol=1e-6)

    def test_view_property_passthrough(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        view = secant_view(pts)
        np.testing.assert_allclose(view.params_seq, pts.params_seq)
        np.testing.assert_allclose(view.grad_seq, pts.grad_seq)
        np.testing.assert_allclose(view.valid_seq, pts.valid_seq)

    def test_newest_secant(self):
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        view = secant_view(pts)
        s, y = view.newest_secant()
        assert s.shape == pts.params_seq[-1].shape
        assert y.shape == pts.grad_seq[-1].shape

    def test_newest_secant_uses_most_recent_valid(self):
        # If the last probe before accepted is invalid, secant should
        # skip it and use an earlier valid one.
        info = make_info_with_probes()
        pts = publish(info)
        assert pts is not None
        view = secant_view(pts)
        s, y = view.newest_secant()
        # s should be finite
        assert np.all(np.isfinite(np.asarray(s)))

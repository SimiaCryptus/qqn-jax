import jax
from jax import numpy as jnp

from qqn_jax.regions.psd_secant import PSDSecantRegion
from qqn_jax.regions.types import RegionInfo


def _make_info(params, new_params, grad, new_grad):
    return RegionInfo(
        params=params,
        new_params=new_params,
        grad=grad,
        new_grad=new_grad,
        pred_reduction=jnp.asarray(1.0),
        actual_reduction=jnp.asarray(1.0),
        t=jnp.asarray(1.0),
        step_size=jnp.asarray(1.0),
    )


def test_empty_history_is_isotropic_clip():

    region = PSDSecantRegion(window=4, gamma=4.0, radius=2.0)
    params = jnp.zeros(3)
    state = region.init(params)

    candidate = jnp.array([3.0, 0.0, 0.0])
    out = region.project(params, candidate, state)
    step = out - params

    mnorm = jnp.sqrt(4.0) * jnp.linalg.norm(step)
    assert jnp.allclose(mnorm, 2.0, atol=1e-5)


def test_large_radius_is_identity():
    region = PSDSecantRegion(window=4, gamma=1.0, radius=1e12)
    params = jnp.zeros(3)
    state = region.init(params)
    candidate = jnp.array([1.0, -2.0, 0.5])
    out = region.project(params, candidate, state)
    assert jnp.allclose(out, candidate, atol=1e-6)


def test_secant_makes_stiff_direction_harder():

    region = PSDSecantRegion(window=4, gamma=1.0, radius=1.0)
    params = jnp.zeros(2)
    state = region.init(params)

    info = _make_info(
        params=jnp.zeros(2),
        new_params=jnp.array([1.0, 0.0]),
        grad=jnp.zeros(2),
        new_grad=jnp.array([10.0, 0.0]),
    )
    state = region.update(state, info)

    stiff = region.project(params, jnp.array([1.0, 0.0]), state) - params
    soft = region.project(params, jnp.array([0.0, 1.0]), state) - params
    assert jnp.linalg.norm(stiff) < jnp.linalg.norm(soft)


def test_jit_compatible():
    region = PSDSecantRegion(window=4)
    params = jnp.zeros(3)
    state = region.init(params)
    proj = jax.jit(region.project)
    out = proj(params, jnp.ones(3), state)
    assert out.shape == params.shape

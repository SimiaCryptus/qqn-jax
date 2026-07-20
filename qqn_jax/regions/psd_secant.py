from typing import NamedTuple

import jax
from jax import numpy as jnp

from qqn_jax.regions.types import Region, _tree_sub, _tree_add


class PSDSecantState(NamedTuple):
    """Bounded window of realized secant pairs backing a PSD metric.

    Attributes:
        s_history: window of iterate deltas ``s = x_new - x``, shape ``(m, n)``.
        y_history: window of gradient deltas ``y = ∇f_new - ∇f``, shape ``(m, n)``.
        prev_params: previous accepted iterate (flat ``(n,)``).
        prev_grad: previous accepted gradient (flat ``(n,)``).
        step_count: number of valid columns currently stored.
    """

    s_history: jnp.ndarray
    y_history: jnp.ndarray
    prev_params: jnp.ndarray
    prev_grad: jnp.ndarray
    step_count: jnp.ndarray


def PSDSecantRegion(
    window: int = 10,
    gamma: float = 1.0,
    radius: float = 1.0,
    reg: float = 1e-8,
) -> Region:
    """Project the step in a PSD metric inferred from realized secants.

    Quasi-Newton oracles measure curvature through secant pairs

        s_i = x_{i+1} − x_i        y_i = ∇f_{i+1} − ∇f_i,

    which — when curvature is positive (``⟨s_i, y_i⟩ > 0``) — sample the
    Hessian along the direction ``s_i``. This region *reuses that measured
    curvature geometrically*: rather than steering the search (an oracle's
    job) it reshapes the **feasible set**, allowing generous motion along
    soft (low-curvature) directions while walling motion along stiff
    (high-curvature) ones.

    Concretely it maintains a bounded window of the last ``window`` accepted
    secant pairs and forms the low-rank PSD metric

        M = γ·I + Σ_i (y_i y_iᵀ) / ⟨s_i, y_i⟩          (BFGS-flavored curvature),

    the same rank-one accumulation whose inverse the BFGS/L-BFGS update
    approximates. The proposed step ``step = candidate − x`` is then confined
    to the ``M``-ellipsoid of squared radius ``radius²``::

        q       = ⟨step, M · step⟩
        scale   = min(1, radius / √(q + reg))
        s_proj  = scale · step

    so the step shrinks *anisotropically* — hardest along the stiff
    directions the secants exposed, gently along the flat ones. With an empty
    history (``step_count = 0``) the metric is ``γ·I`` and the region reduces
    to an isotropic trust-region-style clip of radius ``radius/√γ``, and with
    ``radius → ∞`` it is the identity, preserving the un-regioned optimizer.

    Args:
        window: number of secant pairs retained (buffer depth ``m``).
        gamma: isotropic floor ``γ`` added to the metric (keeps ``M`` PD even
            with no/curvature-degenerate history).
        radius: ellipsoid radius bounding the ``M``-norm of the step.
        reg: small stabilizer added under the square root.
    """
    eps = 1e-12

    def _flatten(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.concatenate([jnp.ravel(l) for l in leaves])

    def init(params):
        flat = _flatten(params)
        n = flat.shape[0]
        zeros = jnp.zeros((window, n), dtype=flat.dtype)
        return PSDSecantState(
            s_history=zeros,
            y_history=zeros,
            prev_params=flat,
            prev_grad=jnp.zeros((n,), dtype=flat.dtype),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def _apply_metric(state, v):
        """Return ``M · v`` for a flat vector ``v`` (low-rank + γI)."""
        s = state.s_history
        y = state.y_history
        m = s.shape[0]
        active = jnp.arange(m) < state.step_count
        sy = jnp.sum(s * y, axis=1)
        curvature_ok = jnp.logical_and(active, sy > eps)
        # coefficient c_i = ⟨y_i, v⟩ / ⟨s_i, y_i⟩ , masked when unusable
        yv = y @ v
        coeff = jnp.where(curvature_ok, yv / jnp.where(curvature_ok, sy, 1.0), 0.0)
        low_rank = coeff @ y
        return gamma * v + low_rank

    def project(params, candidate, state):
        flat_params = _flatten(params)
        flat_cand = _flatten(candidate)
        step = flat_cand - flat_params

        Mstep = _apply_metric(state, step)
        q = jnp.vdot(step, Mstep)
        scale = jnp.minimum(1.0, radius / jnp.sqrt(q + reg))

        s_proj_flat = scale * step

        # unflatten back onto the parameter pytree structure
        leaves, treedef = jax.tree_util.tree_flatten(params)
        sizes = [l.size for l in leaves]
        splits = jnp.cumsum(jnp.asarray(sizes))[:-1]
        chunks = jnp.split(s_proj_flat, splits) if len(leaves) > 1 else [s_proj_flat]
        s_leaves = [c.reshape(l.shape) for c, l in zip(chunks, leaves)]
        s_proj = jax.tree_util.tree_unflatten(treedef, s_leaves)
        return _tree_add(params, s_proj)

    def update(state, info):
        new_params = _flatten(info.new_params)
        new_grad = _flatten(info.new_grad)
        s = new_params - state.prev_params
        y = new_grad - state.prev_grad

        has_grad = jnp.any(state.prev_grad != 0.0) | (state.step_count > 0)
        valid = jnp.logical_and(has_grad, jnp.vdot(s, y) > eps)

        rolled_s = jnp.roll(state.s_history, shift=1, axis=0).at[0].set(s)
        rolled_y = jnp.roll(state.y_history, shift=1, axis=0).at[0].set(y)
        new_s = jnp.where(valid, rolled_s, state.s_history)
        new_y = jnp.where(valid, rolled_y, state.y_history)
        new_count = jnp.where(
            valid, jnp.minimum(state.step_count + 1, window), state.step_count
        )
        return PSDSecantState(
            s_history=new_s,
            y_history=new_y,
            prev_params=new_params,
            prev_grad=new_grad,
            step_count=new_count,
        )

    return Region(init=init, project=project, update=update)
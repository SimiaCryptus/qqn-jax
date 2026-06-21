"""Line search strategies for QQN.

The line search is a *first-class component* of QQN. It operates over the
quadratic path direction ``d`` (already constructed) and selects a step size
``α`` satisfying sufficient decrease (Armijo) and, optionally, the curvature
(strong Wolfe) condition.

Both searches are written with ``lax`` control flow so they are fully
JIT/vmap compatible.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_add_scaled, tree_vdot


class LineSearchResult(NamedTuple):
    """Result of a line search.

    Attributes:
        step_size: chosen step size ``α``.
        new_value: function value at ``params + α·d``.
        new_grad: gradient at ``params + α·d``.
        new_params: the updated parameters.
        done: whether the search satisfied its conditions.
    """

    step_size: jnp.ndarray
    new_value: jnp.ndarray
    new_grad: jnp.ndarray
    new_params: jnp.ndarray
    done: jnp.ndarray


def backtracking_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-4,
    shrink: float = 0.5,
    max_iter: int = 30,
) -> LineSearchResult:
    """Backtracking line search with the Armijo sufficient-decrease condition.

        f(x + α·d) <= f(x) + c1·α·∇fᵀd
    """
    dphi0 = tree_vdot(grad, direction)

    def cond(carry):
        i, alpha, done, *_ = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(done))

    def body(carry):
        i, alpha, _done, _nv, _ng, _np = carry
        new_params = tree_add_scaled(params, alpha, direction)
        new_value, new_grad = value_and_grad_fn(new_params, *args)
        armijo = new_value <= value + c1 * alpha * dphi0
        # If the directional derivative is non-negative we cannot make
        # progress; accept to avoid an infinite loop.
        not_descent = dphi0 >= 0.0
        done = jnp.logical_or(armijo, not_descent)
        next_alpha = jnp.where(done, alpha, alpha * shrink)
        return (i + 1, next_alpha, done, new_value, new_grad, new_params)

    init_params = tree_add_scaled(params, init_step, direction)
    init_value, init_grad = value_and_grad_fn(init_params, *args)
    init_carry = (
        jnp.asarray(0, jnp.int32),
        jnp.asarray(init_step),
        jnp.asarray(False),
        init_value,
        init_grad,
        init_params,
    )

    i, alpha, done, nv, ng, np_ = jax.lax.while_loop(
        cond, body, init_carry
    )
    # ``alpha`` at exit is the *next* step; the accepted step is alpha/shrink
    # when done was set inside the body. Recompute cleanly for correctness.
    accepted_alpha = jnp.where(done, alpha, alpha)
    return LineSearchResult(
        step_size=accepted_alpha,
        new_value=nv,
        new_grad=ng,
        new_params=np_,
        done=done,
    )


def strong_wolfe_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 30,
) -> LineSearchResult:
    """Strong Wolfe line search (bracketing + zoom).

    Enforces both the Armijo sufficient-decrease condition and the strong
    curvature condition:

        f(x + α·d) <= f(x) + c1·α·φ'(0)
        |φ'(α)|    <= c2·|φ'(0)|

    Satisfying strong Wolfe is what keeps the L-BFGS curvature updates
    well-conditioned.
    """
    dphi0 = tree_vdot(grad, direction)

    def phi(alpha):
        new_params = tree_add_scaled(params, alpha, direction)
        v, g = value_and_grad_fn(new_params, *args)
        dphi = tree_vdot(g, direction)
        return v, dphi, g, new_params

    # Bracketing phase: find an interval [alpha_lo, alpha_hi] known to
    # contain a point satisfying strong Wolfe.
    class Bracket(NamedTuple):
        i: jnp.ndarray
        alpha_prev: jnp.ndarray
        phi_prev: jnp.ndarray
        alpha: jnp.ndarray
        lo: jnp.ndarray
        hi: jnp.ndarray
        phi_lo: jnp.ndarray
        dphi_lo: jnp.ndarray
        phi_hi: jnp.ndarray
        done: jnp.ndarray
        found: jnp.ndarray

    v0 = value

    def bracket_cond(b: Bracket):
        return jnp.logical_and(b.i < max_iter, jnp.logical_not(b.done))

    def bracket_body(b: Bracket):
        phi_a, dphi_a, _g, _p = phi(b.alpha)

        armijo_violated = jnp.logical_or(
            phi_a > v0 + c1 * b.alpha * dphi0,
            jnp.logical_and(phi_a >= b.phi_prev, b.i > 0),
        )
        wolfe_ok = jnp.abs(dphi_a) <= -c2 * dphi0
        curvature_pos = dphi_a >= 0.0

        # Case A: bracket found via Armijo violation / non-decrease.
        caseA = armijo_violated
        # Case B: strong Wolfe already satisfied -> done, found.
        caseB = jnp.logical_and(jnp.logical_not(caseA), wolfe_ok)
        # Case C: positive slope -> bracket [alpha, alpha_prev].
        caseC = jnp.logical_and(
            jnp.logical_not(caseA),
            jnp.logical_and(jnp.logical_not(caseB), curvature_pos),
        )

        done = jnp.logical_or(caseA, jnp.logical_or(caseB, caseC))
        found = caseB

        # Determine lo/hi when a bracket is established (A or C).
        # For A: lo = alpha_prev, hi = alpha.
        # For C: lo = alpha, hi = alpha_prev.
        lo = jnp.where(caseC, b.alpha, b.alpha_prev)
        hi = jnp.where(caseC, b.alpha_prev, b.alpha)
        phi_lo = jnp.where(caseC, phi_a, b.phi_prev)
        phi_hi = jnp.where(caseC, b.phi_prev, phi_a)
        # dphi_lo only well-defined for caseC (we have dphi_a). For caseA we
        # recompute in zoom; store dphi0 as a placeholder for lo=alpha_prev.
        dphi_lo = jnp.where(caseC, dphi_a, dphi0)

        # Otherwise expand: alpha_prev <- alpha, alpha <- 2*alpha.
        next_alpha_prev = jnp.where(done, b.alpha_prev, b.alpha)
        next_phi_prev = jnp.where(done, b.phi_prev, phi_a)
        next_alpha = jnp.where(done, b.alpha, 2.0 * b.alpha)

        return Bracket(
            i=b.i + 1,
            alpha_prev=next_alpha_prev,
            phi_prev=next_phi_prev,
            alpha=next_alpha,
            lo=jnp.where(done, lo, b.lo),
            hi=jnp.where(done, hi, b.hi),
            phi_lo=jnp.where(done, phi_lo, b.phi_lo),
            dphi_lo=jnp.where(done, dphi_lo, b.dphi_lo),
            phi_hi=jnp.where(done, phi_hi, b.phi_hi),
            done=done,
            found=found,
        )

    init_bracket = Bracket(
        i=jnp.asarray(0, jnp.int32),
        alpha_prev=jnp.asarray(0.0),
        phi_prev=v0,
        alpha=jnp.asarray(init_step),
        lo=jnp.asarray(0.0),
        hi=jnp.asarray(init_step),
        phi_lo=v0,
        dphi_lo=dphi0,
        phi_hi=v0,
        done=jnp.asarray(False),
        found=jnp.asarray(False),
    )

    b = jax.lax.while_loop(bracket_cond, bracket_body, init_bracket)

    # Zoom phase: bisection within [lo, hi] until strong Wolfe holds.
    class Zoom(NamedTuple):
        i: jnp.ndarray
        lo: jnp.ndarray
        hi: jnp.ndarray
        phi_lo: jnp.ndarray
        best_alpha: jnp.ndarray
        done: jnp.ndarray

    def zoom_cond(z: Zoom):
        return jnp.logical_and(z.i < max_iter, jnp.logical_not(z.done))

    def zoom_body(z: Zoom):
        alpha_j = 0.5 * (z.lo + z.hi)
        phi_j, dphi_j, _g, _p = phi(alpha_j)

        armijo = phi_j <= v0 + c1 * alpha_j * dphi0
        below_lo = phi_j < z.phi_lo
        wolfe_ok = jnp.abs(dphi_j) <= -c2 * dphi0

        satisfied = jnp.logical_and(
            jnp.logical_and(armijo, below_lo), wolfe_ok
        )

        # If Armijo fails or phi_j >= phi_lo: shrink hi to alpha_j.
        shrink_hi = jnp.logical_or(
            jnp.logical_not(armijo), jnp.logical_not(below_lo)
        )

        # Otherwise move lo to alpha_j (and possibly swap hi/lo on sign).
        sign_flip = (dphi_j * (z.hi - z.lo)) >= 0.0
        new_hi = jnp.where(
            shrink_hi,
            alpha_j,
            jnp.where(sign_flip, z.lo, z.hi),
        )
        new_lo = jnp.where(shrink_hi, z.lo, alpha_j)
        new_phi_lo = jnp.where(shrink_hi, z.phi_lo, phi_j)

        best_alpha = jnp.where(satisfied, alpha_j, alpha_j)

        return Zoom(
            i=z.i + 1,
            lo=new_lo,
            hi=new_hi,
            phi_lo=new_phi_lo,
            best_alpha=best_alpha,
            done=satisfied,
        )

    init_zoom = Zoom(
        i=jnp.asarray(0, jnp.int32),
        lo=b.lo,
        hi=b.hi,
        phi_lo=b.phi_lo,
        best_alpha=jnp.where(b.found, b.alpha, 0.5 * (b.lo + b.hi)),
        done=b.found,
    )

    z = jax.lax.while_loop(zoom_cond, zoom_body, init_zoom)

    final_alpha = jnp.where(b.found, b.alpha, z.best_alpha)
    # Guard against a degenerate zero step.
    final_alpha = jnp.where(final_alpha > 0.0, final_alpha, init_step)

    new_params = tree_add_scaled(params, final_alpha, direction)
    new_value, new_grad = value_and_grad_fn(new_params, *args)

    return LineSearchResult(
        step_size=final_alpha,
        new_value=new_value,
        new_grad=new_grad,
        new_params=new_params,
        done=jnp.logical_or(b.found, z.done),
    )
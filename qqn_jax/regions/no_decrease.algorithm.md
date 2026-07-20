# NoDecreaseRegion

## Overview

`NoDecreaseRegion` is a **step-projection region** for iterative optimization.
It constrains each proposed optimization step so that it never *increases* a
secondary (protected) objective `g`, while leaving the primary optimization
free to move in every direction that does not climb `g`.

This is the geometry of **continual learning** and **constrained
fine-tuning**: we optimize a primary objective but must not degrade some
protected quantity (e.g. accuracy on a previous task, a safety metric, a
regularization target).

## Mathematical Formulation

Given:

- Current parameters `x` (`params`),
- A candidate next point `candidate` proposed by the optimizer,
- A secondary objective `g` with gradient `∇g = secondary_grad_fn(x)`,

define the proposed step

    step = candidate - x

We wish to forbid steps that increase `g` to first order. The directional
change in `g` along `step` is

    c = ⟨∇g, step⟩

- If `c ≤ 0`, the step already descends (or is neutral on) `g`, and is
  accepted unchanged.
- If `c > 0`, the step would increase `g`. We remove exactly the offending
  component by orthogonally projecting `step` onto the half-space
  `{ s : ⟨∇g, s⟩ ≤ 0 }`:

    s_proj = step - relu(c) / (‖∇g‖² + eps) · ∇g

Here `relu(c) = max(c, 0)` guarantees that only the *positive*,
g-increasing component is removed. Descent on `g` passes through untouched.

The region finally returns

    x_new = x + s_proj

### Why this is an orthogonal projection

Removing `c / ‖∇g‖² · ∇g` from `step` subtracts precisely the projection of
`step` onto the direction of `∇g`. After subtraction, the residual is
orthogonal to `∇g`, so `⟨∇g, s_proj⟩ = 0` when `c > 0`. This is the closest
point (in Euclidean norm) inside the allowed half-space to the original
`step` — the defining property of the projection onto a half-space.

The `eps = 1e-12` term stabilizes the division when `∇g` is (near) zero. In
that degenerate case `coeff → 0` and the step is left unchanged, which is the
correct behavior: with no gradient information there is no direction to
remove.

## Implementation Notes

The implementation operates over arbitrary PyTree parameter structures:

- `_tree_sub(candidate, params)` computes the step `candidate - x`.
- The inner products `c` and `gg = ‖∇g‖²` are accumulated with `jnp.vdot`
  over corresponding tree leaves, giving flattened dot products.
- `coeff = max(c, 0) / (gg + eps)` is a scalar applied uniformly across all
  leaves via `tree_map`.
- `_tree_add(params, s_proj)` reconstructs the projected next point.

The region reuses the identity lifecycle hooks:

- `init = _identity_init` — no persistent state is required.
- `update = _identity_update` — nothing to update between iterations.

Only `project` carries logic; the region is effectively **stateless**.

## Interface

```python
def NoDecreaseRegion(secondary_grad_fn: Callable) -> Region
```

- **`secondary_grad_fn(params) -> ∇g`**: a callable returning the gradient
  of the protected objective `g` at the current parameters, with the same
  PyTree structure as `params`.

Returns a `Region` with `init`, `project`, and `update` members compatible
with the `qqn_jax.regions.strategy` framework.

## Discussion

### Behavioral properties

- **Non-intrusive:** When the primary step already respects the constraint
  (`c ≤ 0`), the step is passed through with zero modification. The region
  only acts when there is a genuine conflict.
- **Minimal correction:** Because the correction is an orthogonal
  projection, it perturbs the step as little as possible while satisfying
  the constraint. No arbitrary damping or scaling is introduced.
- **First-order guarantee:** The constraint is linear in `step`, based on the
  local gradient `∇g`. It guarantees non-increase of `g` only to first
  order; for large steps or highly curved `g`, second-order effects may
  cause small violations. Pairing with a suitably small step size or a line
  search tightens the guarantee.

### Relationship to gradient surgery

This projection is closely related to **gradient surgery** techniques (e.g.
PCGrad) used in multi-task and continual learning, where conflicting
gradient components are projected away. Here the same idea is applied at the
level of the *step* against a single protected objective's gradient.

### Limitations and extensions

- It protects against a **single** secondary objective. Multiple protected
  objectives would require projecting onto the intersection of several
  half-spaces (an active-set or Dykstra-style iteration).
- The guarantee is local/linear; strongly nonlinear `g` may need
  re-evaluation of `∇g` or a trust-region step-size control.
- The choice of `relu(c)` (rather than removing all of `c`) is deliberate:
  it turns an equality projection into a one-sided half-space projection,
  permitting free descent on `g`.
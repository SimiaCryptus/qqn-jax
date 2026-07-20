# QuantizationRegion Algorithm

## Overview

`QuantizationRegion` is a projection-based *region* constraint that confines
each weight coordinate to the **rounding cell** of its starting value on a
uniform quantization grid. It is designed to steer an optimizer toward
grid-representable (quantized) values without hard-snapping during the
search, thereby preserving gradient information while biasing the solution
toward low rounding error.

## Motivation

A uniform quantizer with `bits` levels over the interval `[lo, hi]` defines a
grid of representable values spaced by

```
Δ = (hi − lo) / (2**bits − 1)
```

or by an explicit `step`. The natural regularizer associated with rounding is
the **L1 rounding-delta penalty** `|x − round(x)|`. This is a *sawtooth*
function:

- its **minima** (zero error) sit exactly on the grid points `g_k`,
- its **maxima** (largest error, `Δ/2`) sit on the *midpoints* between
  adjacent grid points.

The midpoints (local maxima of the rounding delta) partition the real line
into hypercubic **cells**:

```
cell_k = [g_k − Δ/2,  g_k + Δ/2],   with   g_k = lo + k·Δ
```

Each cell contains exactly one grid point at its center. `QuantizationRegion`
walls the search at these midpoints so a coordinate can freely explore its
own cell but never drift into a neighbour's cell (and thus never cross a
rounding-delta maximum).

## Algorithm

Given `params` (the iterate at the **start of the step**), a `candidate`
(the proposed post-step value), and region `state`, the projection operates
leaf-wise (`jax.tree_util.tree_map`) and coordinate-wise on tensors.

For each coordinate with value `x` (from `params`) and candidate `c`:

1. **Clip to range.** Compute `x_clipped = clip(x, lo, hi)`. Values outside
   the quantization range are pulled to the boundary before rounding.

2. **Find the nearest grid index.**
   ```
   k = round((x_clipped − lo) / Δ)
   ```

3. **Clamp the index** to the valid grid range so `g_k` stays within
   `[lo, hi]`:
   ```
   k_max = floor((hi − lo) / Δ)
   k     = clip(k, 0, k_max)
   ```

4. **Compute the cell center** (the grid point / attractor):
   ```
   center = lo + k·Δ
   ```

5. **Project the candidate.**
   - If `lock=True`: return `center` directly — a hard-snap to the grid
     point (true quantization, no exploration).
   - If `lock=False` (default): build the (possibly narrowed) cell box and
     clip the candidate into it:
     ```
     half    = window · Δ / 2
     cell_lo = max(center − half, lo)
     cell_hi = min(center + half, hi)
     return clip(c, cell_lo, cell_hi)
     ```

The anchoring in step 1–4 uses **`params`** (start-of-step value), while the
clamping in step 5 acts on **`candidate`** (proposed value). This is what
makes the cell "follow" the iterate: the wall is placed around wherever the
coordinate currently sits, and the candidate is confined within it.

## Parameters

| Parameter | Meaning |
|-----------|---------|
| `bits`    | Number of quantization bits; grid has `2**bits` levels over `[lo, hi]`. |
| `step`    | Explicit grid spacing `Δ`; **overrides** `bits` when supplied. |
| `lo`, `hi`| Quantization range. Values are clipped to `[lo, hi]` before rounding. |
| `lock`    | If `True`, hard-snap to nearest grid point. If `False` (default), allow roaming within the cell. |
| `window`  | Fraction of the half-cell explorable when `lock=False`. `1.0` exposes the full cell; smaller values tighten the box symmetrically about the grid point. |

At least one of `bits` or `step` must be provided; otherwise a `ValueError`
is raised.

## Grid-spacing computation

```python
def _delta(dtype):
    if step is not None:
        return step                       # explicit spacing
    levels = (2**bits) - 1
    return (hi - lo) / levels             # derived from bit width
```

The delta is materialized in the candidate's dtype to keep the arithmetic
numerically consistent with the tensor being projected.

## Region interface

The returned `Region` reuses the identity lifecycle hooks:

- `init = _identity_init` — no per-region state is required.
- `project = project` — the core cell projection described above.
- `update = _identity_update` — no state mutation between steps.

Because there is no internal state, the region is stateless and idempotent
with respect to `init`/`update`; all behavior lives in `project`.

## Discussion

### Why anchor to the grid point rather than penalize?

Rather than adding a soft rounding penalty to the loss (which perturbs the
optimization landscape and requires tuning a penalty weight),
`QuantizationRegion` enforces a **hard geometric constraint** via projection.
The cell center — the grid point — is the local minimum of rounding error,
so confining the search to the cell naturally attracts the optimizer toward
a quantized value while still permitting local exploration.

### `lock` vs. free exploration

- **`lock=True`** yields immediate, exact quantization. Useful for a final
  "hardening" phase or for measuring quantized performance.
- **`lock=False`** keeps the coordinate differentiable-friendly by allowing
  it to move within its cell, which preserves useful gradient signal during
  training. The `window` knob interpolates between full-cell freedom
  (`1.0`) and near-lock behavior (`→ 0`).

### Boundary handling

Steps 1 and 3 guarantee that the chosen grid point and its cell stay within
`[lo, hi]`. The `cell_lo`/`cell_hi` `max`/`min` guards in step 5 additionally
prevent the explorable box from spilling outside the quantization range for
grid points near the boundaries.

### Numerical considerations

- All constants (`Δ`, `lo`, `hi`, `window`) are cast to the leaf dtype to
  avoid implicit upcasting and to keep the projection type-stable.
- `round`/`floor`/`clip` are elementwise and vectorized, so the projection
  is efficient across arbitrary tensor shapes via `tree_map`.

## Complexity

The projection is `O(N)` in the number of parameters, elementwise, with no
additional memory beyond transient intermediates. It is fully JAX-traceable
and compatible with `jit`/`vmap`.
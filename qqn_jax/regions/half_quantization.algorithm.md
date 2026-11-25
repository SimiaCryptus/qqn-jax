# HalfQuantizationRegion Algorithm

## Overview

`HalfQuantizationRegion` generalizes `OrthantRegion` (OWL-QN's sign-orthant projection) from a single boundary hyperplane at $0$ to the **half-precision quantization lattice** $\mathcal{Q} = \{q_k\}$ (such as 16-bit `bfloat16`, 16-bit IEEE `float16`, or a symmetric 16-bit uniform grid when optimizing 32-bit `float32` weights).

Whereas `QuantizationRegion` places walls at the **midpoints** $q_k \pm \Delta/2$ around each grid value $q_k$ (so clamping at a wall lands on a *worst-case* midpoint tie and leaves the $|x - q_k|$ kink inside the cell), `HalfQuantizationRegion` places walls **directly on the representable 16-bit values** $\{q_k\}$:

$$\mathcal{C}(x_i) = \big[\lfloor x_i \rfloor_{\mathcal{Q}},\; \lceil x_i \rceil_{\mathcal{Q}}\big] = [q_{\text{lo}}(x_i),\; q_{\text{hi}}(x_i)].$$

Consequently:
1. **Every region boundary is an exact 16-bit value:** Any coordinate clamped onto either endpoint $q_{\text{lo}}$ or $q_{\text{hi}}$ rounds to the target 16-bit representation with **zero quantization error**.
2. **Sign-orthant refinement & smooth $L_1$ interior:** Because $0 \in \mathcal{Q}$, no open interval $(q_k, q_{k+1})$ straddles zero. Thus $\|\theta\|_1 = \sum_i |\theta_i|$ is strictly linear ($C^\infty$, zero curvature kink) inside every cell.

## Mathematical Formulation

Given start-of-step parameter leaf $x$ and candidate leaf $c$:

1. **Lattice Bracketing (`half_quant_bounds`):**
   - **IEEE half-floats (`qtype="bfloat16"` or `"float16"`):** Cast $x$ to nearest target float $r_T = \operatorname{RN}_T(x)$ and lift back to $r = \operatorname{float32}(r_T)$. Using `jnp.nextafter` in $T$:
     - If $r = x$: $q_{\text{lo}} = q_{\text{hi}} = x$ ($x$ already lies on a 16-bit boundary).
     - If $r < x$: $q_{\text{lo}} = r$, $q_{\text{hi}} = \operatorname{float32}(\operatorname{nextafter}_T(r_T, +\infty))$.
     - If $r > x$: $q_{\text{lo}} = \operatorname{float32}(\operatorname{nextafter}_T(r_T, -\infty))$, $q_{\text{hi}} = r$.
   - **Symmetric uniform grid (`qtype="uniform"`):** With spacing $\Delta = \text{hi} / (2^{b-1} - 1)$ on $[-\text{hi}, +\text{hi}]$, $0 = 0 \cdot \Delta \in \mathcal{Q}$ is exact, $q_{\text{lo}} = \lfloor x / \Delta \rfloor \Delta$, and $q_{\text{hi}} = \lceil x / \Delta \rceil \Delta$.

2. **Candidate Projection (`boundary_mode`):**
   - If $q_{\text{lo}} < q_{\text{hi}}$ ($x$ is strictly interior to a 16-bit cell):
     $$y = \operatorname{clip}(c,\, q_{\text{lo}},\, q_{\text{hi}}).$$
   - If $q_{\text{lo}} = q_{\text{hi}} = q_k$ ($x$ already sits on a 16-bit boundary):
     - `"sticky"` *(default)*: $y = q_k$ (absorbing boundary, directly mirroring `OrthantRegion`'s $x = 0 \implies 0$ rule).
     - `"owl"`: For $q_k \ne 0$, permits one-cell directional entry into $[\max(q_{k-1}, 0), q_k]$ (when $c < q_k$, $q_k > 0$) or $[q_k, \min(q_{k+1}, 0)]$ (when $c > q_k$, $q_k < 0$), pinning $q_k = 0$ at $0$.
     - `"owl_snap"`: Same multi-cell directional reach as `"owl"`, but transitions from $q_k$ to $q_{k\pm 1}$ only when $c$ reaches that adjacent boundary—preventing already-quantized coordinates from drifting back into the 32-bit interior.

## Complexity & Transform Compatibility

Purely elementwise (`O(P)` time, zero persistent state), using only `astype`, `nextafter`, `floor`/`ceil`, `clip`, and `where` across PyTree leaves—fully compatible with `jax.jit`, `jax.vmap`, `jax.grad`, and distributed FSDP `NamedSharding`.

# EntropyGatedRegion — EG-PTGP as a projective region

Implements `docs/entropy_gated_regions.md`. The `Region` protocol
(`init` / `project` / `update`) is a natural fit: the paper's step
`θ_{t+1} = θ_t − η d*` with `d* = Π_C(d_obj)` is exactly a projection of
the proposed displacement `step = candidate − x` onto a polyhedral cone.

## Sign convention

The paper writes constraints on the *descent direction* `d`
(`Δm_i ≈ −η⟨v_i, d⟩ ≤ ...`). A region sees the *displacement*
`step = −η d`, so `Δm_i ≈ ⟨v_i, step⟩` and "do not decrease the margin by
more than ε" becomes

    ⟨−v_i, step⟩ ≤ ε_i .

Every region constraint is stored in this `⟨a_c, step⟩ ≤ ε_c` form with
`a_c = −v̄_c`, so the region is a strict generalisation of
`NoDecreaseRegion` (one half-space, `ε = 0`) to `k` half-spaces with
certainty-dependent slack.

## Per-step algorithm (`project`)

1. Forward the whole gate set: logits `(N, K)` → entropy `H_i`, margin
   `m_i`, runner-up `r_i`, correctness `c_i`, stability
   `s_i = c_i·σ(−β(H_i − H_mem))`.
2. Select the memorized set `M`: the top-`max_constraints` samples by `s_i`
   with `s_i > s_min` via `lax.top_k` (fixed size ⇒ static shapes; samples
   padded in with weight 0 contribute nothing to any region).
3. Margin gradients `v_i = ∇_θ m_i` via `vmap(grad)` on the flat parameter
   vector; optional `param_mask` for head-only constraints.
4. Per-sample slack `ε_i = κ(1 − s_i)‖v_i‖`.
5. Region assignment by `policy`:
   `mean` (k=1) · `class` (k=K) · `pair` (k=K², region = y·K + r) ·
   `gradient` (weighted spherical k-means on `v_i`, centroids warm-started
   from state) · `sample` (k=|M|) · `random` (fixed random partition).
6. Aggregate: `s`-weighted means `v̄_c`, `ε̄_c`, validity mask
   (non-empty regions), dispersion radius
   `σ_c = max_{i∈R_c}‖v_i − v̄_c‖`; tighten
   `ε̄_c ← max(0, ε̄_c − ζ σ_c ‖step‖)` (Prop. 2 leakage bound).
7. Dual NNLS `min_{λ≥0} ½λᵀGλ − λᵀ(b − ε̄)` with `G = AAᵀ`, `b = A·step`,
   solved by Hildreth's projected Gauss–Seidel (`dual_sweeps` sweeps; exact
   after one sweep when k=1). Invalid regions are masked to the identity
   so their `λ_c = 0`.
8. `s* = step − Aᵀλ = step + Σ_c λ_c v̄_c`.
9. Dead-zone guard (`dead_zone_ratio = δ > 0`): if `‖s*‖ < δ‖step‖`,
   re-solve with `ε̄_c + relax·‖v̄_c‖‖step‖` and take that instead
   (`guard = 1`). If the *relaxed* solve is still inside the dead zone and
   `dead_zone_floor` is set, emit `δ·step` — a shortened, unprojected step —
   so the line search is never handed a ~zero direction (`guard = 2`).
   Constraint satisfaction is traded for progress only where the region
   would otherwise stop the optimizer entirely.
10. Warmup: while `step_count < warmup_steps` the whole channel (steps 1–9)
    is skipped with `lax.cond` and the raw step is returned. The recorder
    sees a `ratio = cos = 1`, all-else-zero row for those steps.

## `update`

Recomputes the gate at the accepted iterate for diagnostics
(`n_memorized`, `n_objective`, `mean_entropy`, `train_accuracy`) and, for
the `gradient` policy, refreshes the centroids every `refresh_every`
accepted steps. The refresh (which needs `|M|` fresh margin gradients) is
inside a `lax.cond`, so non-refresh steps do not pay for it.

## Diagnostics (`ProjectionRecorder`)

One host callback per projection with `‖s*‖/‖step‖`, `cos(s*, step)`,
`#{λ_c > 0}`, `#valid`, `|M|`, `λ_max`, the guard code, the max primal
residual, and the mean `‖v̄_c‖` / `ε̄_c`. The two means are over *valid*
regions only — averaging over all `k` slots would under-report by
`#valid / k` for `class`, `pair`, `random` and `gradient`. Invariants worth
asserting in tests: `cos ≥ ‖s*‖/‖step‖ > 0` (Euclidean projection onto a
convex set containing 0) and `residual ≈ 0` (dual converged).

## Objective channel

`make_gated_loss` builds `Σ α_i ℓ_i / Σ α_i` with `α_i = 1 − s_i` held
constant under differentiation — the entropy hinge of §3.2. Pair it with
the region for the full method; use either alone for the §7.4 ablations.

## Cost

One extra forward over the gate set plus `|M|` per-sample gradients per
projection (`O(|M|·P)`), a `(k×k)` Gram and `dual_sweeps·k²` dual work.
Bound `|M|` with `max_constraints`; restrict `P` with `param_mask`; cap peak
activation memory of the per-sample gradients with `grad_chunk`. Warmup
steps and non-refresh `update` calls are free (see above). Hildreth's
sweep is a sequential `fori_loop` over the `k` coordinates, so the
`sample` policy (`k = |M|`) is dominated by `dual_sweeps·k` tiny
dependent ops rather than by FLOPs.
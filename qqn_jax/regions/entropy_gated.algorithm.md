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
   with `s_i > s_min` (fixed size ⇒ static shapes).
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
9. Dead-zone guard: if `‖s*‖ < δ‖step‖`, re-solve with
   `ε̄_c + relax·‖v̄_c‖‖step‖` and take that instead.
10. Warmup: return the raw step while `step_count < warmup_steps`.

## `update`

Recomputes the gate at the accepted iterate for diagnostics
(`n_memorized`, `n_objective`, `mean_entropy`, `train_accuracy`) and, for
the `gradient` policy, refreshes the centroids every `refresh_every`
accepted steps.

## Objective channel

`make_gated_loss` builds `Σ α_i ℓ_i / Σ α_i` with `α_i = 1 − s_i` held
constant under differentiation — the entropy hinge of §3.2. Pair it with
the region for the full method; use either alone for the §7.4 ablations.

## Cost

One extra forward over the gate set plus `|M|` per-sample gradients per
projection (`O(|M|·P)`), a `(k×k)` Gram and `dual_sweeps·k²` dual work.
Bound `|M|` with `max_constraints`; restrict `P` with `param_mask`.
# Point-History Storage and Secant-View Algorithm

## Overview

This document describes the algorithm implemented in `point_history.py`,
a module that centralizes the point-bookkeeping logic shared across
quasi-Newton (QN) oracles. The design replaces per-oracle duplication of
probe-ordering and secant-anchoring logic with two composable, JIT-friendly
layers built on immutable `NamedTuple` array containers.

## Motivation

Historically each QN oracle re-implemented:

* ordering of line-search probes by step size `alpha`,
* masking of invalid probes,
* capping the number of collinear probes folded into an update
  (the old `_ordered_probe_secants` helper),
* anchoring / differencing to derive curvature pairs `(s, y)`.

Consolidating these into a single module yields one source of truth,
reduces bugs, and keeps the numerics consistent between oracles.

## Architecture: Two Layers

### Layer 1 — `PublishedPoints` (raw ordered buffer)

A plain `NamedTuple` of arrays representing an ordered batch of measured
points. Invariants:

* All sequences run **oldest-first**, i.e. increasing line-search `alpha`.
* The **accepted iterate is always the final entry** and always valid.
* An **anchor** point (`anchor_params`, `anchor_grad`) captures the iterate
  *before* the step (`d'(0)` anchor), used to form the first delta.

Fields:

| Field           | Shape    | Meaning                                    |
|-----------------|----------|--------------------------------------------|
| `params_seq`    | `(k, n)` | iterates                                   |
| `grad_seq`      | `(k, n)` | gradients at those iterates                |
| `alpha_seq`     | `(k,)`   | line-search step sizes (accepted last)     |
| `valid_seq`     | `(k,)`   | boolean validity mask                      |
| `anchor_params` | `(n,)`   | iterate before the step                    |
| `anchor_grad`   | `(n,)`   | gradient at the anchor                     |

### Layer 2 — `SecantStoreView` (cached derived view)

A thin, **cached** view over one `PublishedPoints` batch that computes the
curvature information oracles consume, exactly once:

* **Chained (pairwise-local) secants** `s_i`, `y_i`
* **Anchored secants** `Δx_i = x_t − x_i`, `Δg_i = ∇f_t − ∇f_i`
* The **newest accepted** `(params, grad)` and its secant pair.

Because the derived arrays are stored, an oracle folding them into state
pays the anchoring / differencing cost only once.

## Algorithm: `publish(info, max_replay)`

The single source of truth for turning the solver's raw probe buffers into
an ordered, validity-masked batch.

### Steps

1. **Guard.** If any of `probe_params`, `probe_alphas`, `probe_valid` is
   `None`, return `None`. This signals the caller to fall back to a cheap
   accepted-point-only update built directly from `info`.

2. **Replay capping (optional).** When `max_replay` is provided, keep only
   the probes *closest to the accepted step* — those with the largest
   `alpha` among valid probes:

   * Rank invalid probes to `-inf` so they sort last.
   * `argsort(-ranked_alpha)` in descending order; take the first
     `n_keep = min(max_replay, k)` indices.
   * Subselect params/grads/valid/alphas by those indices.

   This bounds how many collinear probes fold into an update.

3. **Inner ordering.** Re-sort the (possibly capped) kept probes by
   ascending `alpha`, pushing invalid probes to the end
   (`where(valid, alpha, +inf)`), producing oldest-first order.

4. **Append accepted iterate.** Concatenate the accepted point
   (`new_params`, `new_grad`) as the final entry with `valid = True`.
   The accepted step size uses `info.step_size` (defaulting to `1.0`),
   cast to the probe alpha dtype.

5. **Emit.** Return a `PublishedPoints` with the solver's pre-step
   `info.params` / `info.grad` as the anchor.

## Algorithm: `secant_view(points)`

Derives and caches all secant flavors.

1. **Anchor-prefix** the params and grads:
   `anchored_p = [anchor_params, *params_seq]` (similarly for grads).
2. **Chained deltas** — consecutive differences:
   `deltas = anchored_p[1:] − anchored_p[:-1]` and likewise `gdeltas`.
   These are the pairwise-local secants including the anchor→first-point
   step.
3. **Anchored displacements** relative to the newest accepted point:
   `anch_dx = x_new − x_i`, `anch_dg = ∇f_new − ∇f_i`.

## Algorithm: `SecantStoreView.newest_secant()`

Returns `(s, y)` for the accepted step versus the **most-recent valid**
preceding point (a chained BB1-style secant).

1. Form shifted anchor sequences that prepend the batch anchor:
   `anchor_p = [anchor_params, *params_seq[:-1]]` (grads analogously).
2. Look at the validity of preceding points `valid_seq[:-1]`.
3. Select the **largest index** whose preceding point is valid
   (`max(where(prev_valid, arange, 0))`), i.e. the most recent valid
   predecessor.
4. Compute `s = params_seq[-1] − p_prev`, `y = grad_seq[-1] − g_prev`.

## Design Rationale & Discussion

* **JIT-friendliness.** All containers are `NamedTuple`s of arrays and all
  operations are functional (`jnp` ops, no Python mutation), so the objects
  compose cleanly with `jax.lax` control flow. There is no in-place state.

* **Masking vs. dynamic shapes.** Rather than dropping invalid probes
  (which would produce data-dependent shapes and break `jit`), the module
  keeps fixed-size arrays and encodes validity via `valid_seq` plus sentinel
  values (`±inf`) during sorting/selection.

* **Single accepted-point fallback.** Returning `None` from `publish` when
  probes are unpopulated cleanly separates the batched path from the trivial
  single-secant path, keeping callers simple.

* **Separation of concerns.** `publish` owns ordering/masking/capping;
  `secant_view` owns differencing/anchoring; oracles only request the flavor
  of secant they need. This eliminates duplicated, drift-prone logic.

## Potential Impacts & Follow-ups

* Oracles previously using `_ordered_probe_secants` should migrate to
  `publish` + `secant_view`.
* `publish` depends on the `OracleInfo` fields: `probe_params`,
  `probe_grads`, `probe_alphas`, `probe_valid`, `new_params`, `new_grad`,
  `step_size`, `params`, `grad`. Changes to that contract must be reflected
  here.
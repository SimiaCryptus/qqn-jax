---
related:
  - algorithm.md
  - positioning.md
  - conclusions.md
  - future_work.md
---

# The Ideal Problem for QQN

QQN's guarantees are **contingent**, not universal. The assumptions behind its
convergence claims do more than qualify the math — they *define the class of
problems on which QQN is the right tool*. This document states those
assumptions explicitly and translates them into a practical profile of the
ideal problem.

## The Assumptions Behind the Guarantees

QQN's global-convergence, superlinear-convergence, and descent guarantees (see
[`algorithm.md`](algorithm.md)) all rest on the following premises:

1. **Smooth objective with Lipschitz gradient.** The along-path directional
   derivative `⟨∇f, d'(0)⟩ = -‖∇f‖²` and the sufficient-decrease argument
   assume `f ∈ C¹` with a (locally) Lipschitz-continuous gradient. Sharp kinks
   (e.g. ReLU networks, L1 terms) violate the smoothness premise; QQN can still
   make progress, but the formal descent argument is then inherited *only* from
   the line search's sufficient-decrease test, not from path geometry.

2. **A line search that actually attains sufficient decrease.** The default
   `armijo`/`backtracking` search is *inexact and finite* (`max_iter`). The
   claim "a valid decreasing step always exists" is true in the limit because
   `d'(0) = -∇f`, but it is **not operationally guaranteed** within a fixed
   iteration budget. On pathological curvature the search can exhaust its
   budget without satisfying Armijo. Ideal problems leave enough headroom that
   the inexact search reliably succeeds.

3. **Reliable, low-noise curvature information.** The L-BFGS oracle (and any
   history-based oracle) assumes the curvature pairs `(s, y)` describe a
   stable, deterministic function. **Re-sampled / mini-batched gradients
   invalidate the secant history**, degrading the `t = 1` endpoint. Ideal
   problems are **full-batch or low-noise** so curvature memory stays
   trustworthy.

4. **Feasible-path smoothness (when regions are active).** Projective regions
   can make `d_R(t)` discontinuous — the Orthant region is explicitly
   non-smooth at sign flips. The guarantees then hold on the *feasible*
   projected path, and again rest on the line search's sufficient-decrease
   check tolerating those discontinuities, rather than on a smooth-path
   argument. A rigorous descent-preserving proof for the projected
   non-smooth case is **open work**, not a delivered guarantee.

## The Resulting Problem Profile

Combining the above, QQN is at its best on problems that are:

- **Smooth** (`C¹`, ideally `C²`), so the path tangent and curvature model are
  meaningful.
- **Full-batch or low-noise / deterministic**, so curvature memory and an
  exact-ish line search are trustworthy.
- **Ill-conditioned or anisotropically curved**, so genuine second-order
  information pays off and a fixed step rule struggles — this is where QQN's
  adaptive blend earns its keep.
- **Affordable to line-search**, i.e. each `f` (and `∇f`) evaluation is cheap
  enough that walking the path with a robust search is worthwhile.

## When the Profile Breaks

Each violated assumption pushes QQN out of its sweet spot:

| Violated assumption          | Consequence                              | Preferred alternative |
|------------------------------|------------------------------------------|-----------------------|
| Smoothness (kinks)           | Descent only via line-search check       | still QQN, but expect cheaper searches to win; revisit guarantees |
| Sufficient-decrease attained | Search may exhaust budget                | shorten steps / stronger search / smaller problem |
| Low-noise curvature          | `(s, y)` history corrupted, oracle weak  | **Adam / SGD**        |
| Feasible-path smoothness     | Discontinuous `d_R(t)`                   | fixed-radius regions; rely on line search |

These are the precise conditions probed by the proposed benchmarks in
[`future_work.md`](future_work.md): each entry there is a test of one of the
assumptions above.

## See Also

- [`algorithm.md`](algorithm.md) — the guarantees and their footnotes.
- [`positioning.md`](positioning.md) — where QQN fits relative to Adam/L-BFGS.
- [`future_work.md`](future_work.md) — benchmarks that stress each assumption.
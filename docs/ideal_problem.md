---
related:
  - algorithm.md
  - positioning.md
  - conclusions.md
  - future_work.md
---

# The Ideal Problem for QQN

QQN's guarantees are **contingent**, not universal — but the contingency is
*milder* than it first appears. The strongest-sounding premises (smoothness,
Lipschitz gradients, exact line minimization) sharpen the *math* and benefit
the *components* QQN composes, but the **hybrid algorithm itself needs only
`C⁰` continuity** to make monotone progress. This document separates what QQN
*requires* from what merely *helps*, and translates the distinction into a
practical profile of the ideal problem.

## What QQN Actually Requires vs. What Merely Helps

A central and easily-missed point: QQN's *operational* requirement is far
weaker than the assumptions used to prove its sharpest convergence rates.

- **The hybrid algorithm needs only zero-order continuity.** Progress is
  enforced by the line search's **sufficient-decrease test** on the value
  `f(x + d(t))`. That test is a comparison of function *values*, so as long as
  `f` is continuous (`C⁰`) along the path — even with kinks, ridges, or
  non-differentiable seams — a small enough step on the gradient-dominated end
  of the path (`d'(0) = -∇f` wherever the gradient exists) yields decrease.
  QQN does **not** need a globally smooth landscape to descend.

- **More regularity improves the *components*, not the *requirement*.** Extra
  smoothness makes the L-BFGS **oracle** more trustworthy (its secant pairs
  `(s, y)` describe a stabler curvature model) and lets stronger **line
  searches** (strong-Wolfe, exact minimization) realize faster rates. These
  are *upgrades to the substrategies*, not preconditions for the hybrid to
  function. Strip them away and QQN degrades gracefully toward gradient
  descent rather than failing.

This is already **more permissive than many deep-learning problem
formulations provide**: ReLU networks, max-pooling, hinge/L1 terms, and
clipping all produce merely-continuous (or piecewise-smooth) objectives that
*violate* the textbook `C²` premise yet still satisfy QQN's actual `C⁰`
requirement on the path.

## A Counter-Intuitive Sweet Spot: Useful-but-Unreliable Curvature

The most interesting region for QQN may **not** be the pristine, globally
smooth, full-batch ideal. It is the *messier middle*: landscapes that are
**only weakly smooth**, where second-order structure is **informative but
unreliable** — locally meaningful curvature that cannot be trusted globally.

Here QQN's architecture is uniquely suited:

- **Curvature is worth exploiting** — the oracle's `t = 1` endpoint captures
  genuine anisotropy that a fixed step rule would mishandle.
- **Curvature cannot be trusted blindly** — and QQN *never* trusts it blindly.
  The path's `d'(0) = -∇f` anchor and the sufficient-decrease check
  automatically **retreat toward the gradient** whenever the oracle direction
  fails to pay off. An unreliable oracle costs at most a short line search, not
  divergence.

In other words, the very property that breaks pure quasi-Newton methods —
curvature that is locally helpful but globally untrustworthy — is the regime
QQN was *built* to absorb. Methods that commit to the `t = 1` step (standalone
L-BFGS, Newton) are brittle here; pure first-order methods (SGD) leave the
available curvature signal on the table. QQN's geometric blend sits exactly in
between.

## The Assumptions, Re-stated as a Hierarchy

| Assumption                                                   | Status for QQN           | Effect when violated                                                                                 |
|--------------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------|
| `C⁰` continuity along the path                               | **Required**             | Sufficient-decrease test is meaningless; descent not guaranteed                                      |
| (Sub)gradient exists at iterates                             | **Required**             | No path tangent `d'(0)`; cannot form `d(t)`                                                          |
| A line search that attains sufficient decrease within budget | **Strongly preferred**   | Search may exhaust `max_iter` on pathological curvature                                              |
| Lipschitz / `C¹` gradient                                    | **Helps the rate proof** | Formal descent inherited *only* from the line-search check, not path geometry                        |
| `C²` / low-noise curvature                                   | **Helps the oracle**     | `(s, y)` history less trustworthy; oracle weaker, but still safely overridden by the gradient anchor |
| Feasible-path smoothness (regions active)                    | **Helps; open theory**   | Discontinuous `d_R(t)`; descent rests on the line-search check (rigorous proof is open work)         |

Read top-to-bottom, the table is a **ladder of regularity**: only the first
two rungs are load-bearing for *progress*; the rest govern *speed* and
*oracle quality*.

## The Resulting Problem Profile

Combining the above, QQN is at its best on problems that are:

- **At least `C⁰` along the path**, ideally piecewise-smooth — enough for the
  sufficient-decrease test to bite. Full global `C²` smoothness is a *bonus*,
  not a prerequisite.
- **Ill-conditioned or anisotropically curved**, so genuine second-order
  information pays off and a fixed step rule struggles — this is where QQN's
  adaptive blend earns its keep.
- **Endowed with curvature that is locally useful even if globally
  unreliable**, since QQN exploits it opportunistically and falls back to the
  gradient anchor when it misleads.
- **Full-batch or low-noise / deterministic** *enough* that the oracle's
  curvature memory carries signal — though QQN tolerates more noise than pure
  L-BFGS because the gradient anchor and line search guard every step.
- **Affordable to line-search**, i.e. each `f` (and `∇f`) evaluation is cheap
  enough that walking the path with a robust search is worthwhile.

## When the Profile Breaks

Each *load-bearing* violation pushes QQN out of its sweet spot; the
*helps-only* violations merely slow it down or weaken the oracle.

| Violated assumption          | Severity  | Consequence                          | Mitigation / alternative                          |
|------------------------------|-----------|--------------------------------------|---------------------------------------------------|
| `C⁰` continuity on the path  | **Fatal** | Sufficient-decrease test meaningless | restore continuity; smooth the seam               |
| No (sub)gradient available   | **Fatal** | Cannot form the path tangent         | derivative-free method                            |
| Sufficient-decrease attained | Serious   | Search may exhaust budget            | shorten steps / stronger search / smaller problem |
| Lipschitz / `C¹` gradient    | Mild      | Rate proof weakens; still descends   | still QQN; expect cheaper searches to win         |
| Low-noise curvature          | Mild      | Oracle weaker, but auto-overridden   | still QQN; or **Adam / SGD** if noise dominates   |
| Feasible-path smoothness     | Mild/open | Discontinuous `d_R(t)`               | fixed-radius regions; rely on line search         |

The key revision from a naïve reading: **only the first two rows are truly
disqualifying.** The classic "QQN needs smoothness" caution applies to its
*rate guarantees and oracle quality*, not to its ability to make monotone
progress.

These are the precise conditions probed by the proposed benchmarks in
[`future_work.md`](project/future_work.md): each entry there is a test of one of the
assumptions above — and several deliberately target the *useful-but-unreliable
curvature* regime where QQN should shine.

## See Also

- [`algorithm.md`](theory/algorithm.md) — the guarantees and their footnotes.
- [`positioning.md`](positioning.md) — where QQN fits relative to Adam/L-BFGS.
- [`future_work.md`](project/future_work.md) — benchmarks that stress each assumption.
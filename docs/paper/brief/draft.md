# Brief: One Simple Change to L-BFGS

## The Direction Dilemma

Unconstrained smooth optimization has long been split between two families
of methods. First-order methods — gradient descent, momentum — use only
the gradient `∇f(x)`. They are cheap, memory-light, and *robust*: the
negative gradient is always a descent direction. But they crawl on
ill-conditioned problems whose loss landscapes form long, narrow valleys.
Quasi-Newton methods — most prominently L-BFGS — use curvature information
`H ≈ ∇²f⁻¹` to take larger, better-aimed steps. They converge *fast*
(superlinearly near a minimum) but are *fragile*: the direction `-H∇f`
is only guaranteed to descend when `H` is positive-definite, a condition
that fails on non-convex objectives or when the curvature history is stale
or degenerate.

The classical reconciliation is to **pick one direction and then run a
line search** along it. If the quasi-Newton direction is good, the line
search accepts a full step; if it is bad, the search backtracks. But this
still commits to a *single* direction per iteration. When the oracle
direction is poor, a backtracking search along it can waste evaluations
without ever exploring the reliable gradient direction.

## The Minimal Enhancement

We propose refusing the binary choice. Instead of selecting either the
gradient or the quasi-Newton direction, construct a **continuous quadratic
path** that smoothly connects them:

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

This single curve has three decisive properties:

- `d(0) = 0` — the path starts at the current iterate `x`.
- `d'(0) = -∇f` — the path *begins* tangent to steepest descent, so it is
  guaranteed to decrease `f` for small `t` whenever `∇f ≠ 0`.
- `d(1) = -H∇f` — the path *ends* exactly at the quasi-Newton step.

A permissive line search then walks `t ∈ [0, 1]` directly. Each probe
`x + d(t)` is a *state* on the curve, not a direction to be re-scaled by a
separate inner search — the curve itself is the one-dimensional search
space. Near `t = 0` the path *is* gradient descent (robustness); near
`t = 1` it *is* the quasi-Newton step (speed).

That is the entire change. No new hyperparameters beyond a parametric
bound and the line search's own tolerances; no restructuring of L-BFGS's
curvature machinery. The two-loop recursion still supplies `-H∇f`; we
simply stop feeding that direction straight into a line search and instead
feed it into the `t = 1` endpoint of a parabola.

## Why It Works

The reframing turns *"which direction?"* into *"where on the curve?"*.
The tangent anchor `d'(0) = -∇f` guarantees that a valid decreasing step
always exists along the path for sufficiently small `t`, regardless of how
poor the oracle direction is — so global convergence is inherited from
steepest descent. Meanwhile the endpoint `d(1) = -H∇f` keeps the fast
quasi-Newton step reachable, so superlinear behavior returns near the
optimum when the oracle direction dominates (the selected `t` approaches
`1`). The line search discovers the right blend automatically, with no
manual tuning. Because acceptance rests only on a sufficient-decrease test
of function *values*, the argument requires only `C⁰` continuity along the
path — so it holds up even on piecewise-smooth objectives such as ReLU
networks.

## Headline Result

On a 4-layer `tanh, gelu, tanh` MLP (335k parameters) trained full-batch
on Fashion-MNIST, this simple variant — a deep-memory L-BFGS oracle on the
quadratic path with a permissive line search — decisively wins both the
iteration race *and* wall-clock time to the `2e-2` loss target. It
achieves a **2.64× iteration speedup over standalone L-BFGS** while also
being cheaper per iteration (16.08 ms/it vs 20.71 ms/it). The Pareto
frontier (loss vs. time) is entirely occupied by the quadratic-path
variant; standalone L-BFGS is dominated, and the first-order baselines
(SGD, Adam) exhaust their budgets well short of the target.

Two features of the result deserve emphasis. First, **the speedup widens
as the target tightens**: the second-order advantage compounds in the
fine-tuning regime, exactly where first-order baselines stall. Second, and
more surprising, **this simplest variant beats the elaborate ones**. We
built and benchmarked a large cross-product of enhancements — cubic-Hermite
splines that reuse every probe's gradient, trust regions and box
constraints, exotic oracles (momentum, Adam, secant, Anderson, Shampoo),
stochastic temperature acceptance, per-layer partitioning — and none of
them beat the plain quadratic path with a deep L-BFGS oracle and a
permissive line search. The minimal change is not merely competitive; it
is the strongest configuration we found.

## What Follows

The rest of this paper generalizes and justifies this one move. The
**Theory** part shows that the quadratic path is not an arbitrary curve
but the simplest object satisfying its endpoint/tangent constraints, and
reframes optimization as four orthogonal, independently swappable axes —
gradient, oracle, search, region — of which classical methods (L-BFGS,
Newton, momentum, OWL-QN, trust region) are special cases. The
**Methodology** part details the fairness invariants that make the
headline comparison trustworthy. The **Results** part unpacks the full
empirical story across problems and activation functions. But the essential
idea is already complete: once you stop asking "which direction?" and start
asking "where on the curve?", the straight line stops looking like a law of
nature and starts looking like a hard-coded default nobody refactored.
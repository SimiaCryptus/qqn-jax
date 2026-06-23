# Positioning: Where QQN Fits

## TL;DR

QQN is **not** a drop-in replacement for Adam or L-BFGS on every problem. It is
a **configurable geometric solver framework** — a principled, largely
parameter-free optimizer whose value compounds on **ill-curved, anisotropic
landscapes** where naïve direction choices stall, oscillate, or diverge. For
everyday large-scale stochastic training, **Adam remains faster per step and
more memory efficient**. QQN earns its keep when curvature structure matters
and when you want a single framework that can *become* the right classical
method for the problem at hand — with globalization thrown in for free.

---

## What QQN Is

QQN is a **combiner** of four orthogonal components — *gradient*, *oracle*,
*search*, and *region* (see [`algorithm.md`](theory/algorithm.md)). Rather than
committing to a hand-tuned step rule, QQN constructs a quadratic interpolation
path

```
d(t) = t(1 - t)·(-∇f) + t²·(-H∇f),   t ∈ [0, 1]
```

between the steepest-descent direction (the `t → 0` tangent) and a
curvature-aware oracle direction (the `t = 1` endpoint), then lets a line
search *discover* the right blend by walking the path directly. The direction
is not chosen by a heuristic schedule — it **emerges** from the geometry of the
path combined with the sufficient-decrease guarantee of the search.

A key consequence is that QQN introduces **no hyperparameters of its own
beyond those of the components it composes**. There is no global learning rate
to sweep, no `β₁/β₂` schedule, no warmup added by QQN itself. The balance
between first- and second-order behavior is selected *per iteration* by the
line search rather than fixed in advance.

---

## What QQN Is Not

- **Not a faster Adam.** For general-purpose, high-dimensional *stochastic*
  optimization (typical deep-learning training loops), **Adam is still faster
  per step and more memory efficient**. QQN's per-iteration cost (oracle
  direction + path evaluation + line search) and its L-BFGS history (`O(m·n)`)
  make it a poor fit when gradients are noisy and curvature memory is
  unreliable.

  > **No contradiction with the benchmark wins.** Where QQN variants have won a
  > *majority* of benchmark problems, those benchmarks were **smooth and
  > full-batch** — the convex/deterministic regime, not Adam's noisy/stochastic
  > home turf. Winning deterministic benchmarks and losing to Adam on noisy
  > minibatch training are claims about *different problem classes*.

- **Not a single algorithm.** QQN is a *configuration space*, not one method.
  Fixing one or two of its axes to canonical choices reproduces L-BFGS, Newton,
  momentum, Barzilai-Borwein, trust-region, OWL-QN, and projected-gradient
  methods (see [`equivalences.md`](theory/equivalences.md)). Its identity is the
  *framework*, not any single corner of it.

- **Not dependent on global smoothness.** A common misreading is that QQN
  "requires" a `C²` landscape. The hybrid algorithm needs only `C⁰` continuity
  *along the path* to make monotone progress; smoothness sharpens the rate
  proofs and strengthens the oracle, but is not a precondition for descent (see
  [`ideal_problem.md`](ideal_problem.md)).

---

## Where QQN Pays Off

QQN's advantage is **concentrated, not diffuse**. It pays off mainly on
landscapes where:

- **Curvature is strongly anisotropic or ill-conditioned**, so a fixed step
  rule either crawls through narrow valleys or diverges on steep walls.
- **The right amount of "second-order aggressiveness" varies** across the
  landscape, making any hand-tuned schedule brittle.
- **Curvature is useful but unreliable** — locally informative yet not globally
  trustworthy. This *messier middle* is QQN's true sweet spot: the path exploits
  curvature opportunistically and retreats to the gradient anchor whenever the
  oracle misleads, so an unreliable oracle costs at most a short line search
  rather than divergence (see [`ideal_problem.md`](ideal_problem.md)).
- **A robust line search is affordable** — smooth, full-batch, or low-noise
  objectives where each `f` (and `∇f`) evaluation is cheap enough to walk the
  path reliably.
- **You want globalization for free.** Because `d'(0) = -∇f`, every
  configuration contains gradient descent as its `t → 0` limit, so even an
  aggressive or poorly-conditioned oracle stays globally convergent.

In these regimes the geometric path lets QQN transition automatically between
conservative gradient steps and aggressive quasi-Newton steps *without* manual
tuning — precisely where Adam's fixed adaptive-moment heuristics and L-BFGS's
fixed `t = 1` commitment both struggle.

---

## Decision Guide

| Situation                                                      | Prefer           |
|----------------------------------------------------------------|------------------|
| Large-scale, noisy, stochastic minibatch training              | **Adam**         |
| Tight memory budget, very high dimension                       | **Adam / SGD**   |
| Smooth, full-batch, ill-conditioned objective                  | **QQN**          |
| Complex / anisotropic curvature where step tuning is brittle   | **QQN**          |
| Curvature that is locally useful but globally unreliable       | **QQN**          |
| You want a parameter-free, self-tuning blend of GD and L-BFGS  | **QQN**          |
| You need one framework that can *become* L-BFGS, TR, OWL-QN, … | **QQN**          |
| Bound / orthant / trust constraints alongside curvature        | **QQN + region** |

---

## Summary

QQN is a **principled geometric optimizer** and a **configurable solver
framework**. Its defining virtue is that the gradient/oracle blend is
*discovered geometrically* rather than *tuned numerically*, making it
effectively parameter-free apart from the substrategies it composes. It does
not aim to beat Adam on raw speed or memory for general stochastic problems —
Adam wins there. QQN's value emerges on **ill-curved, anisotropic
landscapes**, and especially in the regime of *useful-but-unreliable
curvature*, where its adaptive, globally-convergent path turns hard-to-tune
second-order behavior into a free byproduct of the geometry.

---

## See Also

- [`algorithm.md`](theory/algorithm.md) — the full QQN algorithm and its four axes.
- [`equivalences.md`](theory/equivalences.md) — classical optimizers as QQN special
  cases.
- [`ideal_problem.md`](ideal_problem.md) — what QQN actually requires vs. what
  merely helps.
- [`oracles.md`](theory/oracles.md) — the `t = 1` endpoint sources.
- [`regions.md`](theory/regions.md) — projective constraints and structure.
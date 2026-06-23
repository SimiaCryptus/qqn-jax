# Positioning: Where QQN Fits

## TL;DR

QQN is **not** a drop-in replacement for Adam or L-BFGS on every problem. It is
a **configurable geometric solver framework** — a principled, largely
parameter-free optimizer whose value compounds on **highly complex, ill-curved
manifolds** where naïve direction choices stall or oscillate. For everyday
large-scale stochastic training, **Adam remains faster and more memory
efficient**. QQN earns its keep when curvature structure matters and when you
want one framework that can *become* the right classical method for the problem
at hand.

---

## What QQN Is

QQN is a **combiner** of four orthogonal components — *gradient*, *oracle*,
*search*, and *region* (see [`algorithm.md`](algorithm.md)). Rather than
hand-tuning a step rule, QQN constructs a quadratic interpolation path

```
d(t) = t(1 - t)·(-∇f) + t²·(-H∇f),   t ∈ [0, 1]
```

between the steepest-descent direction (`t → 0`) and a curvature-aware oracle
direction (`t = 1`), and lets a line search *discover* the right blend by
walking the path directly. This makes QQN a **principled geometric optimizer**:
the direction is not picked by a heuristic schedule but emerges from the
geometry of the path plus the sufficient-decrease guarantee of the search.

A key consequence is that QQN is, **except for its substrategies,
free of *additional* hyperparameters beyond those of the components it
composes**. There is no global learning rate to sweep, no `β₁/β₂` schedule,
no warmup introduced by QQN itself. The blend between
first- and second-order behavior is selected per-iteration by the line search
rather than fixed in advance.

---

## What QQN Is Not

- **Not a faster Adam.** For general-purpose, high-dimensional *stochastic*
  optimization (e.g. typical deep-learning training loops), **Adam is still
  faster per step and more memory efficient**. QQN's per-iteration cost
  (oracle direction + path evaluation + line search) and its L-BFGS history
  (`O(m·n)`) make it a poor fit when gradients are noisy and curvature memory
  is unreliable. (Where QQN *has* won a majority of benchmark problems, those
  benchmarks were smooth and full-batch — the convex/deterministic regime,
  not Adam's noisy/stochastic home turf. The two claims do not conflict.)
- **Not a single algorithm.** QQN is a *configuration space*. Fixing one or two
  of its axes to canonical choices reproduces L-BFGS, Newton, momentum,
  Barzilai-Borwein, trust-region, OWL-QN, and projected-gradient methods (see
  [`equivalences.md`](equivalences.md)). Its identity is the framework, not any
  one corner of it.

---

## Where QQN Pays Off

QQN's advantage is concentrated, not diffuse. It pays off mainly in
**highly complex curvature manifolds** — problems where:

- Curvature is strongly anisotropic or ill-conditioned, so a fixed step rule
  either crawls or diverges.
- The right amount of "second-order aggressiveness" varies across the
  landscape, so a hand-tuned schedule is brittle.
- A robust, *exact-ish* line search is affordable (smooth, full-batch or
  low-noise objectives), letting the path blend be discovered reliably.
- You want **globalization for free**: because `d'(0) = -∇f`, every
  configuration contains gradient descent as its `t → 0` limit, so even an
  aggressive or poorly-conditioned oracle stays globally convergent.

In these regimes the geometric path lets QQN automatically transition between
conservative gradient steps and aggressive quasi-Newton steps *without* manual
tuning — exactly where Adam's fixed adaptive-moment heuristics and L-BFGS's
fixed `t = 1` commitment both struggle.

---

## Decision Guide

| Situation                                                        | Prefer            |
|------------------------------------------------------------------|-------------------|
| Large-scale, noisy, stochastic minibatch training               | **Adam**          |
| Tight memory budget, very high dimension                        | **Adam / SGD**    |
| Smooth, full-batch, ill-conditioned objective                   | **QQN**           |
| Complex / anisotropic curvature where step tuning is brittle    | **QQN**           |
| You want a parameter-free, self-tuning blend of GD and L-BFGS   | **QQN**           |
| You need one framework that can *become* L-BFGS, TR, OWL-QN, …  | **QQN**           |
| Bound / orthant / trust constraints alongside curvature         | **QQN + region**  |

---

## Summary

QQN is a **principled geometric optimizer** and a **configurable solver
framework**. Its defining virtue is that the gradient/oracle blend is
*discovered geometrically* rather than *tuned numerically*, making it
effectively parameter-free apart from the substrategies it composes. It does
not aim to beat Adam on raw speed or memory for general problems — Adam wins
there. QQN's value emerges on **highly complex curvature manifolds**, where its
adaptive, globally-convergent path turns hard-to-tune second-order behavior
into a free byproduct of the geometry.

---

## See Also

- [`algorithm.md`](algorithm.md) — the full QQN algorithm and its four axes.
- [`equivalences.md`](equivalences.md) — classical optimizers as QQN special
  cases.
- [`oracles.md`](oracles.md) — the `t = 1` endpoint sources.
- [`regions.md`](regions.md) — projective constraints and structure.